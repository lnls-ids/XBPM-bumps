## Refined GUI Direction
## User Choice: Remove Analyzer

User chooses Option A: delete `xbpm_bumps/ui/analyzer.py`. Preserve asynchronous GUI behavior only if needed by adding a minimal Qt worker later; do not recreate the old analyzer pipeline. Before deletion, verify there are no live imports/instances (current search found none). Objective consequences: the declared but unconnected `analysisRequested` signal no longer has a consumer; analysis currently will not run until `main_window.py` directly invokes a worker/service. All `main_window.py` references to `self.analyzer.app` are invalid and must be replaced as part of reconnecting analysis/export.

## Processor/Data/ROI Cleanup Facts

- Correct data path after HDF5 selection: `self.dataset.beamlinedata[beamline]` -> `BeamlineData`; processor input is `workdata.raw_data.blade_avg`, with `workdata.prm` and runtime `Prm`.
- `XBPMProcessor.__init__` currently accepts `(BladeAvgData, BeamlinePrm, Prm)`. `xbpm_position_calculation` does not receive blade data because the processor retains it in `self.blade_avg`; it should receive only explicit reference positions or, preferably, a typed position/reference input later. A caller must create the processor first and call `processor.xbpm_position_calculation(pos_nom_h, pos_nom_v, ...)`.
- Current processor is not yet dataclass-consistent: many methods still call `self.blade_avg.keys()`/`items()` and expect old dictionary blade arrays, while `BladeAvgData` now contains `nom` and `Blades`; `suppression_matrix` references nonexistent `self.blades_h`/`self.blades_v`; `xbpm_position_calculation` calls `analyze_central_sweeps(show=False)` although the method no longer accepts `show`.
- `_roi_slice_indices` duplicates ROI-bound calculation. Replace it with the selected `BeamlinePrm.roi` after normalizing analysis arrays to 2D. `_extract_roi_slice` is not entirely useless: it performs extraction, but can become a simple `array[roi.sl_v, roi.sl_h]` helper or be removed if all arrays are guaranteed 2D.
- Parameter refresh point: `ParameterPanel` emits `parametersChanged`; `XBPMMainWindow._on_parameters_changed()` reads `roi_v_spin`/`roi_h_spin` and should assign `self.prm.roi = ROISlice.update(self.grid_shape, [roi_v, roi_h])`. The same helper should be called after HDF5 beamline selection and before processor construction. Guard it when `self.workdata` is not loaded.
- `ROISlice.update()` currently computes centered slices, but `ROISlice.__post_init__()` resets them to default slices. This must be corrected before relying on ROI updates; `__post_init__` should not overwrite explicitly supplied slices.

## Current-Code Fact: GUI Chain Is Disconnected

A workspace search found no live `XBPMAnalyzer(...)` construction and no `analysisRequested.connect(...)` or `analysisComplete.connect(...)` wiring in `main_window.py`. The signal is declared and emitted, but no visible current path consumes it. `main_window.py` does successfully load `HDF5DataReader`, select `self.workdata = self.dataset.beamlinedata[self.workbeamline]`, and expose `self.workdata.raw_data.blade_avg` and `self.workdata.prm`.

Therefore the current `self.app` references in `analyzer.py` are fake legacy remnants: `self.app` is never assigned. The `self.analyzer.app` references in `main_window.py` are also invalid legacy assumptions; they must be replaced with explicit `self.workdata`, `self.prm`, and a real analysis result/service reference before export can work. `XBPMAnalyzer` itself is currently not part of the live GUI path.

`XBPMProcessor` is also not correctly connected: its constructor needs `(BladeAvgData, BeamlinePrm, Prm)`, while analyzer code attempts `XBPMP(self.rawdata, self.prm)`. The correct source after selection is `self.workdata.raw_data.blade_avg` plus `self.workdata.prm` plus runtime `Prm`.


The GUI workflows to preserve are HDF5 loading/beamline selection, central sweeps and suppression matrix, XBPM positions/plots, BPM/blade visualizations, and text/HDF5/figure exports. Keep asynchronous execution.

Recommended boundary: `main_window.py` owns Qt widgets, beamline selection, user parameters, canvas updates, and export commands. A small core `AnalysisService` owns typed calculation orchestration: it receives selected `BeamlineData` plus runtime parameters, constructs/reuses `XBPMProcessor`, and returns dataclass-based analysis results. This service is not a replacement UI pipeline. `analyzer.py` should be reduced to a thin Qt worker/presenter that invokes the service and emits typed results, or removed once its signal/thread behavior is relocated. It must not retain legacy step methods, result dictionaries, plotting calls, fake `self.app` state, or export preparation.

**Clarification of service role**

The service is useful because `main_window.py` should not know how to sequence BPM, central sweeps, suppression, and XBPM calculations, while `XBPMProcessor` should not know about Qt, canvases, signals, or file export. The service is the narrow middle layer: typed dataclass inputs in, typed dataclass analysis result out. It does not read HDF5 and does not render figures.

## Plan: Central Sweeps Processor Boundary

Keep `CentralSweeps` as the typed result container, but keep calculation ownership in `XBPMProcessor`. The HDF5 reader should deserialize `BeamlineData` only; an application analysis boundary should instantiate one processor from the selected beamline's `raw_data.blade_avg`, `BeamlineData.prm`, and runtime `Prm`, then request a typed `CentralSweeps` result. `CentralSweeps.compute()` should not silently depend on hidden global or previously-created processor state.

**Steps**
1. Confirm and correct the current contract: `XBPMProcessor` requires `(BladeAvgData, BeamlinePrm, Prm)`, while `XBPMAnalyzer._initialize_and_run_analysis()` currently passes only `(self.app.data, self.app.prm)`. Identify the selected `BeamlineData` and ensure `blade_avg` and beamline parameters come from that object.
2. Move the computation-facing API to the processor boundary: change `XBPMProcessor.analyze_central_sweeps()` to call `central_sweep_h()`/`central_sweep_v()` and return `CentralSweeps` (with `None` for unavailable one-dimensional directions). Preserve `central_sweep_h/v` as instance methods because they depend on `blade_avg`, nominal ranges, and processor parameters.
3. Reduce `CentralSweeps.compute()` to either an assembly helper accepting explicit computed lines (`compute(h=..., v=...)`) or remove it in favor of the processor method. Do not make the dataclass instantiate or discover a processor, and do not use no-argument calls. Remove the top-level processor import from `data_structure.py` once no other dataclass computation requires it, avoiding a domain-model-to-calculator dependency/cycle.
4. Add a small non-Qt application service or analysis-boundary method near `XBPMAnalyzer` that owns processor lifetime for one analysis run: construct `XBPMProcessor(selected_blade_avg, selected_beamline.prm, runtime_prm)`, request central sweeps, and retain the processor only as run-scoped state if later suppression/position calculations need the same inputs. The reader remains responsible only for loading HDF5 into `BeamlineData`.
5. Update `_step_central_sweeps()` to consume the returned `CentralSweeps` object instead of unpacking legacy tuples. Keep visualization and suppression calculations as separate consumers of the typed result during this migration; do not mix export changes into this slice.
6. Add focused tests around the processor boundary: complete 2D data produces both lines; one-dimensional data produces one line and one `None`; the method returns the expected dataclass fields; and processor construction uses `BeamlineData.raw_data.blade_avg` plus `BeamlineData.prm`, not the reader itself. A cheap initial check is a direct construction with a fixture and `processor.analyze_central_sweeps()`.

**Relevant files**
- [xbpm_bumps/core/data_structure.py](xbpm_bumps/core/data_structure.py) — `CentralSweeps`, `CentralSweepLine`, `BeamlineData`, and the currently incorrect processor import/call.
- [xbpm_bumps/core/processors.py](xbpm_bumps/core/processors.py) — `XBPMProcessor.__init__`, `analyze_central_sweeps`, `central_sweep_h`, and `central_sweep_v`.
- [xbpm_bumps/core/reader_hdf5.py](xbpm_bumps/core/reader_hdf5.py) — keep `HDF5DataReader` as the HDF5-to-`BeamlineData` boundary.
- [xbpm_bumps/ui/analyzer.py](xbpm_bumps/ui/analyzer.py) — `_initialize_and_run_analysis`, `_step_central_sweeps`, and selected beamline/data wiring.
- [tests/](tests/) — add or extend narrow central-sweep and processor-construction tests.
- [docs/dataclass_migration_visualization_plan-20260817.md](docs/dataclass_migration_visualization_plan-20260817.md) — source migration intent; exports remain outside this slice.

**Verification**
1. Run the narrow central-sweep tests, including 2D and 1D fixtures.
2. Run a direct processor construction check with `BladeAvgData`, `BeamlinePrm`, and `Prm` and verify `analyze_central_sweeps()` returns `CentralSweeps`.
3. Search for no-argument `XBPMProcessor.central_sweep_h()`/`central_sweep_v()` calls and legacy tuple unpacking of `analyze_central_sweeps()`.
4. Run the existing relevant processor/visualizer tests after the API update; export and full HDF5 migration tests are follow-up scope.

**Decisions**
- Do not instantiate `XBPMProcessor` in `HDF5DataReader`: the reader has no runtime `Prm`, may load multiple beamlines, and should remain reusable as a data loader.
- Instantiate one processor after HDF5 loading and beamline selection, at the analysis service/analyzer boundary. Reuse it for central sweep, position, and suppression calculations during one run when those operations share the same source data and parameters.
- Prefer explicit dependencies over hidden retained state. Retaining a run-scoped processor is fine; making `CentralSweeps.compute()` locate one implicitly is not.
- Keep exports out of this implementation slice, as requested.

**Further Considerations**
1. Decide whether the final API should be `XBPMProcessor.analyze_central_sweeps() -> CentralSweeps` (recommended) or retain a temporary `CentralSweeps.compute(processor)` adapter. The former aligns with the migration plan and avoids coupling dataclasses to processors.
2. Resolve the naming mismatch between `CentralSweeps`/`CentralSweepLine` and older `CentralSweep`/`SweepLine` plan terminology before broad visualizer migration; this slice can preserve current names.
