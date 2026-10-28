## Plan: Make CentralSweep the Visualization Contract

Use `CentralSweep` (`SweepLine` plus nested `Blades`) as the sole analysis-to-rendering and HDF5-to-rendering contract. Processors own every calculation and produce this immutable analysis result; visualizers only configure and draw Matplotlib artists. The reader converts HDF5 into the same dataclass before plotting, so there are no live/reconstruction plotting branches and no dictionary adapters.

**Steps**
1. Normalize the processor boundary: change `XBPMProcessor.analyze_central_sweeps()` to return one `CentralSweep` result, rather than a four-item tuple and cached mutable state. It must produce all position values and fit uncertainties. One-dimensional scans are supported, so change `CentralSweep.h`/`.v` to explicit optional typed fields and define a complete result invariant: at least one line exists. The renderer branches only on this declared domain state, never on unvalidated legacy data.
2. Replace `BladeCurrentVisualizer`'s public API with one renderer, e.g. `plot(central_sweep: CentralSweep, prm: BeamlinePrm) -> Figure`. It reads only `SweepLine.index`, `Blades.{to,ti,bi,bo}`, and `Blades.{sto,sti,sbi,sbo}`, draws error bars, labels, legends, titles, layout, and returns the figure. Move all `np.polyfit` current fitting out of the visualizer; either calculate and store current fit results in a dedicated dataclass or omit fit curves from the blade-current figure.
3. Rename/reduce `SweepVisualizer.plot_from_arrays()` to one typed renderer accepting `CentralSweep` plus display context (`BeamlinePrm` or source-XBPM distance). It draws precomputed `calc_pos`, `fit_pos`, and `fit_pos_err` only, leaving the unused direction blank or omitted according to the defined one-dimensional layout. This depends on step 1.
4. Introduce or reshape the non-Qt application service that owns analysis recalculation: it accepts the current typed parameters (including ROI), invokes processors, and returns a new typed analysis result containing `CentralSweep` and other results. This is the re-calculation boundary used when ROI or other parameters change; it blocks GUI rendering but can be unit-tested without Qt.
5. Retain a thin GUI presenter only: it observes parameter changes, asks the application service for a new result, invokes visualizer render methods, and places returned Figures on canvases. It must not call individual processor methods, maintain results dictionaries, or use checkbox flags as computational switches. This depends on step 4.
6. Refactor `Exporter._write_sweeps_group` to serialize `CentralSweep` fields directly instead of unpacking the current legacy `sweeps_data` tuples/dictionaries. This is the storage migration required before deleting legacy reconstruction adapters.
7. Make `HDF5DataReader` use the existing `CentralSweep.from_hdf5()` and call the same typed renderer used by live analysis. Keep all HDF5 parsing in `reader_hdf5.py`; remove `*_from_hdf5` methods from visualizers. This depends on step 6's final schema.
8. Remove obsolete visualizer methods after the typed renderer tests pass: `BladeCurrentVisualizer.plot_from_hdf5`, `plot_blade_center_from_dicts`, `plot_central_sweeps`, `_plot_blades_common`, `_plot_side`, `_plot_blade`, and `_fit_blade`; and `SweepVisualizer.plot_from_hdf5`. They are either legacy dict paths, duplicate calculations, or private code reachable only through those legacy paths.
9. Remove processor legacy state and its dead facade now: `self.blades_h`, `self.blades_v`, and `XBPMProcessor.show_blades_at_center`. Replace the current `XBPMAnalyzer` figure-result orchestration with the thin presenter described in step 5; no visualizer should call a processor or consult UI flags.
10. Add focused tests before and after deletion: create typed `Blades`, `SweepLine`, and complete/one-dimensional `CentralSweep` fixtures; assert both typed renderers produce expected artists/error bars; recalculate through the application service after an ROI change; and round-trip one persisted `CentralSweep` through exporter/reader to assert it renders through exactly the same renderer.

**Relevant files**
- `/home/agolivei/CNPEM/repos/XBPM/XBPM-bumps/xbpm_bumps/core/data_structure.py` — `Blades`, `SweepLine`, and the existing `CentralSweep` aggregation must define the canonical contract.
- `/home/agolivei/CNPEM/repos/XBPM/XBPM-bumps/xbpm_bumps/core/processors.py` — producer of `CentralSweep`; remove cached dictionary-era blade state and `show_blades_at_center`.
- `/home/agolivei/CNPEM/repos/XBPM/XBPM-bumps/xbpm_bumps/core/visualizers.py` — retain only typed, computation-free renderers; remove all dict/HDF5 adapters and plotting-time fits.
- `/home/agolivei/CNPEM/repos/XBPM/XBPM-bumps/xbpm_bumps/core/exporters.py` — `_write_sweeps_group` currently accepts the legacy tuple/dictionary `sweeps_data`; migrate this to `CentralSweep`.
- `/home/agolivei/CNPEM/repos/XBPM/XBPM-bumps/xbpm_bumps/core/reader_hdf5.py` — deserialize `CentralSweep` then invoke typed renderers.
- `/home/agolivei/CNPEM/repos/XBPM/XBPM-bumps/xbpm_bumps/ui/analyzer.py` — delete or replace stale UI-client orchestration, including incorrect legacy tuple unpacking and result dictionaries.
- `/home/agolivei/CNPEM/repos/XBPM/XBPM-bumps/tests/test_visualizers.py` — add typed-rendering and HDF5 round-trip coverage.

**Verification**
1. `pytest tests/test_visualizers.py tests/test_hdf5_roundtrip.py` with fixtures covering a complete central sweep and, if supported, each one-dimensional case.
2. Confirm no `dict` conversion or `np.polyfit` remains in blade/sweep visualizer methods.
3. Search for removed symbols and confirm no callers remain.
4. Exercise the graph-presentation boundary with a freshly computed `CentralSweep` and a deserialized `CentralSweep`; both must invoke the same render method.

**Decisions**
- Processors, not visualizers, own all calculations. This includes blade-current fit coefficients if displayed.
- A visualizer should have no `h5py`/HDF5-aware API and should never call a processor.
- `CentralSweep` is the canonical aggregate; there must be one renderer per figure kind, not separate live and HDF5 plotting implementations.
- Do not replace `None` guards blindly. `CentralSweep` currently declares both axes required while the processor returns `None` for a one-dimensional scan. Resolve that data-model mismatch upstream, then renderer code can rely on the final invariant.
- The thin Qt GUI is a presenter: it requests a complete recalculation on parameter changes and renders resulting dataclasses. It does not calculate, maintain legacy result dictionaries, or contain an alternative data source.

## Plan: Remove Plotting From Processors and Consolidate Renderers

`processors.py` must become NumPy/dataclass-only: no matplotlib imports or configuration, visualizer imports, Figure values, `savefig`, `outputfile` plotting decisions, or visualizer instances returned in result payloads. The GUI presenter creates figures from the completed typed results and owns optional image export.

**Processor cleanup**
1. In `/home/agolivei/CNPEM/repos/XBPM/XBPM-bumps/xbpm_bumps/core/processors.py`, remove `matplotlib`, `PositionVisualizer`, `SweepVisualizer`, `BladeCurrentVisualizer`, and `FIGDPI` imports, along with the processor-local `rcParams` assignments and commented sweep plot block.
2. Refactor `_scale_positions` to return a typed position-analysis result only: nominal/measured `Positions`, ROI slice/result, RMS statistics, calculation type, transform state, and titles/context as data where needed. It must not construct `PositionVisualizer` or return a `visualizer` object.
3. Refactor `_compile_results` and `xbpm_position_calculation` to return only typed calculation results. Delete `pairwise_figure` / `cross_figure` return fields and processor-side `save_figure` calls. Move image naming/saving to the Qt presenter or an explicit export service.
4. Remove any `BPMProcessor.self.fig` save path and stale `self.bana` dependency; a BPM processor returns `BPMAnalysis`/typed arrays only. The presenter invokes the BPM rendering method.

**Visualizer consolidation**
5. Replace the five current visualization classes with one `AnalysisVisualizer` class or, preferably, one `visualizers.py` module containing one public function per graph/tab. Since no renderer needs persistent state beyond its local `Figure`, module functions make ownership obvious and avoid artificial class instances.
6. Keep exactly these public renderers, accepting typed analysis dataclasses and returning a `Figure`:
   - `plot_bpm_positions(bpm_analysis: BPMAnalysis)`
   - `plot_blade_map(blade_map: BladeMap, prm: BeamlinePrm)`
   - `plot_central_sweep_positions(central_sweep: CentralSweep, prm: BeamlinePrm)`
   - `plot_blade_currents(central_sweep: CentralSweep, prm: BeamlinePrm)`
   - `plot_xbpm_positions(position_analysis: XBPMAnalysis or equivalent)`
   This is one public entry point per tab/analysis result. Pairwise/raw/transformed/partial variants are parameter values of the XBPM result, not separate plotting implementations.
7. Merge `PositionVisualizer` and `BPMVisualizer`: they are duplicate three-panel position-grid plotters. Implement one small private helper for the repeated scatter panel and one for RMS heatmap, or keep the few Matplotlib statements inline when only used once. Retain a single shared equal-axis-limits helper only if both panels call it; remove HDF5 geometry-reconstruction helpers once reader deserializes typed analysis objects.
8. Avoid visualizer calculation: no `np.polyfit`, no delta/sigma, no suppression application, no ROI selection/reconstruction, no HDF5 reads, and no conversion from dictionaries. The only permissible local work is plot presentation such as deriving axis limits, normalizing artist layout, choosing a 1D versus 2D rendering form from the typed result, and creating legends/colorbars.
9. Refactor `HDF5DataReader` to deserialize typed result objects then call the same public plotting functions; delete all `plot_from_hdf5` visualizer APIs. Refactor exporter writes to serialize those result objects instead of legacy dict/tuple result payloads.
10. Replace `XBPMAnalyzer` with a thin presenter/application boundary: parameter change (including ROI) -> application service recalculates typed results -> presenter calls the relevant rendering function -> canvas displays returned figure. Figure saving also lives here or in an explicit export service, never in a processor.

**Verification**
1. Add a dependency check that `xbpm_bumps/core/processors.py` has no matplotlib/visualizer imports and no `savefig` or `Figure` values in processor result types.
2. Unit-test each public tab renderer with a complete typed fixture and a one-dimensional `CentralSweep` fixture.
3. Test raw/scaled/pairwise/cross (and future partial) XBPM results through the single `plot_xbpm_positions` method to ensure title/data variants do not create new plotting code paths.
4. Test GUI ROI changes by verifying the presenter invokes recalculation and replaces the canvas figure with one rendered from the newly returned immutable result.

**Scope boundary**
- This eliminates renderer redundancy and all processor plotting first. It does not require removing the Qt shell; the Qt shell remains display-only.
- Do not introduce a `visualization_geometry.py` module preemptively. Keep shared helpers private in `visualizers.py` until actual reuse beyond the unified position plot proves a separate module helps.
