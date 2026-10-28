I'll re-read the current `ROISlice` definition and the processor's ROI helpers/call sites, since the code changed since the earlier review (e.g., `_scale_positions` lost its `calc_type` parameter).

Read , lines 65 to 100

Read , lines 1 to 100

Searched for regex `_roi_slice_indices|_extract_roi_slice|self\.roi|roisize_h|roisize_v|from_upto|xbpm_position_calculation` (`**/xbpm_bumps/core/processors.py`), 18 results

Read , lines 195 to 440

I confirmed both suspicions by reading the live code, and found two additional bugs directly relevant to this pipeline.

## 1. Is the `_extract_roi_slice` test redundant?

`ROISlice.update()` in `data_structure.py:80-98` is now an exact duplicate of `XBPMProcessor._roi_slice_indices()` in `processors.py:175-178` — same `min(roisize, n)`, same centered `from/upto` formula, just renamed variables. So `_roi_slice_indices` is fully redundant and should be deleted.

The `_extract_roi_slice` branch test:

```python
if array.shape[0] == 1:
    return array[0:1, fr_col:up_col]
elif array.shape[1] == 1:
    return array[fr_row:up_row, 0:1]
else:
    return array[fr_row:up_row, fr_col:up_col]
```

is mathematically redundant **when the bounds were computed from that same array's own shape.** If `n_lin == 1`, then `n_roi_v = min(roisize_v, 1) = 1`, so `fr_row, up_row = 0, 1` automatically — identical to the special-cased branch. So the general `array[fr_row:up_row, fr_col:up_col]` already produces the same result.

The test stops being redundant only in one case: **when the bounds are computed once from one array's shape and then applied to a different array whose shape may not match.** That is exactly what `processors.py:362` does today:

```python
from_upto = self._roi_slice_indices(pos_nom_h)          # computed once, from pos_nom_h
pos_nom_v_roi = self._extract_roi_slice(pos_nom_v, *from_upto)
pos_roi_pair_h = self._extract_roi_slice(pos_h, *from_upto)
pos_roi_cross_h = self._extract_roi_slice(pos_cross_h, *from_upto)
```

If `pos_cross_h` ever had a degenerate axis that `pos_nom_h` didn't (e.g. a 1-D sweep represented inconsistently across arrays), applying `pos_nom_h`'s row bounds to it could silently return an empty or wrong slice — no error, just bad data. The test guards against exactly that mismatch, per-array, at slice time.

**Conclusion:** the test is not something `ROISlice` needs to replicate. The correct fix is architectural: compute the ROI once per canonical grid shape, and require every array sliced against it to already share that shape (all of `pos_nom_h/v`, `pos_h/v`, `pos_cross_h/v` are built from the same `blade_avg` grid, so this is already guaranteed by construction). Then plain `array[roi.sl_v, roi.sl_h]` is correct and the special-case branch becomes unnecessary rather than "coincidentally equivalent." If a shape mismatch ever occurs, it's a real upstream bug that should raise, not be silently patched over.

## 2. Where `ROISlice.update()` must be called

Two bugs make this currently unsafe to wire up as-is — flagging first since they block a correct answer:

- **`ROISlice.__post_init__` discards the centering `update()` computed** (`data_structure.py:75-78`): it unconditionally sets `sl_v = slice(0, self.sz_v)` and `sl_h = slice(0, self.sz_h)`, overwriting the `sl_v`/`sl_h` passed into `update()`'s `cls(...)` call. Any ROI built via `update()` currently loses its centering offset. This must be fixed before propagating ROI anywhere — `__post_init__` should not touch `sl_v`/`sl_h` at all if they were explicitly provided.
- **`XBPMProcessor.__init__` reads `self.prm_bml.roisize`** (`processors.py:81-82`), but `BeamlinePrm` has no `roisize` field — only `roi: ROISlice`. That line will raise `AttributeError` the first time it executes. `roisize_v`/`roisize_h` (and therefore `_roi_slice_indices`) are dead code.

With those fixed, the pipeline has exactly three call points:

1. **After HDF5 load + beamline selection**, in `main_window.py`: compute the grid shape from `workdata.raw_data.blade_avg.nom` and call `ROISlice.update(grid_shape, [roi_v_spin.value(), roi_h_spin.value()])`, assigning the result to `workdata.prm.roi`. This must happen **before** any `XBPMProcessor` is constructed for that beamline.
2. **On `parametersChanged`** (user edits the ROI spin boxes), in the same handler already discussed for parameter refresh: recompute `ROISlice.update()` with the stored grid shape and the new spin values, and reassign `workdata.prm.roi`.
3. **Inside `XBPMProcessor.__init__`**, replace `self.roi = self.prm_bml.roi` with a re-derivation against the processor's own grid: `self.roi = ROISlice.update((len(self.range_v), len(self.range_h)), [self.prm_bml.roi.sz_v, self.prm_bml.roi.sz_h])`. This is the important one — since `self.roi` is captured once at construction time, if `main_window.py` mutates `workdata.prm.roi` after a processor already exists, the processor's cached reference goes stale. Re-deriving from the processor's own grid shape at construction time both fixes staleness and guarantees the ROI is always valid for the data the processor actually holds.

No other call site should invoke `.update()` — `_scale_positions`, `xbpm_position_calculation`, and friends should only read `self.roi.sl_v`/`self.roi.sl_h`.

One unrelated pre-existing bug worth noting since it's adjacent code you'll touch: the cross-blade call to `_scale_positions` in `processors.py:406-412` still passes `'cross', ..., nosuppress` as positional arguments, but the method signature no longer accepts `calc_type`/`nosuppress` — that call will raise `TypeError` at runtime.

Also, `BPMProcessor` (`processors.py:805-806`) assigns `self.roisize_v = prm_bml.roi.sl_v` (a slice, not a size) and later calls `self.bana.roi.update(...)` where `self.bana` doesn't exist — that whole ROI path in `BPMProcessor` is already broken and will need the same `ROISlice`-based rewrite.
