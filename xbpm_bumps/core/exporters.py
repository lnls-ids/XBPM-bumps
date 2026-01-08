"""Data export functionality.

Includes HDF5 export with figures and derived results.
"""

import numpy as np
from dataclasses import asdict
from datetime import datetime
from io import BytesIO
import logging
from typing import Optional

from .parameters import Prm


logger = logging.getLogger(__name__)


class Exporter:
    """Handles persistence of calc. artifacts (positions, blades, supmat)."""

    def __init__(self, prm: Prm):
        """Store parameters used during export."""
        self.prm = prm

    def write_supmat(self, supmat: np.ndarray,
                     write_file: bool = False) -> None:
        """Write suppression matrix to disk.

        Args:
            supmat: Suppression matrix numpy array
            write_file: Whether to write to .dat file (default: False).
                       Supmat is always exported to HDF5 by write_hdf5().
        """
        if not write_file:
            return
        outfile = f"supmat_{self.prm.beamline}.dat"
        with open(outfile, 'w') as fs:
            for lin in supmat:
                for col in lin:
                    fs.write(f" {col:12.6f}")
                fs.write("\n")

    def data_dump(self, data, positions, sup: str = "") -> None:
        """Dump blades data and calculated positions to files."""
        outfile = f"xbpm_blades_values_{self.prm.beamline}.dat"
        print(f"\n Writing out data to file {outfile} ...", end='')
        with open(outfile, 'w') as df:
            for key, val in data.items():
                df.write(f"{key[0]}  {key[1]}")
                for vv in val:
                    df.write(f"  {vv[0]} {vv[1]}")
                df.write("\n")

        pos_pair, pos_cr = positions

        outfilep = f"xbpm_positions_pair_{sup}_{self.prm.beamline}.dat"
        print("\n Writing out pairwise blade calculated positions to file"
              f" {outfilep} ...", end='')
        with open(outfilep, 'w') as fp:
            for key, val in pos_pair.items():
                fp.write(f"{key[0]}  {key[1]}")
                fp.write(f"  {val[0]} {val[1]}\n")

        outfilec = f"xbpm_positions_cross_{sup}_{self.prm.beamline}.dat"
        print("\n Writing out cross-blade calculated positions to file"
              f" {outfilec} ...", end='')
        with open(outfilec, 'w') as fc:
            for key, val in pos_cr.items():
                fc.write(f"{key[0]}  {key[1]}")
                fc.write(f"  {val[0]} {val[1]}\n")

        print("done.\n")

    def write_hdf5(self, filepath: str, data: dict, results: dict,
                   include_figures: bool = True) -> None:
        """Write full analysis package to an HDF5 file."""
        try:
            import h5py  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "h5py is required for HDF5 export. Please install it"
            ) from exc

        prm_dict = asdict(self.prm)
        grid_x, grid_y = self._build_grid_arrays(data)
        table = self._build_nominal_index_table(data, grid_x, grid_y)

        with h5py.File(filepath, 'w') as h5:
            self._write_meta_and_params(h5, prm_dict)
            if table is not None:
                self._write_data(h5, table)
            self._write_derived(h5, results)
            if include_figures:
                self._write_figures(h5, results, data)

        print(f"HDF5 export written to {filepath}")

    @staticmethod
    def _build_grid_arrays(data: dict):
        keys = np.array(list(data.keys())) if data else np.empty((0, 2))
        grid_x = np.unique(keys[:, 0]) if keys.size else np.array([])
        grid_y = np.unique(keys[:, 1]) if keys.size else np.array([])
        return grid_x, grid_y

    @staticmethod
    def _build_blades_dataset(data: dict, grid_x: np.ndarray,
                              grid_y: np.ndarray):
        ny = grid_y.shape[0] if grid_y.size else 0
        nx = grid_x.shape[0] if grid_x.size else 0
        if not (nx and ny):
            return None

        blades_ds = np.full((ny, nx, 4, 2), np.nan, dtype=float)
        x_to_idx = {v: i for i, v in enumerate(grid_x)}
        y_to_idx = {v: i for i, v in enumerate(grid_y[::-1])}
        for (x, y), arr in data.items():
            ii = y_to_idx.get(y)
            jj = x_to_idx.get(x)
            if ii is None or jj is None:
                continue
            a = np.asarray(arr)
            if a.shape == (4, 2):
                blades_ds[ii, jj, :, :] = a
        return blades_ds

    def _build_nominal_index_table(self, data: dict,
                                   grid_x: np.ndarray,
                                   grid_y: np.ndarray):
        """Build blade measurements table indexed by nominal positions.

        Contains only blade measurements - position calculations are
        stored separately in /positions datasets.
        """
        if not (grid_x.size and grid_y.size):
            return None

        fields = [
            ('x_nom', 'f8'), ('y_nom', 'f8'),
            ('to_mean', 'f8'), ('to_err', 'f8'),
            ('ti_mean', 'f8'), ('ti_err', 'f8'),
            ('bi_mean', 'f8'), ('bi_err', 'f8'),
            ('bo_mean', 'f8'), ('bo_err', 'f8'),
        ]
        dtype = np.dtype(fields)

        rows = []
        for y in grid_y[::-1]:
            for x in grid_x:
                to_mean = to_err = ti_mean = ti_err = np.nan
                bi_mean = bi_err = bo_mean = bo_err = np.nan
                arr = data.get((float(x), float(y)))
                if arr is not None:
                    a = np.asarray(arr)
                    if a.shape == (4, 2):
                        to_mean, to_err = float(a[0, 0]), float(a[0, 1])
                        ti_mean, ti_err = float(a[1, 0]), float(a[1, 1])
                        bi_mean, bi_err = float(a[2, 0]), float(a[2, 1])
                        bo_mean, bo_err = float(a[3, 0]), float(a[3, 1])

                row = (
                    float(x), float(y),
                    to_mean, to_err,
                    ti_mean, ti_err,
                    bi_mean, bi_err,
                    bo_mean, bo_err,
                )
                rows.append(row)

        table = np.array(rows, dtype=dtype)
        return table

    @staticmethod
    def _write_data(h5file, table: np.ndarray) -> None:
        """Write blade measurements table indexed by nominal positions.

        Single storage: /raw_data/measurements
        """
        raw_grp = h5file.create_group('raw_data')
        raw_grp.create_dataset('measurements', data=table)

    def _write_meta_and_params(self, h5file, prm_dict: dict) -> None:
        meta = h5file.create_group('meta')
        meta.attrs['version'] = 1
        meta.attrs['created'] = datetime.utcnow().isoformat() + 'Z'
        meta.attrs['beamline'] = self.prm.beamline

        gprm = h5file.create_group('parameters')
        for k, v in prm_dict.items():
            if v is None:
                continue
            try:
                gprm.attrs[k] = v
            except TypeError:
                gprm.attrs[k] = str(v)

    @staticmethod
    def _write_positions(group, positions: dict) -> None:
        """Write position datasets to HDF5.

        Handles two formats:
        1. Direct dict: positions = {'xbpm_raw': {'measured': ...,
            'nominal': ...}}
        2. Pairwise/Cross list: positions = [pairwise_dict, cross_dict]
        """
        # Check if positions is a list [pairwise, cross] from processors
        if isinstance(positions, list) and len(positions) == 2:
            # Handled later when converting dicts to arrays
            return

        # Standard format:
        # positions = {'xbpm_raw': {'measured': ..., 'nominal': ...}}
        for name in ('xbpm_raw', 'xbpm_scaled', 'bpm'):
            if name not in positions or positions[name] is None:
                continue
            val = positions[name]
            meas = (np.asarray(val.get('measured'))
                    if isinstance(val, dict) else None)
            nom = (np.asarray(val.get('nominal'))
                   if isinstance(val, dict) else None)
            # Create single dataset with nominal and measured positions
            Exporter._write_position_table(group, name, meas, nom)

    @staticmethod
    def _dict_to_arrays(pos_dict: dict):
        """Convert {(x_nom, y_nom): [x_meas, y_meas]} to arrays."""
        if not pos_dict:
            return None, None

        keys = list(pos_dict.keys())
        nom = np.array(keys)  # Shape (N, 2)
        meas = np.array([pos_dict[k] for k in keys])  # Shape (N, 2)
        return meas, nom

    @staticmethod
    def _write_position_table(group, name: str,
                              meas: Optional[np.ndarray],
                              nom: Optional[np.ndarray]) -> None:
        """Create position dataset with named fields.

        Creates a structured array per position type (xbpm_raw, xbpm_scaled,
        bpm) with columns named by position type: x_nom, y_nom,
        x_raw/x_scaled/x, y_raw/y_scaled/y.
        """
        try:
            if meas is None or nom is None:
                return
            # If 2D grids (shape (ny, nx, 2)), reshape to 1D
            meas_1d = (meas.reshape(-1, meas.shape[-1]) if meas.ndim > 2
                       else meas)
            nom_1d = (nom.reshape(-1, nom.shape[-1]) if nom.ndim > 2
                      else nom)

            if (meas_1d.shape[0] != nom_1d.shape[0]
                or meas_1d.shape[1] != 2 or nom_1d.shape[1] != 2):
                return

            # Determine field names based on position type
            # Handle pairwise/cross suffixes
            # (e.g., xbpm_raw_pairwise -> xbpm_raw)
            base_name = name.replace('_pairwise', '').replace('_cross', '')

            if base_name == 'xbpm_raw':
                field_names = ['x_nom', 'y_nom', 'x_raw', 'y_raw']
            elif base_name == 'xbpm_scaled':
                field_names = ['x_nom', 'y_nom', 'x_scaled', 'y_scaled']
            elif base_name == 'bpm':
                field_names = ['x_nom', 'y_nom', 'x', 'y']
            else:
                field_names = ['x_nom', 'y_nom', 'x', 'y']

            # Create structured dtype with field names
            dtype = np.dtype([
                (field_names[0], 'f8'),  # x_nom
                (field_names[1], 'f8'),  # y_nom
                (field_names[2], 'f8'),  # x_raw/x_scaled/x
                (field_names[3], 'f8'),  # y_raw/y_scaled/y
            ])

            # Stack nominal and measured data
            combined = np.column_stack((nom_1d, meas_1d)).astype(float)

            # Create structured array
            struct_data = np.array(
                [tuple(row) for row in combined],
                dtype=dtype
            )

            # Write structured dataset
            dset = group.create_dataset(name, data=struct_data)

            # Add figure metadata directly to dataset
            dset.attrs['xlabel'] = 'x [μm]'
            dset.attrs['ylabel'] = 'y [μm]'
            dset.attrs['title'] = f"{name.replace('_', ' ').title()} Positions"

            # Calculate and store statistics
            diff_x = meas_1d[:, 0] - nom_1d[:, 0]
            diff_y = meas_1d[:, 1] - nom_1d[:, 1]
            diff_rms = np.sqrt(diff_x**2 + diff_y**2)
            dset.attrs['rms_mean'] = float(np.mean(diff_rms))
            dset.attrs['rms_std'] = float(np.std(diff_rms))
            dset.attrs['n_points'] = len(nom_1d)
        except Exception:
            logger.warning(
                "Failed to build position table for %s", name,
                exc_info=True,
            )

    def _write_derived(self, h5file, results: dict) -> None:
        """Write analysis results to HDF5 file."""
        positions = (results.get('positions', {})
                     if isinstance(results, dict) else {})
        supmat = (results.get('supmat')
                  if isinstance(results, dict) else None)
        sweeps = (results.get('sweeps_data')
                  if isinstance(results, dict) else None)

        analysis = h5file.create_group('analysis')
        apos = analysis.create_group('positions')

        raw_full = results.get('positions_raw_full')
        scaled_full = results.get('positions_scaled_full')

        self._write_full_positions(apos, raw_full, scaled_full)
        self._write_bpm_positions(apos, positions)
        self._write_fallback_positions(apos, results, positions,
                                       raw_full, scaled_full)
        self._write_supmat_dataset(analysis, supmat)
        self._write_sweeps_group(analysis, sweeps)

    def _write_full_positions(self, apos, raw_full, scaled_full) -> None:
        self._write_raw_full(apos, raw_full)
        self._write_scaled_full(apos, scaled_full)

    def _write_raw_full(self, apos, raw_full) -> None:
        if not (raw_full and isinstance(raw_full, dict)):
            return
        raw_positions = raw_full.get('positions')
        if isinstance(raw_positions, list) and len(raw_positions) == 2:
            pairwise_dict, cross_dict = raw_positions
            pairwise_meas, pairwise_nom = self._dict_to_arrays(
                pairwise_dict
            )
            cross_meas, cross_nom = self._dict_to_arrays(cross_dict)
            self._write_position_table(
                apos, 'xbpm_raw_pairwise', pairwise_meas, pairwise_nom
            )
            self._write_position_table(
                apos, 'xbpm_raw_cross', cross_meas, cross_nom
            )

    def _write_scaled_full(self, apos, scaled_full) -> None:
        if not (scaled_full and isinstance(scaled_full, dict)):
            return
        scaled_positions = scaled_full.get('positions')
        if (isinstance(scaled_positions, list)
                and len(scaled_positions) == 2):
            pairwise_dict, cross_dict = scaled_positions
            pairwise_meas, pairwise_nom = self._dict_to_arrays(
                pairwise_dict
            )
            cross_meas, cross_nom = self._dict_to_arrays(cross_dict)
            self._write_position_table(
                apos, 'xbpm_scaled_pairwise', pairwise_meas, pairwise_nom
            )
            self._write_position_table(
                apos, 'xbpm_scaled_cross', cross_meas, cross_nom
            )

    def _write_bpm_positions(self, apos, positions) -> None:
        if not (isinstance(positions, dict) and 'bpm' in positions):
            return
        bpm_data = positions['bpm']
        if isinstance(bpm_data, dict):
            meas = (np.asarray(bpm_data.get('measured'))
                    if 'measured' in bpm_data else None)
            nom = (np.asarray(bpm_data.get('nominal'))
                   if 'nominal' in bpm_data else None)
            self._write_position_table(apos, 'bpm', meas, nom)

    def _write_fallback_positions(self, apos, results, positions,
                                  raw_full, scaled_full) -> None:
        if raw_full or scaled_full:
            return
        if isinstance(positions, list) and len(positions) == 2:
            pairwise_dict, cross_dict = positions
            pairwise_meas, pairwise_nom = self._dict_to_arrays(
                pairwise_dict
            )
            cross_meas, cross_nom = self._dict_to_arrays(cross_dict)
            has_scales = 'scales' in results
            pos_type = 'xbpm_scaled' if has_scales else 'xbpm_raw'
            self._write_position_table(
                apos, f'{pos_type}_pairwise', pairwise_meas, pairwise_nom
            )
            self._write_position_table(
                apos, f'{pos_type}_cross', cross_meas, cross_nom
            )
            return
        # Standard dict format (old tests)
        self._write_positions(apos, positions)

    def _write_supmat_dataset(self, analysis, supmat) -> None:
        if supmat is None:
            return
        supmat_arr = np.asarray(supmat)
        analysis.create_dataset('suppression_matrix', data=supmat_arr)
        analysis.create_dataset('optimized_suppression_matrix',
                                data=supmat_arr)

    def _write_sweeps_group(self, analysis, sweeps) -> None:
        if not sweeps:
            return
        sweeps_new = analysis.create_group('sweeps')
        self._write_sweeps_dual(None, sweeps_new, sweeps)

    def _write_sweeps_dual(self, root_group, analysis_group, sweeps) -> None:
        """Write sweep data to both root and /analysis groups."""
        # Handle old (4), intermediate (6), and new (8) element formats
        if len(sweeps) == 8:
            (range_h, range_v, blades_h, blades_v,
             pos_h, pos_v, fit_h, fit_v) = sweeps
        elif len(sweeps) == 6:
            range_h, range_v, blades_h, blades_v, pos_h, pos_v = sweeps
            fit_h, fit_v = None, None
        else:
            range_h, range_v, blades_h, blades_v = sweeps
            pos_h, pos_v = None, None
            fit_h, fit_v = None, None

        for target in (root_group, analysis_group):
            if target is None:
                continue
            # Horizontal sweeps
            if isinstance(blades_h, dict) and range_h is not None:
                self._write_sweep_table(
                    target, 'blades_h', range_h, blades_h,
                    self.prm.xbpmdist or 1.0, pos_calc=pos_h, fit_coef=fit_h
                )
            # Vertical sweeps
            if isinstance(blades_v, dict) and range_v is not None:
                self._write_sweep_table(
                    target, 'blades_v', range_v, blades_v,
                    self.prm.xbpmdist or 1.0, pos_calc=pos_v, fit_coef=fit_v
                )

    def _write_sweep_table(self, group, name: str, index_range: np.ndarray,
                           blades_dict: dict, xbpmdist: float = 1.0,
                           pos_calc: np.ndarray = None,
                           fit_coef: np.ndarray = None) -> None:
        """Write sweep data as indexed table with blades and positions.

        Args:
            group: HDF5 group where the table dataset will be created.
            name: Name of the sweep table ('blades_h' or 'blades_v').
            index_range: Array of index values for the sweep dimension.
            blades_dict: Dictionary containing blade measurements
                    (to, ti, bi, bo).
            xbpmdist: Distance from XBPM to source
                    (default 1.0 for normalized).
            pos_calc: Pre-calculated positions from analyze_central_sweeps
                (if None, will recalculate from blade data).
            fit_coef: Linear fit coefficients [k, delta] from np.polyfit
                (shape: (2, 2) with values and errors).
        """
        try:
            blade_order = ['to', 'ti', 'bi', 'bo']
            blade_arrays = self._extract_blade_arrays(
                blade_order, blades_dict, len(index_range), name
            )
            if blade_arrays is None:
                return

            scaled_index = np.asarray(index_range) * xbpmdist
            has_errors = Exporter._has_error_arrays(blade_arrays)

            parts, cols = Exporter._build_sweep_columns(
                name, scaled_index, blade_order, blade_arrays,
                has_errors, xbpmdist, pos_calc
            )
            Exporter._append_fit_columns(
                parts, cols, name, index_range, fit_coef, xbpmdist
            )

            struct_data = Exporter._build_structured_dataset(
                len(index_range), parts, cols
            )

            dset = group.create_dataset(name, data=struct_data)
            Exporter._set_sweep_metadata(dset, name, fit_coef)
        except Exception:
            logger.warning(
                "Failed to write sweep table %s", name,
                exc_info=True,
            )

    @staticmethod
    def _has_error_arrays(blade_arrays: list) -> bool:
        """Check if blade arrays have (N, 2) shape for value+error."""
        if not blade_arrays:
            return False
        return all(
            (arr.ndim == 2 and arr.shape[1] == 2) or (arr.ndim == 1)
            for arr in blade_arrays
        ) and all(arr.ndim == blade_arrays[0].ndim for arr in blade_arrays)

    @staticmethod
    def _extract_blade_arrays(blade_order: list, blades_dict: dict,
                              expected_len: int, name: str):
        """Extract blade arrays and validate length."""
        blade_arrays = [
            np.asarray(blades_dict.get(k, [])) for k in blade_order
        ]
        if not all(len(arr) == expected_len for arr in blade_arrays):
            logger.warning(
                "Sweep %s: blade arrays length mismatch with index", name
            )
            return None
        return blade_arrays

    @staticmethod
    def _build_sweep_columns(name: str, index_range: np.ndarray,
                             blade_order: list,
                             blade_arrays: list,
                             has_errors: bool,
                             xbpmdist: float = 1.0,
                             pos_calc: np.ndarray = None) -> tuple:
        """Build sweep table columns and names."""
        parts = []
        cols = []
        perp_axis = np.zeros(len(index_range))

        # Index and perpendicular columns
        if name == 'blades_h':
            parts.extend([np.asarray(index_range), perp_axis])
            cols.extend(['x_index', 'y_fixed'])
        else:  # blades_v
            parts.extend([perp_axis, np.asarray(index_range)])
            cols.extend(['x_fixed', 'y_index'])

        # Blade columns with optional errors
        Exporter._append_blade_columns(
            parts, cols, blade_order, blade_arrays, has_errors
        )

        # Position calculation columns
        Exporter._append_position_columns(
            parts, cols, name, blade_arrays, has_errors, xbpmdist, pos_calc
        )

        return parts, cols

    @staticmethod
    def _append_blade_columns(parts: list, cols: list,
                              blade_order: list,
                              blade_arrays: list,
                              has_errors: bool) -> None:
        """Add blade value and error columns to table."""
        if has_errors and blade_arrays[0].ndim == 2:
            for k, arr in zip(blade_order, blade_arrays, strict=True):
                parts.append(arr[:, 0])
                parts.append(arr[:, 1])
                cols.append(k)
                cols.append(f's_{k}')
        else:
            for k, arr in zip(blade_order, blade_arrays, strict=True):
                parts.append(arr)
                cols.append(k)

    @staticmethod
    def _append_position_columns(parts: list, cols: list,
                                 name: str,
                                 blade_arrays: list,
                                 has_errors: bool,
                                 xbpmdist: float = 1.0,
                                 pos_calc: np.ndarray = None) -> None:
        """Add position columns to table.

        Uses pre-calculated positions if available, otherwise calculates them.
        Positions are scaled by xbpmdist for physical units.
        """
        if pos_calc is not None:
            # Use pre-calculated positions from analyze_central_sweeps
            pos_result = pos_calc * xbpmdist
        else:
            # Fallback: calculate positions from blade data
            to_arr, ti_arr, bi_arr, bo_arr = blade_arrays

            if name == 'blades_h':
                # Horizontal: y = (to + ti - bo - bi) / (to + ti + bo + bi)
                pos_to_ti = (to_arr + ti_arr)
                pos_bi_bo = (bo_arr + bi_arr)
                pos_result = (pos_to_ti - pos_bi_bo) / (pos_to_ti + pos_bi_bo)
            else:
                # Vertical: x = (to + bo - ti - bi) / (to + bo + ti + bi)
                pos_to_bo = (to_arr + bo_arr)
                pos_ti_bi = (ti_arr + bi_arr)
                pos_result = (pos_to_bo - pos_ti_bi) / (pos_to_bo + pos_ti_bi)

            # Handle NaN and Inf values
            with np.errstate(divide='ignore', invalid='ignore'):
                pos_result = np.nan_to_num(pos_result, nan=0.0, posinf=0.0,
                                           neginf=0.0)
            # Scale by xbpmdist
            pos_result = pos_result * xbpmdist

        # Extract value column from position result
        if has_errors and pos_result.ndim == 2:
            pos_val = pos_result[:, 0]
            pos_err = pos_result[:, 1]
        else:
            pos_val = pos_result
            pos_err = None

        col_name = 'y_calc' if name == 'blades_h' else 'x_calc'
        parts.append(pos_val)
        cols.append(col_name)

        if pos_err is not None:
            parts.append(pos_err)
            cols.append(f's_{col_name}')

    @staticmethod
    def _append_fit_columns(parts: list, cols: list, name: str,
                            index_range: np.ndarray,
                            fit_coef: np.ndarray,
                            xbpmdist: float = 1.0) -> None:
        """Add fit line columns based on coefficients."""
        if fit_coef is None:
            return

        k_val, delta_val = Exporter._extract_fit_values(fit_coef)
        if k_val is None or delta_val is None:
            return

        fit_line = (k_val * index_range + delta_val) * xbpmdist
        fit_col_name = 'y_fit' if name == 'blades_h' else 'x_fit'
        parts.append(fit_line)
        cols.append(fit_col_name)

    @staticmethod
    def _extract_fit_values(fit_coef: np.ndarray):
        """Normalize fit coefficient shapes to values."""
        if fit_coef is None:
            return None, None
        if fit_coef.ndim == 2 and fit_coef.shape == (2, 2):
            return fit_coef[0, 0], fit_coef[1, 0]
        if fit_coef.ndim == 1 and len(fit_coef) == 2:
            return fit_coef[0], fit_coef[1]
        return None, None

    @staticmethod
    def _build_structured_dataset(nrows: int, parts: list,
                                  cols: list) -> np.ndarray:
        """Create structured array from column parts and names."""
        dtype = np.dtype([(c, 'f8') for c in cols])
        struct_data = np.zeros(nrows, dtype=dtype)
        for col_name, values in zip(dtype.names, parts, strict=True):
            struct_data[col_name] = values
        return struct_data

    @staticmethod
    def _set_sweep_metadata(dset, name: str, fit_coef: np.ndarray) -> None:
        """Attach sweep plot metadata and fit attributes."""
        if name == 'blades_h':
            dset.attrs['xlabel'] = 'x [μm]'
            dset.attrs['ylabel'] = 'y [μm]'
            dset.attrs['title'] = 'Central Horizontal Sweeps'
            dset.attrs['xlabel_blades'] = 'x [μrad]'
        else:
            dset.attrs['xlabel'] = 'y [μm]'
            dset.attrs['ylabel'] = 'x [μm]'
            dset.attrs['title'] = 'Central Vertical Sweeps'
            dset.attrs['xlabel_blades'] = 'y [μrad]'

        if fit_coef is None:
            return

        if fit_coef.ndim == 2 and fit_coef.shape == (2, 2):
            dset.attrs['k'] = float(fit_coef[0, 0])
            dset.attrs['s_k'] = float(fit_coef[0, 1])
            dset.attrs['delta'] = float(fit_coef[1, 0])
            dset.attrs['s_delta'] = float(fit_coef[1, 1])
        elif fit_coef.ndim == 1 and len(fit_coef) == 2:
            dset.attrs['k'] = float(fit_coef[0])
            dset.attrs['delta'] = float(fit_coef[1])

    def _write_figures(self, h5file, results: dict, data: dict) -> None:
        """Write blade heatmap to /analysis/blade_map/ with metadata.

        Note: All figure metadata is now stored directly with the data in
        /analysis/ (positions, sweeps). Only blade_map needs special handling
        since it contains processed grid data (expensive to recalculate).
        """
        # Blade heatmap: store processed grid data in /analysis/blade_map/
        if results.get('blade_figure'):
            analysis = h5file['analysis']
            self._write_blade_map_data(analysis, data)

    def _write_blade_map_data(self, analysis_group, data: dict) -> None:
        """Store blade heatmap data in /analysis/blade_map/.

        This includes processed heatmap grids and metadata.
        Special case: we store processed grids since they are expensive
        to recalculate.
        """
        from .processors import XBPMProcessor

        processor = XBPMProcessor(data, self.prm)
        blades, stddevs = processor.data_parse()
        to, ti, bi, bo = blades

        blade_grp = analysis_group.create_group('blade_map')

        for name, data in [('to', to), ('ti', ti), ('bi', bi), ('bo', bo)]:
            dset = blade_grp.create_dataset(
                name, data=data.astype(np.float32), compression='gzip'
            )
            dset.attrs['description'] = f'{name.upper()} blade intensity map'
            dset.attrs['unit'] = ('counts' if self.prm.beamline[:3]
                                 in ['MGN', 'MNC'] else 'A')

        # Store grid coordinates
        data_keys = np.array(list(processor.data.keys()))
        x_coords = np.unique(data_keys[:, 0])
        y_coords = np.unique(data_keys[:, 1])
        blade_grp.create_dataset('x_coords', data=x_coords.astype(np.float32))
        blade_grp.create_dataset('y_coords', data=y_coords.astype(np.float32))

        # Add figure metadata directly to group
        blade_grp.attrs['xlabel'] = 'x [μrad]'
        blade_grp.attrs['ylabel'] = 'y [μrad]'
        blade_grp.attrs['title'] = 'Blade Intensity Map'

    @staticmethod
    def _fig_to_array(fig):
        """Render matplotlib figure to RGBA numpy array.

        Uses PNG format (lossless, 8-bit per channel RGBA) at 300 DPI
        for high quality visualization in HDF5 viewers. Transparent
        background avoids colormap artifacts in silx.
        """
        try:
            from PIL import Image  # type: ignore
            buf = BytesIO()
            # Use transparent background to avoid colormap issues in silx
            fig.savefig(
                buf,
                format='png',
                dpi=300,
                bbox_inches='tight',
                facecolor='none',  # Transparent background
                edgecolor='none',
                transparent=True,
            )
            buf.seek(0)
            # Read PNG bytes into memory before closing buffer
            png_bytes = buf.getvalue()
            # Open image from bytes
            img_buf = BytesIO(png_bytes)
            img = Image.open(img_buf)
            # Keep as RGBA to preserve transparency
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            # Convert to numpy array and ensure it's copied
            rgba_array = np.array(img, copy=True)
            return rgba_array
        except Exception:
            logger.warning("Failed to render figure to array", exc_info=True)
            return None
