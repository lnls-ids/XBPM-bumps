"""Data reading from files and pickle directories."""

import os
import re
import sys
import logging
import numpy as np
from typing import Optional

# Reading trusted local experiment data; HDF5 path will replace this
import pickle  # noqa: S403

from .config import Config
from .parameters import Prm, ParameterBuilder


logger = logging.getLogger(__name__)


class DataReader:
    """Handles reading XBPM data from files or pickle directories.

    This class provides a unified interface for reading XBPM measurement
    data from two different sources:
    - Text files with optional header metadata
    - Directories containing pickle files

    Attributes:
        prm (Prm): Parameters dataclass containing configuration
                   and runtime data.
        data (dict): Dictionary containing parsed measurement data.
    """

    def __init__(self, prm: Prm, builder: Optional[ParameterBuilder] = None):
        """Initialize DataReader with parameters.

        Args:
            prm (Prm): Parameters dataclass instance.
            builder: Optional ParameterBuilder instance to share state.
        """
        self.prm     = prm
        self.builder = builder or ParameterBuilder(prm)
        self.data    = {}
        self.rawdata = None
        self._hdf5_path = None
        self.analysis_meta = {}

    def read(self, beamline_selector=None) -> dict:
        """Read data from working directory or file.

        Automatically determines whether to read from a text file or
        pickle directory based on the workdir path.

        Returns:
            dict: Parsed measurement data dictionary.
        """
        if os.path.isfile(self.prm.workdir):
            # Detect HDF5 file by extension
            lower = (self.prm.workdir or '').lower()
            if lower.endswith('.h5') or lower.endswith('.hdf5'):
                self._read_from_hdf5()
            else:
                self._read_from_file(beamline_selector)
        else:
            self._read_from_directory(beamline_selector)

        self._print_summary()
        return self.data

    def _read_from_file(self, beamline_selector=None) -> None:
        """Read data from a text file with optional header metadata."""
        self.data = {}
        self.rawdata = []
        header_meta = {}

        with open(self.prm.workdir, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                if self._is_empty(line):
                    continue
                if self._is_header(line):
                    self._parse_header_line(line)
                    continue
                self._parse_data_line(line)

        # Capture minimal header info for beamline extraction fallback
        if self.prm.beamline:
            header_meta['beamline'] = self.prm.beamline
        if header_meta:
            self.rawdata = [(header_meta, self.data)]

        # Handle beamline selection and xbpmdist configuration
        self._handle_beamline_selection_file(beamline_selector)
        self._configure_xbpmdist()
        self._infer_gridstep()

    def _read_from_hdf5(self) -> None:
        """Read data and parameters from an HDF5 file."""
        try:
            import h5py  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "h5py is required to read HDF5 files. Please install it"
            ) from exc

        self.data = {}
        self.rawdata = None
        with h5py.File(self.prm.workdir, 'r') as h5:
            self._load_hdf5_parameters(h5)
            grid_x, grid_y, pairs, pairs_nom = self._load_hdf5_grid(
                h5, None, None
            )
            self._load_hdf5_data(h5, grid_x, grid_y, pairs_nom=pairs_nom)
            self._infer_gridstep_from_grid(grid_x, grid_y)
            self._fallback_beamline_from_meta(h5)
            self._load_hdf5_analysis_meta(h5)

            bpm_rawdata = self._load_hdf5_bpm_data(h5)
            if bpm_rawdata:
                self.rawdata = bpm_rawdata

        # Store HDF5 path for later figure loading
        self._hdf5_path = self.prm.workdir

    def _load_hdf5_parameters(self, h5file):
        """Load parameter attributes from HDF5 into prm."""
        if 'parameters' in h5file:
            gprm = h5file['parameters']
            for k, v in gprm.attrs.items():
                try:
                    setattr(self.prm, k, v)
                except Exception:
                    logger.warning(
                        "Skipping parameter attr %s from HDF5", k,
                        exc_info=True,
                    )
        return None, None

    def _load_hdf5_grid(self, h5file, grid_x, grid_y):
        """Extract grid info from /data table or legacy /grid group."""
        pairs = None
        pairs_nom = None

        # New format: extract from /data table
        data_dset = self._get_data_table(h5file)
        if (data_dset is not None and hasattr(data_dset, 'dtype') and
            data_dset.dtype.names):
            table = np.array(data_dset)
            grid_x = np.unique(table['x_nom'])
            grid_y = np.unique(table['y_nom'])
            pairs_nom = np.column_stack([table['x_nom'], table['y_nom']])
            self.grid_pairs_nom = pairs_nom
            return grid_x, grid_y, pairs, pairs_nom

        # Legacy format: read from /grid group
        if 'grid' not in h5file:
            return grid_x, grid_y, pairs, pairs_nom

        ggrid = h5file['grid']
        grid_x = np.array(ggrid.get('x')) if 'x' in ggrid else None
        grid_y = np.array(ggrid.get('y')) if 'y' in ggrid else None
        if 'pairs' in ggrid:
            pairs = np.array(ggrid['pairs'])
        if 'pairs_nom' in ggrid:
            pairs_nom = np.array(ggrid['pairs_nom'])
        # Expose for callers if needed
        self.grid_pairs = pairs
        self.grid_pairs_nom = pairs_nom
        return grid_x, grid_y, pairs, pairs_nom

    def _load_hdf5_analysis_meta(self, h5file):
        """Load analysis extras (scales, BPM stats) if present.

        Coordinates loading of scales, sweeps metadata, and BPM statistics.
        """
        meta = {}
        analysis = h5file.get('analysis')
        if analysis is None:
            self.analysis_meta = meta
            return

        self._load_scales_meta(analysis, meta)
        self._load_sweeps_meta(analysis, meta)
        self._load_bpm_stats_meta(analysis, meta)
        self._load_roi_bounds_fallback(analysis, meta)

        self.analysis_meta = meta

    def _load_scales_meta(self, analysis, meta):
        """Load scale information from position datasets or legacy group."""
        scales = self._read_scale_attrs(analysis)
        if not scales:
            scales_grp = analysis.get('scales')
            if scales_grp is not None:
                scales = self._read_scales_group(scales_grp)
        if scales:
            meta['scales'] = scales

    def _load_sweeps_meta(self, analysis, meta):
        """Load sweeps metadata (positions and blade fits)."""
        sweeps_meta = self._read_sweeps_meta(analysis)
        if sweeps_meta:
            meta['sweeps'] = sweeps_meta

    def _load_hdf5_bpm_data(self, h5file):
        """Rebuild raw acquisition sweeps from /raw_data when present."""
        raw_grp = h5file.get('raw_data') if h5file else None
        if raw_grp is None:
            logger.info(
                "No raw_data group found in HDF5; BPM/XBPM re-run limited"
            )
            return None

        raw_entries = []
        for name in sorted(raw_grp.keys()):
            if not name.startswith('sweep_'):
                continue
            sweep_grp = raw_grp[name]
            header = self._read_bpm_header(sweep_grp)
            gap_info = self._read_bpm_gap_info()
            bpm_dict = self._read_bpm_dict(sweep_grp)
            raw_entries.append((header, gap_info, bpm_dict))
        return raw_entries

    def _read_bpm_header(self, sweep_grp):
        import h5py

        header = {}
        meta_attrs = {
            str(k).replace('meta_', ''): v
            for k, v in sweep_grp.attrs.items()
            if str(k).startswith('meta_')
        }
        header.update(meta_attrs)

        for key, obj in sweep_grp.items():
            if (self._looks_like_machine_key(key) and
                not isinstance(obj, h5py.Group)):
                header[key] = self._read_machine_dataset(obj)

        if not header and self.prm.beamline:
            header[self.prm.beamline] = {}
        return header

    def _read_bpm_gap_info(self):
        code = (self.prm.beamline or '')[:3].lower()
        if code and self.prm.phaseorgap is not None:
            return {code: self.prm.phaseorgap}
        return {}

    def _read_bpm_dict(self, sweep_grp):
        import h5py

        bpm = {
            str(k): v for k, v in sweep_grp.attrs.items()
            if not str(k).startswith('meta_')
        }

        if 'orb' in sweep_grp:
            orb = np.array(sweep_grp['orb'])
            if hasattr(orb, 'dtype') and orb.dtype.names:
                if 'orbx_val' in orb.dtype.names:
                    bpm['orbx'] = orb['orbx_val']
                if 'orby_val' in orb.dtype.names:
                    bpm['orby'] = orb['orby_val']

        for key, obj in sweep_grp.items():
            if key == 'orb':
                continue
            if (self._looks_like_machine_key(key) and
                not isinstance(obj, h5py.Group)):
                bpm[key] = self._read_machine_dataset(obj)
            else:
                bpm[key] = np.array(obj)

        return bpm

    @staticmethod
    def _looks_like_machine_key(key: str) -> bool:
        return bool(re.match(r'^[A-Z]{3,4}[0-9]?$', str(key)))

    @staticmethod
    def _read_machine_dataset(ds):
        """Read machine compound dataset with value and unit columns.

        Dataset has N rows with columns like:
        val_A, val_A_unit, val_B, val_B_unit, A_range, B_range, etc.

        Returns dict with arrays:
        {'val_A': (array, unit),
        'val_B': (array, unit),
        'A_range': array, ...}
        """
        arr = np.array(ds)
        result = {}

        if hasattr(arr, 'dtype') and arr.dtype.names:
            processed_names = set()
            for name in arr.dtype.names:
                # Skip unit columns (they're processed with their value column)
                if name.endswith('_unit') or name in processed_names:
                    continue

                # Extract value array
                value = arr[name]

                # Check if there's a corresponding unit column
                unit_col = f'{name}_unit'
                if unit_col in arr.dtype.names:
                    # Extract unit (should be same for all rows)
                    unit = arr[unit_col][0]
                    result[name] = (value, unit)
                    processed_names.add(unit_col)
                else:
                    result[name] = value

                processed_names.add(name)
        else:
            result['value'] = arr

        prefix = ds.attrs.get('prefix') if hasattr(ds, 'attrs') else None
        if prefix is not None:
            result['prefix'] = prefix
        return result

    def _load_bpm_stats_meta(self, analysis, meta):
        """Load BPM statistics including ROI bounds."""
        bpm_grp = analysis.get('bpm_stats')
        if bpm_grp is None:
            return

        stats = {k: v for k, v in bpm_grp.attrs.items()}
        roi_bounds = self._load_roi_bounds_from_group(bpm_grp)
        if roi_bounds:
            stats['roi_bounds'] = roi_bounds
        meta['bpm_stats'] = stats

    @staticmethod
    def _load_roi_bounds_from_group(bpm_grp):
        """Extract ROI bounds from bpm_stats group."""
        if 'roi_bounds' in bpm_grp:
            return {k: v for k, v in bpm_grp['roi_bounds'].attrs.items()}

        # Backward-compatible: bounds as subgroup
        rb = bpm_grp.get('roi_bounds')
        if rb is not None:
            return {k: v for k, v in rb.attrs.items()}
        return None

    def _load_roi_bounds_fallback(self, analysis, meta):
        """Fallback: ROI bounds stored as attrs on positions group."""
        positions_grp = analysis.get('positions')
        if positions_grp is None:
            return

        try:
            rb_attrs = {
                k: v for k, v in positions_grp.attrs.items()
                if k in ('x_min', 'x_max', 'y_min', 'y_max',
                        'roi_bounds_title')
            }
            if rb_attrs:
                meta.setdefault('bpm_stats', {})['roi_bounds'] = rb_attrs
        except Exception:
            logger.debug(
                "Failed to read roi_bounds attrs from positions",
                exc_info=True
            )

    @staticmethod
    def _read_scales_group(scales_grp):
        result = {}
        for scope in ('raw', 'scaled'):
            sub = scales_grp.get(scope)
            if sub is None:
                continue
            scoped = {}
            for key in ('pair', 'cross'):
                child = sub.get(key)
                if child is None:
                    continue
                scoped[key] = {k: child.attrs[k] for k in child.attrs}
            if scoped:
                result[scope] = scoped
        return result

    @staticmethod
    def _read_scale_attrs(analysis_grp):
        """Extract scale_* attrs from position datasets, if present."""
        positions = analysis_grp.get('positions') if analysis_grp else None
        if positions is None:
            return {}

        def extract(dset):
            if dset is None:
                return None
            attrs = {}
            for key in ('scale_kx', 'scale_ky', 'scale_dx', 'scale_dy'):
                if key in dset.attrs:
                    attrs[key.replace('scale_', '')] = dset.attrs[key]
            return attrs or None

        result = {}
        for scope, p_name, c_name in (
            ('raw', 'xbpm_raw_pairwise', 'xbpm_raw_cross'),
            ('scaled', 'xbpm_scaled_pairwise', 'xbpm_scaled_cross'),
        ):
            pair  = (extract(positions.get(p_name))
                     if p_name in positions else None)
            cross = (extract(positions.get(c_name))
                     if c_name in positions else None)
            scoped = {}
            if pair:
                scoped['pair'] = pair
            if cross:
                scoped['cross'] = cross
            if scoped:
                result[scope] = scoped
        return result

    @staticmethod
    def _read_sweeps_meta(analysis_grp):
        sweeps = analysis_grp.get('sweeps') if analysis_grp else None
        if sweeps is None:
            return {}

        h_ds = sweeps.get('blades_h')
        v_ds = sweeps.get('blades_v')

        positions = DataReader._collect_sweep_positions(h_ds, v_ds)
        blades = DataReader._collect_sweep_blade_trends(h_ds, v_ds)

        meta = {}
        if positions:
            meta['positions'] = positions
        if blades:
            meta['blades'] = blades
        return meta

    @staticmethod
    def _collect_sweep_positions(h_ds, v_ds):
        positions = {}
        for axis, dataset in (('horizontal', h_ds), ('vertical', v_ds)):
            fit = DataReader._read_sweep_fit_attrs(dataset)
            if fit:
                positions[axis] = fit
        return positions

    @staticmethod
    def _collect_sweep_blade_trends(h_ds, v_ds):
        blades = {}
        axis_map = (
            ('horizontal', h_ds, 'x_index'),
            ('vertical', v_ds, 'y_index'),
        )
        for axis, dataset, coord_field in axis_map:
            fits = DataReader._fit_blade_trends(dataset, coord_field)
            if fits:
                blades[axis] = fits
        return blades

    @staticmethod
    def _read_sweep_fit_attrs(ds):
        if ds is None:
            return None
        attrs = {
            key: ds.attrs[key]
            for key in ('k', 'delta', 's_k', 's_delta')
            if key in ds.attrs
        }
        return attrs or None

    @staticmethod
    def _fit_blade_trends(ds, coord_field):
        if ds is None:
            return None
        try:
            data = np.array(ds)
            if data.size == 0 or coord_field not in data.dtype.names:
                return None

            x = data[coord_field]
            fits = {}
            for blade in ('to', 'ti', 'bi', 'bo'):
                if blade not in data.dtype.names:
                    continue
                y = data[blade]
                try:
                    coef = np.polyfit(x, y, deg=1)
                    fits[blade] = {
                        'k': float(coef[0]),
                        'delta': float(coef[1]),
                    }
                except Exception:
                    logger.debug(
                        "Failed to fit blade trend",
                        exc_info=True,
                        extra={
                            'blade': blade,
                            'coord_field': coord_field,
                        },
                    )
                    continue
            return fits or None
        except Exception:
            return None

    def _load_hdf5_data(self, h5file, grid_x, grid_y, pairs_nom=None):
        """Load blades dataset and reconstruct data dict.

        Preference order for keys:
        1) pairs_nom (nominal positions) when present
        2) grid_x/grid_y fallback
        """
        blades = self._get_blades_array(h5file)
        if blades is None:
            return

        ny, nx = blades.shape[:2]
        # Prefer by-nominal pairs if present and sized correctly
        if pairs_nom is not None and pairs_nom.shape == (ny * nx, 2):
            self._populate_data_from_pairs_nom(blades, pairs_nom, ny, nx)
            return

        # Else require grid_x/y to reconstruct
        self._populate_data_from_grid(blades, grid_x, grid_y, ny, nx)

    def _get_blades_array(self, h5file) -> Optional[np.ndarray]:
        """Extract and validate blades data from HDF5.

        Handles both old format (data/blades 4D array) and new format
        (data structured table).
        """
        table_dset = self._get_data_table(h5file)
        if table_dset is not None:
            # Structured table
            if hasattr(table_dset, 'dtype') and table_dset.dtype.names:
                return self._blades_from_table(table_dset)
            # Legacy 4D array
            blades = np.array(table_dset)
            return blades if blades.size else None

        return None

    def _get_data_table(self, h5file):
        """Locate the raw measurement table.

        Preference order:
        1) /raw_data/measurements_<beamline>
        2) /data (legacy fallback)
        3) /data/blades (very old)
        """
        # Try new standard path first: pick any dataset named measurements_*
        if 'raw_data' in h5file:
            raw_grp = h5file['raw_data']
            for name, obj in raw_grp.items():
                if isinstance(obj, (np.ndarray,)):
                    # h5py Datasets are array-like; direct check via attrs
                    pass
                if hasattr(obj, 'shape') and name.startswith('measurements_'):
                    return obj

        # Legacy fallbacks
        if 'data' in h5file:
            if hasattr(h5file['data'], 'keys') and 'blades' in h5file['data']:
                return h5file['data']['blades']
            return h5file['data']

        return None

    def _blades_from_table(self, table_dset) -> Optional[np.ndarray]:
        """Reconstruct blades array from structured table.

        Extracts blade measurements directly into data dict using
        nominal coords.
        """
        table = np.array(table_dset)
        if not table.size:
            return None

        # Extract data directly into self.data using nominal coordinates
        for row in table:
            x_nom = row['x_nom']
            y_nom = row['y_nom']
            blades = np.array([
                [row['to_mean'], row['to_err']],
                [row['ti_mean'], row['ti_err']],
                [row['bi_mean'], row['bi_err']],
                [row['bo_mean'], row['bo_err']],
            ])
            self.data[(x_nom, y_nom)] = blades

        # Return None to skip the array-based processing
        return None

    def _populate_data_from_pairs_nom(
        self, blades: np.ndarray, pairs_nom: np.ndarray, ny: int, nx: int
    ) -> None:
        """Populate self.data using nominal position pairs."""
        flat = blades.reshape(ny * nx, 4, 2)
        # Use strict=True in Python 3.10+; noqa ZIP002 for Python <3.10
        if sys.version_info >= (3, 10):
            for (xn, yn), arr in zip(pairs_nom, flat, strict=True):
                self.data[(float(xn), float(yn))] = arr
        else:
            for (xn, yn), arr in zip(pairs_nom, flat):  # noqa
                self.data[(float(xn), float(yn))] = arr

    def _populate_data_from_grid(
        self, blades: np.ndarray, grid_x: Optional[np.ndarray],
        grid_y: Optional[np.ndarray], ny: int, nx: int
    ) -> None:
        """Populate self.data using grid x/y arrays."""
        if grid_x is None or grid_y is None:
            logger.warning("Missing grid axes; cannot reconstruct data map")
            return
        if ny != grid_y.shape[0] or nx != grid_x.shape[0]:
            logger.warning(
                "Blades dataset shape (%s, %s) does not match grid (%s, %s)",
                ny, nx, grid_y.shape[0], grid_x.shape[0],
            )
            return
        for ii in range(ny):
            for jj in range(nx):
                y = grid_y[::-1][ii]
                x = grid_x[jj]
                self.data[(float(x), float(y))] = blades[ii, jj]

    def _infer_gridstep_from_grid(self, grid_x, grid_y):
        """Infer gridstep from HDF5 grid datasets when missing."""
        if (getattr(self.prm, 'gridstep', None) not in (None, 0)
            or grid_x is None or grid_y is None):
            return
        try:
            dx = np.min(np.diff(np.sort(grid_x))) if grid_x.size > 1 else None
            dy = np.min(np.diff(np.sort(grid_y))) if grid_y.size > 1 else None
            steps = [v for v in (dx, dy) if v not in (None, 0)]
            if steps:
                self.prm.gridstep = float(min(steps))
        except Exception:
            logger.warning(
                "Failed to infer gridstep from HDF5 grid", exc_info=True
            )

    def _fallback_beamline_from_meta(self, h5file):
        """Populate beamline from meta group if still unset."""
        if self.prm.beamline or 'meta' not in h5file:
            return
        bl = h5file['meta'].attrs.get('beamline')
        if isinstance(bl, (str, bytes)):
            self.prm.beamline = bl.decode() if isinstance(bl, bytes) else bl

    def _handle_beamline_selection_file(self, beamline_selector) -> None:
        """Handle beamline selection logic for file-based input."""
        # If beamline still unknown, try selector to avoid terminal prompt
        if beamline_selector and not self.prm.beamline:
            beamlines = self._extract_beamlines_from_header()
            if not beamlines:
                beamlines = self._extract_beamlines_fallback(self.rawdata)

            if len(beamlines) == 1:
                chosen = beamlines[0]
                self.prm.beamline = chosen
                self.builder.prm.beamline = chosen
            elif beamlines:
                selected = beamline_selector(beamlines)
                # If user cancels, pick first to avoid terminal prompt
                chosen = selected or beamlines[0]
                self.prm.beamline = chosen
                self.builder.prm.beamline = chosen

    def _configure_xbpmdist(self) -> None:
        """Configure XBPM distance from source."""
        if self.prm.xbpmdist is None:
            try:
                self.prm.xbpmdist = Config.XBPMDISTS[self.prm.beamline]
            except Exception:
                self.prm.xbpmdist = 1.0
            print("\n WARNING: distance from source to XBPM not provided."
                  f" Using default value: {self.prm.xbpmdist} m")

    def _read_from_directory(self, beamline_selector=None) -> None:
        """Read data from pickle files in a directory."""
        self.rawdata = self._get_pickle_data()[self.prm.skip:]

        # If beamline not set, try provided selector to avoid terminal prompt
        if beamline_selector and not self.prm.beamline:
            beamlines = self._extract_beamlines(self.rawdata)
            if not beamlines:
                beamlines = self._extract_beamlines_fallback(self.rawdata)
            if len(beamlines) == 1:
                chosen = beamlines[0]
                self.prm.beamline = chosen
                self.builder.prm.beamline = chosen
            elif beamlines:
                selected = beamline_selector(beamlines)
                # If user cancels, pick first to avoid terminal prompt
                chosen = selected or beamlines[0]
                self.prm.beamline = chosen
                self.builder.prm.beamline = chosen

        # Enrich parameters (will skip prompt if beamline already set)
        self.prm = self.builder.enrich_from_data(
            self.rawdata,
            selected_beamline=self.prm.beamline,
            beamline_selector=beamline_selector,
        )
        self.data = self._blades_fetch()

    @staticmethod
    def _extract_beamlines(rawdata):
        """Extract unique beamlines from raw data list."""
        try:
            beamlines = set()
            for record in rawdata:
                if not (isinstance(record, (list, tuple)) and len(record) > 0):
                    continue

                header = record[0]

                if isinstance(header, dict):
                    # Beamlines are dict keys (e.g., 'MNC1', 'MNC2', 'CAT')
                    for key in header.keys():
                        if isinstance(key, str):
                            beamlines.add(key)
                elif isinstance(header, (list, tuple)):
                    for item in header:
                        if isinstance(item, str):
                            beamlines.add(item)
            return list(beamlines)
        except Exception:
            return []

    @staticmethod
    def _extract_beamlines_fallback(rawdata):
        """Fallback: gather all header keys/values as beamline candidates."""
        try:
            beamlines = set()
            for record in rawdata:
                if not (isinstance(record, (list, tuple)) and len(record) > 0):
                    continue
                header = record[0]
                if isinstance(header, dict):
                    # Try to get keys first (primary beamline identifiers)
                    for key in header.keys():
                        if isinstance(key, str):
                            beamlines.add(key)
                elif isinstance(header, (list, tuple)):
                    for item in header:
                        if isinstance(item, str):
                            beamlines.add(item)
            return list(beamlines)
        except Exception:
            return []

    def _extract_beamlines_from_header(self):
        """Extract beamline from already parsed header if present."""
        return [self.prm.beamline] if self.prm.beamline else []

    def _get_pickle_data(self) -> list:
        """Open pickle files in directory and extract data.

        Returns:
            list: All data collected from pickle files in working directory.
        """
        allfiles = os.listdir(self.prm.workdir)
        picklefiles = [pf for pf in allfiles if pf.endswith("pickle")]

        if len(picklefiles) == 0:
            print(f"No pickle files found in directory '{self.prm.workdir}'."
                  "Aborting.")
            sys.exit(0)

        sfiles = sorted(picklefiles,
                        key=lambda name: self.builder.lastfield(name, '_'))
        rawdata = []
        for file in sfiles:
            with open(self.prm.workdir + "/" + file, 'rb') as df:
                rawdata.append(pickle.load(df))  # noqa: S301

        return rawdata

    @staticmethod
    def _is_empty(line: str) -> bool:
        """Check if a line is empty or contains only whitespace."""
        return bool(re.match(r'^\s*$', line))

    @staticmethod
    def _is_header(line: str) -> bool:
        """Check if a line is a header/comment line (starts with #)."""
        return bool(re.match(r'^\s*#', line))

    def _parse_header_line(self, line: str) -> None:
        """Extract metadata from a header line.

        Expected format: # Key: value
        """
        cline = line.strip('#')
        # Split only on the first ':' to avoid accidental extra splits
        kprm, vprm = cline.split(':', 1)
        key = kprm.strip().lower()
        val = vprm.strip()

        # Helper to extract a numeric value from strings like "199.86 mA"
        def _extract_float(text: str) -> float:
            m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
                          text.strip())
            if not m:
                raise ValueError(f"Could not parse float from '{text}'")
            return float(m.group(0))

        if re.search(r'current', key, re.IGNORECASE):
            self.prm.current = _extract_float(val)
        elif re.search(r'beamline', key, re.IGNORECASE):
            blval = val
            # Prefer code inside parentheses, e.g., "Caterete (CAT)" -> "CAT"
            m = re.search(r"\(([^)]+)\)", blval)
            if m:
                self.prm.beamline = m.group(1).strip().upper()
            else:
                # Try to find the beamline code by looking up the full name
                try:
                    self.prm.beamline = self.builder.found_key(
                        Config.BEAMLINENAME, blval.strip()
                        )
                except StopIteration:
                    # Fallback: if exact match not found, keep original
                    self.prm.beamline = blval
        elif re.search(r'gap', key, re.IGNORECASE):
            self.prm.phaseorgap = val
        elif re.search(r'xbpm', key, re.IGNORECASE):
            self.prm.xbpmdist = _extract_float(val)
        elif re.search(r'inter bpm', key, re.IGNORECASE):
            self.prm.bpmdist = _extract_float(val)
        else:
            self.prm[key] = val

    def _parse_data_line(self, line: str) -> None:
        """Parse a data line and add it to the data dictionary.

        Expected format: nom_x nom_y to err_to ti err_ti bi err_bi bo err_bo
        """
        parts = line.strip().split()
        key = (float(parts[0]), float(parts[1]))
        self.data[key] = np.array([
            [float(parts[ii]), float(parts[ii + 1])]
            for ii in range(2, len(parts), 2)
        ])

    def _infer_gridstep(self) -> None:
        """Infer grid step from data keys if not explicitly provided."""
        try:
            xs = np.unique([k[0] for k in self.data.keys()])
            ys = np.unique([k[1] for k in self.data.keys()])
            dx = np.min(np.diff(np.sort(xs))) if xs.size > 1 else None
            dy = np.min(np.diff(np.sort(ys))) if ys.size > 1 else None
            steps = [v for v in (dx, dy) if v not in (None, 0)]
            if steps:
                self.prm.gridstep = float(min(steps))
        except Exception as err:
            print(f"\nWARNING: could not infer grid step from data: {err}.")

    def _print_summary(self) -> None:
        """Print a summary of the loaded data and parameters."""
        bl = self.prm.beamline or "N/A"
        blname = Config.BEAMLINENAME[bl[:3]]
        xbpm_str = f"{self.prm.xbpmdist} m" if self.prm.xbpmdist else "N/A"
        bpm_str  = f"{self.prm.bpmdist} m" if self.prm.bpmdist else "N/A"

        print(f"""
### Working beamline:\t   {blname} ({bl})
### Storage ring current : {self.prm.current}
### Grid step:             {self.prm.gridstep}
### Distance source-XBPM : {xbpm_str}
### Distance between BPMs: {bpm_str}
### Gap or phase:          {self.prm.phaseorgap}
"""
)

    def _blades_fetch(self) -> dict:
        """Retrieve each blade's data and average over their values."""
        data = dict()
        beamline = self.prm.beamline

        for dt in self.rawdata:
            try:
                xbpm = dt[0][beamline]
                vals = list()
                for blade in Config.BLADEMAP[beamline].values():
                    av, sd = self._blade_average(xbpm[f'{blade}_val'])
                    vals.append((av, sd))
                bpm_x = dt[2]['agx']
                bpm_y = dt[2]['agy']
                data[bpm_x, bpm_y] = np.array(vals)
            except Exception as err:
                print("\n WARNING: when fetching blades' values and averaging:"
                      f" {err}\n")
        return data

    def _blade_average(self, blade):
        """Calculate the average of blades' values for current beamline."""
        if self.prm.beamline in ["MGN", "MNC"]:
            return np.average(blade), np.std(blade)

        vals = np.array([
            vv * Config.AMPSUB[un] for vv, un in blade
        ])
        return np.average(vals), np.std(vals)

    def load_figures_from_hdf5(self, hdf5_path: str = None) -> dict:
        """Load and reconstruct figures from HDF5 file.

        Parameters
        ----------
        hdf5_path : str, optional
            Path to HDF5 file. If None, uses self._hdf5_path.

        Returns:
            dict: Dictionary with reconstructed figures, or empty dict if
                  no figures exist in the file.
        """
        try:
            import h5py  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "h5py is required to load figures from HDF5 files."
                "Please install it"
            ) from exc

        path = hdf5_path or self._hdf5_path
        if not path:
            return {}

        results = {}

        # Load analysis metadata up front (scales, BPM stats, sweeps fits)
        try:
            with h5py.File(path, 'r') as h5:
                self._load_hdf5_analysis_meta(h5)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load analysis metadata from HDF5")

        # Map HDF5 figure names to result dictionary keys
        figure_map = {
            'blade_map': 'blade_figure',
            'sweeps': 'sweeps_figure',
            'blades_center': 'blades_center_figure',
            'blades_at_sweeps': 'blades_center_figure',
            'blades_sweeps': 'blades_center_figure',
            'xbpm_raw_positions': 'xbpm_raw_pairwise_figure',
            'xbpm_scaled_positions': 'xbpm_scaled_pairwise_figure',
            'bpm_positions': 'bpm_figure',
            'bpm': 'bpm_figure',
        }

        # Additional variants for position figures
        position_variants = [
            ('xbpm_raw_pairwise', 'xbpm_raw_pairwise_figure'),
            ('xbpm_raw_cross', 'xbpm_raw_cross_figure'),
            ('xbpm_scaled_pairwise', 'xbpm_scaled_pairwise_figure'),
            ('xbpm_scaled_cross', 'xbpm_scaled_cross_figure'),
        ]

        for hdf5_name, result_key in figure_map.items():
            try:
                fig = reconstruct_figure_from_hdf5(path, hdf5_name)
                if fig is not None:
                    results[result_key] = fig
                    logger.info("Loaded figure: %s", hdf5_name)
            except (ValueError, KeyError) as exc:
                logger.debug(
                    "Figure %s not found in HDF5",
                    hdf5_name,
                    exc_info=exc,
                )

        # Try loading position variants
        for hdf5_name, result_key in position_variants:
            try:
                fig = reconstruct_figure_from_hdf5(path, hdf5_name)
                if fig is not None:
                    results[result_key] = fig
                    logger.info("Loaded figure: %s", hdf5_name)
            except (ValueError, KeyError) as exc:
                logger.debug(
                    "Figure %s not found in HDF5",
                    hdf5_name,
                    exc_info=exc,
                )

        return results


def reconstruct_figure_from_hdf5(h5_file, figure_name: str):
    """Reconstruct matplotlib figure from stored HDF5 data.

    Args:
        h5_file: Open h5py.File object or path to HDF5 file.
        figure_name: Name of figure to reconstruct. Options:
            - 'blade_map': Blade intensity heatmaps (2x2 subplots)
            - 'sweeps': Central horizontal/vertical sweeps
            - 'blades_center': Blade currents at center
            - 'xbpm_raw_pairwise': Raw XBPM pairwise position comparison
            - 'xbpm_raw_cross': Raw XBPM cross-blade position comparison
            - 'xbpm_scaled_pairwise': Scaled XBPM pairwise position comparison
            - 'xbpm_scaled_cross': Scaled XBPM cross-blade position comparison
            - 'bpm_positions': BPM position comparison

            Legacy names (still supported):
            - 'xbpm_raw_positions': Same as xbpm_raw_pairwise
            - 'xbpm_scaled_positions': Same as xbpm_scaled_pairwise

    Returns:
        matplotlib.figure.Figure: Reconstructed figure.

    Example:
        >>> import h5py
        >>> with h5py.File('analysis.h5', 'r') as h5:
        ...     fig = reconstruct_figure_from_hdf5(h5, 'sweeps')
        ...     fig.savefig('sweeps_reconstructed.png')
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py required for figure reconstruction") from exc

    # Handle file path or file object
    if isinstance(h5_file, str):
        with h5py.File(h5_file, 'r') as h5:
            return reconstruct_figure_from_hdf5(h5, figure_name)

    if 'analysis' not in h5_file:
        raise ValueError("No /analysis/ group found in HDF5 file")

    analysis_grp = h5_file['analysis']

    # Dispatch to reconstruction functions that read from /analysis/
    if figure_name == 'blade_map':
        return _reconstruct_blade_map(analysis_grp)
    elif figure_name == 'sweeps':
        return _reconstruct_sweeps(analysis_grp)
    elif figure_name == 'blades_center':
        return _reconstruct_blades_center(analysis_grp)
    else:
        # Handle all position figure types
        # For pairwise/cross/bpm, use figure_name directly as dataset name
        # For legacy names, map to base types
        legacy_map = {
            'xbpm_raw_positions': 'xbpm_raw',
            'xbpm_scaled_positions': 'xbpm_scaled',
            'bpm_positions': 'bpm',
        }

        # If it's a legacy name, use the mapped name, otherwise use
        # figure_name as-is
        dataset_name = legacy_map.get(figure_name, figure_name)

        # Validate figure name
        valid_names = ['xbpm_raw_pairwise', 'xbpm_raw_cross',
                      'xbpm_scaled_pairwise', 'xbpm_scaled_cross',
                      'bpm_positions', 'bpm', 'xbpm_raw', 'xbpm_scaled']
        if dataset_name not in valid_names:
            available = ['blade_map', 'sweeps', 'blades_center'] + valid_names
            raise ValueError(
                f"Unknown figure '{figure_name}'. Available: {available}"
            )

        return _reconstruct_positions(analysis_grp, dataset_name)


def _reconstruct_blade_map(analysis_grp):
    """Reconstruct blade intensity map from /analysis/blade_map/."""
    from .visualizers import BladeMapVisualizer

    if 'blade_map' not in analysis_grp:
        raise ValueError("No blade_map in /analysis/ group")

    return BladeMapVisualizer.plot_from_hdf5(analysis_grp['blade_map'])


def _reconstruct_sweeps(analysis_grp):
    """Reconstruct central sweeps from /analysis/ sweeps group.

    Accepts current name "sweeps" and legacy variants if present.
    """
    from .visualizers import SweepVisualizer

    sweeps_grp = (
        analysis_grp.get('sweeps') or
        analysis_grp.get('sweep') or
        analysis_grp.get('central_sweeps')
    )

    if sweeps_grp is None:
        raise ValueError("No sweeps in /analysis/ group")

    # Accept legacy dataset names as fallbacks
    h_data = (
        sweeps_grp.get('blades_h') or
        sweeps_grp.get('h_sweep') or
        sweeps_grp.get('horizontal')
    ) if sweeps_grp else None

    v_data = (
        sweeps_grp.get('blades_v') or
        sweeps_grp.get('v_sweep') or
        sweeps_grp.get('vertical')
    ) if sweeps_grp else None

    # Heuristic fallback: pick datasets by their fields if names differ
    if sweeps_grp is not None:
        for _, ds in sweeps_grp.items():
            if h_data is None and 'x_index' in ds.dtype.names:
                h_data = ds
            if v_data is None and 'y_index' in ds.dtype.names:
                v_data = ds
            if h_data is not None and v_data is not None:
                break

    # If neither dataset exists, indicate missing data
    if h_data is None and v_data is None:
        raise ValueError("No sweeps datasets found in sweeps group")

    return SweepVisualizer.plot_from_hdf5(h_data, v_data)


def _reconstruct_blades_center(analysis_grp):
    """Reconstruct blade currents at center from /analysis/sweeps/."""
    from .visualizers import BladeCurrentVisualizer

    if 'sweeps' not in analysis_grp:
        raise ValueError("No sweeps in /analysis/ group")

    sweeps_grp = analysis_grp['sweeps']
    h_data = sweeps_grp.get('blades_h')
    v_data = sweeps_grp.get('blades_v')

    return BladeCurrentVisualizer.plot_from_hdf5(h_data, v_data)


def _reconstruct_positions(analysis_grp, dataset_name):
    """Reconstruct position comparison figure from /analysis/positions/.

    Args:
        analysis_grp: HDF5 /analysis/ group
        dataset_name: Dataset name in /analysis/positions/
                     (e.g., 'xbpm_raw_pairwise', 'xbpm_scaled_cross', 'bpm')

    Returns:
        matplotlib figure with 3 subplots (full grid, ROI, RMS differences)
    """
    from .visualizers import PositionVisualizer

    if 'positions' not in analysis_grp:
        raise ValueError("No positions in /analysis/ group")

    positions_grp = analysis_grp['positions']

    if dataset_name not in positions_grp:
        raise ValueError(f"No {dataset_name} in /analysis/positions/")

    return PositionVisualizer.plot_from_hdf5(positions_grp[dataset_name])
