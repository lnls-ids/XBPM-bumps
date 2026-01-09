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

        # Comprehensive parameter descriptions for HDF5 metadata
        self.PARAM_DESCRIPTIONS = {
            # Command-line/runtime flags
            'showblademap': 'Display blade map visualization',
            'centralsweep': 'Process only central sweep data',
            'showbladescenter': 'Display blade center positions',
            'xbpmpositions': 'Calculate XBPM positions from BPM data',
            'xbpmfrombpm': 'Derive XBPM positions using BPM measurements',
            'xbpmpositionsraw': 'Calculate raw (unsuppressed) XBPM positions',
            'outputfile': 'Path to output HDF5 file',
            'workdir': 'Working directory for analysis',
            'skip': 'Number of initial sweeps to skip in processing',
            'gridstep': 'Grid step size for scan range',
            'maxradangle': 'Maximum radiation angle for analysis (mrad)',

            # Beamline configuration
            'beamline': 'Beamline identifier (e.g., MANACA, CATERETE)',
            'section': 'Beamline section/location',
            'current': 'Storage ring current during measurement (mA)',
            'phaseorgap': 'Undulator phase or gap value during measurement',

            # Distance parameters
            'bpmdist': 'Distance between BPM detectors (m)',
            'xbpmdist': 'Distance from radiation source to XBPM detector (m)',

            # Blade configuration
            'blademap': 'XBPM blade configuration mapping',
            'nroi': 'Number of regions of interest per blade',
        }

        # Analysis group and dataset descriptions
        self.ANALYSIS_DESCRIPTIONS = {
            'positions': 'Calculated beam positions from various methods',
            'positions/bpm': 'BPM-measured beam positions (x, y coordinates)',
            'positions/xbpm_raw_pairwise': (
                'Raw XBPM positions using pairwise blade analysis'
            ),
            'positions/xbpm_scaled_pairwise': (
                'Scaled XBPM positions with suppression correction'
            ),
            'positions/xbpm_raw_cross': (
                'Raw XBPM positions using cross-blade analysis'
            ),
            'positions/xbpm_scaled_cross': (
                'Scaled XBPM positions with cross-blade correction'
            ),

            'matrices': 'Suppression matrices for XBPM position correction',
            'matrices/standard': 'Standard suppression matrix (1/-1 pattern)',
            'matrices/calculated': (
                'Calculated suppression matrix from fitted blade slopes'
            ),
            'matrices/optimized': (
                'Optimized suppression matrix from beam orbit analysis'
            ),

            'scales': 'Scaling coefficients for position transformations',
            'scales/raw': 'Scaling factors for raw XBPM positions',
            'scales/scaled': 'Scaling factors for corrected XBPM positions',

            'sweeps': 'Blade sweep measurement data',
            'sweeps/blades_h': (
                'Horizontal blade sweep positions and currents'
            ),
            'sweeps/blades_v': (
                'Vertical blade sweep positions and currents'
            ),
        }

        # Descriptions for BPM statistics saved on positions/bpm
        self.BPM_STATS_DESCRIPTIONS = {
            'sigma_h': 'Horizontal BPM position std deviation',
            'sigma_v': 'Vertical BPM position std deviation',
            'sigma_total': 'Combined BPM position std deviation',
            'diff_max_h': 'Max horizontal |x_meas - x_nom| [μm]',
            'diff_max_v': 'Max vertical |y_meas - y_nom| [μm]',
        }

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
                   include_figures: bool = True, rawdata: list = None) -> None:
        """Write full analysis package to an HDF5 file.

        Args:
            filepath: Path to output HDF5 file.
            data: Blade measurement data dictionary.
            results: Analysis results dictionary.
            include_figures: Whether to include figure objects.
            rawdata: Raw data tuples list [(meta, grid, bpm_dict), ...].
                     If provided, BPM monitoring data will be stored for
                     complete re-analysis capability.
        """
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
            raw_grp = None
            if table is not None:
                raw_grp = self._write_data(h5, table)
            # If no table was written, ensure raw_data group exists so sweeps
            # have a place to live.
            if raw_grp is None:
                raw_grp = h5.create_group('raw_data')

            self._write_derived(h5, results)
            if rawdata:
                self._write_bpm_data(raw_grp, rawdata)
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

    def _write_data(self, h5file, table: np.ndarray):
        """Write blade measurements table indexed by nominal positions.

        Stores dataset as /raw_data/measurements_<beamline> to indicate
        which beamline the averaging belongs to.
        """
        raw_grp = h5file.create_group('raw_data')
        beamline = getattr(self.prm, 'beamline', None) or 'unknown'
        raw_grp.create_dataset(f'measurements_{beamline}', data=table)
        return raw_grp

    def _write_meta_and_params(self, h5file, prm_dict: dict) -> None:
        meta = h5file.create_group('meta')
        meta.attrs['version'] = 1
        meta.attrs['created'] = datetime.utcnow().isoformat() + 'Z'
        meta.attrs['beamline'] = self.prm.beamline

        # Create parameters group with description
        gprm = h5file.create_group('parameters')
        gprm.attrs['description'] = (
            'Analysis input parameters and beamline configuration'
        )

        # Write each parameter value and its description
        for k, v in prm_dict.items():
            if v is None:
                continue
            try:
                gprm.attrs[k] = v
            except TypeError:
                gprm.attrs[k] = str(v)

            # Add description for this parameter
            desc_key = f'{k}_description'
            if k in self.PARAM_DESCRIPTIONS:
                gprm.attrs[desc_key] = self.PARAM_DESCRIPTIONS[k]

    @staticmethod
    def _write_bpm_data(raw_grp, rawdata: list) -> None:
        """Write raw acquisition sweeps to HDF5 under /raw_data.

        Stores the third element of each rawdata tuple (BPM/XBPM dict)
        in /raw_data/sweep_XXXX/ groups for complete re-analysis capability.

        Args:
            raw_grp: HDF5 group to store sweeps in (raw_data group).
            rawdata: List of tuples [(metadata, grid_data, bpm_dict), ...].
        """
        if not rawdata or raw_grp is None:
            return

        import logging
        import re
        logger = logging.getLogger(__name__)

        machine_pattern = re.compile(r'^[A-Z]{3,4}[0-9]?$')

        for i, entry in enumerate(rawdata):
            if not Exporter._validate_rawdata_entry(entry, i, logger):
                continue

            metadata, grid_data, bpm_dict = entry[:3]
            sweep_grp = raw_grp.create_group(f'sweep_{i:04d}')

            # Store non-machine metadata as attributes
            Exporter._store_metadata_attrs(sweep_grp, metadata)

            # Store machine data from metadata (MNC1, MNC2, etc.) as datasets
            if isinstance(metadata, dict):
                for key, val in metadata.items():
                    is_machine = machine_pattern.match(str(key))
                    if is_machine and isinstance(val, dict):
                        try:
                            Exporter._store_machine_group(sweep_grp, key, val)
                        except Exception as exc:
                            logger.debug(
                                "Could not store metadata['%s'] for "
                                "sweep %d: %s", key, i, exc
                            )

            # Store BPM dict data
            Exporter._store_bpm_dict(sweep_grp, bpm_dict, i, logger)

    @staticmethod
    def _validate_rawdata_entry(entry, index: int, logger) -> bool:
        """Validate a rawdata entry has the expected structure.

        Args:
            entry: Rawdata entry to validate.
            index: Entry index for logging.
            logger: Logger instance.

        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            logger.warning(
                "Skipping rawdata[%d]: expected 3-tuple, got %s",
                index, type(entry)
            )
            return False
        return True

    @staticmethod
    def _store_metadata_attrs(group, metadata) -> None:
        """Store metadata dictionary as HDF5 attributes.

        Machine data (keys like MNC1, MNC2, CAT, etc.) are excluded
        and should be handled separately by _store_bpm_dict.

        Args:
            group: HDF5 group to store attributes in.
            metadata: Metadata dictionary.
        """
        if not isinstance(metadata, dict):
            return

        # Keys that look like machine names (uppercase, 3-4 chars)
        # should not be stored as metadata attributes
        import re
        machine_pattern = re.compile(r'^[A-Z]{3,4}[0-9]?$')

        for key, val in metadata.items():
            if val is None:
                continue
            # Skip machine data keys - they'll be stored as datasets
            if machine_pattern.match(str(key)):
                continue
            try:
                group.attrs[f'meta_{key}'] = val
            except (TypeError, ValueError):
                group.attrs[f'meta_{key}'] = str(val)

    @staticmethod
    def _store_bpm_dict(group, bpm_dict, sweep_index: int, logger) -> None:
        """Store BPM dictionary with proper hierarchy.

        Coordinates three-pass storage: metadata, orbit, machine data.
        """
        if not isinstance(bpm_dict, dict):
            logger.warning(
                "Skipping BPM data for sweep %d: expected dict, got %s",
                sweep_index, type(bpm_dict)
            )
            return

        metadata_keys = {'agx', 'agy', 'posx', 'posy', 'current'}
        orbit_keys = {'orbx', 'orby'}

        Exporter._store_bpm_metadata_attrs(group, bpm_dict, metadata_keys,
                                          sweep_index, logger)
        Exporter._store_bpm_orbit_data(group, bpm_dict, orbit_keys,
                                      sweep_index, logger)
        Exporter._store_bpm_machine_data(group, bpm_dict, metadata_keys,
                                        orbit_keys, sweep_index, logger)

    @staticmethod
    def _store_bpm_metadata_attrs(group, bpm_dict, metadata_keys,
                                 sweep_index: int, logger) -> None:
        """Store BPM metadata fields as HDF5 attributes."""
        for key in metadata_keys:
            if key in bpm_dict and bpm_dict[key] is not None:
                try:
                    Exporter._store_scalar_attr(group, key, bpm_dict[key])
                except Exception as exc:
                    logger.debug(
                        "Could not store bpm_dict['%s'] for sweep %d: %s",
                        key, sweep_index, exc
                    )

    @staticmethod
    def _store_bpm_orbit_data(group, bpm_dict, orbit_keys,
                             sweep_index: int, logger) -> None:
        """Extract and store orbit data as compound dataset."""
        orbit_data = {k: v for k, v in bpm_dict.items()
                     if k in orbit_keys and v is not None}
        if orbit_data:
            try:
                Exporter._store_orbit_dataset(group, orbit_data)
            except Exception as exc:
                logger.debug(
                    "Could not store orbit data for sweep %d: %s",
                    sweep_index, exc
                )

    @staticmethod
    def _store_bpm_machine_data(group, bpm_dict, metadata_keys,
                               orbit_keys, sweep_index: int, logger) -> None:
        """Store machine data (MNC1, MNC2, etc.) as compound datasets."""
        import numpy as np

        for key, val in bpm_dict.items():
            if val is None or key in metadata_keys or key in orbit_keys:
                continue

            try:
                if isinstance(val, dict):
                    Exporter._store_machine_group(group, key, val)
                else:
                    group.create_dataset(key, data=np.asarray(val))
            except Exception as exc:
                logger.debug(
                    "Could not store bpm_dict['%s'] for sweep %d: %s",
                    key, sweep_index, exc
                )

    @staticmethod
    def _store_scalar_attr(group, key: str, val) -> None:
        """Store scalar value as attribute."""
        try:
            group.attrs[key] = val
        except (TypeError, ValueError):
            group.attrs[key] = str(val)

    @staticmethod
    def _store_orbit_dataset(group, orbit_data: dict) -> None:
        """Store orbit data (orbx, orby) as single compound dataset.

        Creates compound dataset with orbx_val and orby_val columns only.
        Orbit values do not have explicit units.

        Args:
            group: HDF5 group to store data in.
            orbit_data: Dictionary with 'orbx' and/or 'orby' keys.
        """
        import numpy as np

        if not orbit_data:
            return

        # Determine array length from first orbit data
        first_key = next(iter(orbit_data))
        first_val = np.asarray(orbit_data[first_key])

        # Determine length (extract just values, ignore units)
        if (isinstance(first_val, tuple) or
            (first_val.ndim == 2 and first_val.shape[1] == 2)):
            arr_len = len(first_val)
        else:
            arr_len = len(first_val)

        # Build dtype: orbx_val, orby_val (values only, no units)
        dtype_fields = []
        for key in ['orbx', 'orby']:
            if key in orbit_data:
                dtype_fields.append((f'{key}_val', 'f8'))

        if not dtype_fields:
            return

        dtype = np.dtype(dtype_fields)
        data = np.empty(arr_len, dtype=dtype)

        # Fill in data (extract values, ignore units)
        for key in ['orbx', 'orby']:
            if key not in orbit_data:
                continue

            arr = np.asarray(orbit_data[key])

            if arr.ndim == 2 and arr.shape[1] == 2:
                # (value, unit) tuple structure - extract only values
                data[f'{key}_val'] = arr[:, 0]
            else:
                # Just values
                data[f'{key}_val'] = arr

        group.create_dataset('orb', data=data)

    @staticmethod
    def _store_machine_group(group, key: str, val) -> None:
        """Store machine data (MNC1, MNC2, etc.) as a compound dataset.

        Requirements:
        - The machine (e.g., MNC1) is a dataset directly under the sweep group.
        - Each parameter (A_val, A_range, B_val, ...) becomes a column.
        - Units are stored as attributes per column ("<param>_unit").
        - "prefix" is stored as a dataset attribute (metadata), not a column.
        """
        if not isinstance(val, dict):
            Exporter._create_simple_machine_dataset(group, key, val)
            return

        fields, values, prefix, n_rows = Exporter._prepare_machine_fields(val)
        if not fields:
            return

        data = Exporter._build_machine_array(fields, values, n_rows)
        ds = group.create_dataset(key, data=data)
        Exporter._write_prefix_attr(ds, prefix)

    @staticmethod
    def _split_value_and_unit(value):
        """Extract values and units from parameter data.

        Handles two formats:
        1. List of (value, unit) tuples: [(v1, u1), (v2, u2), ...]
           Returns: array([v1, v2, ...]), u1 (assumes units are same)
        2. List of scalars: [v1, v2, ...]
           Returns: array([v1, v2, ...]), None
        3. Single tuple: (value, unit)
           Returns: value, unit
        4. Scalar: value
           Returns: value, None
        """
        import numpy as np

        # Check if it's a list of tuples (e.g., [(49702, 0), (49997, 0), ...])
        if (isinstance(value, list) and len(value) > 0 and
            isinstance(value[0], (list, tuple))):
            # Extract values and units separately
            values = [item[0] for item in value]
            units = [item[1] for item in value]
            # Assume all units are the same, take first
            unit = units[0] if units else None
            return np.array(values), unit

        # Single tuple (value, unit)
        if (isinstance(value, (tuple)) and len(value) == 2 and not
            isinstance(value[0], (list, tuple))):
            return value[0], value[1]

        # List of scalars or single scalar
        return value, None

    @staticmethod
    def _create_simple_machine_dataset(group, key: str, val) -> None:
        import numpy as np
        group.create_dataset(key, data=np.asarray(val))

    @staticmethod
    def _prepare_machine_fields(val: dict):
        import numpy as np

        fields = []
        values = {}
        prefix = val.get('prefix')
        n_rows = None

        for name, raw in val.items():
            if name == 'prefix':
                continue
            value, unit = Exporter._split_value_and_unit(raw)
            arr = np.asarray(value, dtype='f8')

            # Detect number of rows from first array parameter
            if n_rows is None and arr.ndim > 0:
                n_rows = arr.size

            # Add value column
            fields.append((name, 'f8'))

            # Add unit column for val_* parameters
            if unit is not None:
                fields.append((f'{name}_unit', 'f8'))

            values[name] = (arr, unit)

        if n_rows is None:
            n_rows = 1

        return fields, values, prefix, n_rows

    @staticmethod
    def _build_machine_array(fields, values, n_rows):
        import numpy as np

        dtype = np.dtype(fields)
        data = np.empty(n_rows, dtype=dtype)

        for name, (value, unit) in values.items():
            arr = np.asarray(value, dtype='f8')

            # Fill value column
            if arr.ndim == 0:
                # Scalar: replicate to all rows
                data[name][:] = float(arr)
            else:
                # Array: use values for each row
                data[name][:] = arr.flatten()[:n_rows]

            # Fill unit column if present
            if unit is not None and f'{name}_unit' in dtype.names:
                try:
                    data[f'{name}_unit'][:] = float(unit)
                except (ValueError, TypeError):
                    data[f'{name}_unit'][:] = np.nan

        return data

    @staticmethod
    def _write_prefix_attr(dataset, prefix):
        if prefix is None:
            return
        try:
            dataset.attrs['prefix'] = prefix
            dataset.attrs['prefix_description'] = (
                'Common field prefix used in this dataset'
            )
        except Exception:
            dataset.attrs['prefix'] = str(prefix)
            dataset.attrs['prefix_description'] = (
                'Common field prefix used in this dataset'
            )

    @staticmethod
    def _write_unit_attrs(dataset, values: dict) -> None:
        for name, (_value, unit) in values.items():
            # Only write units for parameters that have them
            # (e.g., val_A, val_B).
            # Skip parameters without units (e.g., A_range, B_range)
            if unit is None:
                continue
            try:
                dataset.attrs[f'{name}_unit'] = float(unit)
            except (ValueError, TypeError):
                dataset.attrs[f'{name}_unit'] = str(unit)
            dataset.attrs[f'{name}_unit_description'] = (
                f"Unit for field '{name}'"
            )

    def _write_positions(self, group, positions: dict) -> None:
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
            self._write_position_table(group, name, meas, nom)

    @staticmethod
    def _dict_to_arrays(pos_dict: dict):
        """Convert {(x_nom, y_nom): [x_meas, y_meas]} to arrays."""
        if not pos_dict:
            return None, None

        keys = list(pos_dict.keys())
        nom = np.array(keys)  # Shape (N, 2)
        meas = np.array([pos_dict[k] for k in keys])  # Shape (N, 2)
        return meas, nom

    def _write_position_table(self, group, name: str,
                              meas: Optional[np.ndarray],
                              nom: Optional[np.ndarray]):
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

            # Add dataset description
            desc_key = f'positions/{name}'
            if desc_key in self.ANALYSIS_DESCRIPTIONS:
                dset.attrs['description'] = (
                    self.ANALYSIS_DESCRIPTIONS[desc_key]
                )

            # Add figure metadata directly to dataset
            dset.attrs['xlabel'] = 'x [μm]'
            dset.attrs['xlabel_description'] = (
                'X axis label for plotting positions'
            )
            dset.attrs['ylabel'] = 'y [μm]'
            dset.attrs['ylabel_description'] = (
                'Y axis label for plotting positions'
            )
            dset.attrs['title'] = (
                f"{name.replace('_', ' ').title()} Positions"
            )
            dset.attrs['title_description'] = (
                'Plot title for this position dataset'
            )

            # Calculate and store statistics
            diff_x = meas_1d[:, 0] - nom_1d[:, 0]
            diff_y = meas_1d[:, 1] - nom_1d[:, 1]
            diff_rms = np.sqrt(diff_x**2 + diff_y**2)
            dset.attrs['rms_mean'] = float(np.mean(diff_rms))
            dset.attrs['rms_mean_description'] = (
                'Mean RMS of deviation sqrt(dx^2 + dy^2)'
            )
            dset.attrs['rms_std'] = float(np.std(diff_rms))
            dset.attrs['rms_std_description'] = (
                'Standard deviation of RMS deviation'
            )
            dset.attrs['n_points'] = len(nom_1d)
            dset.attrs['n_points_description'] = (
                'Number of position points'
            )
            return dset
        except Exception:
            logger.warning(
                "Failed to build position table for %s", name,
                exc_info=True,
            )
        return None

    def _write_derived(self, h5file, results: dict) -> None:
        """Write analysis results to HDF5 file."""
        positions = (results.get('positions', {})
                     if isinstance(results, dict) else {})
        supmat = (results.get('supmat')
                  if isinstance(results, dict) else None)
        supmat_standard = (results.get('supmat_standard')
                          if isinstance(results, dict) else None)
        sweeps = (results.get('sweeps_data')
                  if isinstance(results, dict) else None)
        bpm_stats = (results.get('bpm_stats')
                     if isinstance(results, dict) else None)

        analysis = h5file.create_group('analysis')
        analysis.attrs['description'] = (
            'Analysis results including positions, matrices, and sweep data'
        )

        apos = analysis.create_group('positions')
        apos.attrs['description'] = self.ANALYSIS_DESCRIPTIONS['positions']

        raw_full = results.get('positions_raw_full')
        scaled_full = results.get('positions_scaled_full')

        self._write_full_positions(apos, raw_full, scaled_full)
        self._write_bpm_positions(apos, positions, bpm_stats)
        self._write_fallback_positions(apos, results, positions,
                                       raw_full, scaled_full)
        self._attach_roi_bounds_attrs(apos, bpm_stats)
        self._write_supmat_dataset(analysis, supmat, supmat_standard)
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
            scales = (raw_full.get('scales')
                      if isinstance(raw_full, dict) else None)
            pair_dset = self._write_position_table(
                apos, 'xbpm_raw_pairwise', pairwise_meas, pairwise_nom
            )
            cross_dset = self._write_position_table(
                apos, 'xbpm_raw_cross', cross_meas, cross_nom
            )
            self._attach_scale_attrs(pair_dset, scales, 'pair')
            self._attach_scale_attrs(cross_dset, scales, 'cross')

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
            scales = (scaled_full.get('scales')
                      if isinstance(scaled_full, dict) else None)
            pair_dset = self._write_position_table(
                apos, 'xbpm_scaled_pairwise', pairwise_meas, pairwise_nom
            )
            cross_dset = self._write_position_table(
                apos, 'xbpm_scaled_cross', cross_meas, cross_nom
            )
            self._attach_scale_attrs(pair_dset, scales, 'pair')
            self._attach_scale_attrs(cross_dset, scales, 'cross')

    def _write_bpm_positions(self, apos, positions,
                             bpm_stats: dict = None) -> None:
        if not (isinstance(positions, dict) and 'bpm' in positions):
            return
        bpm_data = positions['bpm']
        if not isinstance(bpm_data, dict):
            return

        meas = (np.asarray(bpm_data.get('measured'))
                if 'measured' in bpm_data else None)
        nom = (np.asarray(bpm_data.get('nominal'))
               if 'nominal' in bpm_data else None)

        dset = self._write_position_table(apos, 'bpm', meas, nom)
        self._attach_bpm_stats_attrs(dset, bpm_stats)

    def _attach_bpm_stats_attrs(self, dset, bpm_stats: dict) -> None:
        """Attach BPM statistics attributes and descriptions to dataset."""
        if dset is None or not isinstance(bpm_stats, dict):
            return
        for key, desc in self.BPM_STATS_DESCRIPTIONS.items():
            val = bpm_stats.get(key)
            if val is None:
                continue
            try:
                dset.attrs[key] = float(val)
            except Exception:
                dset.attrs[key] = val
            dset.attrs[f'{key}_description'] = desc

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
            dset_pair = self._write_position_table(
                apos, f'{pos_type}_pairwise', pairwise_meas, pairwise_nom
            )
            dset_cross = self._write_position_table(
                apos, f'{pos_type}_cross', cross_meas, cross_nom
            )
            self._attach_scale_attrs(dset_pair,
                                     results.get('scales', {}), 'pair')
            self._attach_scale_attrs(dset_cross,
                                     results.get('scales', {}), 'cross')
            return
        # Standard dict format (old tests)
        self._write_positions(apos, positions)

    def _write_supmat_dataset(self, analysis, supmat,
                             supmat_standard=None) -> None:
        """Write suppression matrices to HDF5 under analysis/matrices.

        Structure:
            analysis/
              matrices/
                standard    (#1)
                calculated  (#2)
                optimized   (#3 - placeholder for now)
        """
        # Ensure matrices group exists
        matrices = analysis.create_group('matrices')
        try:
            matrices.attrs['description'] = (
                self.ANALYSIS_DESCRIPTIONS['matrices']
            )
        except Exception:
            logger.warning(
                "Failed to write matrices description attribute",
                exc_info=True
            )

        # Write standard matrix (#1) if provided
        if supmat_standard is not None:
            ds_std = matrices.create_dataset('standard',
                                             data=np.asarray(supmat_standard))
            try:
                ds_std.attrs['description'] = (
                    self.ANALYSIS_DESCRIPTIONS['matrices/standard']
                )
            except Exception:
                logger.warning(
                    "Failed to set standard matrix description",
                    exc_info=True,
                )

        # Write calculated matrix (#2)
        if supmat is not None:
            supmat_arr = np.asarray(supmat)
            ds_calc = matrices.create_dataset('calculated', data=supmat_arr)
            try:
                ds_calc.attrs['description'] = (
                    self.ANALYSIS_DESCRIPTIONS['matrices/calculated']
                )
            except Exception:
                logger.warning(
                    "Failed to set calculated matrix description",
                    exc_info=True,
                )

            # Optimized placeholder (#3)
            opt_ds = matrices.create_dataset('optimized', data=supmat_arr)
            try:
                opt_ds.attrs['description'] = (
                    self.ANALYSIS_DESCRIPTIONS['matrices/optimized']
                )
                opt_ds.attrs['is_placeholder'] = True
            except Exception:
                logger.warning(
                    "Failed to set is_placeholder on optimized matrix",
                    exc_info=True
                )

        # Backward-compatible dataset at /analysis/suppression_matrix
        # Prefer calculated if available, otherwise standard
        try:
            if supmat is not None:
                analysis.create_dataset('suppression_matrix',
                                        data=np.asarray(supmat))
            elif supmat_standard is not None:
                analysis.create_dataset('suppression_matrix',
                                        data=np.asarray(supmat_standard))
        except Exception:
            logger.warning(
                "Failed to write backward-compatible suppression_matrix",
                exc_info=True,
            )

    def _write_scales(self, analysis, raw_full, scaled_full) -> None:
        """Persist scaling coefficients for raw and scaled runs."""
        scales_raw = (raw_full.get('scales')
                      if isinstance(raw_full, dict) else None)
        scales_scl = (scaled_full.get('scales')
                      if isinstance(scaled_full, dict) else None)
        if not scales_raw and not scales_scl:
            return

        grp = analysis.create_group('scales')
        grp.attrs['description'] = self.ANALYSIS_DESCRIPTIONS['scales']
        if scales_raw:
            self._write_scale_group(grp, 'raw', scales_raw)
            try:
                grp['raw'].attrs['description'] = (
                    self.ANALYSIS_DESCRIPTIONS['scales/raw']
                )
            except Exception:
                logger.warning(
                    "Failed to set description on scales/raw",
                    exc_info=True,
                )
        if scales_scl:
            self._write_scale_group(grp, 'scaled', scales_scl)
            try:
                grp['scaled'].attrs['description'] = (
                    self.ANALYSIS_DESCRIPTIONS['scales/scaled']
                )
            except Exception:
                logger.warning(
                    "Failed to set description on scales/scaled",
                    exc_info=True,
                )

    @staticmethod
    def _write_scale_group(parent, name: str, scales: dict) -> None:
        sub = parent.create_group(name)
        for key in ('pair', 'cross'):
            val = scales.get(key)
            if not isinstance(val, dict):
                continue
            child = sub.create_group(key)
            for attr in ('kx', 'ky', 'dx', 'dy'):
                if attr in val and val[attr] is not None:
                    child.attrs[attr] = float(val[attr])

    @staticmethod
    def _attach_scale_attrs(dset, scales: dict, key: str) -> None:
        if dset is None or not isinstance(scales, dict):
            return
        vals = scales.get(key)
        if not isinstance(vals, dict):
            return
        for attr in ('kx', 'ky', 'dx', 'dy'):
            if attr in vals and vals[attr] is not None:
                dset.attrs[f'scale_{attr}'] = float(vals[attr])
                dset.attrs[f'scale_{attr}_description'] = (
                    f"Scale parameter '{attr}' for {key} positions"
                )

    @staticmethod
    def _write_bpm_stats(analysis, stats: dict) -> None:
        if not stats:
            return
        grp = analysis.create_group('bpm_stats')
        for key, val in stats.items():
            if isinstance(val, dict):
                sub = grp.create_group(key)
                for sk, sv in val.items():
                    sub.attrs[sk] = float(sv)
            else:
                grp.attrs[key] = float(val)

    @staticmethod
    def _attach_roi_bounds_attrs(positions_group, bpm_stats: dict) -> None:
        """Persist ROI bounds as attributes on /analysis/positions."""
        if positions_group is None or not bpm_stats:
            return
        roi = (bpm_stats.get('roi_bounds')
               if isinstance(bpm_stats, dict) else None)
        if not isinstance(roi, dict):
            return
        positions_group.attrs['roi_bounds_title'] = 'ROI bounds'
        positions_group.attrs['roi_bounds_title_description'] = (
            'Label for region-of-interest bounds metadata'
        )
        for key in ('x_min', 'x_max', 'y_min', 'y_max'):
            if key in roi and roi[key] is not None:
                positions_group.attrs[key] = float(roi[key])
                positions_group.attrs[f'{key}_description'] = (
                    f"ROI bound for '{key.split('_')[0]}' [μm]"
                )

    def _write_sweeps_group(self, analysis, sweeps) -> None:
        if not sweeps:
            return
        sweeps_new = analysis.create_group('sweeps')
        sweeps_new.attrs['description'] = (
            self.ANALYSIS_DESCRIPTIONS['sweeps']
        )
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
            Exporter._attach_blade_fit_attrs(
                dset, blade_order, blade_arrays, has_errors,
                np.asarray(index_range) * xbpmdist
            )
            self._set_sweep_metadata(dset, name, fit_coef)
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

    def _set_sweep_metadata(
        self, dset, name: str, fit_coef: np.ndarray
    ) -> None:
        """Attach sweep plot metadata and fit attributes."""
        # Add dataset description
        desc_key = f'sweeps/{name}'
        if desc_key in self.ANALYSIS_DESCRIPTIONS:
            dset.attrs['description'] = (
                self.ANALYSIS_DESCRIPTIONS[desc_key]
            )

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

        # Axis and title attribute descriptions
        dset.attrs['xlabel_description'] = (
            'Label for calculated position axis'
        )
        dset.attrs['ylabel_description'] = (
            'Label for calculated position axis'
        )
        dset.attrs['title_description'] = (
            'Plot title for central sweep data'
        )
        dset.attrs['xlabel_blades_description'] = (
            'Label for blade angle axis'
        )

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

        # Fit attributes descriptions
        dset.attrs['k_description'] = (
            'Slope of fitted line from central sweep'
        )
        dset.attrs['delta_description'] = (
            'Intercept of fitted line from central sweep'
        )
        if 's_k' in dset.attrs:
            dset.attrs['s_k_description'] = (
                'Uncertainty of slope (standard error)'
            )
        if 's_delta' in dset.attrs:
            dset.attrs['s_delta_description'] = (
                'Uncertainty of intercept (standard error)'
            )

    @staticmethod
    def _attach_blade_fit_attrs(dset, blade_order, blade_arrays,
                                has_errors: bool,
                                index_vals: np.ndarray) -> None:
        """Store per-blade linear fit coefficients (k, delta) as attrs."""
        if dset is None or blade_arrays is None:
            return

        for blade, arr in zip(blade_order, blade_arrays, strict=True):
            try:
                y = arr[:, 0] if has_errors and arr.ndim == 2 else arr
                if y is None or len(y) != len(index_vals):
                    continue
                if not np.any(np.isfinite(y)) or np.nanstd(y) == 0:
                    continue
                weights = None
                if has_errors and arr.ndim == 2:
                    err = arr[:, 1]
                    if np.any(err <= 0) or not np.all(np.isfinite(err)):
                        err = None
                    if err is not None:
                        weights = 1.0 / err
                coef = np.polyfit(index_vals, y, deg=1, w=weights)
                dset.attrs[f'k_{blade}'] = float(coef[0])
                dset.attrs[f'delta_{blade}'] = float(coef[1])
                dset.attrs[f'k_{blade}_description'] = (
                    f"Slope of fitted line for blade '{blade}'"
                )
                dset.attrs[f'delta_{blade}_description'] = (
                    f"Intercept of fitted line for blade '{blade}'"
                )
            except Exception as exc:
                logger.debug("Failed to fit blade %s: %s", blade, exc)
                continue

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
