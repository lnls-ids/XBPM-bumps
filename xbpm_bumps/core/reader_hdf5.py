
"""HDF5 backend for XBPM DataReader."""
import h5py
import logging
import numpy as np


# --- Object-Oriented HDF5 Figure Reconstructor ---
class HDF5FigureReconstructor:
    """Encapsulates figure reconstruction from HDF5 files."""

    def __init__(self, hdf5_path):
        """Initialize with the path to the HDF5 file."""
        self.hdf5_path = hdf5_path

    def load_figures(self, analysis_meta_loader=None):
        """Load and reconstruct figures from the HDF5 file.

        Args:
            analysis_meta_loader (callable, optional): Function to load
                analysis meta.

        Returns:
            dict: Dictionary with reconstructed figures, or empty dict
                if none exist.
        """
        import logging
        logger = logging.getLogger(__name__)
        import h5py
        results = {}
        path = self.hdf5_path
        if analysis_meta_loader is not None:
            try:
                with h5py.File(path, 'r') as h5:
                    analysis_meta_loader(h5)
            except Exception:
                logger.exception("Failed to load analysis metadata from HDF5")

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

        position_variants = [
            ('xbpm_raw_pairwise', 'xbpm_raw_pairwise_figure'),
            ('xbpm_raw_cross', 'xbpm_raw_cross_figure'),
            ('xbpm_scaled_pairwise', 'xbpm_scaled_pairwise_figure'),
            ('xbpm_scaled_cross', 'xbpm_scaled_cross_figure'),
        ]

        for hdf5_name, result_key in figure_map.items():
            try:
                fig = self.reconstruct_figure(hdf5_name)
                if fig is not None:
                    results[result_key] = fig
                    logger.info("Loaded figure: %s", hdf5_name)
            except (ValueError, KeyError) as exc:
                logger.debug(
                    "Figure %s not found in HDF5",
                    hdf5_name,
                    exc_info=exc,
                )
        for hdf5_name, result_key in position_variants:
            try:
                fig = self.reconstruct_figure(hdf5_name)
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

    def reconstruct_figure(self, figure_name):
        """Reconstruct a matplotlib figure from stored HDF5 data.

        Args:
            figure_name (str): Name of figure to reconstruct.

        Returns:
            matplotlib.figure.Figure: Reconstructed figure, or None.
        """
        import h5py
        with h5py.File(self.hdf5_path, 'r') as h5:
            analysis_grp = self._find_analysis_group(h5)
            if analysis_grp is None:
                raise ValueError("No /analysis* group found in HDF5 file")
            if figure_name == 'blade_map':
                return self._reconstruct_blade_map(analysis_grp)
            elif figure_name == 'sweeps':
                return self._reconstruct_sweeps(analysis_grp)
            elif figure_name == 'blades_center':
                return self._reconstruct_blades_center(analysis_grp)
            else:
                legacy_map = {
                    'xbpm_raw_positions': 'xbpm_raw',
                    'xbpm_scaled_positions': 'xbpm_scaled',
                    'bpm_positions': 'bpm',
                }
                dataset_name = legacy_map.get(figure_name, figure_name)
                valid_names = [
                    'xbpm_raw_pairwise',
                    'xbpm_raw_cross',
                    'xbpm_scaled_pairwise',
                    'xbpm_scaled_cross',
                    'bpm_positions',
                    'bpm',
                    'xbpm_raw',
                    'xbpm_scaled'
                    ]

                if dataset_name not in valid_names:
                    available = (['blade_map', 'sweeps', 'blades_center'] +
                                 valid_names)
                    raise ValueError(
                        f"Unknown figure '{figure_name}'."
                        f" Available: {available}"
                    )
                return self._reconstruct_positions(analysis_grp, dataset_name)

    @staticmethod
    def _find_analysis_group(h5_file):
        """Find analysis group, prefer analysis_<beamline>."""
        for key in h5_file.keys():
            if key.startswith('analysis_'):
                return h5_file[key]
        if 'analysis' in h5_file:
            return h5_file['analysis']
        return None

    @staticmethod
    def _reconstruct_blade_map(analysis_grp):
        """Reconstruct blade map figure from analysis group."""
        from .visualizers import BladeMapVisualizer
        if 'blade_map' not in analysis_grp:
            raise ValueError("No blade_map in /analysis/ group")
        return BladeMapVisualizer.plot_from_hdf5(analysis_grp['blade_map'])

    @staticmethod
    def _reconstruct_sweeps(analysis_grp):
        """Reconstruct sweeps figure from analysis group."""
        from .visualizers import SweepVisualizer
        sweeps_grp = (
            analysis_grp.get('sweeps') or
            analysis_grp.get('sweep') or
            analysis_grp.get('central_sweeps')
        )
        if sweeps_grp is None:
            raise ValueError("No sweeps in /analysis/ group")
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
        if sweeps_grp is not None:
            for _, ds in sweeps_grp.items():
                if h_data is None and 'x_index' in ds.dtype.names:
                    h_data = ds
                if v_data is None and 'y_index' in ds.dtype.names:
                    v_data = ds
                if h_data is not None and v_data is not None:
                    break
        if h_data is None and v_data is None:
            raise ValueError("No sweeps datasets found in sweeps group")
        return SweepVisualizer.plot_from_hdf5(h_data, v_data)

    @staticmethod
    def _reconstruct_blades_center(analysis_grp):
        """Reconstruct blades center figure from analysis group."""
        from .visualizers import BladeCurrentVisualizer
        if 'sweeps' not in analysis_grp:
            raise ValueError("No sweeps in /analysis/ group")
        sweeps_grp = analysis_grp['sweeps']
        h_data = sweeps_grp.get('blades_h')
        v_data = sweeps_grp.get('blades_v')
        return BladeCurrentVisualizer.plot_from_hdf5(h_data, v_data)

    @staticmethod
    def _reconstruct_positions(analysis_grp, dataset_name):
        """Reconstruct position figure from analysis group."""
        from .visualizers import PositionVisualizer
        if 'positions' not in analysis_grp:
            raise ValueError("No positions in /analysis/ group")
        positions_grp = analysis_grp['positions']
        if dataset_name not in positions_grp:
            raise ValueError(f"No {dataset_name} in /analysis/positions/")
        return PositionVisualizer.plot_from_hdf5(positions_grp[dataset_name])


# --- Object-Oriented HDF5 Data Reader ---
class HDF5DataReader:
    """Encapsulates HDF5 file access and data extraction for XBPM."""

    def __init__(self, path):
        """Initialize HDF5DataReader with file path."""
        self.path = path
        self.h5 = None
        self.rawdata = None
        self.measured_data = None
        self.beamlines = None
        self.analysis_meta = None

    def __enter__(self):
        """Enter context: open HDF5 file."""
        self.h5 = h5py.File(self.path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: close HDF5 file."""
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def load_all(self):
        """Load all data: rawdata, measured_data, beamlines."""
        if 'raw_data' not in self.h5:
            self.rawdata, self.measured_data, self.beamlines = [], [], []
            return
        raw_grp = self.h5['raw_data']
        measurement_datasets = [
            k for k, v in raw_grp.items()
            if isinstance(v, h5py.Dataset) and k.startswith('measurements_')
        ]
        self.beamlines = self.extract_beamlines(raw_grp, measurement_datasets)
        self.measured_data = [
            self.parse_measurement_dataset(raw_grp, bl)
            for bl in self.beamlines
        ]
        self.rawdata = self.extract_sweeps(raw_grp)

    @staticmethod
    def extract_beamlines(raw_grp, measurement_datasets):
        """Extract available beamlines from measurement datasets."""
        beamlines = [name.replace('measurements_', '')
                     for name in measurement_datasets
                     if name.startswith('measurements_')]
        avail = raw_grp.attrs.get('available_beamlines')
        if avail:
            if isinstance(avail, bytes):
                avail = avail.decode()
            if isinstance(avail, str):
                beamlines = [
                    b.strip()
                    for b in avail.split(',')
                    if b.strip()
                    ]
            elif isinstance(avail, (list, tuple, np.ndarray)):
                beamlines = list(avail)
        elif measurement_datasets:
            first_ds = raw_grp[measurement_datasets[0]]
            avail = first_ds.attrs.get('available_beamlines')
            if avail:
                if isinstance(avail, bytes):
                    avail = avail.decode()
                if isinstance(avail, str):
                    beamlines = [
                        b.strip()
                        for b in avail.split(',')
                        if b.strip()
                        ]
                elif isinstance(avail, (list, tuple, np.ndarray)):
                    beamlines = list(avail)
        return beamlines

    @staticmethod
    def parse_measurement_dataset(raw_grp, beamline):
        """Parse measurement dataset for a given beamline."""
        ds_name = f"measurements_{beamline}"
        if ds_name not in raw_grp:
            return None
        dset = raw_grp[ds_name]
        meta = dict(dset.attrs.items())
        grid = None  # Not present in new structure
        bpm_dict = {}
        if hasattr(dset, 'dtype') and dset.dtype.names:
            for name in dset.dtype.names:
                bpm_dict[name] = dset[name][()]
        else:
            bpm_dict = np.array(dset)
        return (meta, grid, bpm_dict)

    @staticmethod
    def extract_sweeps(raw_grp):
        """Extract sweep data from raw_data, enforcing canonical structure."""
        import numpy as np
        rawdata = []
        expected_keys = [
            'current', 'agx', 'agy', 'posx', 'posy', 'orbx', 'orby',
            'MNC1', 'MNC2', 'cnb', 'mnc', 'orb'
        ]
        for key in raw_grp:
            if key.startswith('sweep_'):
                sweep_grp = raw_grp[key]
                meta = dict(sweep_grp.attrs.items())
                grid = (np.array(sweep_grp['grid'])
                        if 'grid' in sweep_grp else None)
                bpm_dict = {}
                # Collect all datasets and attrs
                for ds_name, ds in sweep_grp.items():
                    if ds_name == 'grid':
                        continue
                    arr = np.array(ds)
                    bpm_dict[ds_name] = arr
                for param in expected_keys:
                    # Prefer dataset, fallback to attribute
                    if param not in bpm_dict and param in sweep_grp.attrs:
                        bpm_dict[param] = sweep_grp.attrs[param]
                # Ensure all expected keys are present
                # (fill with None if missing)
                for param in expected_keys:
                    if param not in bpm_dict:
                        bpm_dict[param] = None
                rawdata.append((meta, grid, bpm_dict))
        return rawdata

    def get_analysis_meta(self, beamline=None):
        """Get analysis metadata for the given beamline."""
        meta = {}
        h5file = self.h5
        if beamline is None:
            for key in h5file.keys():
                if key.startswith('analysis_'):
                    beamline = key.replace('analysis_', '')
                    break
        group_name = f'analysis_{beamline}' if beamline else None
        if group_name and group_name in h5file:
            analysis = h5file[group_name]
            self._load_scales_meta(analysis, meta)
            self._load_sweeps_meta(analysis, meta)
            self._load_bpm_stats_meta(analysis, meta)
            self._load_roi_bounds_fallback(analysis, meta)
            self._load_supmat_meta(analysis, meta)
        return meta

    @staticmethod
    def _load_scales_meta(analysis, meta):
        scales = HDF5DataReader._read_scale_attrs(analysis)
        if not scales:
            scales_grp = analysis.get('scales')
            if scales_grp is not None:
                scales = HDF5DataReader._read_scales_group(scales_grp)
        if scales:
            meta['scales'] = scales

    @staticmethod
    def _load_sweeps_meta(analysis, meta):
        sweeps_meta = HDF5DataReader._read_sweeps_meta(analysis)
        if sweeps_meta:
            meta['sweeps'] = sweeps_meta

    @staticmethod
    def _load_supmat_meta(analysis, meta):
        mats = analysis.get('matrices') if analysis else None
        loaded_any = False
        if mats is not None:
            std = mats.get('standard')
            calc = mats.get('calculated')
            opt = mats.get('optimized')
            if std is not None:
                meta['supmat_standard'] = np.array(std)
                loaded_any = True
            if calc is not None:
                meta['supmat'] = np.array(calc)
                loaded_any = True
            if opt is not None:
                meta['supmat_optimized'] = np.array(opt)
                loaded_any = True
        if loaded_any:
            return
        if analysis is None:
            return
        supmat_calc = analysis.get('suppression_matrix')
        if supmat_calc is not None:
            meta['supmat'] = np.array(supmat_calc)
        opt_supmat = analysis.get('optimized_suppression_matrix')
        if opt_supmat is not None:
            meta['supmat_optimized'] = np.array(opt_supmat)
        if 'supmat_standard' not in meta:
            from .processors import XBPMProcessor
            meta['supmat_standard'] = (
                XBPMProcessor.standard_suppression_matrix()
                )

    @staticmethod
    def _load_bpm_stats_meta(analysis, meta):
        positions_grp = analysis.get('positions')
        if positions_grp is None:
            return
        bpm_ds = positions_grp.get('bpm')
        if bpm_ds is None:
            return
        stats = {}
        for key in ('sigma_h', 'sigma_v', 'sigma_total',
                    'diff_max_h', 'diff_max_v'):
            if key in bpm_ds.attrs:
                stats[key] = float(bpm_ds.attrs[key])
        roi_bounds = meta.get('bpm_stats', {}).get('roi_bounds')
        if not roi_bounds:
            roi_bounds = (
                HDF5DataReader._load_roi_bounds_fallback_attrs(positions_grp)
                )
            if roi_bounds:
                stats['roi_bounds'] = roi_bounds
        if stats:
            meta['bpm_stats'] = stats

    @staticmethod
    def _load_roi_bounds_fallback(analysis, meta):
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
            logging.debug("Failed to read roi_bounds attrs from positions",
                          exc_info=True)

    @staticmethod
    def _load_roi_bounds_fallback_attrs(positions_grp):
        try:
            rb_attrs = {
                k: v for k, v in positions_grp.attrs.items()
                if k in ('x_min', 'x_max', 'y_min', 'y_max',
                         'roi_bounds_title')
            }
            return rb_attrs if rb_attrs else None
        except Exception:
            return None

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

        positions = HDF5DataReader._collect_sweep_positions(h_ds, v_ds)
        blades = HDF5DataReader._collect_sweep_blade_trends(h_ds, v_ds)

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
            fit = HDF5DataReader._read_sweep_fit_attrs(dataset)
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
            fits = HDF5DataReader._fit_blade_trends(dataset, coord_field)
            if fits:
                blades[axis] = fits
        return blades

    @staticmethod
    def _read_sweep_fit_attrs(ds):
        if ds is None:
            return None
        attrs = {key: ds.attrs[key]
                 for key in ('k', 'delta', 's_k', 's_delta')
                  if key in ds.attrs}
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
                        'delta': float(coef[1])
                        }
                except Exception:
                    logging.debug("Failed to fit blade trend",
                                  exc_info=True,
                                  extra={'blade': blade,
                                         'coord_field': coord_field})
                    continue
            return fits or None
        except Exception:
            return None


# DEBUG
# print("\n\n #### DEBUG (reader_hdf5.read_hdf5): ####\n")
# print(f" rawdata type: {type(rawdata)}")
# print(f" rawdata [0][2]: {rawdata[0][2].keys() if rawdata else 'None'}")
# print(" rawdata[0][2].keys() ="
#       f" {rawdata[0][2].keys() if rawdata else 'None'}")
# print(" rawdata[0][2]['current'] = "
#       f"{rawdata[0][2]['current'] if rawdata else 'None'}")
# print("\n ########## END DEBUG reader_hdf5.read_hdf5 ##########\n\n")
# END DEBUG
