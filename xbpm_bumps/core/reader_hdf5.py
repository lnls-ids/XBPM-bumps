"""HDF5 backend for XBPM DataReader."""

import h5py
import logging
import numpy as np
import sys
import matplotlib

from .config import Config
from data_structure import BladeAvgData, SweepData


# --- Object-Oriented HDF5 Data Reader ---
class HDF5DataReader:
    """Encapsulates HDF5 file access and data extraction for XBPM."""

    def __init__(self, filepath: str) -> None:
        """Initialize HDF5DataReader with file path.
        
        Args:
            filepath (str) : the HDF5 file with data.
        
        """
        self.filepath      = filepath
        self.h5            = None
        self.rawdata       = None
        self.beamlines     = None
        self.measured_data = None
        self.analysis_meta = None

    def __enter__(self: "HDF5DataReader") -> "HDF5DataReader":
        """Enter context: open HDF5 file."""
        self.h5 = h5py.File(self.filepath, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: close HDF5 file."""
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def _read_raw_data(self, bldata: h5py.Group, beamline: str) -> tuple:
        """Read raw data for a given beamline from HDF5 group."""
        sweeps = {}
        for key, data in bldata.items():
            # Blade averages.
            if key == "blade_averages":
                blade_avg = BladeAvgData.from_hdf5_group(data=data)

            # Sweep data.
            elif key.startswith('sweep_'):
                # Extract sweep number
                num         = int(key.split('_')[1])
                sweeps[num] = SweepData.from_hdf5_group(data=data)

            else:
                print(f" WARNING: Unknown key '{key}'"
                        f" in beamline '{beamline}'. Skipping.")

        return sweeps, blade_avg

    def load_data(self, filepath: str) -> None:
        self.rawdata = {}
        with h5py.File(filepath, 'r') as hf:
            for beamline, bldata in hf.items():
                if beamline not in Config.BLADEMAP.keys():
                    print(f" WARNING: Unknown beamline '{beamline}'"
                          " defined in HDF5 file. Skipping.")
                    continue

                # Get raw data.
                sweeps, blade_avg = self._read_raw_data(bldata, beamline)

                # Assemble the extracted data in the rawdata dictionary.
                self.rawdata[beamline] = {
                    'sweeps': sweeps,
                    'bladeavg': blade_avg
                    }


# --- Object-Oriented HDF5 Figure Reconstructor ---
class HDF5FigureReconstructor:
    """Encapsulates figure reconstruction from HDF5 files."""

    def __init__(self, hdf5_path: str) -> None:
        """Initialize with the path to the HDF5 file."""
        self.hdf5_path = hdf5_path

    def load_figures(self,
                     analysis_meta_loader: callable = None) -> dict:
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
            if result_key in results:
                continue
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

    def reconstruct_figure(self, figure_name: str):
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
                return self._reconstruct_positions(
                    h5, analysis_grp, dataset_name
                )

    @staticmethod
    def _find_analysis_group(h5_file: h5py.File) -> h5py.Group | None:
        """Find analysis group, prefer analysis_<beamline>."""
        for key in h5_file.keys():
            if key.startswith('analysis_'):
                return h5_file[key]
        if 'analysis' in h5_file:
            return h5_file['analysis']
        return None

    @staticmethod
    def _reconstruct_blade_map(
        analysis_grp: h5py.Group) -> "matplotlib.figure.Figure":
        """Reconstruct blade map figure from analysis group."""
        from .visualizers import BladeMapVisualizer
        if 'blade_map' not in analysis_grp:
            raise ValueError("No blade_map in /analysis/ group")
        return BladeMapVisualizer.plot_from_hdf5(analysis_grp['blade_map'])

    @staticmethod
    def _reconstruct_sweeps(
        analysis_grp: h5py.Group) -> "matplotlib.figure.Figure":
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
    def _reconstruct_blades_center(
        analysis_grp: h5py.Group) -> "matplotlib.figure.Figure":
        """Reconstruct blades center figure from analysis group.

        Uses the canonical plotting function with blade data extracted
        from HDF5, ensuring consistency with live analysis.
        """
        from .visualizers import BladeCurrentVisualizer

        sweeps_grp = (analysis_grp.get('sweeps')
                      or analysis_grp.get('central_sweeps'))
        if sweeps_grp is None:
            raise ValueError("No sweeps in /analysis/ group")
        h_data = sweeps_grp.get('blades_h')
        v_data = sweeps_grp.get('blades_v')

        return BladeCurrentVisualizer.plot_from_hdf5(h_data, v_data)

    @staticmethod
    def _reconstruct_positions(h5_file: h5py.File,
                analysis_grp: h5py.Group,
                dataset_name: str) -> "matplotlib.figure.Figure":
        """Reconstruct position figure from analysis group."""
        from .visualizers import PositionVisualizer
        if 'positions' not in analysis_grp:
            raise ValueError("No positions in /analysis/ group")
        positions_grp = analysis_grp['positions']

        # Support legacy/new naming variants for the same semantic figure.
        candidates = [dataset_name]
        if dataset_name == 'bpm':
            candidates.append('bpm_positions')
        elif dataset_name == 'bpm_positions':
            candidates.append('bpm')

        selected_name = None
        for name in candidates:
            if name in positions_grp:
                selected_name = name
                break
        if selected_name is None and dataset_name in ('bpm', 'bpm_positions'):
            return HDF5FigureReconstructor._reconstruct_bpm_from_raw(
                h5_file, analysis_grp
            )
        if selected_name is None:
            raise ValueError(
                f"No {dataset_name} in /analysis/positions/ "
                f"(tried {candidates})"
            )

        roi_bounds = {}
        for key in ('x_min', 'x_max', 'y_min', 'y_max'):
            if key in positions_grp.attrs:
                roi_bounds[key] = positions_grp.attrs[key]

        # Prefer explicitly stored ROI sizes over coordinate inference.
        roi_h_size = analysis_grp.attrs.get('roi_h_size', None)
        roi_v_size = analysis_grp.attrs.get('roi_v_size', None)
        stored_roi_size = None
        if roi_h_size is not None and roi_v_size is not None:
            stored_roi_size = (int(roi_h_size), int(roi_v_size))

        # Infer title context from dataset/group naming conventions.
        name = (selected_name or '').lower()
        calc = 'pairwise' if 'pair' in name else ('cross' if 'cross' in name
                                                   else '')
        if 'raw' in name:
            rort = 'raw'
        elif 'scaled' in name or 'transf' in name:
            rort = 'Transf.'
        else:
            rort = ''

        beamline = analysis_grp.attrs.get('beamline', '')
        if not beamline:
            grp_name = analysis_grp.name.rsplit('/', 1)[-1]
            if grp_name.startswith('analysis_'):
                beamline = grp_name.split('analysis_', 1)[1]

        return PositionVisualizer.plot_from_hdf5(
            positions_grp[selected_name],
            beamline=beamline,
            rort=rort,
            calc=calc,
            roi_bounds=roi_bounds or None,
            roi_size=stored_roi_size,
        )

    @staticmethod
    def _reconstruct_bpm_from_raw(h5_file: h5py.File,
            analysis_grp: h5py.Group) -> "matplotlib.figure.Figure":
        """Fallback reconstruction of BPM positions from raw_data sweeps."""
        from .parameters import Prm
        from .processors import BPMProcessor
        from .config import Config

        raw_grp = h5_file.get('raw_data')
        if raw_grp is None:
            raise ValueError("No BPM dataset in positions and no raw_data")

        rawdata = HDF5DataReader.extract_sweeps(raw_grp)
        if not rawdata:
            raise ValueError("No BPM dataset in positions and raw_data is empty")

        beamline = analysis_grp.attrs.get('beamline', '')
        if not beamline:
            grp_name = analysis_grp.name.rsplit('/', 1)[-1]
            if grp_name.startswith('analysis_'):
                beamline = grp_name.split('analysis_', 1)[1]

        base_bl = beamline[:3] if beamline else ''
        section = Config.SECTIONS.get(base_bl)
        if not section:
            raise ValueError(
                "No BPM dataset in positions and section is unavailable "
                "to reconstruct from raw_data"
            )

        prm = Prm()
        prm.beamline = beamline
        prm.section = section
        prm.bpmdist = Config.BPMDISTS.get(base_bl)
        prm.xbpmdist = analysis_grp.attrs.get(
            'xbpmdist', Config.XBPMDISTS.get(beamline,
                                             Config.XBPMDISTS.get(base_bl))
        )

        bpm_processor = BPMProcessor(rawdata, prm)
        bpm_processor.calculate_positions()
        return bpm_processor.fig



    # @staticmethod
    # def parse_measurement_dataset(raw_grp: h5py.Group,
    #                               beamline: str) -> tuple:
    #     """Parse measurement dataset for a given beamline."""
    #     ds_name = f"measurements_{beamline}"
    #     if ds_name not in raw_grp:
    #         return None
    #     dset = raw_grp[ds_name]
    #     meta = dict(dset.attrs.items())
    #     grid = None  # Not present in new structure
    #     bpm_dict = {}
    #     if hasattr(dset, 'dtype') and dset.dtype.names:
    #         for name in dset.dtype.names:
    #             bpm_dict[name] = dset[name][()]
    #     else:
    #         bpm_dict = np.array(dset)
    #     return (meta, grid, bpm_dict)

    def get_analysis_meta(self, beamline=None) -> dict:
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
            bpm_stats = meta.get('bpm_stats', {}) if isinstance(meta, dict) else {}
            needs_bpm_derivation = not (
                isinstance(bpm_stats, dict) and
                any(key in bpm_stats for key in ('sigma_h', 'sigma_v', 'sigma_total'))
            )
            if needs_bpm_derivation:
                bpm_stats = self._derive_bpm_stats_from_raw(
                    h5file, analysis, beamline
                )
                if isinstance(bpm_stats, dict) and bpm_stats:
                    meta['bpm_stats'] = bpm_stats
            self._load_supmat_meta(analysis, meta)
        return meta

    @staticmethod
    def _derive_bpm_stats_from_raw(h5file, analysis, beamline):
        """Derive BPM stats from /raw_data when /positions/bpm is missing."""
        raw_grp = h5file.get('raw_data') if h5file is not None else None
        if raw_grp is None:
            return None

        rawdata = HDF5DataReader.extract_sweeps(raw_grp)
        if not rawdata:
            return None

        from .parameters import Prm
        from .processors import BPMProcessor
        from .config import Config
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO

        beamline_val = beamline or analysis.attrs.get('beamline', '')
        base_bl = beamline_val[:3] if beamline_val else ''

        section = analysis.attrs.get('section') or Config.SECTIONS.get(base_bl)
        bpmdist = analysis.attrs.get('bpmdist') or Config.BPMDISTS.get(base_bl)
        xbpmdist = (analysis.attrs.get('xbpmdist')
                    or Config.XBPMDISTS.get(beamline_val)
                    or Config.XBPMDISTS.get(base_bl))

        if section is None or bpmdist is None or xbpmdist is None:
            return None

        prm = Prm()
        prm.beamline = beamline_val
        prm.section = section
        prm.bpmdist = float(bpmdist)
        prm.xbpmdist = float(xbpmdist)

        bpm_processor = BPMProcessor(rawdata, prm)
        sink = StringIO()
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                bpm_processor.calculate_positions()
        except Exception:
            return None
        return getattr(bpm_processor, 'last_stats', None)

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
            stddev = mats.get('stddev')
            if std is not None:
                meta['supmat_standard'] = np.array(std)
                loaded_any = True
            if calc is not None:
                meta['supmat'] = np.array(calc)
                loaded_any = True
            if opt is not None:
                meta['supmat_optimized'] = np.array(opt)
                loaded_any = True
            if stddev is not None:
                meta['stddevmat'] = np.array(stddev)
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
            from .config import Config
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Recalculating standard suppression matrix from processor "
                "(not found in HDF5). If this is unexpected, verify HDF5 "
                "file contains matrices/standard dataset."
            )
            supmat_standard, stddevmat_standard = (
                Config.standard_suppression_matrix()
            )
            meta['supmat_standard'] = supmat_standard
            meta.setdefault('stddevmat', stddevmat_standard)

    @staticmethod
    def _load_bpm_stats_meta(analysis, meta):
        positions_grp = analysis.get('positions')
        if positions_grp is None:
            return
        bpm_ds = positions_grp.get('bpm') or positions_grp.get(
            'bpm_positions'
        )
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
            for key in (
                'scale_kx', 'scale_skx', 'scale_dx', 'scale_sdx',
                'scale_ky', 'scale_sky', 'scale_dy', 'scale_sdy',
                'scale_s_kx', 'scale_s_dx', 'scale_s_ky', 'scale_s_dy',
            ):
                if key in dset.attrs:
                    normalized = key.replace('scale_', '')
                    legacy_map = {
                        's_kx': 'skx',
                        's_dx': 'sdx',
                        's_ky': 'sky',
                        's_dy': 'sdy',
                    }
                    attrs[legacy_map.get(normalized, normalized)] = dset.attrs[key]
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
        if analysis_grp is None:
            return {}
        sweeps = analysis_grp.get('sweeps') or analysis_grp.get(
            'central_sweeps'
        )
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
