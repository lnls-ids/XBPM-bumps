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

    def write_supmat(self, supmat: np.ndarray) -> None:
        """Write suppression matrix to disk."""
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
        blades_ds = self._build_blades_dataset(data, grid_x, grid_y)
        table = self._build_nominal_index_table(data, grid_x, grid_y)

        with h5py.File(filepath, 'w') as h5:
            self._write_meta_and_params(h5, prm_dict)
            self._write_grid_and_data(
                h5, grid_x, grid_y, blades_ds, xbpmdist=self.prm.xbpmdist
            )
            if table is not None:
                self._write_tables(h5, table)
            self._write_derived(h5, results)
            if include_figures:
                self._write_figures(h5, results)

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
        if not (grid_x.size and grid_y.size):
            return None

        pos_maps = self._compute_pos_maps_for_table(data)

        fields = [
            ('x_nom', 'f8'), ('y_nom', 'f8'),
            ('to_mean', 'f8'), ('to_err', 'f8'),
            ('ti_mean', 'f8'), ('ti_err', 'f8'),
            ('bi_mean', 'f8'), ('bi_err', 'f8'),
            ('bo_mean', 'f8'), ('bo_err', 'f8'),
            ('raw_pair_x', 'f8'), ('raw_pair_y', 'f8'),
            ('scaled_pair_x', 'f8'), ('scaled_pair_y', 'f8'),
            ('raw_cross_x', 'f8'), ('raw_cross_y', 'f8'),
            ('scaled_cross_x', 'f8'), ('scaled_cross_y', 'f8'),
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

                rp = pos_maps['raw_pair'].get((float(x), float(y)))
                sp = pos_maps['scaled_pair'].get((float(x), float(y)))
                rc = pos_maps['raw_cross'].get((float(x), float(y)))
                sc = pos_maps['scaled_cross'].get((float(x), float(y)))

                row = (
                    float(x), float(y),
                    to_mean, to_err,
                    ti_mean, ti_err,
                    bi_mean, bi_err,
                    bo_mean, bo_err,
                    (float(rp[0]) if rp else np.nan),
                    (float(rp[1]) if rp else np.nan),
                    (float(sp[0]) if sp else np.nan),
                    (float(sp[1]) if sp else np.nan),
                    (float(rc[0]) if rc else np.nan),
                    (float(rc[1]) if rc else np.nan),
                    (float(sc[0]) if sc else np.nan),
                    (float(sc[1]) if sc else np.nan),
                )
                rows.append(row)

        table = np.array(rows, dtype=dtype)
        return table

    @staticmethod
    def _write_tables(h5file, table: np.ndarray) -> None:
        gtables = h5file.create_group('tables')
        gtables.create_dataset('nominal_index', data=table)

    def _compute_pos_maps_for_table(self, data: dict) -> dict:
        maps = {
            'raw_pair': {},
            'raw_cross': {},
            'scaled_pair': {},
            'scaled_cross': {},
        }
        try:
            from .processors import XBPMProcessor  # type: ignore
            proc = XBPMProcessor(data, self.prm)
            raw_res = proc.calculate_raw_positions(showmatrix=False)
            sca_res = proc.calculate_scaled_positions(showmatrix=False)

            def _extract(res, key):
                if not isinstance(res, dict):
                    return
                lst = res.get('positions', [])
                if not (isinstance(lst, list) and len(lst) == 2):
                    return
                pair_dict, cross_dict = lst
                if isinstance(pair_dict, dict):
                    maps[f'{key}_pair'] = {
                        k: tuple(v) for k, v in pair_dict.items()
                    }
                if isinstance(cross_dict, dict):
                    maps[f'{key}_cross'] = {
                        k: tuple(v) for k, v in cross_dict.items()
                    }

            _extract(raw_res, 'raw')
            _extract(sca_res, 'scaled')
        except Exception:
            logger.warning(
                "Failed to compute positions for table; continuing without",
                exc_info=True,
            )
        return maps

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
    def _create_grid_datasets(ggrid, grid_x, grid_y, xbpmdist: float) -> None:
        if grid_x.size:
            ggrid.create_dataset('x', data=grid_x)
        if grid_y.size:
            ggrid.create_dataset('y', data=grid_y)
        if xbpmdist and grid_x.size:
            ggrid.create_dataset('x_nom', data=(grid_x * xbpmdist))
        if xbpmdist and grid_y.size:
            ggrid.create_dataset('y_nom', data=(grid_y * xbpmdist))
        # Explicit grid pairs for indexing, in storage order (y desc, x asc)
        if grid_x.size and grid_y.size:
            yy = grid_y[::-1]
            xgrid, ygrid = np.meshgrid(grid_x, yy, indexing='xy')
            pairs = np.column_stack([xgrid.ravel(), ygrid.ravel()])
            ggrid.create_dataset('pairs', data=pairs)
            if xbpmdist:
                pairs_nom = np.column_stack([
                    (xgrid * xbpmdist).ravel(),
                    (ygrid * xbpmdist).ravel(),
                ])
                ggrid.create_dataset('pairs_nom', data=pairs_nom)

    @staticmethod
    def _attach_blades_scales(ggrid, blades) -> None:
        try:
            if 'y_nom' in ggrid:
                blades.dims[0].attach_scale(ggrid['y_nom'])
                blades.dims[0].label = 'y_nom'
            elif 'y' in ggrid:
                blades.dims[0].attach_scale(ggrid['y'])
                blades.dims[0].label = 'y'
            if 'x_nom' in ggrid:
                blades.dims[1].attach_scale(ggrid['x_nom'])
                blades.dims[1].label = 'x_nom'
            elif 'x' in ggrid:
                blades.dims[1].attach_scale(ggrid['x'])
                blades.dims[1].label = 'x'
        except Exception as exc:
            logger.warning("Failed attaching dimension scales to blades: %s",
                           exc)

    @staticmethod
    def _write_grid_and_data(h5file, grid_x, grid_y, blades_ds,
                             xbpmdist: float = 0.0) -> None:
        ggrid = h5file.create_group('grid')
        Exporter._create_grid_datasets(ggrid, grid_x, grid_y, xbpmdist)

        gdata = h5file.create_group('data')
        if blades_ds is not None and getattr(blades_ds, 'size', 0):
            blades = gdata.create_dataset('blades', data=blades_ds)
            try:
                import h5py  # type: ignore
                gdata.attrs['blade_order'] = np.array(
                    ['to', 'ti', 'bi', 'bo'],
                    dtype=h5py.string_dtype(),
                )
            except Exception:
                gdata.attrs['blade_order'] = 'to,ti,bi,bo'
            Exporter._attach_blades_scales(ggrid, blades)

    @staticmethod
    def _write_positions(group, positions: dict) -> None:
        for name in ('xbpm_raw', 'xbpm_scaled', 'bpm'):
            if name not in positions or positions[name] is None:
                continue
            sub = group.create_group(name)
            val = positions[name]
            meas = (np.asarray(val.get('measured'))
                    if isinstance(val, dict) else None)
            nom = (np.asarray(val.get('nominal'))
                   if isinstance(val, dict) else None)
            # Export measured/nominal preserving their original shape
            if meas is not None:
                sub.create_dataset('measured', data=meas)
            if nom is not None:
                sub.create_dataset('nominal', data=nom)
            # Also create 1D by_nominal table for easy access
            Exporter._write_by_nominal_table(sub, name, meas, nom)

    @staticmethod
    def _write_by_nominal_table(sub, name: str,
                                meas: Optional[np.ndarray],
                                nom: Optional[np.ndarray]) -> None:
        """Create `by_nominal` dataset with indexed 1D access.

        If measured/nominal are 2D grids, reshape to 1D for by_nominal table.
        If they're already 1D (point-indexed), use directly.
        """
        try:
            if meas is None or nom is None:
                return
            # If 2D grids (shape (ny, nx)), reshape to 1D
            meas_1d = (meas.reshape(-1, meas.shape[-1]) if meas.ndim > 2
                       else meas)
            nom_1d = (nom.reshape(-1, nom.shape[-1]) if nom.ndim > 2
                      else nom)

            if (meas_1d.shape[0] != nom_1d.shape[0]
                or meas_1d.shape[1] != 2 or nom_1d.shape[1] != 2):
                return
            by_nom = np.column_stack((nom_1d, meas_1d)).astype(float)
            dset = sub.create_dataset('by_nominal', data=by_nom)
            # Annotate columns by dataset type
            if name == 'xbpm_raw':
                cols = ['x_nom', 'y_nom', 'x_raw', 'y_raw']
            elif name == 'xbpm_scaled':
                cols = ['x_nom', 'y_nom', 'x_scaled', 'y_scaled']
            elif name == 'bpm':
                cols = ['x_nom', 'y_nom', 'x', 'y']
            else:
                cols = ['x_nom', 'y_nom', 'x', 'y']
            try:
                import h5py  # type: ignore
                dtype = h5py.string_dtype()
                dset.attrs['columns'] = np.array(cols, dtype=dtype)
            except Exception:
                dset.attrs['columns'] = ','.join(cols)
        except Exception:
            logger.warning(
                "Failed to build by_nominal for %s", name,
                exc_info=True,
            )

    def _write_derived(self, h5file, results: dict) -> None:
        gdrv = h5file.create_group('derived')
        positions = (results.get('positions', {})
                     if isinstance(results, dict) else {})
        gpos = gdrv.create_group('positions')
        self._write_positions(gpos, positions)

        supmat = (results.get('supmat')
                  if isinstance(results, dict) else None)
        if supmat is not None:
            gdrv.create_dataset('suppression_matrix', data=np.asarray(supmat))

        sweeps = (results.get('sweeps_data')
                  if isinstance(results, dict) else None)
        if sweeps:
            range_h, range_v, blades_h, blades_v = sweeps
            gsweeps = gdrv.create_group('sweeps')
            if range_h is not None:
                gsweeps.create_dataset('range_h', data=np.asarray(range_h))
            if range_v is not None:
                gsweeps.create_dataset('range_v', data=np.asarray(range_v))
            if isinstance(blades_h, dict):
                gh = gsweeps.create_group('blades_h')
                for k, arr in blades_h.items():
                    gh.create_dataset(str(k), data=np.asarray(arr))
            if isinstance(blades_v, dict):
                gv = gsweeps.create_group('blades_v')
                for k, arr in blades_v.items():
                    gv.create_dataset(str(k), data=np.asarray(arr))

    def _write_figures(self, h5file, results: dict) -> None:
        # Store figure arrays as 2D/3D image data (viewable in silx, etc.)
        figs = h5file.create_group('figures')
        figure_keys = {
            'blade_map': results.get('blade_figure'),
            'sweeps': results.get('sweeps_figure'),
            'blades_center': results.get('blades_center_figure'),
            'xbpm_raw_pairwise': results.get('xbpm_raw_pairwise_figure'),
            'xbpm_raw_cross': results.get('xbpm_raw_cross_figure'),
            'xbpm_scaled_pairwise': results.get(
                'xbpm_scaled_pairwise_figure'
            ),
            'xbpm_scaled_cross': results.get('xbpm_scaled_cross_figure'),
        }
        for name, fig in figure_keys.items():
            try:
                if fig is not None:
                    # Render figure to RGB array (height, width, 3)
                    img_array = self._fig_to_array(fig)
                    if img_array is not None:
                        dset = figs.create_dataset(name, data=img_array)
                        dset.attrs['format'] = 'RGBA'
                        dset.attrs['description'] = (
                            'Figure rendered as RGBA array (H x W x 4)'
                        )
            except Exception:
                logger.warning(
                    "Skipping figure %s during HDF5 export", name,
                    exc_info=True,
                )

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
