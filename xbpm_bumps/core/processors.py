"""XBPM and BPM data processors."""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .parameters  import Prm                     # noqa: E272
from .visualizers import PositionVisualizer
from .constants   import ROI_SIZE_V, ROI_SIZE_H, FIGDPI    # noqa: E272
from .config      import Config                  # noqa: E272

_Title = Config.get_plot_title   # shorthand for plot titles
# from .exporters import Exporter

# Keep math font consistent with visualizers (Computer Modern / cm).
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'
matplotlib.rcParams['font.family'] = 'serif'


class XBPMProcessor:
    """Processes XBPM data to calculate beam positions.

    This class handles all calculation logic for XBPM position analysis:
    - Central sweep analysis to determine suppression matrices
    - Pairwise and cross-blade position calculations
    - Raw (no suppression) and scaled (with suppression) positions
    - Blade behavior analysis at central positions

    Attributes:
        data (dict): Measurement data dictionary.
        prm (Prm): Parameters dataclass.
        range_h (np.ndarray): Horizontal sweep range.
        range_v (np.ndarray): Vertical sweep range.
        blades_h (dict): Blade measurements along horizontal center line.
        blades_v (dict): Blade measurements along vertical center line.
        suppression_matrix_val: Calculated suppression matrix.
    """

    def __init__(self, data: dict, prm: Prm):
        """Initialize processor with data and parameters.

        Args:
            data: Measurement data dictionary.
            prm: Parameters dataclass instance.
        """
        self.data     = data
        self.prm      = prm
        self.range_h  = None
        self.range_v  = None
        self.blades_h = None
        self.blades_v = None
        self.suppression_matrix_val = None
        self.roi_h_size = prm.roisize[0] if prm.roisize else ROI_SIZE_H
        self.roi_v_size = prm.roisize[1] if prm.roisize else ROI_SIZE_V

    def analyze_central_sweeps(self, show: bool = False) -> tuple:
        """Analyze blade behavior at central sweep positions.

        Examines blade measurements along central horizontal and vertical
        lines to understand blade response and calculate suppression factors.

        Args:
            show: Whether to display sweep plots.

        Returns:
            Tuple of (range_h, range_v, blades_h, blades_v, pos_h, pos_v)
            where pos_h and pos_v are the calculated positions.
        """
        keys = np.array(list(self.data.keys()))
        self.range_h = np.unique(keys[:, 0])
        self.range_v = np.unique(keys[:, 1])

        # Run through central horizontal line if data is not just a point
        if len(self.range_h) > 1:
            (pos_ch_v, fit_ch_v,
             self.blades_h) = self._central_sweep_horizontal()
        else:
            pos_ch_v = np.zeros(len(self.range_h))
            fit_ch_v, self.blades_h = None, None

        # Run through central vertical line if data is not just a point
        if len(self.range_v) > 1:
            pos_cv_h, fit_cv_h, self.blades_v = self._central_sweep_vertical()
        else:
            pos_cv_h = np.zeros(len(self.range_v))
            fit_cv_h, self.blades_v = None, None

        if show:
            self._central_sweeps_show(pos_ch_v, fit_ch_v, pos_cv_h, fit_cv_h)

        return (self.range_h, self.range_v, self.blades_h, self.blades_v,
                pos_ch_v, pos_cv_h, fit_ch_v, fit_cv_h)

    def _central_sweep_horizontal(self) -> tuple:
        """Extract blade measurements along horizontal center line."""
        try:
            to_ch = np.array([self.data[jj, 0][0] for jj in self.range_h])
            ti_ch = np.array([self.data[jj, 0][1] for jj in self.range_h])
            bi_ch = np.array([self.data[jj, 0][2] for jj in self.range_h])
            bo_ch = np.array([self.data[jj, 0][3] for jj in self.range_h])
            blades_h = {"to": to_ch, "ti": ti_ch, "bi": bi_ch, "bo": bo_ch}
        except Exception as err:
            print("\n WARNING: horizontal sweeping interrupted,"
                  f" data grid may be incomplete: {err}")
            blades = {bl: np.array([[1., 0] for _ in self.range_h])
                      for bl in ["to", "ti", "bi", "bo"]}
            return None, None, blades

        pos_to_ti_v = (to_ch + ti_ch)
        pos_bi_bo_v = (bo_ch + bi_ch)
        pos_ch_v = (pos_to_ti_v - pos_bi_bo_v) / (pos_to_ti_v + pos_bi_bo_v)
        fit_ch_v = np.polyfit(self.range_h, pos_ch_v, deg=1)

        return pos_ch_v, fit_ch_v, blades_h

    def _central_sweep_vertical(self) -> tuple:
        """Extract blade measurements along vertical center line."""
        try:
            to_cv = np.array([self.data[0, jj][0] for jj in self.range_v])
            ti_cv = np.array([self.data[0, jj][1] for jj in self.range_v])
            bi_cv = np.array([self.data[0, jj][2] for jj in self.range_v])
            bo_cv = np.array([self.data[0, jj][3] for jj in self.range_v])
            blades_v = {"to": to_cv, "ti": ti_cv, "bi": bi_cv, "bo": bo_cv}
        except Exception as err:
            print("\n WARNING: vertical sweeping interrupted,"
                  f" data grid may be incomplete: {err}")
            blades = {bl: np.array([[1., 0] for _ in self.range_v])
                      for bl in ["to", "ti", "bi", "bo"]}
            return None, None, blades

        pos_to_bo_h = (to_cv + bo_cv)
        pos_ti_bi_h = (ti_cv + bi_cv)
        pos_cv_h = (pos_to_bo_h - pos_ti_bi_h) / (pos_to_bo_h + pos_ti_bi_h)
        fit_cv_h = np.polyfit(self.range_v, pos_cv_h, deg=1)

        return pos_cv_h, fit_cv_h, blades_v

    def _central_sweeps_show(self, pos_ch_v, fit_ch_v, pos_cv_h, fit_cv_h):
        """Plot results from fittings on central sweeps."""
        from .visualizers import SweepVisualizer

        # Extract fit coefficients if available
        fit_h = fit_ch_v[:, 0] if fit_ch_v is not None else None
        fit_v = fit_cv_h[:, 0] if fit_cv_h is not None else None

        # Extract position values
        pos_h = pos_ch_v[:, 0] if pos_ch_v is not None else None
        pos_v = pos_cv_h[:, 0] if pos_cv_h is not None else None

        fig = SweepVisualizer.plot_from_arrays(
            self.range_h, self.range_v,
            pos_h, pos_v,
            fit_h, fit_v,
            xbpm_dist=self.prm.xbpmdist
        )

        if self.prm.outputfile:
            outfile = f"xbpm_sweeps_{self.prm.beamline}.png"
            fig.savefig(outfile, dpi=FIGDPI)
            print(f" Figure of central sweeps saved to file {outfile}.\n")

    def show_blades_at_center(self) -> None:
        """Display blade measurements along central sweeping points."""
        from .visualizers import BladeCurrentVisualizer

        # Ensure we have sweep data
        if self.range_h is None or self.range_v is None:
            self.analyze_central_sweeps(show=False)

        if self.blades_h is None and self.blades_v is None:
            print("\n WARNING: could not retrieve blades' currents,"
                  " maybe there is insufficient data."
                  " Skipping central analysis.")
            return

        fig = BladeCurrentVisualizer.plot_from_dicts(
            self.blades_h, self.blades_v,
            self.range_h, self.range_v,
            beamline=self.prm.beamline
        )

        if self.prm.outputfile:
            outfile = f"central_sweep_{self.prm.beamline}.png"
            fig.savefig(outfile, dpi=FIGDPI)
            print("\n Figure of blades behaviour at central sweeps"
                  f" saved to file {outfile}.\n")

    def _roi_slice_indices(self, array):
        """Extract centered ROI slice indices from an array, handling 1D/2D."""
        n_lin, n_col = array.shape
        n_roi_h  = min(self.roi_h_size, n_col)
        n_roi_v  = min(self.roi_v_size, n_lin)
        fr_col     = max(0, int((n_col - n_roi_h) / 2))
        up_col     = min(n_col, fr_col + n_roi_h)
        fr_row     = max(0, int((n_lin - n_roi_v) / 2))
        up_row     = min(n_lin, fr_row + n_roi_v)

        if fr_col == up_col:
            dim = 'h'
        elif fr_row == up_row:
            dim = 'v'
        else:
            dim = '2d'

        return (fr_col, up_col, fr_row, up_row), dim

    def _extract_roi_slice(self, array, dim,
                           fr_col, up_col, fr_row, up_row):
        """Check whether array is 1D along one axis and extract accordingly.
        
        Args:
            array: Input array to extract ROI from.
            fr_col: Starting index for horizontal slice.
            up_col: Ending index for horizontal slice.
            fr_row: Starting index for vertical slice.
            up_row: Ending index for vertical slice.

        Returns:
            Extracted ROI slice from the input array.
        """
        if dim == 'h':
            return array[0, fr_col:up_col]
        elif dim == 'v':
            return array[fr_row:up_row, 0]
        else:
            return array[fr_row:up_row, fr_col:up_col]

    def _process_position_type(self, calc_type,  # noqa: D417
                               pos_all_h, pos_all_v,
                               pos_roi_h, pos_roi_v,
                               pos_nom_h, pos_nom_v,
                               pos_nom_h_roi,
                               pos_nom_v_roi, nosuppress, dim):
        """Process a single position type (pairwise or cross-blade).

        Args:
            calc_type: 'pairwise' or 'cross'
            pos_all_h/v: Full position array (measured)
            pos_nom_h/v: Nominal position array (reference)
            pos_nom_h/v_roi: ROI slice of nominal positions
            nosuppress: If True, label results as raw mode.
            dim : Dimension of ROI ('h', 'v', or '2d') for RMS calculation.

        Returns:
            Dict with scaled positions, scales, stats, visualizer.
        """
        # Perform scaling fit
        label = "Δ/Σ" if calc_type == "pairwise" else "Partial Δ/Σ"
        (kx, deltax, ky, deltay) = XBPMProcessor.scaling_fit(
            pos_roi_h, pos_roi_v,
            pos_nom_h_roi, pos_nom_v_roi, label
        )

        # Set raw (R) or transformed (T) graph type.
        transform = "R" if nosuppress else "T"

        # Build title map for visualizer with formatted titles from registry.
        title_map = {
            'total'   : _Title('xbpm_positions', 'total',
                               beamline=self.prm.beamline,
                               rort=transform,
                               calc_type=calc_type),
            'roi'     : _Title('xbpm_positions', 'roi',
                               beamline=self.prm.beamline,
                               rort=transform,
                               calc_type=calc_type),
            'heatmap' : _Title('xbpm_positions', 'heatmap',
                               beamline=self.prm.beamline,
                               rort=transform,
                               calc_type=calc_type),
        }

        # Scale full positions
        pos_all_h_scaled = kx * pos_all_h + deltax
        pos_all_v_scaled = ky * pos_all_v + deltay
        pos_roi_h_scaled = kx * pos_roi_h  + deltax
        pos_roi_v_scaled = ky * pos_roi_v  + deltay

        # Compute statistics
        diffx2_roi = (pos_roi_h_scaled - pos_nom_h_roi) ** 2
        diffy2_roi = (pos_roi_v_scaled - pos_nom_v_roi) ** 2
        stats      = self._calculate_roi_stats(diffx2_roi, diffy2_roi)
        diffxyroi  = np.sqrt(diffx2_roi + diffy2_roi)

        if dim == 'h':
            diffroi = diffx2_roi
        elif dim == 'v':
            diffroi = diffy2_roi
        else:
            diffroi = diffxyroi

        # Visualize
        visualizer = PositionVisualizer(self.prm, titles=title_map)
        visualizer.show_position_results(
            pos_nom_h, pos_nom_v,
            pos_all_h_scaled, pos_all_v_scaled,
            pos_roi_h_scaled, pos_roi_v_scaled,
            pos_nom_h_roi, pos_nom_v_roi,
            diffroi
        )

        return {
            'h_scaled'     : pos_all_h_scaled,
            'v_scaled'     : pos_all_v_scaled,
            'h_roi_scaled' : pos_roi_h_scaled,
            'v_roi_scaled' : pos_roi_v_scaled,
            'kx'           : kx,
            'ky'           : ky,
            'dx'           : deltax,
            'dy'           : deltay,
            'stats'        : stats,
            'visualizer'   : visualizer,
        }

    def _compile_results(self, pair_result, cross_result,
                         supmat, nosuppress,
                         pos_nom_h, pos_nom_v):
        """Compile and save final results from pairwise and cross-blade."""
        pair_visualizer  = pair_result['visualizer']
        cross_visualizer = cross_result['visualizer']

        # Save figures if requested
        if self.prm.outputfile:
            outdir = '.'
            sup = "raw" if nosuppress else "scaled"
            bl = self.prm.beamline

            outfile_p = os.path.join(outdir, f"xbpm_pair_pos_{sup}_{bl}.png")
            pair_visualizer.save_figure(outfile_p)

            outfile_c = os.path.join(outdir, f"xbpm_cross_pos_{sup}_{bl}.png")
            cross_visualizer.save_figure(outfile_c)

        # Build position dictionaries for export
        scaled_pos_pair  = dict()
        scaled_pos_cross = dict()
        for ii, lin in enumerate(pos_nom_h):
            for jj, xx in enumerate(lin):
                yy = pos_nom_v[ii, jj]
                scaled_pos_pair[xx, yy] = [
                    pair_result['h_scaled'][ii, jj],
                    pair_result['v_scaled'][ii, jj]
                ]
                scaled_pos_cross[xx, yy] = [
                    cross_result['h_scaled'][ii, jj],
                    cross_result['v_scaled'][ii, jj]
                ]

        return {
            'positions'       : [scaled_pos_pair, scaled_pos_cross],
            'pairwise_figure' : pair_visualizer.fig,
            'cross_figure'    : cross_visualizer.fig,
            'scales' : {
                'pair'   : {
                    'kx' : pair_result['kx'],
                    'ky' : pair_result['ky'],
                    'dx' : pair_result['dx'],
                    'dy' : pair_result['dy'],
                },
                'cross'  : {
                    'kx' : cross_result['kx'],
                    'ky' : cross_result['ky'],
                    'dx' : cross_result['dx'],
                    'dy' : cross_result['dy'],
                },
            },
            'supmat' : supmat,
            'xbpm_stats' : {
                'pairwise' : pair_result['stats'],
                'cross'    : cross_result['stats'],
            },
        }

    def xbpm_position_calculation(self, pos_nom_h, pos_nom_v,
                                  nosuppress: bool = False,
                                  showmatrix: bool = True) -> dict:
        """Orchestrate position calculation for pairwise and cross-blade.

        Delegates to helpers for reduced complexity while maintaining
        full analysis pipeline.
        """
        # Ensure sweep data is available for suppression matrix estimation.
        if (self.range_h is None or self.range_v is None or
                self.blades_h is None or self.blades_v is None):
            self.analyze_central_sweeps(show=False)

        # Parse and compute core data
        blades, _ = self.data_parse()
        supmat = self.suppression_matrix(showmatrix=showmatrix,
                                         nosuppress=nosuppress)

        # Extract nominal ROI slices.
        from_upto, dim = self._roi_slice_indices(pos_nom_h)
        pos_nom_h_roi  = self._extract_roi_slice(pos_nom_h, dim, *from_upto)
        pos_nom_v_roi  = self._extract_roi_slice(pos_nom_v, dim, *from_upto)

        # Pairwise calculation (Delta/Sigma).
        pos_pair = self.beam_position_pair(supmat)
        (_, _, pos_h, pos_v) = self.position_dict_parse(pos_pair)

        # Extract ROI slices from measured data.
        pos_roi_h = self._extract_roi_slice(pos_h, dim, *from_upto)
        pos_roi_v = self._extract_roi_slice(pos_v, dim, *from_upto)

        # Process data: fitting, scaling, stats, visualization.
        pairwise_result = self._process_position_type(
                'pairwise', pos_h, pos_v, pos_roi_h, pos_roi_v,
                pos_nom_h, pos_nom_v, pos_nom_h_roi, pos_nom_v_roi,
                nosuppress, dim
            )

        # Cross-blade calculation (partial Delta/Sigma).
        pos_h, pos_v = self.beam_position_cross(blades)

        # Extract ROI slices from measured data.
        pos_roi_h = self._extract_roi_slice(pos_h, dim, *from_upto)
        pos_roi_v = self._extract_roi_slice(pos_v, dim, *from_upto)

        # Process data: fitting, scaling, stats, visualization.
        cross_result = self._process_position_type(
            'cross', pos_h, pos_v, pos_roi_h, pos_roi_v,
            pos_nom_h, pos_nom_v, pos_nom_h_roi, pos_nom_v_roi,
            nosuppress, dim
            )

        # Compile and return results
        return self._compile_results(
            pairwise_result, cross_result,
            supmat, nosuppress,
            pos_nom_h, pos_nom_v,
        )

    @staticmethod
    def standard_suppression_matrix():
        """Return the standard suppression matrix with 1/-1 pattern.

        This is the fixed pattern used for raw position calculations,
        independent of blade behavior or data fitting.

        Returns:
            np.ndarray: 4x4 standard suppression matrix
        """
        return np.array([
            [1, -1, -1,  1],
            [1,  1,  1,  1],
            [1,  1, -1, -1],
            [1,  1,  1,  1],
        ], dtype=float)

    def suppression_matrix(self, showmatrix=False, nosuppress=False):
        """Calculate the suppression matrix from blade behavior.

        When nosuppress=True (raw), returns the standard 1/-1 matrix.
        When nosuppress=False (scaled), calculates from fitted slopes.
        """
        if nosuppress:
            # Return standard matrix for raw calculations
            return self.standard_suppression_matrix()

        # Calculate from blade slopes for scaled calculations
        pch = XBPMProcessor.central_line_fit(self.blades_h,
                                             self.range_h, 'h')
        pcv = XBPMProcessor.central_line_fit(self.blades_v,
                                             self.range_v, 'v')

        if len(self.range_h) > 1:
            pch = pch[0] / np.abs(pch)
        else:
            pch = np.ones(8).reshape(4, 2)

        if len(self.range_v) > 1:
            pcv = pcv[0] / np.abs(pcv)
        else:
            pcv = np.ones(8).reshape(4, 2)

        supmat = np.array([
            [pcv[0, 0], -pcv[1, 0], -pcv[2, 0],  pcv[3, 0]],
            [pcv[0, 0],  pcv[1, 0],  pcv[2, 0],  pcv[3, 0]],
            [pch[0, 0],  pch[1, 0], -pch[2, 0], -pch[3, 0]],
            [pch[0, 0],  pch[1, 0],  pch[2, 0],  pch[3, 0]],
        ])

        if showmatrix:
            print("\n Suppression matrix:")
            for lin in supmat:
                for col in lin:
                    print(f" {col:12.6f}", end='')
                print()
            print()

        # Exporter(self.prm).write_supmat(supmat)
        return supmat

    @staticmethod
    def central_line_fit(blades, range_vals, direction):
        """Linear fittings to each blade's data through central line."""
        if blades is None:
            dr = 'horizontal' if direction == 'h' else 'vertical'
            print(f"\n WARNING: (central_line_fit) {dr} blades' values"
                  " not defined. Seetting fitting values to [1, 0].")
            return np.array([[1, 0] for _ in range(4)])

        pc = list()
        for blade in blades.values():
            weight = 1. / blade[:, 1]

            if np.isinf(weight).any():
                weight = None

            pc.append(np.polyfit(range_vals, blade[:, 0], deg=1, w=weight))
        pc = np.array(pc)

        if np.isinf(pc).any() or (pc == 0).any():
            pc = np.array([[1, 0] for _ in range(4)])

        return pc

    def beam_position_pair(self, supmat):
        """Calculate beam position from blades' currents (pairwise)."""
        positions = dict()
        for pos, bld in self.data.items():
            dsps = supmat @ bld[:, 0]
            positions[pos] = np.array([dsps[0] / dsps[1], dsps[2] / dsps[3]])
        return positions

    def position_dict_parse(self, data):
        """Parse XBPM position dict into structured arrays."""
        gridlist = np.array(list(data.keys()))

        grid_lin = np.unique(gridlist[:, 1])
        grid_col = np.unique(gridlist[:, 0])

        gsh_lin = len(grid_lin)
        gsh_col = len(grid_col)

        xbpm_nom_h  = np.zeros((gsh_lin, gsh_col))
        xbpm_nom_v  = np.zeros((gsh_lin, gsh_col))
        xbpm_meas_h = np.zeros((gsh_lin, gsh_col))
        xbpm_meas_v = np.zeros((gsh_lin, gsh_col))

        for ii, y in enumerate(grid_lin):
            for jj, x in enumerate(grid_col):
                # key = (x, y) = (col, lin)
                key = (x, y)
                if key not in data.keys():
                    print(f"\n WARNING: position {key} not found in data,"
                        " Skipping.")
                    continue

                try:
                    xbpm_nom_h[ii, jj]  = x
                    xbpm_nom_v[ii, jj]  = y
                    xbpm_meas_h[ii, jj] = data[key][0]
                    xbpm_meas_v[ii, jj] = data[key][1]
                except Exception as err:
                    print(f"\n WARNING: failed when parsing positions"
                          f" dictionary:\n{err}\n"
                          f" lin, col = {y}, {x}, key = {key}")
                    continue

        return (xbpm_nom_h, xbpm_nom_v, xbpm_meas_h, xbpm_meas_v)

    @staticmethod
    def beam_position_cross(blades):
        """Calculate beam position from blades' currents (cross-blade)."""
        to, ti, bi, bo = blades
        v1 = (to - bi) / (to + bi)
        v2 = (ti - bo) / (ti + bo)
        hpos = (v1 - v2)
        vpos = (v1 + v2)
        return [hpos, vpos]

    @staticmethod
    def scaling_fit(pos_h, pos_v, nom_h, nom_v, calctype=""):
        """Calculate scaling coefficients from fitted positions.
        
        Args:
            pos_h    : Measured horizontal positions array.
            pos_v    : Measured vertical positions array.
            nom_h    : Nominal horizontal positions array.
            nom_v    : Nominal vertical positions array.
            calctype : Type of calculation (for logging purposes).
        
        Returns:
            kx     : Horizontal scaling factor.
            deltax : Horizontal offset.
            ky     : Vertical scaling factor.
            deltay : Vertical offset.
        """
        print(f"\n#### {calctype} blades:")

        h_finitemask = np.isfinite(pos_h)
        pos_h_cln = pos_h[h_finitemask]
        nom_h_cln = nom_h[h_finitemask]

        v_finitemask = np.isfinite(pos_v)
        pos_v_cln = pos_v[v_finitemask]
        nom_v_cln = nom_v[v_finitemask]

        kx, deltax = 1., 0.
        if len(set(nom_h.ravel())) > 1:
            try:
                kx, deltax = np.polyfit(pos_h_cln, nom_h_cln, deg=1)
            except Exception as err:
                print(f"\n WARNING: when calculating horizontal scaling"
                      f" coefficients:\n{err}\n Setting to default values.")

        ky, deltay = 1., 0.
        if len(set(nom_v.ravel())) > 1:
            try:
                ky, deltay = np.polyfit(pos_v_cln, nom_v_cln, deg=1)
            except Exception as err:
                print(f"\n WARNING: when calculating vertical scaling"
                      f" coefficients:\n{err}\n Setting to default values.")

        print(f"kx = {kx:12.4f},   deltax = {deltax:12.4f}")
        print(f"ky = {ky:12.4f},   deltay = {deltay:12.4f}\n")
        return kx, deltax, ky, deltay

    @staticmethod
    def _estimate_spreaded_std_dev(pos_h_scaled, pos_v_scaled,
                                   rawblades):
        """Estimate standard deviations in ROI between scaled and nominal.

        Args:
            pos_h_scaled: Scaled horizontal positions array.
            pos_v_scaled: Scaled vertical positions array.
            rawblades: Dictionary of raw blade measurements in ROI.

        Returns:
            diffx2: Squared horizontal differences in ROI.
            diffy2: Squared vertical differences in ROI.
        """
        # Calculate squared differences
        # for key, val in rawblades.items():
        #     to = val['to']
        #     ti = val['ti']
        #     bo = val['bo']
        #     bi = val['bi']

        # diffx2 = (pos_h_scaled - pos_nom_h) ** 2
        # diffy2 = (pos_v_scaled - pos_nom_v) ** 2
        # return diffx2, diffy2
        pass

    @staticmethod
    def _calculate_roi_stats(diffx2, diffy2):
        """Calculate RMS statistics from squared position differences in ROI.

        Args:
            diffx2: Squared horizontal differences array.
            diffy2: Squared vertical differences array.

        Returns:
            dict: Statistics with sigma_h, sigma_v, sigma_total,
                  diff_max_h, diff_min_h, diff_max_v, diff_min_v.
        """
        nx, ny = diffx2.size, diffy2.size
        sum_diffx2_n = np.sum(diffx2) / nx
        sum_diffy2_n = np.sum(diffy2) / ny
        sigma_h      = np.sqrt(sum_diffx2_n)
        sigma_v      = np.sqrt(sum_diffy2_n)
        sigma_total  = np.sqrt(sum_diffx2_n + sum_diffy2_n)

        diff_max_h   = np.sqrt(np.max(diffx2))
        diff_min_h   = np.sqrt(np.min(diffx2))
        diff_max_v   = np.sqrt(np.max(diffy2))
        diff_min_v   = np.sqrt(np.min(diffy2))

        return {
            'sigma_h'     : sigma_h,
            'sigma_v'     : sigma_v,
            'sigma_total' : sigma_total,
            'diff_max_h'  : diff_max_h,
            'diff_min_h'  : diff_min_h,
            'diff_max_v'  : diff_max_v,
            'diff_min_v'  : diff_min_v,
        }

    def data_parse(self):
        """Extract each blade's data from whole data dict into arrays."""
        dk = np.array(list(self.data.keys()))

        try:
            nh = np.unique(dk[:, 0])
            nv = np.unique(dk[:, 1])
        except:  # noqa: E722
            # Some data are 1-D only
            nh = np.zeros(len(dk))
            nv = np.unique(dk)

        ngrid = (nv.shape[0], nh.shape[0])
        to,  ti  = np.zeros(ngrid), np.zeros(ngrid)
        bo,  bi  = np.zeros(ngrid), np.zeros(ngrid)
        sto, sti = np.zeros(ngrid), np.zeros(ngrid)
        sbo, sbi = np.zeros(ngrid), np.zeros(ngrid)

        for ii, nl in enumerate(nv):
            for jj, nc in enumerate(nh):
                key = (nc, nl)
                ilin = ngrid[0] - ii - 1
                icol = jj

                if key not in self.data.keys():
                    break

                try:
                    to[ilin, icol]  = self.data[key][0, 0]
                    ti[ilin, icol]  = self.data[key][1, 0]
                    bi[ilin, icol]  = self.data[key][2, 0]
                    bo[ilin, icol]  = self.data[key][3, 0]

                    sto[ilin, icol] = self.data[key][0, 1]
                    sti[ilin, icol] = self.data[key][1, 1]
                    sbi[ilin, icol] = self.data[key][2, 1]
                    sbo[ilin, icol] = self.data[key][3, 1]
                except Exception as err:
                    print(f"\n WARNING, when trying to parse blade data: {err}"
                          f"\n nominal position: {err},"
                          f" array index: {ilin}, {icol}"
                          "\n Maybe data grid is incomplete?")

        return [to, ti, bi, bo], [sto, sti, sbi, sbo]


class BPMProcessor:
    """Processes BPM-only data to estimate XBPM positions.

    Encapsulates the legacy BPM calculation flow so orchestration can
    call a single entry point instead of standalone functions.
    """

    def __init__(self, rawdata, prm: Prm):
        """Store raw BPM/XBPM dataset and parameters for later processing."""
        self.rawdata = rawdata
        self.prm     = prm

        self.roi_h_size = prm.roisize[0] if prm.roisize else ROI_SIZE_H
        self.roi_v_size = prm.roisize[1] if prm.roisize else ROI_SIZE_V

    def calculate_positions(self):
        """Calculate and plot XBPM positions derived from BPM data.

        Returns:
            Array of [x, y] coordinates or None if calculation fails.
        """
        sector_idx = self._sector_index()
        tangents = self._tangents_calc(sector_idx)

        print("\n# Distance between BPMs            ="
              f" {self.prm.bpmdist:8.4f}  m")
        print("# Distance between source and XBPM ="
              f" {self.prm.xbpmdist:8.4f} m\n")

        # Calculate positions at XBPM from BPM tangents and distances.
        self.xbpm_pos = self._positions_from_tangents(
            tangents, self.prm.xbpmdist
            )

        # Assemble position data into structured grid arrays.
        self._position_grid_assemble()

        # Estimate standard deviations.
        self.last_stats, self.bpm_roi_diffs = self._std_dev_estimate(
            self.xnom, self.ynom, self.xpos, self.ypos
            )

        # Extract ROI data for closeup view
        (xnom_roi, ynom_roi,
         xpos_roi, ypos_roi) = self._extract_roi_positions()

        # Initialize figure with 1x3 subplots:
        # full grid, roi closeup, differences
        if self.bpm_roi_diffs is None:
            # ROI can be unavailable for sparse/incomplete scans.
            is_1d = True
        else:
            is_1d = (self.bpm_roi_diffs.ndim == 1 or
                     (self.bpm_roi_diffs.ndim == 2 and
                      min(self.bpm_roi_diffs.shape) == 1))
        gridspec = {'width_ratios': [1, 1, 0.1]} if is_1d else None
        self.fig, (ax_all, ax_close, ax_color) = plt.subplots(
            1, 3, figsize=(18, 6), constrained_layout=True,
            gridspec_kw=gridspec
        )

        # Apply layout padding similar to PositionVisualizer for consistency
        try:
            engine = self.fig.get_layout_engine()
            if engine is not None and hasattr(engine, "set"):
                engine.set(
                    w_pad=0.02,   # space figure edge / axes, horizontal
                    h_pad=0.0,    # space figure edge / axes, vertical
                    wspace=0.02,  # space between axes, horizontal
                    hspace=0.0,   # space between axes, vertical
                )
        except Exception:  # noqa: S110
            pass  # Ignore if layout engine unavailable

        # Plot full grid
        self._plot_position_scatter(
            ax_all, self.xnom, self.ynom, self.xpos, self.ypos,
            _Title('bpm', 'total', beamline=self.prm.beamline)
        )

        # Plot ROI closeup
        self._plot_position_scatter(
            ax_close, xnom_roi, ynom_roi, xpos_roi, ypos_roi,
            _Title('bpm', 'roi', beamline=self.prm.beamline)
        )

        # Plot differences heatmap with extent mapping
        self._plot_roi_differences(ax_color, xnom_roi, ynom_roi)

        if self.prm.outputfile:
            outfile = f"bpm_positions_{self.prm.beamline}.png"
            self.fig.savefig(outfile, dpi=FIGDPI)
            print(" Figure of positions calculated by BPM measurements "
                  f"saved to file {outfile}.\n")

        return self._compile_measurement_results()

    def _extract_roi_positions(self):
        """Extract ROI positions from full grid for closeup view.

        Returns:
            Tuple (xnom_roi, ynom_roi, xpos_roi, ypos_roi) of ROI arrays.
        """
        rows, cols = self._roi_slices(self.xnom.shape)
        return (
            self.xnom[rows, cols],
            self.ynom[rows, cols],
            self.xpos[rows, cols],
            self.ypos[rows, cols],
        )

    def _compile_measurement_results(self):
        """Compile measured and nominal coordinates into return format.

        Returns:
            Tuple (measured, nominal) where each is a 2-column array or None.
        """
        measured = (np.column_stack((self.xpos, self.ypos))
                    if self.xpos.size else None)
        nominal = (np.column_stack((self.xnom, self.ynom))
                   if self.xnom.size else None)
        return measured, nominal

    def _plot_roi_differences(self, axdiff, pos_nom_h, pos_nom_v):
        """Plot ROI differences as scatter (1D) or heatmap (2D).

        Args:
            axdiff: Matplotlib axis for ROI differences visualization.
            pos_nom_h: Nominal horizontal positions for extent mapping.
            pos_nom_v: Nominal vertical positions for extent mapping.
        """
        if self.bpm_roi_diffs is None:
            return

        roi_diffs = self.bpm_roi_diffs

        # Treat as 1-D if truly 1-D (shape = (n,)) or effectively 1-D (one
        # dimension is 1, like (1, n) or (n, 1)), or if one nominal axis is
        # constant (single-line sweep).
        h_const = np.nanmax(pos_nom_h) == np.nanmin(pos_nom_h)
        v_const = np.nanmax(pos_nom_v) == np.nanmin(pos_nom_v)
        is_1d = (roi_diffs.ndim == 1 or
             (roi_diffs.ndim == 2 and min(roi_diffs.shape) == 1) or
             h_const or v_const)

        if is_1d:
            # 1D imshow: render as a thin band of square cells
            h_min = np.nanmin(pos_nom_h)
            h_max = np.nanmax(pos_nom_h)

            color_vals = np.ravel(roi_diffs).reshape(-1, 1)
            extent = [0, 1, pos_nom_v.min(), pos_nom_v.max()]
            aspect = 'auto'

            # Make the single column visually wider
            axdiff.set_box_aspect(10)
            axdiff.set_anchor('C')
            axdiff.set_xticks([])

            im = axdiff.imshow(color_vals,
                               cmap='viridis',
                               extent=extent,
                               aspect=aspect,
                               origin='lower')
            cbar = self.fig.colorbar(im, ax=axdiff,
                                      fraction=0.4, pad=0.3)
            cbar.set_label(u"RMS Difference [$\\mu$m]", fontsize=14)

            axdiff.set_xlabel("")
            axdiff.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)
            axdiff.set_title(
                _Title('bpm', 'heatmap', self.prm.beamline), pad=2
                )
            axdiff.grid(False)
        else:
            # 2D heatmap with extent mapping
            h_min = np.nanmin(pos_nom_h)
            h_max = np.nanmax(pos_nom_h)
            v_min = np.nanmin(pos_nom_v)
            v_max = np.nanmax(pos_nom_v)
            extent = [h_min, h_max, v_min, v_max]

            # Calculate aspect ratio to maintain proper physical proportions.
            # Account for both physical extents and array shape to avoid
            # distortion when physical x and y ranges differ significantly.
            n_v, n_h = roi_diffs.shape
            h_extent = h_max - h_min
            v_extent = v_max - v_min
            # aspect = (physical_y_per_pixel) / (physical_x_per_pixel)
            aspect = ((v_extent / n_v) / (h_extent / n_h)
                      if (h_extent > 0 and v_extent > 0) else 1)

            # Use imshow for filled heatmap visualization
            im = axdiff.imshow(
                roi_diffs, cmap='viridis', extent=extent,
                aspect=aspect, origin='lower'
            )
            cbar = self.fig.colorbar(im, ax=axdiff,
                                     fraction=0.046, pad=0.04)
            cbar.set_label(u"RMS Difference [$\\mu$m]", fontsize=14)

            axdiff.set_xlabel(u"$x$ [$\\mu$m]", fontsize=14)  # noqa: W605
            axdiff.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)
            axdiff.set_title(
                _Title('bpm', 'heatmap', self.prm.beamline), pad=2
                )
            axdiff.grid(False)

    def _plot_position_scatter(self, ax, pos_nom_h, pos_nom_v,
                               pos_h, pos_v, title):
        """Plot measured vs nominal positions scatter plot.

        Args:
            ax: Matplotlib axis for plotting.
            pos_nom_h: Nominal horizontal positions.
            pos_nom_v: Nominal vertical positions.
            pos_h: Calculated horizontal positions.
            pos_v: Calculated vertical positions.
            title: Plot title.
        """
        ax.set_title(title, pad=2)
        pos = ax.plot(pos_h, pos_v, 'bo')
        nom = ax.plot(pos_nom_h, pos_nom_v, 'r+')
        ax.set_xlabel(u"$x$ [$\\mu$m]", fontsize=14)  # noqa: W605
        ax.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)

        # Compute common limits to ensure equal aspect ratio with margin.
        # Use only finite values to avoid NaN/Inf axis-limit failures.
        all_h = np.concatenate([np.ravel(pos_h), np.ravel(pos_nom_h)])
        all_v = np.concatenate([np.ravel(pos_v), np.ravel(pos_nom_v)])
        all_h = all_h[np.isfinite(all_h)]
        all_v = all_v[np.isfinite(all_v)]

        if all_h.size == 0 or all_v.size == 0:
            print("\n WARNING: no finite position data available for plotting;"
                  " using default axis limits.")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', adjustable='box')
            ax.grid()
            return

        h_min, h_max = np.min(all_h), np.max(all_h)
        v_min, v_max = np.min(all_v), np.max(all_v)

        h_range = h_max - h_min if h_max > h_min else 1
        v_range = v_max - v_min if v_max > v_min else 1

        # Add 15% margin (30% total: 15% on each side)
        total_range = max(h_range, v_range) * 1.3

        # Center the limits and expand to match max range
        h_center = (h_min + h_max) / 2
        v_center = (v_min + v_max) / 2

        ax.set_xlim(h_center - total_range / 2, h_center + total_range / 2)
        ax.set_ylim(v_center - total_range / 2, v_center + total_range / 2)

        # Force 1:1 aspect ratio after setting limits
        ax.set_aspect('equal', adjustable='box')

        handles, labels = [], []
        if len(nom) > 0:
            handles.append(nom[0])
            labels.append("Nom.")
        if len(pos) > 0:
            handles.append(pos[0])
            labels.append("Calc.")
        if handles:
            ax.legend(handles, labels)
        ax.grid()

    def _position_grid_assemble(self):
        """Assemble position data into structured grid numpy arrays."""
        # Get unique sorted indices for x and y from the position keys.
        xidx = sorted(set([key[0] for key in self.xbpm_pos.keys()]))
        yidx = sorted(set([key[1] for key in self.xbpm_pos.keys()]))
        nx, ny = len(xidx), len(yidx)

        # Initialize numpy arrays for nominal and measured positions.
        self.xnom = np.zeros((ny, nx))
        self.ynom = np.zeros((ny, nx))
        self.xpos = np.full((ny, nx), np.nan)
        self.ypos = np.full((ny, nx), np.nan)

        # Fill the arrays.
        missing = 0
        for iy in range(ny):
            for ix in range(nx):
                key = (xidx[ix], yidx[iy])
                self.xnom[iy, ix] = key[0]
                self.ynom[iy, ix] = key[1]
                if key in self.xbpm_pos:
                    self.xpos[iy, ix] = self.xbpm_pos[key][0]
                    self.ypos[iy, ix] = self.xbpm_pos[key][1]
                else:
                    missing += 1

        if missing > 0:
            print("\n WARNING: sparse BPM grid detected:"
                  f" {missing} points missing from nominal mesh."
                  " Missing points were set to NaN.")

    def _roi_slices(self, shape):
        """Return row/column slices for the centered ROI."""
        nv, nh = shape
        roi_h = min(self.roi_h_size, nh)
        roi_v = min(self.roi_v_size, nv)
        fromh = max(0, int((nh - roi_h) / 2))
        uptoh = min(nh, fromh + roi_h)
        fromv = max(0, int((nv - roi_v) / 2))
        uptov = min(nv, fromv + roi_v)

        if nh == 1 or nv == 1:
            return slice(fromv, uptov), slice(None)

        return slice(fromv, uptov), slice(fromh, uptoh)

    def _sector_index(self) -> int:
        sector = int(self.prm.section.split(':')[1][:2])
        return 8 * (sector - 1) - 1

    def _tangents_calc(self, idx):
        """Calculate tangents of beam angles between neighbour BPMs."""
        nextidx = idx + 1
        offset_x_sect, offset_y_sect = 0, 0
        offset_x_next, offset_y_next = 0, 0
        offsetfound = False

        # Search for zero-angle reference orbit to define the offset.
        for dt in self.rawdata:
            if dt[2]['agx'] == 0 and dt[2]['agy'] == 0:
                offset_x_sect = dt[2]['orbx'][idx]
                offset_y_sect = dt[2]['orby'][idx]
                offset_x_next = dt[2]['orbx'][nextidx]
                offset_y_next = dt[2]['orby'][nextidx]
                offsetfound = True
                break

        # Try and guess offsets by extrapolation if not found.
        if not offsetfound:
            (offset_x_sect, offset_x_next,
             offset_y_sect, offset_y_next) = self._offset_search(idx)

        # Calculate tangents for all angles.
        tangents = dict()
        bdist = self.prm.bpmdist
        for dt in self.rawdata:
            tx = (((dt[2]['orbx'][nextidx] - offset_x_next)) -
                  (dt[2]['orbx'][idx]     - offset_x_sect)) / bdist
            ty = (((dt[2]['orby'][nextidx] - offset_y_next)) -
                  (dt[2]['orby'][idx]     - offset_y_sect)) / bdist
            agx, agy = dt[2]['agx'], dt[2]['agy']
            tangents[agx, agy] = np.array([tx, ty])
        return tangents

    def _offset_search(self, idx):
        """Extrapolate offsets when reference orbit is missing."""
        # Get the angle and orbit data for the current and next BPMs
        # across all measurements.
        nextidx = idx + 1
        agx    = np.array([dt[2]['agx']
                           for dt in self.rawdata])
        orbx   = np.array([dt[2]['orbx'][idx]
                           for dt in self.rawdata])
        n_orbx = np.array([dt[2]['orbx'][nextidx]
                           for dt in self.rawdata])

        # Find the max and min angles to check for variation. If angles are
        # constant, we cannot extrapolate and must raise an error.
        agxmax = np.max(agx)
        agxmin = np.min(agx)
        if np.isclose(agxmax, agxmin):
            raise ValueError(
                "Cannot infer BPM x-offset from data without agx variation "
                "or explicit (agx=0, agy=0) reference point."
            )

        # Use the max and min orbits to extrapolate the offset at zero angle.
        osx = np.array(sorted(list(set(orbx))))
        oxmin, oxmax = osx[0], osx[-1]
        offset_x_sect = ((oxmin * agxmax - oxmax * agxmin) /
                         (agxmax - agxmin))

        onx = np.array(sorted(list(set(n_orbx))))
        onxmin, onxmax = onx[0], onx[-1]
        offset_x_next = ((onxmin * agxmax - onxmax * agxmin) /
                         (agxmax - agxmin))

        # Repeat the same process for the vertical plane.
        agy    = np.array([dt[2]['agy']
                           for dt in self.rawdata])
        orby   = np.array([dt[2]['orby'][idx]
                           for dt in self.rawdata])
        n_orby = np.array([dt[2]['orby'][nextidx]
                           for dt in self.rawdata])

        agymax = np.max(agy)
        agymin = np.min(agy)
        if np.isclose(agymax, agymin):
            raise ValueError(
                "Cannot infer BPM y-offset from data without agy variation "
                "or explicit (agx=0, agy=0) reference point."
            )

        osy = np.array(sorted(list(set(orby))))
        oymin, oymax = osy[0], osy[-1]
        offset_y_sect = ((oymin * agymax - oymax * agymin) /
                         (agymax - agymin))

        ony = np.array(sorted(list(set(n_orby))))
        onymin, onymax = ony[0], ony[-1]
        offset_y_next = ((onymin * agymax - onymax * agymin) /
                         (agymax - agymin))

        return (offset_x_sect, offset_x_next, offset_y_sect, offset_y_next)

    def _positions_from_tangents(self, tangents, xbpm_dist):
        """Calculate beam positions from tangents at BPMs."""
        positions = dict()
        for key, tg in tangents.items():
            newkey = (key[0] * xbpm_dist, key[1] * xbpm_dist)
            positions[newkey] = tg * xbpm_dist
        return positions

    def _std_dev_estimate(self, xnom, ynom, xpos, ypos):
        """Estimate RMS deviations between measured and nominal positions.
        
        Args:
            xnom: Nominal horizontal positions grid.
            ynom: Nominal vertical positions grid.
            xpos: Measured horizontal positions grid.
            ypos: Measured vertical positions grid.

        Returns:
            Tuple (rms_stats, roi_diffs) where:
            rms_stats: Dictionary with overall RMS statistics for all sites.
            roi_diffs: 2D array of differences at ROI or None if ROI
                unavailable.
        """
        # Total differences (all sites).
        nv, nh = xnom.shape[0], xnom.shape[1]
        nsites_total = nv * nh

        # Calculate differences and filter valid (finite) points.
        diff_h = xnom - xpos
        diff_v = ynom - ypos
        valid = np.isfinite(diff_h) & np.isfinite(diff_v)
        nsites = int(np.count_nonzero(valid))

        # Check if any valid points are available for RMS estimation.
        if nsites == 0:
            print("\n WARNING: no valid BPM points found for RMS estimation.")
            rms_stats = {
                'sigma_h'       : np.nan,
                'sigma_v'       : np.nan,
                'sigma_total'   : np.nan,
                'diff_min_h'    : np.nan,
                'diff_min_v'    : np.nan,
                'diff_max_h'    : np.nan,
                'diff_max_v'    : np.nan,
                'roi_available' : False,
            }
            return rms_stats, None

        # Calculate statistics using only valid points to avoid NaN/Inf issues.
        diff_h_valid = diff_h[valid]
        diff_v_valid = diff_v[valid]

        diff_h_min = np.abs(np.min(diff_h_valid))
        diff_h_max = np.abs(np.max(diff_h_valid))
        sig2_h = np.mean(diff_h_valid**2)

        diff_v_min = np.abs(np.min(diff_v_valid))
        diff_v_max = np.abs(np.max(diff_v_valid))
        sig2_v = np.mean(diff_v_valid**2)

        sig_h = np.sqrt(sig2_h)
        sig_v = np.sqrt(sig2_v)
        sig_tot = np.sqrt(sig2_h + sig2_v)

        print("Sigmas:\n"
              f"   (all sites)     H = {sig_h:.4f}\n"
              f"   (all sites)     V = {sig_v:.4f},\n"
              f"   (all sites) total = {sig_tot:.4f}\n"
              "\n  Maximum difference:\n"
              f"   (all sites) H = {diff_h_max:.4f}\n"
              f"   (all sites) V = {diff_v_max:.4f},\n"
              "\n  Minimum difference:\n"
              f"   (all sites) H = {diff_h_min:.4f}\n"
              f"   (all sites) V = {diff_v_min:.4f},\n"
              )

        # Check whether the sweeping is complete.
        if nsites < nsites_total:
            print("\n WARNING: sweeping looks incomplete, no ROI was defined"
              f" ({nsites} valid sites, out of {nsites_total}"
                  " in total). Skipping ROI analysis.")
            rms_stats = {
                'sigma_h'       : sig_h,
                'sigma_v'       : sig_v,
                'sigma_total'   : sig_tot,
                'diff_min_h'    : diff_h_min,
                'diff_min_v'    : diff_v_min,
                'diff_max_h'    : diff_h_max,
                'diff_max_v'    : diff_v_max,
                'roi_available' : False,
            }
            return rms_stats, None

        # Differences at ROI.
        rows, cols = self._roi_slices(xnom.shape)
        xnom_cut = xnom[rows, cols]
        ynom_cut = ynom[rows, cols]
        xpos_cut = xpos[rows, cols]
        ypos_cut = ypos[rows, cols]

        # Calculate differences and filter valid (finite) points in ROI.
        diff_h_cut = np.abs(xnom_cut - xpos_cut)
        diff_v_cut = np.abs(ynom_cut - ypos_cut)
        valid_roi  = np.isfinite(diff_h_cut) & np.isfinite(diff_v_cut)
        nroi_valid = int(np.count_nonzero(valid_roi))
        if nroi_valid == 0:
            print("\n WARNING: no valid ROI points found."
                  " Skipping ROI analysis.")
            rms_stats = {
                'sigma_h'       : sig_h,
                'sigma_v'       : sig_v,
                'sigma_total'   : sig_tot,
                'diff_min_h'    : diff_h_min,
                'diff_min_v'    : diff_v_min,
                'diff_max_h'    : diff_h_max,
                'diff_max_v'    : diff_v_max,
                'roi_available' : False,
            }
            return rms_stats, None

        # Calculate total differences in ROI for visualization.
        diff_cut    = np.sqrt(diff_h_cut**2 + diff_v_cut**2)
        sig2_v_roi  = np.mean((diff_v_cut[valid_roi])**2)
        sig2_h_roi  = np.mean((diff_h_cut[valid_roi])**2)
        roi_sig_h   = np.sqrt(sig2_h_roi)
        roi_sig_v   = np.sqrt(sig2_v_roi)
        roi_sig_tot = np.sqrt(sig2_h_roi + sig2_v_roi)

        print("  Differences in ROI\n"
              f"   (x in [{np.min(xnom_cut)}, {np.max(xnom_cut)}];"
              f"  y in [{np.min(ynom_cut)}, {np.max(ynom_cut)}])\n"
              f"       H = {roi_sig_h:.4f}\n"
              f"       V = {roi_sig_v:.4f},\n"
              f"   total = {roi_sig_tot:.4f}")

        rms_stats = {
            'sigma_h'         : sig_h,
            'sigma_v'         : sig_v,
            'sigma_total'     : sig_tot,
            'diff_min_h'      : diff_h_min,
            'diff_min_v'      : diff_v_min,
            'diff_max_h'      : diff_h_max,
            'diff_max_v'      : diff_v_max,
            'roi_available'   : True,
            'roi_sigma_h'     : roi_sig_h,
            'roi_sigma_v'     : roi_sig_v,
            'roi_sigma_total' : roi_sig_tot,
            'roi_bounds'      : {
                'x_min' : float(np.min(xnom_cut)),
                'x_max' : float(np.max(xnom_cut)),
                'y_min' : float(np.min(ynom_cut)),
                'y_max' : float(np.max(ynom_cut)),
            },
        }

        return rms_stats, diff_cut
