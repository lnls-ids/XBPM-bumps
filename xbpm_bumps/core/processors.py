"""XBPM and BPM data processors."""

import os
import numpy as np
import matplotlib.pyplot as plt

from .parameters  import Prm                     # noqa: E272
from .visualizers import PositionVisualizer
from .constants   import ROI_SIZE_V, ROI_SIZE_H, FIGDPI    # noqa: E272
# from .exporters import Exporter


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

    def calculate_scaled_positions(self, showmatrix: bool = True) -> list:
        """Calculate positions with suppression matrix correction.

        Applies suppression matrices to correct for blade gain variations
        and scales results to physical distances.

        Args:
            showmatrix: If True, display blade behavior matrices.

        Returns:
            List of [pairwise_positions_dict, cross_positions_dict].
        """
        return self.calculate_positions(
            rtitle="scaled XBPM positions",
            nosuppress=False,
            showmatrix=showmatrix
        )

    def calculate_raw_positions(self, showmatrix: bool = True) -> list:
        """Calculate positions without suppression (raw)."""
        return self.calculate_positions(
            rtitle="raw XBPM positions",
            nosuppress=True,
            showmatrix=showmatrix
        )

    def calculate_positions(self, rtitle: str,
                             nosuppress: bool,
                             showmatrix: bool = True) -> list:
        """Orchestrate XBPM position calculations and visualization.

        Ensures central sweeps are analyzed, then delegates to
        `xbpm_position_calc` to compute positions and plot results.
        """
        # Ensure sweep data is available
        if (self.range_h is None or self.range_v is None or
                self.blades_h is None or self.blades_v is None):
            self.analyze_central_sweeps(show=False)

        return self._xbpm_position_calc(
            rtitle=rtitle,
            nosuppress=nosuppress,
            showmatrix=showmatrix,
        )

    def _xbpm_position_calc(self, rtitle: str, nosuppress: bool,
                            showmatrix: bool) -> list:
        """Calculate positions from blades' measured data and visualize."""
        data = self.data
        prm  = self.prm

        blades, _stddevs = self.data_parse()

        supmat = self.suppression_matrix(showmatrix=showmatrix,
                                         nosuppress=nosuppress)

        pos_pair  = self.beam_position_pair(supmat)
        (pos_nom_h, pos_nom_v,
         pos_pair_h, pos_pair_v) = self.position_dict_parse(pos_pair)

        pos_nom_h *= prm.xbpmdist
        pos_nom_v *= prm.xbpmdist

        nh = pos_nom_h.shape[1]
        nv = pos_nom_h.shape[0]
        roi_h = min(self.roi_h_size, nh)
        roi_v = min(self.roi_v_size, nv)
        frh = max(0, int((nh - roi_h) / 2))
        uph = min(nh, frh + roi_h)
        frv = max(0, int((nv - roi_v) / 2))
        upv = min(nv, frv + roi_v)

        if pos_nom_h.shape[0] == 1 or pos_nom_v.shape[0] == 1:
            pos_nom_h_roi = pos_nom_h[0, frv:upv]
            pos_nom_v_roi = pos_nom_v[0, frv:upv]
        elif pos_nom_h.shape[1] == 1 or pos_nom_v.shape[1] == 1:
            pos_nom_h_roi = pos_nom_h[frv:upv, 0]
            pos_nom_v_roi = pos_nom_v[frv:upv, 0]
        else:
            pos_nom_h_roi  = pos_nom_h[frv:upv, frh:uph]
            pos_nom_v_roi  = pos_nom_v[frv:upv, frh:uph]

        if pos_nom_h.shape[0] == 1 or pos_nom_v.shape[0] == 1:
            pos_pair_h_roi = pos_pair_h[0, frv:upv]
            pos_pair_v_roi = pos_pair_v[0, frv:upv]
        elif pos_nom_h.shape[1] == 1 or pos_nom_v.shape[1] == 1:
            pos_pair_h_roi = pos_pair_h[frv:upv, 0]
            pos_pair_v_roi = pos_pair_v[frv:upv, 0]
        else:
            pos_pair_h_roi = pos_pair_h[frv:upv, frh:uph]
            pos_pair_v_roi = pos_pair_v[frv:upv, frh:uph]

        (kxp, deltaxp,
         kyp, deltayp) = XBPMProcessor.scaling_fit(pos_pair_h_roi,
                pos_pair_v_roi, pos_nom_h_roi, pos_nom_v_roi, "Pairwise")
        pos_pair_h_scaled = kxp * pos_pair_h + deltaxp
        pos_pair_v_scaled = kyp * pos_pair_v + deltayp
        pos_pair_h_roi_scaled = kxp * pos_pair_h_roi + deltaxp
        pos_pair_v_roi_scaled = kyp * pos_pair_v_roi + deltayp

        diffx2  = (pos_pair_h_roi_scaled - pos_nom_h_roi) ** 2
        diffy2  = (pos_pair_v_roi_scaled - pos_nom_v_roi) ** 2
        diffpairroi = np.sqrt(diffx2 + diffy2)

        pair_visualizer = PositionVisualizer(prm, f"Pairwise {rtitle}")
        pair_visualizer.show_position_results(
            pos_nom_h, pos_nom_v,
            pos_pair_h_scaled, pos_pair_v_scaled,
            pos_pair_h_roi_scaled, pos_pair_v_roi_scaled,
            pos_nom_h_roi, pos_nom_v_roi,
            diffpairroi
        )

        pos_cr_h, pos_cr_v = XBPMProcessor.beam_position_cross(blades)

        if pos_nom_h.shape[0] == 1 or pos_nom_v.shape[0] == 1:
            pos_cr_h_roi  = pos_cr_h[0, frv:upv]
            pos_cr_v_roi  = pos_cr_v[0, frv:upv]
        elif pos_nom_h.shape[1] == 1 or pos_nom_v.shape[1] == 1:
            pos_cr_h_roi  = pos_cr_h[frv:upv, 0]
            pos_cr_v_roi  = pos_cr_v[frv:upv, 0]
        else:
            pos_cr_h_roi  = pos_cr_h[frv:upv, frh:uph]
            pos_cr_v_roi  = pos_cr_v[frv:upv, frh:uph]

        (kxc, deltaxc,
         kyc, deltayc) = XBPMProcessor.scaling_fit(pos_cr_h_roi,
                                                   pos_cr_v_roi,
                                                   pos_nom_h_roi,
                                                   pos_nom_v_roi, "Cross")
        pos_cr_h_scaled = kxc * pos_cr_h + deltaxc
        pos_cr_v_scaled = kyc * pos_cr_v + deltayc
        pos_cr_h_roi_scaled = kxc * pos_cr_h_roi + deltaxc
        pos_cr_v_roi_scaled = kyc * pos_cr_v_roi + deltayc

        diffx2  = (pos_cr_h_roi_scaled - pos_nom_h_roi) ** 2
        diffy2  = (pos_cr_v_roi_scaled - pos_nom_v_roi) ** 2
        diffcrroi = np.sqrt(diffx2 + diffy2)

        cross_visualizer = PositionVisualizer(prm, f"Cross-blades {rtitle}")
        cross_visualizer.show_position_results(
            pos_nom_h, pos_nom_v,
            pos_cr_h_scaled, pos_cr_v_scaled,
            pos_cr_h_roi_scaled, pos_cr_v_roi_scaled,
            pos_nom_h_roi, pos_nom_v_roi,
            diffcrroi
        )

        if prm.outputfile:
            outdir = '.'
            sup = "raw" if nosuppress else "scaled"
            bl = prm.beamline

            outfile_p = os.path.join(outdir, f"xbpm_pair_pos_{sup}_{bl}.png")
            pair_visualizer.save_figure(outfile_p)

            outfile_c = os.path.join(outdir, f"xbpm_cross_pos_{sup}_{bl}.png")
            cross_visualizer.save_figure(outfile_c)

        scaled_pos_pair = dict()
        scaled_pos_cros = dict()
        for ii, lin in enumerate(pos_nom_h):
            for jj, xx in enumerate(lin):
                yy = pos_nom_v[ii, jj]
                scaled_pos_pair[xx, yy] = [
                    pos_pair_h_scaled[ii, jj],
                    pos_pair_v_scaled[ii, jj]
                    ]
                scaled_pos_cros[xx, yy] = [
                    pos_cr_h_scaled[ii, jj],
                    pos_cr_v_scaled[ii, jj]
                    ]
        return {
            'positions': [scaled_pos_pair, scaled_pos_cros],
            'pairwise_figure': pair_visualizer.fig,
            'cross_figure': cross_visualizer.fig,
            'scales': {
                'pair': {
                    'kx': kxp,
                    'ky': kyp,
                    'dx': deltaxp,
                    'dy': deltayp,
                },
                'cross': {
                    'kx': kxc,
                    'ky': kyc,
                    'dx': deltaxc,
                    'dy': deltayc,
                },
            },
            'supmat': supmat,
        }

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

        # minval_h = np.min(grid_col)
        # maxval_v = np.max(grid_lin)

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

        # for key, val in data.items():
        #     col = int((key[0] - minval_h) / self.prm.gridstep)
        #     lin = int((maxval_v - key[1]) / self.prm.gridstep)
        #
        #     try:
        #         xbpm_nom_h[lin, col]  = key[0]
        #         xbpm_nom_v[lin, col]  = key[1]
        #         xbpm_meas_h[lin, col] = val[0]
        #         xbpm_meas_v[lin, col] = val[1]
        #     except Exception as err:
        #         print(f"\n WARNING: failed when parsing positions"
        #               " dictionary:"
        #               f"\n{err}\n lin, col = {lin}, {col}, key = {key}")
        #         continue

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
        """Calculate scaling coefficients from fitted positions."""
        print(f"\n#### {calctype} blades:")

        hfinitemask = np.isfinite(pos_h)
        ph_cln = pos_h[hfinitemask]
        nh_cln = nom_h[hfinitemask]

        vfinitemask = np.isfinite(pos_v)
        pv_cln = pos_v[vfinitemask]
        nv_cln = nom_v[vfinitemask]

        kx, deltax = 1., 0.
        if len(set(nom_h.ravel())) > 1:
            try:
                kx, deltax = np.polyfit(ph_cln, nh_cln, deg=1)
            except Exception as err:
                print(f"\n WARNING: when calculating horizontal scaling"
                      f" coefficients:\n{err}\n Setting to default values.")

        ky, deltay = 1., 0.
        if len(set(nom_v.ravel())) > 1:
            try:
                ky, deltay = np.polyfit(pv_cln, nv_cln, deg=1)
            except Exception as err:
                print(f"\n WARNING: when calculating vertical scaling"
                      f" coefficients:\n{err}\n Setting to default values.")

        print(f"kx = {kx:12.4f},   deltax = {deltax:12.4f}")
        print(f"ky = {ky:12.4f},   deltay = {deltay:12.4f}\n")
        return kx, deltax, ky, deltay

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

        # DEBUG
        print("\n\n", "#" * 10)
        print("# DEBUG BPMProcessor init ")
        print(f"\n BPMProcessor initialized for beamline {self.prm.beamline}"
              f" section {self.prm.section}.\n")
        # END DEBUG

    def calculate_positions(self):
        """Calculate and plot XBPM positions derived from BPM data.

        Returns:
            Array of [x, y] coordinates or None if calculation fails.
        """
        if self.prm.section is None:
            print("### ERROR: no section defined for the beamline in data set."
                  "\n### Cannot proceed with BPM data analysis. Skipping.")
            return None

        if self.rawdata is None:
            print("### ERROR: no raw BPM data available."
                  "\n### Skipping BPM analysis.")
            return None

        self.last_fig, (axpos, axdiff) = plt.subplots(
            1, 2, figsize=(10.5, 5.5), constrained_layout=True
        )

        sector_idx = self._sector_index()
        tangents = self._tangents_calc(sector_idx)

        print("# Distance between BPMs            ="
              f" {self.prm.bpmdist:8.4f}  m\n"
              "# Distance between source and XBPM ="
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

        axpos.scatter(self.xpos.ravel(), self.ypos.ravel(),
                  c='b', marker='o', label="measured")
        axpos.scatter(self.xnom.ravel(), self.ynom.ravel(),
                  c='r', marker='+', label="nominal")
        axpos.set_xlabel("$x$ [$\\mu$m]")  # noqa: W605
        axpos.set_ylabel("$y$ [$\\mu$m]")  # noqa: W605
        axpos.set_title(f"Beam positions @ {self.prm.beamline}"
                        " (from BPM)")

        limx = np.max(np.abs(self.xnom)) * 1.5
        limy = np.max(np.abs(self.ynom)) * 1.5
        limx = max(limx, limy)
        limy = limy

        # DEBUG
        print("\n#####\n#### BPMProcessor.calculate_positions:"
              f"\n##### x lim = {limx}"
              f"\n##### y lim = {limy}"
              f"\n##### x pos max = {np.max(self.xpos)}"
              f"\n##### y pos max = {np.max(self.ypos)}"
              "\n#####")
        # DEBUG
        axpos.set_xlim(-limx, limx)
        axpos.set_ylim(-limy, limy)
        axpos.set_aspect("equal", adjustable="box")
        axpos.legend()
        axpos.grid()

        if self.bpm_roi_diffs is not None:
            xnom_cut, ynom_cut = self._roi_cut(self.xnom, self.ynom)
            if (self.bpm_roi_diffs.ndim == 2 and xnom_cut is not None
                    and ynom_cut is not None):
                im = axdiff.pcolormesh(
                    xnom_cut, ynom_cut, self.bpm_roi_diffs,
                    cmap='viridis', shading='auto'
                )
            else:
                im = axdiff.scatter(
                    np.ravel(xnom_cut), np.ravel(ynom_cut),
                    c=np.ravel(self.bpm_roi_diffs), cmap='viridis', s=50
                )
            cbar = self.last_fig.colorbar(im, ax=axdiff,
                                          fraction=0.046, pad=0.04)
            cbar.set_label(u"RMS differences (ROI)")

        # axdiff.plot(self.bpm_roi_diffs, 'g+', label="ROI differences")
        axdiff.set_xlabel(u"$x$ [$\\mu$m]")  # noqa: W605
        axdiff.set_ylabel(u"$y$ [$\\mu$m]")
        axdiff.set_title("Differences at ROI")
        axdiff.set_aspect("equal", adjustable="box")
        axdiff.grid(False)

        if self.prm.outputfile:
            outfile = f"bpm_positions_{self.prm.beamline}.png"
            self.last_fig.savefig(outfile, dpi=FIGDPI)
            print(" Figure of positions calculated by BPM measurements "
                  f"saved to file {outfile}.\n")

        # Return measured and nominal coordinates
        measured = (np.column_stack((self.xpos, self.ypos))
                    if self.xpos.size else None)
        nominal  = (np.column_stack((self.xnom, self.ynom))
                    if self.xnom.size else None)
        return measured, nominal

    def _position_grid_assemble(self):
        """Assemble position data into structured grid numpy arrays."""
        # Get unique sorted indices for x and y from the position keys.
        xidx = sorted(set([key[0] for key in self.xbpm_pos.keys()]))
        yidx = sorted(set([key[1] for key in self.xbpm_pos.keys()]))
        nx, ny = len(xidx), len(yidx)

        # Initialize numpy arrays for nominal and measured positions.
        self.xnom = np.zeros((ny, nx))
        self.ynom = np.zeros((ny, nx))
        self.xpos = np.zeros((ny, nx))
        self.ypos = np.zeros((ny, nx))

        # Fill the arrays.
        for iy in range(ny):
            for ix in range(nx):
                key = (xidx[ix], yidx[iy])
                self.xnom[iy, ix] = key[0]
                self.ynom[iy, ix] = key[1]
                self.xpos[iy, ix] = self.xbpm_pos[key][0]
                self.ypos[iy, ix] = self.xbpm_pos[key][1]

    def _roi_cut(self, xnom, ynom):
        """Return ROI slices of nominal coordinate grids."""
        nv, nh = xnom.shape[0], xnom.shape[1]
        roi_h = min(self.roi_h_size, nh)
        roi_v = min(self.roi_v_size, nv)
        fromh = max(0, int((nh - roi_h) / 2))
        uptoh = min(nh, fromh + roi_h)
        fromv = max(0, int((nv - roi_v) / 2))
        uptov = min(nv, fromv + roi_v)

        if nh == 1 or nv == 1:
            return xnom[fromv:uptov], ynom[fromv:uptov]

        return xnom[fromv:uptov, fromh:uptoh], ynom[fromv:uptov, fromh:uptoh]

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
        nextidx = idx + 1
        agx    = np.array([dt[2]['agx']
                           for dt in self.rawdata])
        orbx   = np.array([dt[2]['orbx'][idx]
                           for dt in self.rawdata])
        n_orbx = np.array([dt[2]['orbx'][nextidx]
                           for dt in self.rawdata])

        agxmax = np.max(agx)
        agxmin = np.min(agx)

        osx = np.array(sorted(list(set(orbx))))
        oxmin, oxmax = osx[0], osx[-1]
        offset_x_sect = ((oxmin * agxmax - oxmax * agxmin) /
                         (agxmax - agxmin))

        onx = np.array(sorted(list(set(n_orbx))))
        onxmin, onxmax = onx[0], onx[-1]
        offset_x_next = ((onxmin * agxmax - onxmax * agxmin) /
                         (agxmax - agxmin))

        agy    = np.array([dt[2]['agy']
                           for dt in self.rawdata])
        orby   = np.array([dt[2]['orby'][idx]
                           for dt in self.rawdata])
        n_orby = np.array([dt[2]['orby'][nextidx]
                           for dt in self.rawdata])

        agymax = np.max(agy)
        agymin = np.min(agy)

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
        """Estimate RMS deviations between measured and nominal positions."""
        # Total differences (all sites).
        nv, nh = xnom.shape[0], xnom.shape[1]
        nsites = nv * nh

        diff_h = xnom - xpos
        diff_h_min = np.abs(np.min(diff_h))
        diff_h_max = np.abs(np.max(diff_h))
        sig2_h = np.sum(diff_h**2) / nsites

        diff_v = ynom - ypos
        diff_v_min = np.abs(np.min(diff_v))
        diff_v_max = np.abs(np.max(diff_v))
        sig2_v = np.sum(diff_v**2) / nsites

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
        # nmax = len(set(list(xnom.ravel()))) * len(set(list(ynom.ravel())))
        nmax = (np.unique(xnom.ravel()).shape[0] *
                np.unique(ynom.ravel()).shape[0])
        if nmax < nsites:
            print("\n WARNING: sweeping looks incomplete, no ROI was defined"
                  f" ({nmax} sites measured at most, out of {nsites}"
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
        roi_h = min(self.roi_h_size, nh)
        roi_v = min(self.roi_v_size, nv)
        fromh = max(0, int((nh - roi_h) / 2))
        uptoh = min(nh, fromh + roi_h)
        fromv = max(0, int((nv - roi_v) / 2))
        uptov = min(nv, fromv + roi_v)
        nroi = (uptov - fromv) * (uptoh - fromh)

        if nh == 1 or nv == 1:
            xnom_cut = xnom[fromv:uptov]
            ynom_cut = ynom[fromv:uptov]
            xpos_cut = xpos[fromv:uptov]
            ypos_cut = ypos[fromv:uptov]
        else:
            xnom_cut = xnom[fromv:uptov, fromh:uptoh]
            ynom_cut = ynom[fromv:uptov, fromh:uptoh]
            xpos_cut = xpos[fromv:uptov, fromh:uptoh]
            ypos_cut = ypos[fromv:uptov, fromh:uptoh]

        diff_h_cut = np.abs(xnom_cut - xpos_cut)
        diff_v_cut = np.abs(ynom_cut - ypos_cut)
        diff_cut = np.sqrt(diff_h_cut**2 + diff_v_cut**2)

        sig2_v_roi = np.sum(diff_v_cut**2) / nroi
        sig2_h_roi = np.sum(diff_h_cut**2) / nroi

        roi_sig_h = np.sqrt(sig2_h_roi)
        roi_sig_v = np.sqrt(sig2_v_roi)
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
