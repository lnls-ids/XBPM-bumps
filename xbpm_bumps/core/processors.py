"""XBPM and BPM data processors."""

import numpy as np
import matplotlib.pyplot as plt

from .parameters import Prm
from .visualizers import PositionVisualizer
from .exporters import Exporter
from .constants import STD_ROI_SIZE, FIGDPI


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

    def analyze_central_sweeps(self, show: bool = False) -> tuple:
        """Analyze blade behavior at central sweep positions.

        Examines blade measurements along central horizontal and vertical
        lines to understand blade response and calculate suppression factors.

        Args:
            show: Whether to display sweep plots.

        Returns:
            Tuple of (range_h, range_v, blades_h, blades_v).
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

        return (self.range_h, self.range_v, self.blades_h, self.blades_v)

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
        fig, (axh, axv) = plt.subplots(1, 2, figsize=(12, 5))

        if fit_ch_v is not None:
            hline = ((fit_ch_v[0, 0] * self.range_h + fit_ch_v[1, 0])
                     * self.prm.xbpmdist)
            axh.plot(self.range_h * self.prm.xbpmdist, hline,
                     '^-', label="H fit")
            axh.plot(self.range_h * self.prm.xbpmdist,
                    pos_ch_v[:, 0] * self.prm.xbpmdist, 'o-', label="H sweep")
            axh.set_xlabel(u"$x$ [$\\mu$m]")
            axh.set_ylabel(u"$y$ [$\\mu$m]")
            axh.set_title("Central Horizontal Sweeps")
            ylim = (np.max(np.abs(hline + pos_ch_v[:, 0]
                                  * self.prm.xbpmdist)) * 1.1)
            axh.set_ylim(-ylim, ylim)
            axh.grid(True)
            axh.legend()

        if fit_cv_h is not None:
            vline = ((fit_cv_h[0, 0] * self.range_v + fit_cv_h[1, 0])
                     * self.prm.xbpmdist)
            axv.plot(pos_cv_h[:, 0] * self.prm.xbpmdist,
                     self.range_v * self.prm.xbpmdist,
                     'o-', label="V sweep")
            axv.plot(vline, self.range_v * self.prm.xbpmdist,
                     '^-', label="V fit")
            axv.set_xlabel(u"$x$ [$\\mu$m]")
            axv.set_ylabel(u"$y$ [$\\mu$m]")
            axv.set_title("Central Vertical Sweeps")
            axv.set_xlim((np.min(self.range_v) * 0.005 + fit_cv_h[1, 0])
                         * self.prm.xbpmdist,
                         (np.max(self.range_v) * 0.005 + fit_cv_h[1, 0])
                         * self.prm.xbpmdist)
            axv.grid(True)
            axv.legend()

        if self.prm.outputfile:
            outfile = f"xbpm_sweeps_{self.prm.beamline}.png"
            fig.savefig(outfile, dpi=FIGDPI)
            print(f" Figure of central sweeps saved to file {outfile}.\n")

    def show_blades_at_center(self) -> None:
        """Display blade measurements along central sweeping points."""
        # Ensure we have sweep data
        if self.range_h is None or self.range_v is None:
            self.analyze_central_sweeps(show=False)

        if self.blades_h is None and self.blades_v is None:
            print("\n WARNING: could not retrieve blades' currents,"
                  " maybe there is insufficient data."
                  " Skipping central analysis.")
            return

        fig, (axh, axv) = plt.subplots(1, 2, figsize=(10, 5))

        if self.blades_h is not None:
            for key, blval in self.blades_h.items():
                val = blval[:, 0]
                wval = blval[:, 1]
                weight = 1. / wval if not np.isinf(1. / wval).any else None
                (acoef, bcoef) = np.polyfit(self.range_h, val, deg=1, w=weight)
                axh.plot(self.range_h, self.range_h * acoef + bcoef, "o-",
                         label=f"{key} fit")
                axh.errorbar(self.range_h, val, wval, fmt='^-', label=key)

        if self.blades_v is not None:
            for key, blval in self.blades_v.items():
                val = blval[:, 0]
                wval = blval[:, 1]
                weight = 1. / wval if not np.isinf(1. / wval).any else None
                (acoef, bcoef) = np.polyfit(self.range_v, val, deg=1, w=weight)
                axv.plot(self.range_v, self.range_v * acoef + bcoef, "o-",
                         label=f"{key} fit")
                axv.errorbar(self.range_v, val, wval, fmt='^-', label=key)

        axh.set_title("Horizontal")
        axv.set_title("Vertical")
        axh.legend()
        axv.legend()
        axh.grid()
        axv.grid()
        axh.set_xlabel(u"$x$ $\\mu$rad")
        axv.set_xlabel(u"$y$ $\\mu$rad")

        ylabel = (u"$I$ [# counts]" if self.prm.beamline[:3]
                  in ["MGN", "MNC"] else u"$I$ [A]")
        axh.set_ylabel(ylabel)
        axv.set_ylabel(ylabel)
        fig.tight_layout()

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
        prm = self.prm

        blades, _stddevs = self.data_parse()

        supmat = self.suppression_matrix(showmatrix=showmatrix,
                                        nosuppress=nosuppress)

        pos_pair  = self.beam_position_pair(supmat)
        (pos_nom_h, pos_nom_v,
         pos_pair_h, pos_pair_v) = self.position_dict_parse(pos_pair)

        pos_nom_h *= prm.xbpmdist
        pos_nom_v *= prm.xbpmdist

        keys = np.array(list(data.keys()))
        range_h = np.unique(keys[:, 0])
        range_v = np.unique(keys[:, 1])
        halfh = int(range_h.shape[0] / 2)
        halfv = int(range_v.shape[0] / 2)
        abh = STD_ROI_SIZE if range_h[-1] > STD_ROI_SIZE else halfh
        frh, uph = halfh - abh, halfh + abh + 1
        abv = STD_ROI_SIZE if range_v[-1] > STD_ROI_SIZE else halfv
        frv, upv = halfv - abv, halfv + abv + 1

        if pos_nom_h.shape[0] == 1 or pos_nom_v.shape[0] == 1:
            pos_nom_h_roi = pos_nom_h[0, frv:upv]
            pos_nom_v_roi = pos_nom_v[0, frv:upv]
        elif pos_nom_h.shape[1] == 1 or pos_nom_v.shape[1] == 1:
            pos_nom_h_roi = pos_nom_h[frv:upv, 0]
            pos_nom_v_roi = pos_nom_v[frv:upv, 0]
        else:
            pos_nom_h_roi  = pos_nom_h[frh:uph, frv:upv]
            pos_nom_v_roi  = pos_nom_v[frh:uph, frv:upv]

        if pos_nom_h.shape[0] == 1 or pos_nom_v.shape[0] == 1:
            pos_pair_h_roi = pos_pair_h[0, frv:upv]
            pos_pair_v_roi = pos_pair_v[0, frv:upv]
        elif pos_nom_h.shape[1] == 1 or pos_nom_v.shape[1] == 1:
            pos_pair_h_roi = pos_pair_h[frv:upv, 0]
            pos_pair_v_roi = pos_pair_v[frv:upv, 0]
        else:
            pos_pair_h_roi = pos_pair_h[frh:uph, frv:upv]
            pos_pair_v_roi = pos_pair_v[frh:uph, frv:upv]

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
            pos_cr_h_roi  = pos_cr_h[frh:uph, frv:upv]
            pos_cr_v_roi  = pos_cr_v[frh:uph, frv:upv]

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
        return [scaled_pos_pair, scaled_pos_cros]

    def suppression_matrix(self, showmatrix=False, nosuppress=False):
        """Calculate the suppression matrix and persist it to disk."""
        if nosuppress:
            pch = np.ones(8).reshape(4, 2)
            pcv = np.ones(8).reshape(4, 2)
        else:
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

        Exporter(self.prm).write_supmat(supmat)
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
        gsh_lin = len(np.unique(gridlist[:, 1]))
        gsh_col = len(np.unique(gridlist[:, 0]))

        xbpm_nom_h  = np.zeros((gsh_lin, gsh_col))
        xbpm_nom_v  = np.zeros((gsh_lin, gsh_col))
        xbpm_meas_h = np.zeros((gsh_lin, gsh_col))
        xbpm_meas_v = np.zeros((gsh_lin, gsh_col))

        minval_h = np.min(gridlist[:, 0])
        maxval_v = np.max(gridlist[:, 1])
        for key, val in data.items():
            col = int((key[0] - minval_h) / self.prm.gridstep)
            lin = int((maxval_v - key[1]) / self.prm.gridstep)

            try:
                xbpm_nom_h[lin, col]  = key[0]
                xbpm_nom_v[lin, col]  = key[1]
                xbpm_meas_h[lin, col] = val[0]
                xbpm_meas_v[lin, col] = val[1]
            except Exception as err:
                print(f"\n WARNING: failed when parsing positions dictionary:"
                      f"\n{err}\n lin, col = {lin}, {col}, key = {key}")
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

        nh = np.unique(dk[:, 0])
        nv = np.unique(dk[:, 1])
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
        self.prm = prm

    def calculate_positions(self):
        """Calculate and plot XBPM positions derived from BPM data."""
        if self.prm.section is None:
            print("### ERROR: no section defined for the beamline in data set."
                  "\n### Cannot proceed with BPM data analysis. Skipping.")
            return

        if self.rawdata is None:
            print("### ERROR: no raw BPM data available."
                  "\n### Skipping BPM analysis.")
            return

        fig, ax = plt.subplots()

        sector_idx = self._sector_index()
        tangents = self._tangents_calc(sector_idx)

        print("# Distance between BPMs            ="
              f" {self.prm.bpmdist:8.4f}  m\n"
              "# Distance between source and XBPM ="
              f" {self.prm.xbpmdist:8.4f} m\n")

        xbpm_pos = self._positions_from_tangents(tangents, self.prm.xbpmdist)

        xpos, ypos, xnom, ynom = [], [], [], []
        for key, val in xbpm_pos.items():
            xnom.append(key[0])
            ynom.append(key[1])
            xpos.append(val[0])
            ypos.append(val[1])

        self._std_dev_estimate(xnom, ynom, xpos, ypos)

        ax.plot(xpos, np.array(ypos), 'bo', label="measured")
        ax.plot(xnom, ynom, 'r+', label="nominal")

        ax.set_xlabel("$x$ [$\\mu$m]")  # noqa: W605
        ax.set_ylabel("$y$ [$\\mu$m]")  # noqa: W605
        ax.set_title(f"Beam positions @ {self.prm.beamline} from BPM values")

        lim = np.max(np.abs(xnom + ynom)) * 1.7
        ax.axis("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.legend()
        ax.grid()

        if self.prm.outputfile:
            outfile = f"bpm_positions_{self.prm.beamline}.png"
            fig.savefig(outfile, dpi=FIGDPI)
            print(" Figure of positions calculated by BPM measurements "
                  f"saved to file {outfile}.\n")

    def _sector_index(self) -> int:
        sector = int(self.prm.section.split(':')[1][:2])
        return 8 * (sector - 1) - 1

    def _tangents_calc(self, idx):
        """Calculate tangents of beam angles between neighbour BPMs."""
        nextidx = idx + 1
        offset_x_sect, offset_y_sect = 0, 0
        offset_x_next, offset_y_next = 0, 0
        offsetfound = False

        for dt in self.rawdata:
            if dt[2]['agx'] == 0 and dt[2]['agy'] == 0:
                offset_x_sect = dt[2]['orbx'][idx]
                offset_y_sect = dt[2]['orby'][idx]
                offset_x_next = dt[2]['orbx'][nextidx]
                offset_y_next = dt[2]['orby'][nextidx]
                offsetfound = True
                break

        if not offsetfound:
            (offset_x_sect, offset_x_next,
             offset_y_sect, offset_y_next) = self._offset_search(idx)

        tangents = dict()
        for dt in self.rawdata:
            tx = (((dt[2]['orbx'][nextidx] - offset_x_next)) -
                  (dt[2]['orbx'][idx]     - offset_x_sect)) / self.prm.bpmdist
            ty = (((dt[2]['orby'][nextidx] - offset_y_next)) -
                  (dt[2]['orby'][idx]     - offset_y_sect)) / self.prm.bpmdist
            agx, agy = dt[2]['agx'], dt[2]['agy']
            tangents[agx, agy] = np.array([tx, ty])
        return tangents

    def _offset_search(self, idx):
        """Extrapolate offsets when reference orbit is missing."""
        nextidx = idx + 1
        agx    = np.array([dt[2]['agx']           for dt in self.rawdata])
        orbx   = np.array([dt[2]['orbx'][idx]     for dt in self.rawdata])
        n_orbx = np.array([dt[2]['orbx'][nextidx] for dt in self.rawdata])

        agxmax = np.max(agx)
        agxmin = np.min(agx)

        osx = np.array(sorted(list(set(orbx))))
        oxmin, oxmax = osx[0], osx[-1]
        offset_x_sect = (oxmin * agxmax - oxmax * agxmin) / (agxmax - agxmin)

        onx = np.array(sorted(list(set(n_orbx))))
        onxmin, onxmax = onx[0], onx[-1]
        offset_x_next = (onxmin * agxmax - onxmax * agxmin) / (agxmax - agxmin)

        agy    = np.array([dt[2]['agy']           for dt in self.rawdata])
        orby   = np.array([dt[2]['orby'][idx]     for dt in self.rawdata])
        n_orby = np.array([dt[2]['orby'][nextidx] for dt in self.rawdata])

        agymax = np.max(agy)
        agymin = np.min(agy)

        osy = np.array(sorted(list(set(orby))))
        oymin, oymax = osy[0], osy[-1]
        offset_y_sect = (oymin * agymax - oymax * agymin) / (agymax - agymin)

        ony = np.array(sorted(list(set(n_orby))))
        onymin, onymax = ony[0], ony[-1]
        offset_y_next = (onymin * agymax - onymax * agymin) / (agymax - agymin)

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
        np_x_nom = np.array(xnom)
        np_y_nom = np.array(ynom)
        np_x_pos = np.array(xpos)
        np_y_pos = np.array(ypos)

        nfh = np_x_nom.shape[0]
        nfv = np_y_nom.shape[0]
        diff_h = np.abs(np_x_nom.ravel() - np_x_pos.ravel())
        diff_h_max = np.max(diff_h)
        sig2_h = np.sum(diff_h**2) / nfh

        diff_v = np.abs(np_y_nom.ravel() - np_y_pos.ravel())
        diff_v_max = np.max(diff_v)
        sig2_v = np.sum(diff_v**2) / nfv

        print("Sigmas:\n"
              f"   (all sites)     H = {np.sqrt(sig2_h):.4f}\n"
              f"   (all sites)     V = {np.sqrt(sig2_v):.4f},\n"
              f"   (all sites) total = {np.sqrt(sig2_h + sig2_v):.4f}\n"
              "\n  Maximum difference:\n"
              f"   (all sites) H = {diff_h_max:.4f}\n"
              f"   (all sites) V = {diff_v_max:.4f},\n")

        nsh_x, nsh_y = len(set(xnom)), len(set(ynom))
        nmax = nsh_x * nsh_y

        if nmax > nfh or nmax > nfv:
            print("\n WARNING: sweeping looks incomplete, no ROI was defined. "
                  " (Maybe just one line swept?)")
            return

        frh, uptoh = int(nsh_x / 2 - 2), int(nsh_x / 2 + 2)
        frv, uptov = int(nsh_y / 2 - 2), int(nsh_y / 2 + 2)

        if nsh_x == 1 or nsh_y == 1:
            np_x_nom_cut = np_x_nom.reshape(nsh_x, nsh_y)[0, frv:uptov]
            np_y_nom_cut = np_y_nom.reshape(nsh_x, nsh_y)[0, frv:uptov]
            np_x_pos_cut = np_x_pos.reshape(nsh_x, nsh_y)[0, frv:uptov]
            np_y_pos_cut = np_y_pos.reshape(nsh_x, nsh_y)[0, frv:uptov]
        else:
            np_x_nom_cut = np_x_nom.reshape(nsh_x, nsh_y)[frv:uptov, frh:uptoh]
            np_y_nom_cut = np_y_nom.reshape(nsh_x, nsh_y)[frv:uptov, frh:uptoh]
            np_x_pos_cut = np_x_pos.reshape(nsh_x, nsh_y)[frv:uptov, frh:uptoh]
            np_y_pos_cut = np_y_pos.reshape(nsh_x, nsh_y)[frv:uptov, frh:uptoh]

        sig2_v_roi = np.sum((np_y_nom_cut.ravel() -
                             np_y_pos_cut.ravel())**2) / nfv
        sig2_h_roi = np.sum((np_x_nom_cut.ravel() -
                             np_x_pos_cut.ravel())**2) / nfh

        print("  Differences in ROI\n"
              f"   (x in [{np.min(np_x_nom_cut)}, {np.max(np_x_nom_cut)}];"
              f"  y in [{np.min(np_y_nom_cut)}, {np.max(np_y_nom_cut)}])\n"
              f"       H = {np.sqrt(sig2_h_roi):.4f}\n"
              f"       V = {np.sqrt(sig2_v_roi):.4f},\n"
              f"   total = {np.sqrt(sig2_h_roi + sig2_v_roi):.4f}")