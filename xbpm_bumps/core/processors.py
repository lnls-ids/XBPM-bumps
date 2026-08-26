"""XBPM and BPM data processors."""

# import os
# import matplotlib
import numpy as np

# from .visualizers import PositionVisualizer as PSV
# from .visualizers import SweepVisualizer as SWV
# from .visualizers import BladeCurrentVisualizer as BCV

from .config         import Config    
from .constants      import FIGDPI
from .data_structure import (
    Positions,
    BeamlineData,
    BeamlinePrm,
    BeamlineRawData,
    Prm,
    CentralSweeps,
    CentralSweepLine,
    Blades,
    RMSGridStatistics,
    RMSStatistics,
    ROISlice,
    )

# _Title = Config.get_plot_title   # shorthand for plot titles
# from .exporters import Exporter

# Keep math font consistent with visualizers (Computer Modern / cm).
# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['mathtext.rm'] = 'serif'
# matplotlib.rcParams['font.family'] = 'serif'


class XBPMProcessor:
    """Processes XBPM data to calculate beam positions.

    This class handles all calculation logic for XBPM position analysis:
    - Central sweep analysis to determine suppression matrices
    - Pairwise and cross-blade position calculations
    - Raw (no suppression) and scaled (with suppression) positions
    - Blade behavior analysis at central positions

    Attributes:
        blade_avg (BladeAvgData): Blade average data structure.
        prm (Prm): Parameters dataclass instance.
        prm_bl (BeamlinePrm): Beamline parameters dataclass instance.
    """

    def __init__(self,
                 beamlinedata: BeamlineData,
                 prm_bml: BeamlinePrm,
                 prm_gen: Prm
                 ) -> None:
        """Initialize processor with data and parameters.

        Args:
            blade_avg : BladeAvgData instance containing measurement data.
            prm_gen   : General parameters dataclass instance.
            prm_bml   : Beamline parameters dataclass instance.
        """
        # Get blade data structures.
        self.blade_avg  = beamlinedata.raw_data.blade_avg
        self.blades     = self.blade_avg.blades
        self.prm_avg    = self.blade_avg.prm
        self.prm_gen    = prm_gen

        # Analysis data.
        self.analysis = beamlinedata.analysis
        self.supmat   = self.analysis.supmat

        # Beamline parameters.
        self.prm_bml    = prm_bml
        self.beamline   = self.prm_bml.beamline
        # ROI defines V and H sizes (sz_h/v), and respective slices (sl_h/v).
        self.roi        = self.prm_bml.roi

        # Nominal positions.
        self.pos_nom = Positions(
            self.blade_avg.pos_nom.x,
            self.blade_avg.pos_nom.y
            )

        # Calculate ranges.
        self.range_h    = np.unique(self.pos_nom.x)
        self.range_v    = np.unique(self.pos_nom.y)

    def analyze_central_sweeps(self) -> "CentralSweeps":
        """Assemble the central sweep analysis.

        Returns:
            CentralSweeps instance containing horizontal and vertical central sweeps.
        """
        # Run through central lines if data is not just a point.
        h = (self.central_sweep_h() if len(self.range_h) > 1 else None)
        v = (self.central_sweep_v() if len(self.range_v) > 1 else None)
        return CentralSweeps(h=h, v=v)

    def central_sweep_h(self) -> "CentralSweepLine":
        """Analyze position calculation along the central horizontal line."""
        # Select blades at y ~ 0 (central horizontal line).
        pos_nom_x = self.blade_avg.pos_nom.x
        pos_nom_y = self.blade_avg.pos_nom.y
        mask  = np.isclose(pos_nom_y, 0)
        idx   = np.argsort(pos_nom_x[mask])
        blds  = self.blade_avg.blades
        to    = blds.to[mask][idx]
        ti    = blds.ti[mask][idx]
        bi    = blds.bi[mask][idx]
        bo    = blds.bo[mask][idx]
        sto   = blds.sto[mask][idx]
        sti   = blds.sti[mask][idx]
        sbi   = blds.sbi[mask][idx]
        sbo   = blds.sbo[mask][idx]
        nom_x = pos_nom_x[mask][idx]
        nom_y = pos_nom_y[mask][idx]

        # Calculate positions using pairwise Δ/Σ formula.
        # calc_pos_v is the calculated set of positions at central line
        # along h direction (fixed nominal y at 0), expected to be zero.
        s_top = to + ti
        s_bot = bo + bi
        pos_calc_v = (s_top - s_bot) / (s_top + s_bot)

        # Fit a linear model to the position data and calculate uncertainties.
        fit, cov   = np.polyfit(self.range_h, pos_calc_v, deg=1, cov=True)
        fit_pos_v  = np.polyval(fit, self.range_h)
        sa, sb     = np.sqrt(np.diag(cov))
        fit_v_err  = np.sqrt((self.range_h * sa)**2 + sb**2)

        # Build the SweepLine data structure for horizontal sweep.
        blades = Blades(to, ti, bi, bo, sto, sti, sbi, sbo, nom_x, nom_y)
        return CentralSweepLine(
            blades=blades,
            pos_index=self.range_h,
            pos_fixed=pos_nom_y[mask][idx],
            pos_calc=pos_calc_v,
            pos_fit=fit_pos_v,
            pos_fit_err=fit_v_err
            )

    def central_sweep_v(self) -> "CentralSweepLine":
        """Analyze position calculation along the central vertical line."""
        # Select blades at x ~ 0 (central vertical line).
        pos_nom_x = self.blade_avg.pos_nom.x
        pos_nom_y = self.blade_avg.pos_nom.y
        mask  = np.isclose(pos_nom_x, 0)
        idx   = np.argsort(pos_nom_y[mask])
        blds  = self.blade_avg.blades
        to    = blds.to[mask][idx]
        ti    = blds.ti[mask][idx]
        bi    = blds.bi[mask][idx]
        bo    = blds.bo[mask][idx]
        sto   = blds.sto[mask][idx]
        sti   = blds.sti[mask][idx]
        sbi   = blds.sbi[mask][idx]
        sbo   = blds.sbo[mask][idx]
        nom_x = pos_nom_x[mask][idx]
        nom_y = pos_nom_y[mask][idx]

        # Calculate positions using pairwise Δ/Σ formula.
        s_left     = to + bo
        s_right    = ti + bi
        pos_calc_h = (s_left - s_right) / (s_left + s_right)

        # Fit a linear model to the position data and calculate uncertainties.
        fit, cov   = np.polyfit(self.range_v, pos_calc_h, deg=1, cov=True)
        pos_fit_h  = np.polyval(fit, self.range_v)
        sa, sb     = np.sqrt(np.diag(cov))
        fit_h_err  = np.sqrt((self.range_v * sa)**2 + sb**2)

        # Build the SweepLine data structure for vertical sweep.
        blades = Blades(to, ti, bi, bo, sto, sti, sbi, sbo, nom_x, nom_y)
        return CentralSweepLine(
            blades=blades,
            pos_index=self.range_v,
            pos_fixed=pos_nom_x[mask][idx],
            pos_calc=pos_calc_h,
            pos_fit=pos_fit_h,
            pos_fit_err=fit_h_err,
            )

#
# Position calculation tabs.
#

    def xbpm_position_calculation(self, suppress: bool = False) -> dict:
        """Orchestrate position calculation for pairwise and cross-blade.

        Delegates to helpers for reduced complexity while maintaining
        full analysis pipeline.
        """
        # Ensure central sweep data is available.
        # This call should be transferred to the orchestrator.
        self.pos_central_sweeps = self.analyze_central_sweeps()

        # Extract nominal ROI slices.
        pos_nom_h_roi = self.pos_nom.x[self.roi.sl_v, self.roi.sl_h]
        pos_nom_v_roi = self.pos_nom.y[self.roi.sl_v, self.roi.sl_h]

        # Compute suppression matrix at the ROI.
        self.suppression_matrix()

        # Pairwise calculation (Delta/Sigma).

        # No correction, uses standard suppression matrix.
        pos_pair_std = self.beam_position_pair(self.supmat.standard)

        # Process data: fitting, scaling, stats, visualization.
        pairwise_result_std = self.scale_positions(pos_pair_std)




        # Cross-blade calculation (partial Delta/Sigma).
        pos_cross_h, pos_cross_v = self.beam_position_cross(self.blades)

        # Extract ROI slices from measured data.
        pos_roi_cross_h = pos_cross_h[self.roi.sl_v, self.roi.sl_h]
        pos_roi_cross_v = pos_cross_v[self.roi.sl_v, self.roi.sl_h]

        # Process data: fitting, scaling, stats, visualization.
        cross_result = self.scale_positions(
            pos_cross_h,
            pos_cross_v,
            pos_roi_cross_h,
            pos_roi_cross_v,
            pos_nom_h_roi,
            pos_nom_v_roi,
            )

        # Compile and return results
        return self._compile_results(
            pairwise_result_std,
            cross_result,
            self.supmat,
            suppress,
            self.pos_nom.x,
            self.pos_nom.y
            )

    def suppression_matrix(self) -> None:
        """Calculate the suppression matrix from blade behavior.

        Args:
            to, ti, bi, bo: Blade measurements.
            sto, sti, sbi, sbo: Blade measurements std dev.

        Returns:
            Tuple of (suppression matrix, standard deviation matrix)
        """
        # Calculate blade slopes for scaled calculations.
        # Blade sweep in horizontal direction.
        pc_h, covs_h = self.blade_central_line_fit(
            self.roi.sl_h, 0, self.range_h[self.roi.sl_h]
            )
        # Blade sweep in vertical direction.
        pc_v, covs_v = self.blade_central_line_fit(
            0, self.roi.sl_v, self.range_v[self.roi.sl_v]
            )

        if len(self.range_h) > 1:
            sdv_h = np.sqrt(covs_h) * pc_h[0, 0] / (pc_h[:, 0]**2)
            pc_h  = pc_h[0] / np.abs(pc_h)
        else:
            pc_h  = np.ones(8).reshape(4, 2)
            sdv_h = np.zeros(4)

        if len(self.range_v) > 1:
            sdv_v = np.sqrt(covs_v) * pc_v[0, 0] / (pc_v[:, 0]**2)
            pc_v  = pc_v[0] / np.abs(pc_v)
        else:
            pc_v  = np.ones(8).reshape(4, 2)
            sdv_v = np.zeros(4)

        self.supmat.calculated = np.array([
            [pc_v[0, 0], -pc_v[1, 0], -pc_v[2, 0],  pc_v[3, 0]],
            [pc_v[0, 0],  pc_v[1, 0],  pc_v[2, 0],  pc_v[3, 0]],
            [pc_h[0, 0],  pc_h[1, 0], -pc_h[2, 0], -pc_h[3, 0]],
            [pc_h[0, 0],  pc_h[1, 0],  pc_h[2, 0],  pc_h[3, 0]],
        ])

        self.supmat.stddev = np.array([
            [sdv_v[0], sdv_v[1], sdv_v[2], sdv_v[3]],
            [sdv_v[0], sdv_v[1], sdv_v[2], sdv_v[3]],
            [sdv_h[0], sdv_h[1], sdv_h[2], sdv_h[3]],
            [sdv_h[0], sdv_h[1], sdv_h[2], sdv_h[3]],
        ])

    def blade_central_line_fit(self,
                               range_vals: np.ndarray
                               ) -> tuple:
        """Linear fittings to each blade's data through central line.
        
        Args:
            roi_h: slice for horizontal ROI.
            roi_v: slice for vertical ROI.
            range_vals: range values corresponding to the ROI.

        Returns:
            Tuple of (fit coefficients, std dev values) for each blade.
        """
        # Define blades values according to ROI for slope calculation.
        to = self.blades.to[self.roi.sl_v, self.roi.sl_h]
        ti = self.blades.ti[self.roi.sl_v, self.roi.sl_h]
        bi = self.blades.bi[self.roi.sl_v, self.roi.sl_h]
        bo = self.blades.bo[self.roi.sl_v, self.roi.sl_h]

        sto = self.blades.sto[self.roi.sl_v, self.roi.sl_h]
        sti = self.blades.sti[self.roi.sl_v, self.roi.sl_h]
        sbi = self.blades.sbi[self.roi.sl_v, self.roi.sl_h]
        sbo = self.blades.sbo[self.roi.sl_v, self.roi.sl_h]

        pc = list()
        covs = list()
        for blade, err in [
            (to, sto), (ti, sti), (bi, sbi), (bo, sbo)
            ]:
            weight = 1. / err[:]

            if np.isinf(weight).any():
                weight = None

            coefs, cov = np.polyfit(range_vals,
                                    blade[:, 0],
                                    deg=1,
                                    w=weight,
                                    cov=True)
            pc.append(coefs)
            covs.append(cov[0, 0])
        pc = np.array(pc)

        if np.isinf(pc).any() or (pc == 0).any():
            pc = np.array([[1, 0] for _ in range(4)]) 
        return pc, covs

    def beam_position_pair(self, supmat: np.ndarray) -> dict:
        """Calculate beam position from blades' currents (pairwise)."""
        blade_meas = np.stack([
            self.blades.to,
            self.blades.ti,
            self.blades.bi,
            self.blades.bo
            ], axis=0)
        Q_deltasum = np.matmul(supmat, blade_meas.reshape(4, -1))
        x = Q_deltasum.T[:, 0] / Q_deltasum.T[:, 1]
        y = Q_deltasum.T[:, 2] / Q_deltasum.T[:, 3]
        return Positions(x, y)

    @staticmethod
    def beam_position_cross(blades) -> list:
        """Calculate beam position from blades' currents (cross-blade)."""
        to, ti, bi, bo = blades
        v1 = (to - bi) / (to + bi)
        v2 = (ti - bo) / (ti + bo)
        hpos = (v1 - v2)
        vpos = (v1 + v2)
        return [hpos, vpos]

    def scale_positions(self, pos_calc: Positions) -> dict:
        """Scale positions, pairwise or cross-blade.

        Args:
            calc_type       : 'pairwise' (Δ/Σ) or 'cross' (partial Δ/Σ).
            pos_all_h/v     : Full position array (measured)
            pos_nom_h/v_roi : ROI slice of nominal positions
            nosuppress      : If True, label results as raw mode.

        Returns:
            Dict with scaled positions, scales, stats, visualizer.
        """
        pos_nom_roi_x = self.pos_nom.x[self.roi.sl_v, self.roi.sl_h]
        pos_nom_roi_y = self.pos_nom.y[self.roi.sl_v, self.roi.sl_h]

        pos_roi_x = pos_calc.x[self.roi.sl_v, self.roi.sl_h]
        pos_roi_y = pos_calc.y[self.roi.sl_v, self.roi.sl_h]

        # Perform scaling fit
        # label = "Δ/Σ" if calc_type == "pairwise" else "Partial Δ/Σ"
        ((scl_x, sig_x),
         (scl_y, sig_y)) = self.scaling_fit(
            pos_roi_x,
            pos_roi_y,
            pos_nom_roi_x,
            pos_nom_roi_y,
            polydeg=self.prm_bml.scalepolydeg,
        )
        (qx, kx, deltax), (sqx, skx, sdeltax) = scl_x, sig_x
        (qy, ky, deltay), (sqy, sky, sdeltay) = scl_y, sig_y

        # Scale all positions.
        pos_all_x_scaled = qx * pos_calc.x**2 + kx * pos_calc.x + deltax
        pos_all_y_scaled = qy * pos_calc.y**2 + ky * pos_calc.y + deltay
        # Scale ROI positions.
        pos_roi_x_scaled = qx * pos_roi_x**2 + kx * pos_roi_x + deltax
        pos_roi_y_scaled = qy * pos_roi_y**2 + ky * pos_roi_y + deltay

        # Compute statistics
        rms_all = calculate_grid_stats(
            pos_calc.x,
            pos_calc.y,
            pos_all_x_scaled,
            pos_all_y_scaled,
        )

        rms_roi = calculate_grid_stats(
            pos_roi_x,
            pos_roi_y,
            pos_roi_x_scaled,
            pos_roi_y_scaled,
        )

        pos_analyzed = {
            'pos_nom' : Positions(pos_nom_roi_x, pos_nom_roi_y),
            'pos_calc' : Positions(pos_roi_x, pos_roi_y),
        }


        return {
            'h_scaled'     : pos_all_x_scaled,
            'v_scaled'     : pos_all_y_scaled,
            'h_roi_scaled' : pos_roi_x_scaled,
            'v_roi_scaled' : pos_roi_y_scaled,
            'qx'           : qx,
            'sqx'          : sqx,
            'kx'           : kx,
            'skx'          : skx,
            'qy'           : qy,
            'sqy'          : sqy,
            'ky'           : ky,
            'sky'          : sky,
            'dx'           : deltax,
            'sdx'          : sdeltax,
            'dy'           : deltay,
            'sdy'          : sdeltay,
            'rms_all'      : rms_all,
            'rms_roi'      : rms_roi,
        }

    def _compile_results(self, pair_result: dict, cross_result: dict,
                         supmat: np.ndarray, stddevmat: np.ndarray,
                         nosuppress: bool,
                         pos_nom_h: np.ndarray, pos_nom_v: np.ndarray) -> dict:
        """Compile and save final results from pairwise and cross-blade."""
        # Build position dictionaries for export.
        # Keys are always derived from the raw scan grid (data.keys() angles ×
        # xbpmdist) so they are regular and sortable regardless of whether the
        # optimisation reference is the nominal grid or BPM-measured positions.
        gridlist  = np.array(list(self.blade_avg.keys()))
        grid_lin  = np.unique(gridlist[:, 1])   # sorted y scan angles
        grid_col  = np.unique(gridlist[:, 0])   # sorted x scan angles
        dist      = self.prm_bml.xbpmdist

        scaled_pos_pair  = dict()
        scaled_pos_cross = dict()
        for ii, y in enumerate(grid_lin):
            for jj, x in enumerate(grid_col):
                xk = x * dist
                yk = y * dist
                scaled_pos_pair[xk, yk] = [
                    pair_result['h_scaled'][ii, jj],
                    pair_result['v_scaled'][ii, jj]
                ]
                scaled_pos_cross[xk, yk] = [
                    cross_result['h_scaled'][ii, jj],
                    cross_result['v_scaled'][ii, jj]
                ]

        return {
            'positions'       : [scaled_pos_pair, scaled_pos_cross],
            'scales' : {
                'pair'    : {
                    'qx'  : pair_result['qx'],
                    'sqx' : pair_result['sqx'],
                    'kx'  : pair_result['kx'],
                    'skx' : pair_result['skx'],
                    'dx'  : pair_result['dx'],
                    'sdx' : pair_result['sdx'],
                    'qy'  : pair_result['qy'],
                    'sqy' : pair_result['sqy'],
                    'ky'  : pair_result['ky'],
                    'sky' : pair_result['sky'],
                    'dy'  : pair_result['dy'],
                    'sdy' : pair_result['sdy'],
                },
                'cross'   : {
                    'qx'  : cross_result['qx'],
                    'sqx' : cross_result['sqx'],
                    'kx'  : cross_result['kx'],
                    'skx' : cross_result['skx'],
                    'dx'  : cross_result['dx'],
                    'sdx' : cross_result['sdx'],
                    'qy'  : cross_result['qy'],
                    'sqy' : cross_result['sqy'],
                    'ky'  : cross_result['ky'],
                    'sky' : cross_result['sky'],
                    'dy'  : cross_result['dy'],
                    'sdy' : cross_result['sdy'],
                },
            },
            'supmat'     : supmat,
            'stddevmat'  : stddevmat,
            'phaseorgap' : self.prm_gen.phaseorgap,
            'xbpm_stats' : {
                'pairwise' : pair_result['stats'],
                'cross'    : cross_result['stats'],
            },
        }

    def scaling_fit(self,
                    pos_cal_h: np.ndarray,
                    pos_cal_v: np.ndarray,
                    pos_nom_h: np.ndarray,
                    pos_nom_v: np.ndarray,
                    polydeg: int = 1,
                    ) -> tuple:
        """Calculate scaling coefficients from fitted positions.
        
        Args:
            pos_cal_h: Measured horizontal positions array.
            pos_cal_v: Measured vertical positions array.
            pos_nom_h: Nominal horizontal positions array.
            pos_nom_v: Nominal vertical positions array.
            polydeg  : Degree of the polynomial fit.
        
        Returns:
            kx     : Horizontal scaling factor.
            deltax : Horizontal offset.
            ky     : Vertical scaling factor.
            deltay : Vertical offset.
            s*     : Standard deviations of the respective coefficients.
        """
        # Clean data by removing NaN or infinite values for fitting.
        h_finitemask = np.isfinite(pos_cal_h)
        pos_cal_h_cln = pos_cal_h[h_finitemask]
        pos_nom_h_cln = pos_nom_h[h_finitemask]

        v_finitemask = np.isfinite(pos_cal_v)
        pos_cal_v_cln = pos_cal_v[v_finitemask]
        pos_nom_v_cln = pos_nom_v[v_finitemask]

        # Fit a polynomial to the cleaned data.
        coeffs_x, sigmas_x = self.poly_fitting(
            pos_nom_h_cln,
            pos_cal_h_cln,
            polydeg
            )
        coeffs_y, sigmas_y = self.poly_fitting(
            pos_nom_v_cln,
            pos_cal_v_cln,
            polydeg
            )

        # Prepend zero for quadratic term if necessary.
        if self.prm_bml.scalepolydeg == 1:
            coeffs_x = [0, *coeffs_x]
            coeffs_y = [0, *coeffs_y]
            sigmas_x = [0, *sigmas_x]
            sigmas_y = [0, *sigmas_y]

        return (
            (coeffs_x, sigmas_x),
            (coeffs_y, sigmas_y)
            )

    def poly_fitting(self,
                     nom: np.ndarray,
                     meas: np.ndarray,
                     polydeg: int = 1,
                     ) -> tuple:
        """Return fitting parameters for scaling fit."""
        # Default values.
        coeffs  = np.zeros(polydeg + 1)
        sigmas  = np.zeros(polydeg + 1)
        covs    = None

        # Check if the nominal values are constant or if
        # there are too few valid points.
        if len(set(nom.ravel())) <= 1 or meas.size < 2:
            return (coeffs, sigmas)

        try:
            coeffs, covs = np.polyfit(meas, nom, deg=polydeg, cov=True)
            sigmas = np.sqrt(np.diag(covs))
        except Exception:
            # Keep fitted coefficients if covariance cannot be estimated.
            try:
                coeffs = np.polyfit(meas, nom, deg=polydeg)
            except Exception as err:
                print(f"\n WARNING: when calculating horizontal scaling"
                        f" coefficients:\n{err}\n"
                        " Setting to default values.")
        return (coeffs, sigmas)


class BPMProcessor:
    """Processes BPM-only data to estimate XBPM positions.

    Encapsulates the legacy BPM calculation flow so orchestration can
    call a single entry point instead of standalone functions.
    """

    def __init__(self,
                 rawdata : BeamlineRawData,
                 prm_bml : BeamlinePrm,
                 ) -> None:
        """Store raw BPM/XBPM dataset and parameters for later processing."""
        self.rawdata = rawdata
        self.sweeps  = rawdata.sweeps_bpm
        self.prm_bml = prm_bml
        self.roi     = prm_bml.roi
        self._print_bpm_info()
        self.calculate_positions()

    def _print_bpm_info(self) -> None:
        """Print BPM position information."""
        print("\n# BPM position calculation"
              f"\n# {'Distance between neighbor BPMs':<35} ="
              f" {self.prm_bml.bpmdist:8.4f}  m")
        print(f"# {'Distance between source and XBPM':<35} ="
              f" {self.prm_bml.xbpmdist:8.4f} m\n")

    def calculate_positions(self) -> tuple:
        """Calculate and plot XBPM positions derived from BPM data.

        Returns:
            Array of [x, y] coordinates or None if calculation fails.
        """
        # Calculate positions at XBPM from BPM tangents and distances.
        self._positions_from_tangents()

        # Estimate standard deviations.
        self.rms_grid_stats = RMSGridStatistics(
            self.nom_x,  self.nom_y,
            self.meas_x, self.meas_y,
            self.prm_bml.roi
            )
        # self.rms_diff_all, self.rms_diff_roi = self.std_dev_estimate()

        # Extract ROI data for closeup view.
        self._extract_roi_positions()

        if self.prm_bml.outputfile:
            outfile = f"bpm_positions_{self.prm_bml.beamline}.png"
            self.fig.savefig(outfile, dpi=FIGDPI)
            print(" Figure of positions calculated by BPM measurements "
                  f"saved to file {outfile}.\n")

        self._stack_measurement_results()

        # self.plot_bpm_positions()

    def _positions_from_tangents(self) -> dict:
        """Calculate beam positions from tangents at BPMs."""
        # Calculate the tangents.
        self._tangents_calc(self._sector_index())

        xbpm_dist = self.prm_bml.xbpmdist
        positions = dict()
        for key, tg in self.tangents.items():
            newkey = (key[0] * xbpm_dist, key[1] * xbpm_dist)
            positions[newkey] = tg * xbpm_dist

        #
        # Assemble position data into structured grid numpy arrays.
        #

        # Get unique sorted indices for x and y from the position keys.
        xidx = sorted(set([key[0] for key in positions.keys()]))
        yidx = sorted(set([key[1] for key in positions.keys()]))
        nx, ny = len(xidx), len(yidx)

        # Initialize numpy arrays for nominal and measured positions.
        self.nom_x  = np.zeros((ny, nx))
        self.nom_y  = np.zeros((ny, nx))
        self.meas_x = np.full((ny, nx), np.nan)
        self.meas_y = np.full((ny, nx), np.nan)

        # Fill the arrays.
        missing = 0
        for iy in range(ny):
            for ix in range(nx):
                key = (xidx[ix], yidx[iy])
                self.nom_x[iy, ix] = key[0]
                self.nom_y[iy, ix] = key[1]
                if key in positions:
                    self.meas_x[iy, ix] = positions[key][0]
                    self.meas_y[iy, ix] = positions[key][1]
                else:
                    missing += 1

        if missing > 0:
            print("\n WARNING: sparse BPM grid detected:"
                  f" {missing} points missing from nominal mesh."
                  " Missing points were set to NaN.")

    def _tangents_calc(self, sector_idx: int) -> dict:
        """Calculate tangents of beam angles between neighbour BPMs."""
        sector_idx_nxt = sector_idx + 1      # Next BPM in the sector.
        offset_x_sect, offset_y_sect = 0, 0
        offset_x_next, offset_y_next = 0, 0
        offsetfound = False

        for swp in self.sweeps:
            agx = swp.prm.get('Angle x')
            agy = swp.prm.get('Angle y')
            if agx == 0 and agy == 0:
                offset_x_sect = swp.bpm.pos.x[sector_idx]
                offset_y_sect = swp.bpm.pos.y[sector_idx]
                offset_x_next = swp.bpm.pos.x[sector_idx_nxt]
                offset_y_next = swp.bpm.pos.y[sector_idx_nxt]
                offsetfound = True
                break

        # Try and guess offsets by extrapolation if not found.
        if not offsetfound:
            (offset_x_sect, offset_x_next,
             offset_y_sect, offset_y_next) = self._offset_search(sector_idx)

        # Calculate tangents for all angles.
        self.tangents = dict()
        bdist = self.prm_bml.bpmdist
        # for dt in self.rawdata:
        for swp in self.sweeps:
            agx = swp.prm.get('Angle x')
            agy = swp.prm.get('Angle y')

            orbx     = swp.bpm.pos.x[sector_idx]
            orby     = swp.bpm.pos.y[sector_idx]
            orbx_nxt = swp.bpm.pos.x[sector_idx_nxt]
            orby_nxt = swp.bpm.pos.y[sector_idx_nxt]
            tx = ((orbx_nxt - offset_x_next) -
                  (orbx - offset_x_sect)) / bdist
            ty = ((orby_nxt - offset_y_next) -
                  (orby - offset_y_sect)) / bdist
            self.tangents[agx, agy] = np.array([tx, ty])

    def _sector_index(self) -> int:
        """Extract sector index from the section string."""
        return 8 * (self.prm_bml.sector[1] - 1) - 1

    def _offset_search(self, sector_idx: int) -> tuple:
        """Extrapolate offsets when reference orbit is missing."""
        # Get the angle and orbit data for the current and next BPMs
        # across all measurements.
        sector_idx_nxt = sector_idx + 1

        def _offset_from_direction(direction: str,
                                         orb: np.ndarray,
                                         orb_nxt: np.ndarray
                                         ) -> tuple:
            """Search offset in given direction."""
            # Get nominal angle value.
            angle = np.array([swp.prm.get(f'Angle {direction}')
                              for swp in self.sweeps])

            ang_min = np.min(angle)
            ang_max = np.max(angle)
            if np.isclose(ang_max, ang_min):
                raise ValueError(
                    f"Cannot infer BPM {direction}-offset from data without {direction} angle variation "
                    "or explicit (agx=0, agy=0) reference point."
                )

            orb_sort = np.array(sorted(list(set(orb))))
            orb_min, orb_max = orb_sort[0], orb_sort[-1]
            offset_sect = ((orb_min * ang_max - orb_max * ang_min) /
                           (ang_max - ang_min))

            orb_sort_nxt = np.array(sorted(list(set(orb_nxt))))
            orb_min_nxt, orb_max_nxt = orb_sort_nxt[0], orb_sort_nxt[-1]
            offset_next = ((orb_min_nxt * ang_max - orb_max_nxt * ang_min) /
                           (ang_max - ang_min))

            return (offset_sect, offset_next)

        orbx     = np.array([swp.bpm.pos.x[sector_idx]
                             for swp in self.sweeps])
        orbx_nxt = np.array([swp.bpm.pos.x[sector_idx_nxt]
                             for swp in self.sweeps])
        (offset_x, offset_x_nxt) = _offset_from_direction('x', orbx, orbx_nxt)

        orby     = np.array([swp.bpm.pos.y[sector_idx]
                             for swp in self.sweeps])
        orby_nxt = np.array([swp.bpm.pos.y[sector_idx_nxt]
                             for swp in self.sweeps])
        (offset_y, offset_y_nxt) = _offset_from_direction('y', orby, orby_nxt)

        return (offset_x, offset_x_nxt, offset_y, offset_y_nxt)

    def _extract_roi_positions(self) -> tuple:
        """Extract ROI positions from full grid for closeup view.

        Returns:
            Tuple (xnom_roi, ynom_roi, xpos_roi, ypos_roi) of ROI arrays.
        """
        rows, cols = self.bana.roi.update(
            self.nom_x.shape, (self.roi.sz_v, self.roi.sz_h)
            )
        self.xnom_roi = self.nom_x[rows, cols]
        self.ynom_roi = self.nom_y[rows, cols]
        self.xpos_roi = self.meas_x[rows, cols]
        self.ypos_roi = self.meas_y[rows, cols]

    def _stack_measurement_results(self) -> tuple:
        """Compile measured and nominal coordinates into return format.

        Returns:
            Tuple (measured, nominal) where each is a 2-column array or None.
        """
        self.measured = (np.column_stack(
            (self.meas_x.ravel(), self.meas_y.ravel()))
            if self.meas_x.size else None)
        self.nominal = (np.column_stack(
            (self.nom_x.ravel(), self.nom_y.ravel()))
            if self.nom_x.size else None)


def calculate_grid_stats(
        nom_x   : np.ndarray,
        nom_y   : np.ndarray,
        meas_x  : np.ndarray,
        meas_y  : np.ndarray,
    ) -> "RMSGridStatistics":
    """Calculate RMS statistics from squared position differences in ROI.

    Args:
        nom_x  : nominal values in x direction
        nom_y  : nominal values in y direction
        meas_x : measured values in x direction
        meas_y : measured values in y direction

    Returns:
        dict: Statistics with rms_h, rms_v, rms_total,
                rms_max_h, rms_min_h, rms_max_v, rms_min_v.
    """
    # Squared differences.
    diff_x2 = (meas_x - nom_x) ** 2
    diff_y2 = (meas_y - nom_y) ** 2

    # Check for valid points, since the sweeping might be interrupted.
    valid  = np.isfinite(diff_x2) & np.isfinite(diff_y2)
    nsites = int(np.count_nonzero(valid))
    if nsites == 0:
        print("\n WARNING: no valid BPM points found for RMS estimation.")
        rms_stats = {
            key : np.nan
            for key in [
                'h', 'v', 'tot',
                'min_h', 'max_h',
                'min_v', 'max_v',
                'mean_h', 'mean_v', 'mean_tot',
                ]}
        return rms_stats

    # Valid points only.
    vld_diff_x2  = diff_x2[valid]
    vld_diff_y2  = diff_y2[valid]

    # Absolute values of differences.
    rms_h        = np.sqrt(vld_diff_x2)
    rms_v        = np.sqrt(vld_diff_y2)
    rms_tot      = np.sqrt(vld_diff_x2 + vld_diff_y2)

    # RMS minimum and maximum values.
    rms_max_h    = np.max(rms_h)
    rms_min_h    = np.min(rms_h)
    rms_max_v    = np.max(rms_v)
    rms_min_v    = np.min(rms_v)

    # RMS global estimates.
    rms_mean_h   = np.sqrt(np.mean(vld_diff_x2))
    rms_mean_v   = np.sqrt(np.mean(vld_diff_y2))
    rms_mean_tot = np.sqrt(np.mean(vld_diff_x2 + vld_diff_y2))

    nsites_total = int(diff_x2.size)
    if nsites < nsites_total:
        print("\n WARNING: sweeping looks incomplete, no ROI was defined"
            f" ({nsites} valid sites, out of {nsites_total}"
                " in total). Skipping ROI analysis.")

    rms = {
        'h'        : rms_h,
        'v'        : rms_v,
        'tot'      : rms_tot,
        'min_h'    : rms_min_h,
        'max_h'    : rms_max_h,
        'min_v'    : rms_min_v,
        'max_v'    : rms_max_v,
        'mean_h'   : rms_mean_h,
        'mean_v'   : rms_mean_v,
        'mean_tot' : rms_mean_tot,
    }
    return RMSStatistics(**rms)


def grid_statistics(nom_x: np.ndarray,
                    nom_y: np.ndarray,
                    meas_x: np.ndarray,
                    meas_y: np.ndarray,
                    roislice: ROISlice,
                    ) -> RMSGridStatistics:
    """Calculate grid statistics from measured and nominal positions.

    Args:
        nom_x  : Nominal x positions (2D array).
        nom_y  : Nominal y positions (2D array).
        meas_x : Measured x positions (2D array).
        meas_y : Measured y positions (2D array).
    """
    # Statistics for the full grid.
    rms_all = calculate_grid_stats(nom_x, nom_y, meas_x, meas_y)

    # Statistics at ROI.
    sl_v, sl_h = roislice.sl_v, roislice.sl_h
    rms_roi = calculate_grid_stats(
        nom_x[sl_v, sl_h],
        nom_y[sl_v, sl_h],
        meas_x[sl_v, sl_h],
        meas_y[sl_v, sl_h]
        )

    return RMSGridStatistics(
        all=rms_all,
        roi=rms_roi,
        roislice=roislice
    )
