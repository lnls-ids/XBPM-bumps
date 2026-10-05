"""XBPM and BPM data processors."""

import numpy as np
from copy import deepcopy
from .config import Config
from . import data_structure as DStr


class XBPMProcessor:
    """Processes XBPM data to calculate beam positions.

    This class handles all calculation logic for XBPM position analysis:
    - Central sweep analysis to determine suppression matrices
    - Pairwise and cross-blade position calculations
    - Raw (no suppression) and scaled (with suppression) positions
    - Blade behavior analysis at central positions

    Attributes:
        blade_avg (BladeAvgData) : Blade average data structure.
        prm       (Prm)          : Parameters dataclass instance.
        prm_bl    (BeamlinePrm)  : Beamline parameters dataclass instance.
    """

    def __init__(self,
                 beamlinedata : DStr.BeamlineData,
                 beamline_prm : DStr.BeamlinePrm,
                 runtime_prm  : DStr.GenPrm,
                 analysis     : DStr.DataAnalysis,
                 ) -> None:
        """Initialize processor with data and parameters.

        Args:
            beamlinedata : DStr.BeamlineData instance containing
                        measurement data.
            runtime_prm  : DStr.Prm instance containing general parameters.
            beamline_prm : DStr.BeamlinePrm instance containing
                        beamline parameters.
        """
        # Get blade data structures.
        self.blade_avg  = beamlinedata.raw_data.blade_avg

        self.blades     = self.blade_avg.blades
        self.prm_avg    = self.blade_avg.prm
        self.prm_gen    = runtime_prm
        # Analysis data.
        self.analysis   = analysis

        # Beamline parameters.
        self.prm_bml    = beamline_prm
        self.beamline   = self.prm_bml.beamline
        # ROI defines V and H sizes (sz_h/v), and respective slices (sl_h/v).
        self.roi        = self.prm_bml.roislice

        # Nominal positions.
        self.pos_nom    = self.blade_avg.pos_nom
        self._central_sweep_slices()

    """
    Position calculation tabs.
    """

    def xbpm_position_calculation(self) -> DStr.AnalyzedPositions:
        """Orchestrate position calculation for pairwise and cross-blade.

        Delegates to helpers for reduced complexity while maintaining
        full analysis pipeline.
        """
        # Define reference positions as nominal or BPM's.
        self.pos_ref = (
            self.analysis.bpm.pos_meas
            if self.prm_bml.usebpmref
            else self.blade_avg.pos_nom
            )

        # Perform scaling fit
        # Compute suppression matrix at the ROI.
        self.supmat = self._calculate_suppression_matrix()
        # Keep track of the calculated matrices.
        self.analysis.supmat = self.supmat

        # Pairwise (pw) calculation (Delta/Sigma).
        # Positions with standard suppression matrix.
        pos_pw_std = self.beam_position_pair(self.supmat.standard)

        # Scale positions for pairwise standard case.
        pw_scale_std, pw_pos_std = self._scale_positions(
            self.pos_ref,
            pos_pw_std,
            )

        # RMS statistics.
        rms_pw_std = grid_statistics(
            self.pos_ref.x,
            self.pos_ref.y,
            pw_pos_std.x,
            pw_pos_std.y,
            self.roi
        )

        # Positions with calculated suppression matrix and scaling.
        pos_pw_sup = self.beam_position_pair(self.supmat.calculated)

        # Scale positions.
        pw_scale_sup, pw_pos_sup = self._scale_positions(
            self.pos_ref,
            pos_pw_sup,
            )

        # RMS statistics.
        rms_pw_sup = grid_statistics(
            self.pos_ref.x,
            self.pos_ref.y,
            pw_pos_sup.x,
            pw_pos_sup.y,
            self.roi
        )

        # Pairwise results instance.
        pairwise_res = DStr.CalculatedPositions(
            roi=deepcopy(self.roi),
            pos_std=pw_pos_std,
            scale_std=pw_scale_std,
            stat_std=rms_pw_std,
            pos_trn=pw_pos_sup,
            scale_trn=pw_scale_sup,
            stat_trn=rms_pw_sup
        )

        # Compute suppression matrix at the ROI.
        # Cross-blade calculation (partial Delta/Sigma).
        # Positions from standard formulae.
        pos_cr_std = self.beam_position_cross(self.blades)

        # Process data: fitting, scaling, stats, visualization.
        cr_scale_std, pos_cr_std_scaled = self._scale_positions(
            self.pos_ref,
            pos_cr_std,
            )

        # RMS statistics.
        rms_cr_std = grid_statistics(
            self.pos_ref.x,
            self.pos_ref.y,
            pos_cr_std_scaled.x,
            pos_cr_std_scaled.y,
            self.roi
        )

        # Positions with linear general transfomation.
        pos_cr_lintr = self.transform_position_cross(pos_cr_std_scaled)
        self.analysis.gl2r = self.gl2rmat

        # Scale positions.
        cr_scale_lintr, pos_cr_lintr_scaled = self._scale_positions(
            self.pos_ref,
            pos_cr_lintr,
            )

        # RMS statistics.
        rms_cr_lintr = grid_statistics(
            self.pos_ref.x,
            self.pos_ref.y,
            pos_cr_lintr_scaled.x,
            pos_cr_lintr_scaled.y,
            self.roi
        )

        # Cross-blade results instance.
        cross_res = DStr.CalculatedPositions(
            roi=deepcopy(self.roi),
            pos_std=pos_cr_std_scaled,
            scale_std=cr_scale_std,
            stat_std=rms_cr_std,
            pos_trn=pos_cr_lintr,
            scale_trn=cr_scale_lintr,
            stat_trn=rms_cr_lintr
        )

        # Compile and return results
        return DStr.AnalyzedPositions(
            nom=deepcopy(self.pos_ref),
            bpm=deepcopy(self.analysis.bpm),
            pairw=pairwise_res,
            cross=cross_res
            )

    """
    Blade behavior at central line sweeps.
    """

    def analyze_central_sweep_blades(self) -> dict:
        """Analyze the central positions of the blades."""
        # Define blade intervals according to ROI for slope calculation.
        blades_h = {
            'to'  : self.cs_bld_h.to[self.roi.slice_h],
            'ti'  : self.cs_bld_h.ti[self.roi.slice_h],
            'bi'  : self.cs_bld_h.bi[self.roi.slice_h],
            'bo'  : self.cs_bld_h.bo[self.roi.slice_h],
            'sto' : self.cs_bld_h.sto[self.roi.slice_h],
            'sti' : self.cs_bld_h.sti[self.roi.slice_h],
            'sbi' : self.cs_bld_h.sbi[self.roi.slice_h],
            'sbo' : self.cs_bld_h.sbo[self.roi.slice_h],
        }
        bld_fit_h = {
            k: (blades_h[k], blades_h[f's{k}'])
            for k in ['to', 'ti', 'bi', 'bo']
        }

        blades_v = {
            'to'  : self.cs_bld_v.to[self.roi.slice_v],
            'ti'  : self.cs_bld_v.ti[self.roi.slice_v],
            'bi'  : self.cs_bld_v.bi[self.roi.slice_v],
            'bo'  : self.cs_bld_v.bo[self.roi.slice_v],
            'sto' : self.cs_bld_v.sto[self.roi.slice_v],
            'sti' : self.cs_bld_v.sti[self.roi.slice_v],
            'sbi' : self.cs_bld_v.sbi[self.roi.slice_v],
            'sbo' : self.cs_bld_v.sbo[self.roi.slice_v],
        }
        bld_fit_v = {
            k: (blades_v[k], blades_v[f's{k}'])
            for k in ['to', 'ti', 'bi', 'bo']
        }

        # Perform central line fit for horizontal and vertical blade analysis.
        hrange = self.range_h[self.roi.slice_h]
        horz   = self._blade_central_line_fit(hrange, bld_fit_h)

        vrange = self.range_v[self.roi.slice_v]
        vert   = self._blade_central_line_fit(vrange, bld_fit_v)

        return DStr.BCA_HV(
            h = DStr.BladeCenterAnalysis(**horz),
            v = DStr.BladeCenterAnalysis(**vert),
        )

    def _blade_central_line_fit(self,
                                range_vals: np.ndarray,
                                blades: dict,
                                ) -> dict:
        """Linear fittings to each blade's data through central line.
        
        Args:
            range_vals : range values corresponding to the ROI.
            blades     : dictionary containing blade data and associated
                         errors within the ROI.

        Returns:
            Dictionary containing the fit coefficients and standard
            deviations for each blade.
        """
        # Loop over each blade and perform linear fitting.
        results = {}
        for bl, (blade, err) in blades.items():
            # Calculate weights as the inverse of the errors, if possible.
            weight = 1. / err[:]
            if np.isinf(weight).any():
                weight = None

            # Fit a linear polynomial to the blade data with weights.
            coefs, cov = np.polyfit(range_vals,
                                    blade,
                                    deg = 1,
                                    w   = weight,
                                    cov = True)

            # Coefficient errors.
            sigmas = np.sqrt(np.diag(cov))

            # Fitted values.
            yfit = np.polyval(coefs, range_vals)

            # Store results in the dataclass for each blade.
            results[bl] = DStr.BladeLineFit(
                k       = coefs[0],
                sk      = sigmas[0],
                d       = coefs[1],
                sd      = sigmas[1],
                pos     = range_vals,
                bld_raw = blade,
                bld_err = err,
                bld_fit = yfit,
            )
        return results

    """
    Position calculation with conventional delta/sigma formula.
    """

    def _calculate_suppression_matrix(self) -> DStr.SuppressionMatrix:
        """Calculate the suppression matrix from blade behavior.

        Args:
            to, ti, bi, bo: Blade measurements.
            sto, sti, sbi, sbo: Blade measurements std dev.

        Returns:
            Tuple of (suppression matrix, standard deviation matrix)
        """
        sw_h = self.analysis.bladecenter.h
        sw_v = self.analysis.bladecenter.v

        pc_h  = np.array([sw_h.to.k,  sw_h.ti.k,  sw_h.bi.k,  sw_h.bo.k])
        pc_v  = np.array([sw_v.to.k,  sw_v.ti.k,  sw_v.bi.k,  sw_v.bo.k])
        sdv_h = np.array([sw_h.to.sk, sw_h.ti.sk, sw_h.bi.sk, sw_h.bo.sk])
        sdv_v = np.array([sw_v.to.sk, sw_v.ti.sk, sw_v.bi.sk, sw_v.bo.sk])

        # Check for the vertical and horizontal cases.
        if len(self.range_h) > 1:
            # Calculate suppresions (1/Gain), normalized by TO blade.
            pc_h  = sw_h.to.k  / np.abs(pc_h)
            sto   = sw_h.to.sk / sw_h.to.k
            sdv_h = (
                (sw_h.to.k / pc_h[:]) * ((sdv_h / pc_h[:])**2 + sto**2)**0.5
                )
        else:
            pc_h  = np.ones(8).reshape(4, 2)
            sdv_h = np.zeros(4)

        if len(self.range_v) > 1:
            # Calculate suppresions (1/Gain), normalized by TO blade.
            pc_v  = sw_v.to.k / np.abs(pc_v)
  
            sdv_v = sdv_v * sw_v.to.k / (pc_v[:]**2)
            sto   = sw_v.to.sk / sw_v.to.k
            sdv_v = (
                (sw_v.to.k / pc_v[:]) * ((sdv_v / pc_v[:])**2 + sto**2)**0.5
                )

        else:
            pc_v  = np.ones(8).reshape(4, 2)
            sdv_v = np.zeros(4)

        # Assemble the calculated suppression matrix.
        calculated = np.array([
            [pc_v[0], -pc_v[1], -pc_v[2],  pc_v[3]],
            pc_v,
            [pc_h[0],  pc_h[1], -pc_h[2], -pc_h[3]],
            pc_h,
        ])

        # Assemble the standard deviation matrix.
        stddev = np.array([
            sdv_v,
            sdv_v,
            sdv_h,
            sdv_h
            ])

        # Get the standard matrix (gains == 1).
        std, _ = Config.standard_suppression_matrix()
        return DStr.SuppressionMatrix(
            standard=std,
            calculated=calculated,
            stddev=stddev,
            optimized=deepcopy(calculated)
            )

    def beam_position_pair(self, supmat: np.ndarray) -> DStr.Positions:
        """Calculate beam position from blades' currents (pairwise).
        
        Args:
            supmat: Suppression matrix to apply to blade measurements.
            supmat may be either the standard or calculated suppression matrix.

        Returns:
            Calculated beam position as a Positions instance.
        """
        # Stack blade measurements into a 4xN array for matrix multiplication.
        blade_meas = np.stack([
            self.blades.to,
            self.blades.ti,
            self.blades.bi,
            self.blades.bo
            ],
            axis=0)

        # Multiply supmat by the stacked values accordingly.
        Q_deltasum = np.einsum('lk,kij->lij', supmat, blade_meas)

        # Calculate positions using the Δ/Σ formula.
        x = Q_deltasum[0] / Q_deltasum[1]
        y = Q_deltasum[2] / Q_deltasum[3]
        return DStr.Positions(x, y)

    def _central_sweep_slices(self) -> None:
        """Extract central sweep blade values for both h, v directions."""
        # Calculate ranges.
        self.range_h    = np.unique(self.pos_nom.x)
        self.range_v    = np.unique(self.pos_nom.y)
        self.grid_shape = (
            len(self.range_v),
            len(self.range_h)
            )

        # Shortcut to blade data.
        blades = self.blade_avg.blades

        # Select blades at the central horizontal line (y ~ 0).
        # Find sites next to the zero position.
        mask_h = np.isclose(self.pos_nom.y, 0)
        if not np.any(mask_h):
            print("WARNING: no counts/currents found near the central"
                  " horizontal line (y ~ 0).")
            self.cs_bld_h = None
        else:
            # Guarantee the order of the indices along the horizontal direction.
            idx_h  = np.argsort(self.pos_nom.x[mask_h])

            self.cs_bld_h = DStr.Blades(
                to      = blades.to[mask_h][idx_h],
                ti      = blades.ti[mask_h][idx_h],
                bi      = blades.bi[mask_h][idx_h],
                bo      = blades.bo[mask_h][idx_h],
                sto     = blades.sto[mask_h][idx_h],
                sti     = blades.sti[mask_h][idx_h],
                sbi     = blades.sbi[mask_h][idx_h],
                sbo     = blades.sbo[mask_h][idx_h],
                pos_nom = DStr.Positions(
                    x = self.pos_nom.x[mask_h][idx_h],
                    y = self.pos_nom.y[mask_h][idx_h]
                )
            )

        # Select blades at the central vertical line (x ~ 0).
        mask_v = np.isclose(self.pos_nom.x, 0)
        if not np.any(mask_v):
            print("WARNING: no counts/currents found near the central"
                  " vertical line (x ~ 0).")
            self.cs_bld_v = None
        else:
            idx_v   = np.argsort(self.pos_nom.y[mask_v])
            self.cs_bld_v = DStr.Blades(
                to      = blades.to[mask_v][idx_v],
                ti      = blades.ti[mask_v][idx_v],
                bi      = blades.bi[mask_v][idx_v],
                bo      = blades.bo[mask_v][idx_v],
                sto     = blades.sto[mask_v][idx_v],
                sti     = blades.sti[mask_v][idx_v],
                sbi     = blades.sbi[mask_v][idx_v],
                sbo     = blades.sbo[mask_v][idx_v],
                pos_nom = DStr.Positions(
                    x = self.pos_nom.x[mask_v][idx_v],
                    y = self.pos_nom.y[mask_v][idx_v]
                )
            )

    """
    Position calculation with partial delta/sigma formula.
    """

    def analyze_central_sweep_positions(self,
                               pairw: bool = False
                               ) -> DStr.CentralSweeps:
        """Assemble the central sweep analysis.

        Returns:
            CentralSweeps instance containing horizontal and vertical central sweeps.
        """
        # Run through central lines if data is not just a point.
        h = (
            self._central_sweep_h(pairw)
            if len(self.range_h) > 1
            else None
            )
        v = (
            self._central_sweep_v(pairw)
            if len(self.range_v) > 1
            else None
            )
        return DStr.CentralSweeps(h=h, v=v)

    def _central_sweep_h(self,
                        pairw: bool = False
                        ) -> DStr.CentralSweepLine:
        """Analyze position calculation along the central horizontal line."""
        # Shortcuts for the central horizontal blades.
        to = self.cs_bld_h.to
        ti = self.cs_bld_h.ti
        bi = self.cs_bld_h.bi
        bo = self.cs_bld_h.bo

        # Calculate positions using pairwise Δ/Σ or cross-blades (partial Δ/Σ) 
        # formula. pos_calc_v is the calculated set of positions at central
        # line along h direction (fixed nominal y at 0), expected to be zero.
        if pairw:
            s_top = to + ti
            s_bot = bo + bi
            pos_calc_v = (s_top - s_bot) / (s_top + s_bot)
        else:
            v1 = (to - bi) / (to + bi)
            v2 = (ti - bo) / (ti + bo)
            pos_calc_v = (v1 + v2)

        # Fit a linear model to the position data and calculate uncertainties.
        range_h = np.linspace(-1, 1, len(self.range_h))
        fit, cov = np.polyfit(range_h, pos_calc_v, deg=1, cov=True)
        pos_fit_v  = np.polyval(fit, range_h)
        sk, sd   = np.sqrt(np.diag(cov))
        fit_err  = np.sqrt((range_h * sk)**2 + sd**2)

        pos_calc_v -= fit[1]
        pos_fit_v  -= fit[1]

        return DStr.CentralSweepLine(
            blades      = self.cs_bld_h,
            pos_index   = range_h,
            pos_fixed   = self.cs_bld_h.pos_nom.y,
            pos_calc    = pos_calc_v,
            pos_fit     = pos_fit_v,
            pos_fit_err = fit_err,
            coeffs      = fit,
            sigmas      = np.sqrt(np.diag(cov))
        )

    def _central_sweep_v(self,
                        pairw: bool = False
                        ) -> DStr.CentralSweepLine:
        """Analyze position calculation along the central vertical line."""
        # Shortcuts for the central vertical blades.
        to = self.cs_bld_v.to
        ti = self.cs_bld_v.ti
        bi = self.cs_bld_v.bi
        bo = self.cs_bld_v.bo

        # Calculate positions using pairwise Δ/Σ or cross-blades (partial Δ/Σ)
        # formula. pos_calc_h is the calculated set of positions at central
        # line along v direction (fixed nominal x at 0), expected to be zero.
        if pairw:
            s_left     = to + bo
            s_right    = ti + bi
            pos_calc_h = (s_left - s_right) / (s_left + s_right)
        else:
            h1 = (to - bi) / (to + bi)
            h2 = (ti - bo) / (ti + bo)
            pos_calc_h = (h1 + h2)

        # Fit a linear model to the position data and calculate uncertainties.
        range_v  = np.linspace(-1, 1, len(self.range_v))
        fit, cov = np.polyfit(range_v, pos_calc_h, deg=1, cov=True)
        pos_fit_h  = np.polyval(fit, range_v)
        sa, sb   = np.sqrt(np.diag(cov))
        fit_err  = np.sqrt((range_v * sa)**2 + sb**2)

        pos_calc_h -= fit[1]
        pos_fit_h  -= fit[1]

        return DStr.CentralSweepLine(
            blades      = self.cs_bld_v,
            pos_index   = range_v,
            pos_fixed   = self.cs_bld_v.pos_nom.x,
            pos_calc    = pos_calc_h,
            pos_fit     = pos_fit_h,
            pos_fit_err = fit_err,
            coeffs      = fit,
            sigmas      = np.sqrt(np.diag(cov))
            )

    @staticmethod
    def beam_position_cross(blades) -> DStr.Positions:
        """Calculate beam position from blades' currents (cross-blade)."""
        v1 = (blades.to - blades.bi) / (blades.to + blades.bi)
        v2 = (blades.ti - blades.bo) / (blades.ti + blades.bo)
        hpos = (v1 - v2)
        vpos = (v1 + v2)
        return DStr.Positions(x=hpos, y=vpos)

    def transform_position_cross(self,
                                 pos_std: DStr.Positions
                                 ) -> DStr.Positions:
        """Transform cross-blade positions using the linear transformation matrix."""
        # Extract the coefficients from the central sweep linear fits.

        # Compute suppression matrix at the ROI.
        csweep = self.analysis.centralsweeps
        self.gl2rmat = self.general_linear_transformation(
            csweep.h.coeffs[0],
            csweep.v.coeffs[0]
            )

        # Stack cross-blade positions into a 2xN array for transformation.
        scross = np.stack((pos_std.x, pos_std.y), axis=0)

        # Apply the linear transformation to the stacked positions.
        pos_tr = np.einsum('ij,jkl->ikl', self.gl2rmat.calc, scross)
        return DStr.Positions(x=pos_tr[0], y=pos_tr[1])

    def general_linear_transformation(self,
                                      kx: float,
                                      ky: float
                                      ) -> np.ndarray:
        """Calculate the linear transformation matrix from position slopes."""
        c = 1. / (kx - ky)
        gl2rmat = c * np.array([
            [ ky, -1],
            [-kx,  1]
        ])
        return DStr.GL2RMatrix(calc = gl2rmat)

    """
    Scaling functions.
    """

    def _scale_positions(self,
                        pos_nom  : DStr.Positions,
                        pos_calc : DStr.Positions
                        ) -> tuple[DStr.Scales,
                                   DStr.Positions]:
        """Scale positions, pairwise or cross-blade.

        Scale positions using polynomial fitting to reference positions (nominal or BPM's). Return calculated scales (polynomial coefficients with errors) and scaled positions.

        Args:
            calc_type   : 'pairwise' (Δ/Σ) or 'cross' (partial Δ/Σ).
            pos_all_h/v : Full position array (measured)
            pos_nom_h/v : nominal h or v positions
            nosuppress  : If True, label results as raw mode.

        Returns:
            Tuple with scales and all scaled positions.
        """
        nom_roi_x = pos_nom.x[self.roi.slice_v, self.roi.slice_h]
        nom_roi_y = pos_nom.y[self.roi.slice_v, self.roi.slice_h]
        calc_roi_x = pos_calc.x[self.roi.slice_v, self.roi.slice_h]
        calc_roi_y = pos_calc.y[self.roi.slice_v, self.roi.slice_h]

        # Perform scaling fit
        # label = "Δ/Σ" if calc_type == "pairwise" else "Partial Δ/Σ"
        ((scl_x, sig_x),
         (scl_y, sig_y)) = self._scaling_fit(
            nom_roi_x,
            nom_roi_y,
            calc_roi_x,
            calc_roi_y,
            polydeg=self.prm_bml.scalepolydeg,
        )
        (qx, kx, deltax), (sqx, skx, sdeltax) = scl_x, sig_x
        (qy, ky, deltay), (sqy, sky, sdeltay) = scl_y, sig_y

        # Scale all positions.
        pos_all_x_scaled = qx * pos_calc.x**2 + kx * pos_calc.x + deltax
        pos_all_y_scaled = qy * pos_calc.y**2 + ky * pos_calc.y + deltay

        # Build scales dataclass instance.
        scales = DStr.Scales(
            qx=qx, sqx=sqx, kx=kx, skx=skx, dx=deltax, sdx=sdeltax,
            qy=qy, sqy=sqy, ky=ky, sky=sky, dy=deltay, sdy=sdeltay
        )
        pos_all_scaled = DStr.Positions(
            x=pos_all_x_scaled,
            y=pos_all_y_scaled
            )
        return scales, pos_all_scaled

    def _scaling_fit(self,
                    nom_h: np.ndarray,
                    nom_v: np.ndarray,
                    cal_h: np.ndarray,
                    cal_v: np.ndarray,
                    polydeg: int = 1,
                    ) -> tuple:
        """Calculate scaling coefficients from fitted positions.
        
        Args:
            nom_h: Nominal horizontal positions array.
            nom_v: Nominal vertical positions array.
            cal_h: Measured horizontal positions array.
            cal_v: Measured vertical positions array.
            polydeg  : Degree of the polynomial fit.
        
        Returns:
            kx     : Horizontal scaling factor.
            deltax : Horizontal offset.
            ky     : Vertical scaling factor.
            deltay : Vertical offset.
            s*     : Standard deviations of the respective coefficients.
        """
        # Clean data by removing NaN or infinite values for fitting.
        h_finitemask = np.isfinite(cal_h)
        cal_h_cln = cal_h[h_finitemask]
        nom_h_cln = nom_h[h_finitemask]

        v_finitemask = np.isfinite(cal_v)
        cal_v_cln = cal_v[v_finitemask]
        nom_v_cln = nom_v[v_finitemask]

        # Fit a polynomial to the cleaned data.
        coeffs_x, sigmas_x = self._poly_fitting(
            nom_h_cln,
            cal_h_cln,
            polydeg
            )
        coeffs_y, sigmas_y = self._poly_fitting(
            nom_v_cln,
            cal_v_cln,
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

    def _poly_fitting(self,
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
                 raw_data : DStr.BeamlineRawData,
                 prm_bml  : DStr.BeamlinePrm,
                 ) -> None:
        """Store raw BPM/XBPM dataset and parameters for later processing."""
        self.raw_data    = raw_data
        self.sweeps_bpm  = raw_data.sweeps_bpm
        self.meta        = raw_data.meta
        self.prm_bml     = prm_bml
        self.roi         = prm_bml.roislice
        self._print_bpm_info()
        self.bpmanalysis = self.calculate_positions()

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
        self.rms_grid_stats = grid_statistics(
            self.nom_x,
            self.nom_y,
            self.meas_x,
            self.meas_y,
            self.prm_bml.roislice
        )

        # Build structures for BPM analysis results.
        pos_meas = DStr.Positions(
            x=self.meas_x,
            y=self.meas_y
            )
        bpmanalysis = DStr.BPMAnalysis(
            pos_meas=pos_meas,
            prm=self.prm_bml,
            rms_diff=self.rms_grid_stats,
        )
        return bpmanalysis

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

        # Select offsets from BPMs with zero angles (reference orbit).
        for ns in self.sweeps_bpm:
            agx = self.meta[ns].get('Angle x')
            agy = self.meta[ns].get('Angle y')
            if agx == 0 and agy == 0:
                offset_x_sect = self.sweeps_bpm[ns].pos.x[sector_idx]
                offset_y_sect = self.sweeps_bpm[ns].pos.y[sector_idx]
                offset_x_next = self.sweeps_bpm[ns].pos.x[sector_idx_nxt]
                offset_y_next = self.sweeps_bpm[ns].pos.y[sector_idx_nxt]
                offsetfound = True
                break

        # Try and guess offsets by extrapolation if not found.
        if not offsetfound:
            (offset_x_sect, offset_x_next,
             offset_y_sect, offset_y_next) = self._offset_search(sector_idx)

        # Calculate tangents for all angles.
        self.tangents = dict()
        bdist = self.prm_bml.bpmdist
        for ns in self.sweeps_bpm:
            agx = self.meta[ns].get('Angle x')
            agy = self.meta[ns].get('Angle y')

            orbx     = self.sweeps_bpm[ns].pos.x[sector_idx]
            orby     = self.sweeps_bpm[ns].pos.y[sector_idx]
            orbx_nxt = self.sweeps_bpm[ns].pos.x[sector_idx_nxt]
            orby_nxt = self.sweeps_bpm[ns].pos.y[sector_idx_nxt]
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
            angle = np.array([
                swp.meta.get(f'Angle {direction}')
                for ns, swp in self.sweeps_bpm.items()
                ])

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
                             for ns, swp in self.sweeps_bpm.items()])
        orbx_nxt = np.array([swp.bpm.pos.x[sector_idx_nxt]
                             for ns, swp in self.sweeps_bpm.items()])
        (offset_x, offset_x_nxt) = _offset_from_direction('x', orbx, orbx_nxt)

        orby     = np.array([swp.bpm.pos.y[sector_idx]
                             for ns, swp in self.sweeps_bpm.items()])
        orby_nxt = np.array([swp.bpm.pos.y[sector_idx_nxt]
                             for ns, swp in self.sweeps_bpm.items()])
        (offset_y, offset_y_nxt) = _offset_from_direction('y', orby, orby_nxt)

        return (offset_x, offset_x_nxt, offset_y, offset_y_nxt)


def calculate_grid_stats(
        nom_x   : np.ndarray,
        nom_y   : np.ndarray,
        meas_x  : np.ndarray,
        meas_y  : np.ndarray,
        ) -> DStr.RMSStatistics:
    """Calculate RMS statistics from squared position differences in ROI.

    Args:
        nom_x  : nominal values in x direction
        nom_y  : nominal values in y direction
        meas_x : measured values in x direction
        meas_y : measured values in y direction

    Returns:
        DStr.RMSStatistics: Statistics with rms_h, rms_v, rms_total,
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
        rms_dict = {
            key: np.nan for key in [
            'min_h', 'max_h', 'min_v', 'max_v',
            'mean_h', 'mean_v', 'mean_tot',
            ]}
        shape = nom_x.shape
        rms_dict.update({
            'h'   : np.full(shape, np.nan),
            'v'   : np.full(shape, np.nan),
            'tot' : np.full(shape, np.nan),
        })
        return DStr.RMSStatistics(**rms_dict)

    # Valid points only.
    diff_x2 = np.where(valid, diff_x2, np.nan)
    diff_y2 = np.where(valid, diff_y2, np.nan)

    # Absolute values of differences.
    rms_h        = np.sqrt(diff_x2)
    rms_v        = np.sqrt(diff_y2)
    rms_tot      = np.sqrt(diff_x2 + diff_y2)

    # RMS minimum and maximum values.
    rms_max_h    = np.nanmax(rms_h)
    rms_min_h    = np.nanmin(rms_h)
    rms_max_v    = np.nanmax(rms_v)
    rms_min_v    = np.nanmin(rms_v)

    # RMS global estimates.
    rms_mean_h   = np.sqrt(np.nanmean(diff_x2))
    rms_mean_v   = np.sqrt(np.nanmean(diff_y2))
    rms_mean_tot = np.sqrt(np.nanmean(diff_x2 + diff_y2))

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
    return DStr.RMSStatistics(**rms)


def grid_statistics(
        nom_x    : np.ndarray,
        nom_y    : np.ndarray,
        meas_x   : np.ndarray,
        meas_y   : np.ndarray,
        roislice : DStr.ROISlice,
        ) -> DStr.RMSGridStatistics:
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
    sl_v, sl_h = roislice.slice_v, roislice.slice_h
    rms_roi = calculate_grid_stats(
        nom_x[sl_v, sl_h],
        nom_y[sl_v, sl_h],
        meas_x[sl_v, sl_h],
        meas_y[sl_v, sl_h]
        )

    return DStr.RMSGridStatistics(
        all=rms_all,
        roi=rms_roi,
        roislice=roislice
    )
