"""XBPM and BPM data processors."""

import os
import numpy as np
import matplotlib

from .visualizers import PositionVisualizer as PSV
from .visualizers import SweepVisualizer as SWV
from .visualizers import BladeCurrentVisualizer as BCV

from .config         import Config    
from .constants      import FIGDPI
from .data_structure import (
    BeamlinePrm,
    BeamlineRawData,
    BladeAvgData,
    Prm,
    SweepData,
    SweepLine,
    Blades,
    RMSGridStatistics
    )

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
        blade_avg (BladeAvgData): Blade average data structure.
        prm (Prm): Parameters dataclass instance.
        prm_bl (BeamlinePrm): Beamline parameters dataclass instance.
    """

    def __init__(self,
                 blade_avg: BladeAvgData,
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
        self.blade_avg  = blade_avg
        self.blades     = blade_avg.blades
        self.prm_avg    = blade_avg.prm
        self.prm_gen    = prm_gen

        # Beamline parameters.
        self.prm_bml    = prm_bml
        self.beamline   = self.prm_bml.beamline
        # ROI defines V and H sizes (sz_h/v), and respective slices (sl_h/v).
        self.roi        = self.prm_bml.roi

        # Nominal positions.
        self.nom_pos_x  = self.blade_avg.nom.x
        self.nom_pos_y  = self.blade_avg.nom.y

        # Calculate ranges.
        self.range_h    = np.unique(self.nom_pos_x)
        self.range_v    = np.unique(self.nom_pos_y)

        # Are these really needed?
        self.blades_h   = None
        self.blades_v   = None
        self._initialize_ranges()
        self.roi_v_size = self.prm_bml.roisize[0]
        self.roi_h_size = self.prm_bml.roisize[1]

    def analyze_central_sweeps(self,
                               show: bool = False
                               ) -> tuple[np.array, np.array,
                                          SweepLine, SweepLine]:
        """Analyze blade behavior at central sweep positions.

        Examines blade measurements along central horizontal and vertical
        lines to understand blade response and calculate suppression factors.

        Args:
            show: Whether to display sweep plots.

        Returns:
            Tuple of (np.array, np.array, SweepLine, SweepLine).
        """
        # Run through central horizontal line if data is not just a point
        self.sweepline_h = (self._central_sweep_h()
                            if len(self.range_h) > 1 else None)

        # Run through central vertical line if data is not just a point
        self.sweepline_v = (self._central_sweep_v()
                            if len(self.range_v) > 1 else None)

        if show:
            fig = SWV.plot_from_arrays(
                self.range_h,
                self.range_v,
                self.sweepline_h,
                self.sweepline_v,
                xbpm_dist=self.prm_bml.xbpmdist
            )

            if self.prm_gen.outputfile:
                outfile = f"xbpm_sweeps_{self.prm_bml.beamline}.png"
                fig.savefig(outfile, dpi=FIGDPI)
                print(f" Figure of central sweeps saved to file {outfile}.\n")

        return (
            self.range_h,
            self.range_v,
            self.sweepline_h,
            self.sweepline_v
            )

    def _central_sweep_h(self) -> "SweepLine":
        """Analyze blade behavior along the central horizontal line."""
        # Select blades at y ~ 0 (central horizontal line).
        pos_nom_x = self.blade_avg.nom.x
        pos_nom_y = self.blade_avg.nom.y
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

        # Calculate positions using pairwise Δ/Σ formula.
        # calc_pos_v is the calculated set of positions at central line
        # along h direction (fixed nominal y at 0), expected to be zero.
        s_top = to + ti
        s_bot = bo + bi
        calc_pos_v = (s_top - s_bot) / (s_top + s_bot)

        # Fit a linear model to the position data and calculate uncertainties.
        fit, cov   = np.polyfit(self.range_h, calc_pos_v, deg=1, cov=True)
        fit_pos_v  = np.polyval(fit, self.range_h)
        sa, sb     = np.sqrt(np.diag(cov))
        fit_v_err  = np.sqrt((self.range_h * sa)**2 + sb**2)

        # Build the SweepLine data structure for horizontal sweep.
        blades = Blades(to, ti, bi, bo, sto, sti, sbi, sbo)
        return SweepLine(
            blades=blades,
            index=self.range_h,
            fixed=pos_nom_y[mask][idx],
            calc_pos=calc_pos_v,
            fit_pos=fit_pos_v,
            fit_pos_err=fit_v_err
            )

    def _central_sweep_v(self) -> "SweepLine":
        """Analyze blade behavior along the central vertical line."""
        # Select blades at x ~ 0 (central vertical line).
        pos_nom_x = self.blade_avg.nom.x
        pos_nom_y = self.blade_avg.nom.y
        mask = np.isclose(pos_nom_x, 0)
        idx  = np.argsort(pos_nom_y[mask])
        blds = self.blade_avg.blades
        to   = blds.to[mask][idx]
        ti   = blds.ti[mask][idx]
        bi   = blds.bi[mask][idx]
        bo   = blds.bo[mask][idx]
        sto  = blds.sto[mask][idx]
        sti  = blds.sti[mask][idx]
        sbi  = blds.sbi[mask][idx]
        sbo  = blds.sbo[mask][idx]

        # Calculate positions using pairwise Δ/Σ formula.
        s_left     = to + bo
        s_right    = ti + bi
        calc_pos_h = (s_left - s_right) / (s_left + s_right)

        # Fit a linear model to the position data and calculate uncertainties.
        fit, cov   = np.polyfit(self.range_v, calc_pos_h, deg=1, cov=True)
        fit_pos_h  = np.polyval(fit, self.range_v)
        sa, sb     = np.sqrt(np.diag(cov))
        fit_h_err  = np.sqrt((self.range_v * sa)**2 + sb**2)

        # Build the SweepLine data structure for vertical sweep.
        blades = Blades(to, ti, bi, bo, sto, sti, sbi, sbo)
        return SweepLine(
            blades=blades,
            index=self.range_v,
            fixed=pos_nom_x[mask][idx],
            calc_pos=calc_pos_h,
            fit_pos=fit_pos_h,
            fit_pos_err=fit_h_err,
            )

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

        fig = BCV.plot_blade_center_from_dicts(
            self.blades_h, self.blades_v,
            self.range_h, self.range_v,
            beamline=self.prm_bml.beamline
            )

        if self.prm_gen.outputfile:
            outfile = f"central_sweep_{self.prm_bml.beamline}.png"
            fig.savefig(outfile, dpi=FIGDPI)
            print("\n Figure of blades behaviour at central sweeps"
                  f" saved to file {outfile}.\n")

    def _roi_slice_indices(self, array: np.ndarray) -> tuple:
        """Extract centered ROI slice indices from an array, handling 1D/2D.

        Analyze 1D and 2D arrays to determine the appropriate slice indices
        for the region of interest (ROI). The ROI is centered and sized according to the specified horizontal and vertical dimensions, while ensuring that the indices remain within the bounds of the input array.

        Args:
            array: Input array from which to extract ROI slice.

        Returns:
            A tuple containing the slice indices (fr_col, up_col, fr_row,
            up_row).
        """
        n_lin, n_col = array.shape
        n_roi_h = min(self.roi_h_size, n_col)
        n_roi_v = min(self.roi_v_size, n_lin)

        # DEBUG
        # print("\n#####\n##### DEBUG: _roi_slice_indices, array shape = "
        #       f" {array.shape}, ROI H = {n_roi_h}, ROI V = {n_roi_v}"
        #       "\n#####\n")
        # DEBUG

        fr_col  = max(0, int((n_col - n_roi_h) / 2))
        up_col  = min(n_col, fr_col + n_roi_h)

        fr_row  = max(0, int((n_lin - n_roi_v) / 2))
        up_row  = min(n_lin, fr_row + n_roi_v)

        # DEBUG
        # print("#####\n##### DEBUG: _roi_slice_indices, "
        #       f" col : {fr_col} - {up_col},"
        #       f" row : {fr_row} - {up_row},"
        #       "\n#####\n")
        #     #   f" dim = {dim}"
        # DEBUG

        return (fr_col, up_col, fr_row, up_row)  #, dim

    def _extract_roi_slice(self, array: np.ndarray,
                           fr_col: int, up_col: int,
                           fr_row: int, up_row: int) -> np.ndarray:
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
        if array.shape[0] == 1:
            return array[0:1, fr_col:up_col]
        elif array.shape[1] == 1:
            return array[fr_row:up_row, 0:1]
        else:
            return array[fr_row:up_row, fr_col:up_col]

    def _scale_positions(self, calc_type: str,
                         pos_all_h: np.ndarray, pos_all_v: np.ndarray,
                         pos_roi_h: np.ndarray, pos_roi_v: np.ndarray,
                         pos_nom_h: np.ndarray, pos_nom_v: np.ndarray,
                         pos_nom_h_roi: np.ndarray,
                         pos_nom_v_roi: np.ndarray, nosuppress: bool
                        ) -> dict:
        """Scale positions, pairwise or cross-blade.

        Args:
            calc_type       : 'pairwise' (Δ/Σ) or 'cross' (partial Δ/Σ).
            pos_all_h/v     : Full position array (measured)
            pos_nom_h/v     : Nominal position array (reference)
            pos_nom_h/v_roi : ROI slice of nominal positions
            nosuppress      : If True, label results as raw mode.

        Returns:
            Dict with scaled positions, scales, stats, visualizer.
        """
        # Perform scaling fit
        label = "Δ/Σ" if calc_type == "pairwise" else "Partial Δ/Σ"
        (scalesx, sigmasx, scalesy, sigmasy) = self.scaling_fit(
            pos_roi_h, pos_roi_v, pos_nom_h_roi, pos_nom_v_roi, label
        )
        (qx, kx, deltax), (sqx, skx, sdeltax) = scalesx, sigmasx
        (qy, ky, deltay), (sqy, sky, sdeltay) = scalesy, sigmasy

        # Set raw (R) or transformed (T) graph type.
        transform = "R" if nosuppress else "T"

        # Build title map for visualizer with formatted titles from registry.
        title_map = {
            'total'   : _Title('xbpm_positions', 'total',
                               beamline=self.prm_bml.beamline,
                               rort=transform,
                               calc_type=calc_type),
            'roi'     : _Title('xbpm_positions', 'roi',
                               beamline=self.prm_bml.beamline,
                               rort=transform,
                               calc_type=calc_type),
            'heatmap' : _Title('xbpm_positions', 'heatmap',
                               beamline=self.prm_bml.beamline,
                               rort=transform,
                               calc_type=calc_type),
        }

        # Scale full positions
        pos_all_h_scaled = qx * pos_all_h**2 + kx * pos_all_h + deltax
        pos_all_v_scaled = qy * pos_all_v**2 + ky * pos_all_v + deltay
        pos_roi_h_scaled = qx * pos_roi_h**2 + kx * pos_roi_h + deltax
        pos_roi_v_scaled = qy * pos_roi_v**2 + ky * pos_roi_v + deltay

        # Compute statistics
        diffx2_roi = (pos_roi_h_scaled - pos_nom_h_roi) ** 2
        diffy2_roi = (pos_roi_v_scaled - pos_nom_v_roi) ** 2
        stats      = self.calculate_grid_stats(diffx2_roi, diffy2_roi)
        diffroi    = np.sqrt(diffx2_roi + diffy2_roi)

        # Visualize
        visualizer = PSV(self.prm_gen, titles=title_map)
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
            'stats'        : stats,
            'visualizer'   : visualizer,
        }

    def _compile_results(self, pair_result: dict, cross_result: dict,
                         supmat: np.ndarray, stddevmat: np.ndarray,
                         nosuppress: bool,
                         pos_nom_h: np.ndarray, pos_nom_v: np.ndarray) -> dict:
        """Compile and save final results from pairwise and cross-blade."""
        pair_visualizer  = pair_result['visualizer']
        cross_visualizer = cross_result['visualizer']

        # Save figures if requested
        if self.prm_gen.outputfile:
            outdir = '.'
            sup = "raw" if nosuppress else "scaled"
            bl = self.prm_bml.beamline

            outfile_p = os.path.join(outdir, f"xbpm_pair_pos_{sup}_{bl}.png")
            pair_visualizer.save_figure(outfile_p)

            outfile_c = os.path.join(outdir, f"xbpm_cross_pos_{sup}_{bl}.png")
            cross_visualizer.save_figure(outfile_c)

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
            'pairwise_figure' : pair_visualizer.fig,
            'cross_figure'    : cross_visualizer.fig,
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

    def xbpm_position_calculation(self,
                                  pos_nom_h: np.ndarray,
                                  pos_nom_v: np.ndarray,
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
        supmat, stddevmat = self.suppression_matrix(
            showmatrix=showmatrix, nosuppress=nosuppress
            )

        # Extract nominal ROI slices.
        from_upto     = self._roi_slice_indices(pos_nom_h)
        pos_nom_h_roi = self._extract_roi_slice(pos_nom_h, *from_upto)
        pos_nom_v_roi = self._extract_roi_slice(pos_nom_v, *from_upto)

        # Pairwise calculation (Delta/Sigma).
        pos_pair = self.beam_position_pair(supmat)
        (_, _, pos_h, pos_v) = self.position_dict_parse(pos_pair)

        # Extract ROI slices from measured data.
        pos_roi_pair_h = self._extract_roi_slice(pos_h, *from_upto)
        pos_roi_pair_v = self._extract_roi_slice(pos_v, *from_upto)

        # Process data: fitting, scaling, stats, visualization.
        pairwise_result = self._scale_positions(
                'pairwise', pos_h, pos_v,
                pos_roi_pair_h, pos_roi_pair_v,
                pos_nom_h, pos_nom_v,
                pos_nom_h_roi, pos_nom_v_roi, nosuppress
            )

        # Cross-blade calculation (partial Delta/Sigma).
        pos_cross_h, pos_cross_v = self.beam_position_cross(blades)

        # Extract ROI slices from measured data.
        pos_roi_cross_h = self._extract_roi_slice(pos_cross_h, *from_upto)
        pos_roi_cross_v = self._extract_roi_slice(pos_cross_v, *from_upto)

        # Process data: fitting, scaling, stats, visualization.
        cross_result = self._scale_positions(
            'cross', pos_cross_h, pos_cross_v,
            pos_roi_cross_h, pos_roi_cross_v,
            pos_nom_h, pos_nom_v,
            pos_nom_h_roi, pos_nom_v_roi,
            nosuppress
            )

        # Compile and return results
        return self._compile_results(pairwise_result,
                                     cross_result, supmat,
                                     stddevmat, nosuppress,
                                     pos_nom_h, pos_nom_v)

    def suppression_matrix(self, showmatrix: bool = False,
                           nosuppress: bool = False) -> tuple:
        """Calculate the suppression matrix from blade behavior.

        Args:
            showmatrix: If True, prints the suppression matrix.
            nosuppress: If True, returns the standard 1/-1 matrix.
                        If False, calculates from fitted slopes.

        Returns:
            Tuple of (suppression matrix, standard deviation matrix)
        """
        if nosuppress:
            # Return standard matrix for raw calculations
            return Config.standard_suppression_matrix()

        # Calculate from blade slopes for scaled calculations
        pch, covs_h = self.central_line_fit(self.blades_h,
                                            self.range_h, 'h')
        pcv, covs_v = self.central_line_fit(self.blades_v,
                                            self.range_v, 'v')

        if len(self.range_h) > 1:
            sdevh = np.sqrt(covs_h) * pch[0, 0] / (pch[:, 0]**2)
            pch = pch[0] / np.abs(pch)
        else:
            pch = np.ones(8).reshape(4, 2)
            sdevh = np.zeros(4)

        if len(self.range_v) > 1:
            sdevv = np.sqrt(covs_v) * pcv[0, 0] / (pcv[:, 0]**2)
            pcv = pcv[0] / np.abs(pcv)
        else:
            pcv = np.ones(8).reshape(4, 2)
            sdevv = np.zeros(4)

        supmat = np.array([
            [pcv[0, 0], -pcv[1, 0], -pcv[2, 0],  pcv[3, 0]],
            [pcv[0, 0],  pcv[1, 0],  pcv[2, 0],  pcv[3, 0]],
            [pch[0, 0],  pch[1, 0], -pch[2, 0], -pch[3, 0]],
            [pch[0, 0],  pch[1, 0],  pch[2, 0],  pch[3, 0]],
        ])

        stddevmat = np.array([
            [sdevv[0], sdevv[1], sdevv[2], sdevv[3]],
            [sdevv[0], sdevv[1], sdevv[2], sdevv[3]],
            [sdevh[0], sdevh[1], sdevh[2], sdevh[3]],
            [sdevh[0], sdevh[1], sdevh[2], sdevh[3]],
        ])

        if showmatrix:
            print(f'\nUndulator phase or gap: {self.prm_gen.phaseorgap}')
            print("\nSuppression matrix:")
            for ii, lin in enumerate(supmat):
                for jj, col in enumerate(lin):
                    print(f" {col:12.6f} (±{stddevmat[ii, jj]:10.6f})", end='')
                print()
            print()

        # Exporter(self.prm).write_supmat(supmat)
        return supmat, stddevmat

    def central_line_fit(self, blades: dict, range_vals: np.ndarray,
                         direction: str) -> tuple:
        """Linear fittings to each blade's data through central line.
        
        Args:
            blades: Dictionary of blade measurements along a central line.
            range_vals: Array of sweep positions (angles or distances).
            direction: 'h' for horizontal, 'v' for vertical.

        Returns:
            Tuple of (fit coefficients, std dev values) for each blade.
        """
        if blades is None:
            dr = 'horizontal' if direction == 'h' else 'vertical'
            print(f"\n WARNING: (central_line_fit) {dr} blades' values"
                  " not defined. Seetting fitting values to [1, 0].")
            pc = np.array([[1, 0] for _ in range(4)])
            covs = np.zeros(4)
            return pc, covs

        pc = list()
        covs = list()
        for blade in blades.values():
            weight = 1. / blade[:, 1]

            if np.isinf(weight).any():
                weight = None

            coefs, cov = np.polyfit(range_vals, blade[:, 0], deg=1, w=weight, cov=True)
            pc.append(coefs)
            covs.append(cov[0, 0])
        pc = np.array(pc)

        if np.isinf(pc).any() or (pc == 0).any():
            pc = np.array([[1, 0] for _ in range(4)]) 
        return pc, covs

    def beam_position_pair(self, supmat: np.ndarray) -> dict:
        """Calculate beam position from blades' currents (pairwise)."""
        positions = dict()
        for pos, bld in self.blade_avg.items():
            dsps = supmat @ bld[:, 0]
            positions[pos] = np.array([dsps[0] / dsps[1], dsps[2] / dsps[3]])
        return positions

    def position_dict_parse(self, data: dict) -> tuple:
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
    def beam_position_cross(blades) -> list:
        """Calculate beam position from blades' currents (cross-blade)."""
        to, ti, bi, bo = blades
        v1 = (to - bi) / (to + bi)
        v2 = (ti - bo) / (ti + bo)
        hpos = (v1 - v2)
        vpos = (v1 + v2)
        return [hpos, vpos]

    def scaling_fit(self, pos_h: np.ndarray, pos_v: np.ndarray,
                    nom_h: np.ndarray, nom_v: np.ndarray, calctype=""):
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

        coeffs_x, deltas_x = self._poly_fitting(nom_h, nom_h_cln, pos_h_cln)
        coeffs_y, deltas_y = self._poly_fitting(nom_v, nom_v_cln, pos_v_cln)
        if self.prm_bml.scalepolydeg == 1:
            qx, kx, deltax    = 0., coeffs_x[0], coeffs_x[1]
            sqx, skx, sdeltax = 0., deltas_x[0], deltas_x[1]

            qy, ky, deltay    = 0., coeffs_y[0], coeffs_y[1]
            sqy, sky, sdeltay = 0., deltas_y[0], deltas_y[1]

            qxtxt, qytxt = "", ""
        elif self.prm_bml.scalepolydeg == 2:
            qx, kx, deltax    = coeffs_x
            sqx, skx, sdeltax = deltas_x

            qy, ky, deltay    = coeffs_y
            sqy, sky, sdeltay = deltas_y

            qxtxt = f"qx = {qx:12.4f} ({sqx:4.1f}),\t"
            qytxt = f"qy = {qy:12.4f} ({sqy:4.1f}),\t"

        print(qxtxt, f"kx = {kx:12.4f} ({skx:4.1f}),"
              f"   deltax = {deltax:12.4f} ({sdeltax:4.1f})")
        print(qytxt, f"ky = {ky:12.4f} ({sky:4.1f}),"
              f"   deltay = {deltay:12.4f} ({sdeltay:4.1f})\n")
        return ((qx, kx, deltax), (sqx, skx, sdeltax),
                (qy, ky, deltay), (sqy, sky, sdeltay))

    def _poly_fitting(self, nom_val: np.ndarray,
                      nom_cln: np.ndarray,
                      pos_cln: np.ndarray) -> tuple:
        """Return fitting parameters for scaling fit."""
        if len(set(nom_val.ravel())) > 1 and pos_cln.size >= 2:
            coeffs = None
            covs   = None
            try:
                coeffs, covs = np.polyfit(
                    pos_cln, nom_cln, deg=self.prm_bml.scalepolydeg, cov=True
                )
            except Exception:
                # Keep fitted coefficients even if covariance cannot be
                # estimated (e.g., small sample count), so scaling is still
                # applied (polyfit crashes if covariance cannot be estimated).
                try:
                    coeffs = np.polyfit(
                        pos_cln, nom_cln, deg=self.prm_bml.scalepolydeg
                        )
                    covs = None
                except Exception as err:
                    print(f"\n WARNING: when calculating horizontal scaling"
                          f" coefficients:\n{err}\n"
                          " Setting to default values.")
                    coeffs = np.zeros(self.prm_bml.scalepolydeg + 1)
                    covs   = None

            # Extract standard deviations from covariance matrix if available.
            if covs is not None:
                deltas = np.sqrt(np.diag(covs))
            else:
                deltas = np.zeros(self.prm_bml.scalepolydeg + 1)
        else:
            coeffs = np.zeros(self.prm_bml.scalepolydeg + 1)
            deltas = np.zeros(self.prm_bml.scalepolydeg + 1)
        return (coeffs, deltas)

    @staticmethod
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
                    'h', 'v', 't',
                    'min_h', 'max_h', 'min_v', 'max_v',
                    'mean_h', 'mean_v', 'mean_t',
                    ]}
            return rms_stats

        # Valid points only.
        vld_diff_x2 = diff_x2[valid]
        vld_diff_y2 = diff_y2[valid]

        # Absolute values of differences.
        rms_h = np.sqrt(vld_diff_x2)
        rms_v = np.sqrt(vld_diff_y2)
        rms_t = np.sqrt(vld_diff_x2 + vld_diff_y2)

        # RMS minimum and maximum values.
        rms_max_h = np.max(rms_h)
        rms_min_h = np.min(rms_h)
        rms_max_v = np.max(rms_v)
        rms_min_v = np.min(rms_v)

        # RMS global estimates.
        rms_mean_h = np.sqrt(np.mean(vld_diff_x2))
        rms_mean_v = np.sqrt(np.mean(vld_diff_y2))
        rms_mean_t = np.sqrt(np.mean(vld_diff_x2 + vld_diff_y2))

        nsites_total = int(diff_x2.size)
        if nsites < nsites_total:
            print("\n WARNING: sweeping looks incomplete, no ROI was defined"
              f" ({nsites} valid sites, out of {nsites_total}"
                  " in total). Skipping ROI analysis.")

        rms = {
            'h'      : rms_h,
            'v'      : rms_v,
            't'      : rms_t,
            'min_h'  : rms_min_h,
            'max_h'  : rms_max_h,
            'min_v'  : rms_min_v,
            'max_v'  : rms_max_v,
            'mean_h' : rms_mean_h,
            'mean_v' : rms_mean_v,
            'mean_t' : rms_mean_t,
        }
        return rms

    def data_parse(self) -> tuple:
        """Extract each blade's data from whole data dict into arrays."""
        dk = np.array(list(self.blade_avg.keys()))

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

                if key not in self.blade_avg.keys():
                    break

                try:
                    to[ilin, icol]  = self.blade_avg[key][0, 0]
                    ti[ilin, icol]  = self.blade_avg[key][1, 0]
                    bi[ilin, icol]  = self.blade_avg[key][2, 0]
                    bo[ilin, icol]  = self.blade_avg[key][3, 0]

                    sto[ilin, icol] = self.blade_avg[key][0, 1]
                    sti[ilin, icol] = self.blade_avg[key][1, 1]
                    sbi[ilin, icol] = self.blade_avg[key][2, 1]
                    sbo[ilin, icol] = self.blade_avg[key][3, 1]
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

    def __init__(self,
                 rawdata : BeamlineRawData,
                 prm_bml : BeamlinePrm,
                 ) -> None:
        """Store raw BPM/XBPM dataset and parameters for later processing."""
        self.rawdata = rawdata
        self.sweeps  = rawdata.sweeps_bpm
        self.prm_bml = prm_bml


        self.roisize_v = prm_bml.roi.sl_v
        self.roisize_h = prm_bml.roi.sl_h

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

        self.plot_bpm_positions()

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
            self.nom_x.shape, (self.roisize_v, self.roisize_h)
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
                'h', 'v', 't',
                'min_h', 'max_h', 'min_v', 'max_v',
                'mean_h', 'mean_v', 'mean_t',
                ]}
        return rms_stats

    # Valid points only.
    vld_diff_x2 = diff_x2[valid]
    vld_diff_y2 = diff_y2[valid]

    # Absolute values of differences.
    rms_h = np.sqrt(vld_diff_x2)
    rms_v = np.sqrt(vld_diff_y2)
    rms_t = np.sqrt(vld_diff_x2 + vld_diff_y2)

    # RMS minimum and maximum values.
    rms_max_h = np.max(rms_h)
    rms_min_h = np.min(rms_h)
    rms_max_v = np.max(rms_v)
    rms_min_v = np.min(rms_v)

    # RMS global estimates.
    rms_mean_h = np.sqrt(np.mean(vld_diff_x2))
    rms_mean_v = np.sqrt(np.mean(vld_diff_y2))
    rms_mean_t = np.sqrt(np.mean(vld_diff_x2 + vld_diff_y2))

    nsites_total = int(diff_x2.size)
    if nsites < nsites_total:
        print("\n WARNING: sweeping looks incomplete, no ROI was defined"
            f" ({nsites} valid sites, out of {nsites_total}"
                " in total). Skipping ROI analysis.")

    rms = {
        'h'      : rms_h,
        'v'      : rms_v,
        't'      : rms_t,
        'min_h'  : rms_min_h,
        'max_h'  : rms_max_h,
        'min_v'  : rms_min_v,
        'max_v'  : rms_max_v,
        'mean_h' : rms_mean_h,
        'mean_v' : rms_mean_v,
        'mean_t' : rms_mean_t,
    }
    return rms
