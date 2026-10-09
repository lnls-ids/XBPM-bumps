"""Visualization classes for blade maps and positions."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import logging
# import os

from typing import Optional
# from pyparsing import Optional

from .constants import FIGDPI
from .config import Config
from .data_structure import (
    BPMAnalysis,
    CentralSweeps,
    Positions,
    BeamlinePrm,
    DataAnalysis,
    BladeMap,
    RMSStatistics,
    ROISlice,
    BCA_HV,
    )


_Title = Config.get_plot_title   # shorthand used throughout this module

# Use Computer Modern (standard LaTeX math font) for all math text,
# so symbols like Δ/Σ in titles render in serif academic style.
# Also set the regular text font to serif so titles and labels
# match the math font family throughout.
matplotlib.rcParams['mathtext.fontset']     = 'cm'
matplotlib.rcParams['mathtext.rm']          = 'serif'
matplotlib.rcParams['font.family']          = 'serif'
matplotlib.rcParams['font.serif']           = [
    'cmr10', 'Computer Modern Roman', 'DejaVu Serif'
]
matplotlib.rcParams['axes.formatter.use_mathtext'] = True
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize']       = 14
matplotlib.rcParams['axes.labelpad']        = 2
matplotlib.rcParams['xtick.labelsize']      = 12
matplotlib.rcParams['ytick.labelsize']      = 12
matplotlib.rcParams['legend.fontsize']      = 10
matplotlib.rcParams['figure.titlesize']     = 'xx-small'
matplotlib.rcParams['legend.handletextpad'] = 0.8

# Module logger
logger = logging.getLogger(__name__)


def render_data_analysis(
        pos_nom  : Positions,
        analysis : DataAnalysis,
        prm      : BeamlinePrm,
        ) -> dict[str, Figure]:
    """Return one Figure per tab key for every populated analysis field."""
    figures: dict[str, Figure] = {}

    if analysis.bpm is not None:
        figures["bpm"] = (
            BPMVisualizer(
                analysis.bpm,
                pos_nom
                ).plot_bpm_positions()
            )

    if analysis.blademap is not None:
        figures["blade_map"] = (
            BladeMapVisualizer(
                analysis.blademap,
                ).plot_blade_map()
            )

    if analysis.centralsweeps is not None:
        cs = analysis.centralsweeps
        if cs.h is not None or cs.v is not None:
            figures["position_central_sweeps"] = (
                CentralSweepVisualizer.plot_central_sweep_positions(
                    csweep   = cs,
                    beamline = prm.beamline,
                    )
                )

    if analysis.bladecenter is not None:
        figures["blade_central_sweeps"] = (
            CentralSweepVisualizer.plot_central_sweep_blades(
                bc_analysis = analysis.bladecenter,
                beamline = prm.beamline
                ))

    # Position visualizations. For each tab, select the appropriate sets:
    # ROI slice, nominal positions, calculated positions (raw or transformed),
    # and RMS statistics at ROI.
    pos = analysis.positions
    if pos is not None:
        prw = analysis.positions.pairw
        crs = analysis.positions.cross

        tab_table = {
            "xbpm_pairwise_raw" : (
                prw.roi, pos.nom, prw.pos_std, prw.stat_std.roi
                ),
            "xbpm_pairwise_trn" : (
                prw.roi, pos.nom, prw.pos_trn, prw.stat_trn.roi
                ),
            "xbpm_cross_raw"    : (
                crs.roi, pos.nom, crs.pos_std, crs.stat_std.roi
                ),
            "xbpm_cross_trn"    : (
                crs.roi, pos.nom, crs.pos_trn, crs.stat_trn.roi
                ),
        }

        # Iterate over each tab and its corresponding data tuple,
        # then generate the figure.
        for tab, (roi, pos_nom, pos_calc, stat_roi) in tab_table.items():
            figures[tab] = (
                PositionVisualizer(prm).plot_position_results(
                    roi        = roi,
                    pos_nom    = pos_nom,
                    pos_calc   = pos_calc,
                    stat_roi   = stat_roi,
                    graph_type = tab,
                    )
                )

    return figures


class BPMVisualizer:
    """Unified visualizer for BPM analysis.

    Creates BPM plots from either:
    - Live analysis data (from processors)
    - HDF5 stored data (from readers)

    This eliminates redundancy between processors.show_bpm_at_center()
    and readers._reconstruct_bpm_center().
    """
    def __init__(self,
                 bpm_ana : BPMAnalysis,
                 pos_nom : Positions,
                 ) -> None:
        self.bana     = bpm_ana
        self.beamline = bpm_ana.prm.beamline
        self.rms_diff = bpm_ana.rms_diff

        # Abbreviated names for plotting convenience.
        self.nom_x  = pos_nom.x
        self.nom_y  = pos_nom.y
        self.meas_x = bpm_ana.pos_meas.x
        self.meas_y = bpm_ana.pos_meas.y

    def plot_bpm_positions(self) -> None:
        """Plot BPM positions and differences in a 1x3 subplot figure."""
        # Initialize figure with 1x3 subplots:
        # full grid, roi closeup, differences
        is_1d = (
            self.rms_diff.roi.diff_tot.ndim == 1 or
            (self.rms_diff.roi.diff_tot.ndim == 2 and
             min(self.rms_diff.roi.diff_tot.shape) == 1)
             )
        gridspec = {'width_ratios': [1, 1, 0.1]} if is_1d else None
        self.fig, bpm_axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(18, 6),
            constrained_layout=True,
            gridspec_kw=gridspec
        )
        # The plotting axes for BPM: all points, roi closeup,
        # and colorbar for differences.
        self.ax_all, self.ax_roi, self.ax_diff = bpm_axes

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
            self.ax_all,
            self.meas_x,
            self.nom_x,
            self.meas_y,
            self.nom_y,
            _Title(
                beamline   = self.beamline,
                graph_type = 'bpm',
                ax_type    = 'total',
                )
        )

        # Plot ROI closeup
        slice_h = self.bana.prm.roislice.slice_h
        slice_v = self.bana.prm.roislice.slice_v
        self.nom_roi_x = self.nom_x[slice_v, slice_h]
        self.nom_roi_y = self.nom_y[slice_v, slice_h]
        self._plot_position_scatter(
            self.ax_roi,
            self.nom_roi_x,
            self.meas_x[slice_v, slice_h],
            self.nom_roi_y,
            self.meas_y[slice_v, slice_h],
            _Title(
                beamline   = self.beamline,
                graph_type = 'bpm',
                ax_type    = 'roi',
                )
        )

        # Plot differences heatmap with extent mapping
        self._plot_roi_differences()

        return self.fig

    def _plot_position_scatter(self,
                               ax     : 'matplotlib.axes.Axes',
                               nom_x  : np.array,
                               meas_x : np.array,
                               nom_y  : np.array,
                               meas_y : np.array,
                               title  : str
                               ) -> None:
        """Plot measured vs nominal positions scatter plot.

        Args:
            ax     : Matplotlib axis for plotting.
            nom_x  : Nominal x positions.
            meas_x : Measured x positions.
            nom_y  : Nominal y positions.
            meas_y : Measured y positions.
            title  : Plot title.
        """
        ax.set_title(title, pad=2)
        pos = ax.plot(meas_x, meas_y, 'bo')
        nom = ax.plot(nom_x, nom_y, 'r+') 
        ax.set_xlabel(u"$x$ [$\\mu$m]", fontsize=14)
        ax.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)

        # Compute common limits to ensure equal aspect ratio with margin.
        # Use only finite values to avoid NaN/Inf axis-limit failures.
        all_h = np.concatenate([np.ravel(meas_x), np.ravel(nom_x)])
        all_v = np.concatenate([np.ravel(meas_y), np.ravel(nom_y)])
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

    def _plot_roi_differences(self) -> None:
        """Plot ROI differences as scatter (1D) or heatmap (2D).

        Args:
            axdiff: Matplotlib axis for ROI differences visualization.
            pos_nom_h: Nominal horizontal positions for extent mapping.
            pos_nom_v: Nominal vertical positions for extent mapping.
        """
        roi_diffs = self.bana.rms_diff.roi

        # Treat as 1-D if truly 1-D (shape = (n,)) or effectively 1-D (one
        # dimension is 1, like (1, n) or (n, 1)), or if one nominal axis is
        # constant (single-line sweep).
        h_const = np.nanmax(self.nom_roi_x) == np.nanmin(self.nom_roi_x)
        v_const = np.nanmax(self.nom_roi_y) == np.nanmin(self.nom_roi_y)
        is_1d = (roi_diffs.diff_h.ndim == 1 or
             (roi_diffs.diff_h.ndim == 2 and
              min(roi_diffs.diff_h.shape) == 1) or
             h_const or v_const
             )

        if is_1d:
            # 1D imshow: render as a thin band of square cells
            h_min = np.nanmin(self.nom_roi_x)
            h_max = np.nanmax(self.nom_roi_x)

            color_vals = np.ravel(roi_diffs.diff_tot).reshape(-1, 1)
            extent = [0, 1, self.nom_roi_y.min(), self.nom_roi_y.max()]
            aspect = 'auto'

            # Make the single column visually wider
            self.ax_diff.set_box_aspect(10)
            self.ax_diff.set_anchor('C')
            self.ax_diff.set_xticks([])

            im = self.ax_diff.imshow(color_vals,
                               cmap='viridis',
                               extent=extent,
                               aspect=aspect,
                               origin='lower')
            cbar = self.fig.colorbar(im, ax=self.ax_diff,
                                      fraction=0.4, pad=0.3)
            cbar.set_label(u"RMS Difference [$\\mu$m]", fontsize=14)

            self.ax_diff.set_xlabel("")
            self.ax_diff.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)
            self.ax_diff.set_title(
                _Title(
                    beamline   = self.beamline,
                    graph_type = 'bpm',
                    ax_type    = 'heatmap',
                    ), pad=2
                )
            self.ax_diff.grid(False)
        else:
            # 2D heatmap with extent mapping
            h_min = np.nanmin(self.nom_roi_x)
            h_max = np.nanmax(self.nom_roi_x)
            v_min = np.nanmin(self.nom_roi_y)
            v_max = np.nanmax(self.nom_roi_y)
            extent = [h_min, h_max, v_min, v_max]

            # Calculate aspect ratio to maintain proper physical proportions.
            # Account for both physical extents and array shape to avoid
            # distortion when physical x and y ranges differ significantly.
            n_v, n_h = roi_diffs.diff_h.shape
            h_extent = h_max - h_min
            v_extent = v_max - v_min
            # aspect = (physical_y_per_pixel) / (physical_x_per_pixel)
            aspect = ((v_extent / n_v) / (h_extent / n_h)
                      if (h_extent > 0 and v_extent > 0) else 1)

            # Use imshow for filled heatmap visualization
            im = self.ax_diff.imshow(
                roi_diffs.diff_tot, cmap='viridis', extent=extent,
                aspect=aspect, origin='lower'
            )
            cbar = self.fig.colorbar(im, ax=self.ax_diff,
                                     fraction=0.046, pad=0.04)
            cbar.set_label(u"RMS Difference [$\\mu$m]", fontsize=14)

            self.ax_diff.set_xlabel(u"$x$ [$\\mu$m]", fontsize=14)
            self.ax_diff.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)
            self.ax_diff.set_title(
                _Title(
                    beamline   = self.beamline,
                    graph_type = 'bpm',
                    ax_type    = 'heatmap',
                    ), pad=2
                )
            self.ax_diff.grid(False)


class BladeMapVisualizer:
    """Visualizes XBPM blade intensity maps.

    This class creates color maps showing the intensity (current) measured
    by each blade across the measurement grid.

    Attributes:
        data (dict): Measurement data dictionary.
        prm (BeamlinePrm): Parameters dataclass.
    """

    def __init__(self,
                 bmap : BladeMap,
                 ) -> None:
        """Initialize visualizer with data and parameters.

        Args:
            bmap: BladeMap instance.
        """
        self.bm_ana = bmap

    def plot_blade_map(self) -> "matplotlib.figure.Figure":
        """Display blade intensity maps for all four blades.

        Creates a 2x2 subplot figure showing heatmaps for:
        - Top-Inner (TI), Top-Outer (TO)
        - Bottom-Inner (BI), Bottom-Outer (BO)

        Arranges blades in quadrants:
        [TI  TO]
        [BI  BO]

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Import here to avoid circular dependency
        to, ti, bi, bo = (
            self.bm_ana.blades.to,
            self.bm_ana.blades.ti,
            self.bm_ana.blades.bi,
            self.bm_ana.blades.bo
        )

        fig, rx = plt.subplots(2, 2, figsize=(10, 6))

        # Calculate extent for proper axis labels
        if (to.ndim < 2 or to.shape[0] <= 1 or to.shape[1] <= 1):
            extent = None
        else:
            x, y = self.bm_ana.pos.x, self.bm_ana.pos.y
            minvalx, maxvalx = np.min(x), np.max(x)
            minvaly, maxvaly = np.min(y), np.max(y)
            extent = (minvalx, maxvalx, minvaly, maxvaly)

        quad = [
            [ti, to],
            [bi, bo]
            ]
        names = [
            ["TI", "TO"],
            ["BI", "BO"]
            ]

        for idy in range(2):
            for idx in range(2):
                rx[idy][idx].imshow(
                    quad[idy][idx],
                    extent=extent,
                    origin='lower'
                    )
                if extent is None:
                    rx[idy][idx].set_xlabel('')
                    rx[idy][idx].set_xticks([])
                else:
                    rx[idy][idx].set_xlabel(u"$x$ [$\\mu$rad]", fontsize=14)
                rx[idy][idx].set_ylabel(u"$y$ [$\\mu$rad]", fontsize=14)
                rx[idy][idx].set_title(names[idy][idx])

        fig.tight_layout(pad=0., w_pad=-17., h_pad=2.)

        return fig


class CentralSweepVisualizer:
    """Unified visualizer for central sweep analysis.
./tests/chat-deepseek-2026-09-29.json
    Creates sweep plots from either:
    - Live analysis data (from processors)
    - HDF5 stored data (from readers)

    This eliminates redundancy between processors._central_sweeps_show()
    and readers._reconstruct_sweeps().
    """
    @staticmethod
    def plot_central_sweep_positions(
        csweep   : CentralSweeps,
        beamline : str = "",
        ) -> "matplotlib.figure.Figure":
        """Create sweep figure from numpy arrays (position reconstruction path).

        This is used when sweeps data is stored as pre-calculated positions
        (HDF5 or other sources). Formatting matches canonical plot_central_sweeps.

        Args:
            csweep   : CentralSweeps object containing horizontal and
                        vertical sweeps
            figsize  : Figure size tuple

        Returns:
            matplotlib.figure.Figure
        """
        # Scale for the beamline (source-XBPM distance).
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(12, 5)
            )

        # Discount offsets, normalize index range for graph and plot it.
        ch  = csweep.h
        if csweep.h is not None:
            # offset_h_y = np.mean(ch.pos_fit)
            offset_h_x = np.mean(ch.pos_index)
            a, b = ch.coeffs
            offset_h_y = a * offset_h_x + b
            hx  = ch.pos_index - offset_h_x
            hy  = ch.pos_calc  - offset_h_y
            hfy = ch.pos_fit   - offset_h_y
            #
            norm_x = np.max(np.abs(hx))
            hx  /= norm_x
            hy  /= norm_x
            hfy /= norm_x

            ax.plot(hx,  hy, 'o-', label="H calc", zorder=2)
            ax.plot(hx, hfy, '^-', label="H fit",  zorder=3)

        cv  = csweep.v
        if csweep.v is not None:
            # offset_v_y = np.mean(ch.pos_fit)
            offset_v_x = np.mean(cv.pos_index)
            a, b = cv.coeffs
            offset_v_y = a * offset_v_x + b
            vx  = cv.pos_index - offset_v_x
            vy  = cv.pos_calc  - offset_v_y
            vfy = cv.pos_fit   - offset_v_y
            #
            norm_y = np.max(np.abs(vx))
            vx  /= norm_y
            vy  /= norm_y
            vfy /= norm_y

            ax.plot(vy,  vx, 'o-', label="V calc", zorder=2)
            ax.plot(vfy, vx, '^-', label="V fit",  zorder=3)

        extent = np.array((-1, 1, -1, 1)) * 1.05
        ax.set_xlabel("$x$ [a.u.]")
        ax.set_ylabel("$y$ [a.u.]")
        ax.set_title(_Title(
            beamline   = beamline,
            graph_type = 'sweeps',
            ax_type    = '')
            )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        return fig

    """Unified visualizer for blade current analysis.

    Creates blade current plots from either:
    - Live analysis data (from processors)
    - HDF5 stored data (from readers)

    This eliminates redundancy between processors.show_blades_at_center()
    and readers._reconstruct_blades_center().
    """

    @staticmethod
    def plot_central_sweep_blades(
        bc_analysis: BCA_HV,
        beamline: str = ""
        ) -> Optional[Figure]:
        """Generate blade currents at center plots (canonical version).

        This is the tuned plotting implementation used by both live analysis
        and HDF5 reconstruction. It encapsulates the exact visualization
        semantics for the "Blades at sweeps" tab.

        Args:
            bc_analysis : BCA_HV instance containing BladeCenterAnalysis
                            data for horizontal and vertical sweeps.
            beamline    : Beamline name for ylabel determination.

        Returns:
            matplotlib.figure.Figure or None if no blade data.
        """
        fig, (axh, axv) = plt.subplots(1, 2, figsize=(10, 5))

        ch = bc_analysis.h
        cv = bc_analysis.v

        # If horizontal sweeps are available.
        if ch is not None:
            hblades = {
                "TO" : ch.to,
                "TI" : ch.ti,
                "BI" : ch.bi,
                "BO" : ch.bo, 
            }
            for key, bld in hblades.items():
                k = f"{key.upper()}"
                axh.errorbar(
                    bld.pos,
                    bld.bld_raw,
                    bld.bld_err,
                    fmt='o-',
                    label=k,
                    zorder=1
                    )
                axh.plot(
                    bld.pos,
                    bld.bld_fit,
                    "^-",
                    label=f"{k} fit",
                    zorder=2
                    )

        # If vertical sweeps are available.
        if cv is not None:
            vblades = {
                "TO" : cv.to,
                "TI" : cv.ti,
                "BI" : cv.bi,
                "BO" : cv.bo
            }
            for key, bld in vblades.items():
                k = f"{key.upper()}"
                axv.errorbar(
                    bld.pos,
                    bld.bld_raw,
                    bld.bld_err,
                    fmt='o-',
                    label=k,
                    zorder=1
                    )
                axv.plot(
                    bld.pos,
                    bld.bld_fit,
                    "^-",
                    label=f"{k} fit",
                    zorder=2
                    )

        axh.set_title(_Title(
            beamline   = beamline,
            graph_type = 'blades_at_sweeps',
            ax_type    = 'h')
        )
        axv.set_title(_Title(
            beamline   = beamline,
            graph_type = 'blades_at_sweeps',
            ax_type    = 'v'
            )
        )
        axh.legend()
        axv.legend()
        axh.grid()
        axv.grid()
        axh.set_xlabel("$x$ [$\\mu$rad]")
        axv.set_xlabel("$y$ [$\\mu$rad]")

        ylabel = ("$I$ [# counts]" if beamline[:3]
                in ["MGN", "MNC"] else "$I$ [A]")
        axh.set_ylabel(ylabel)
        axv.set_ylabel(ylabel)
        fig.tight_layout()

        return fig

class PositionVisualizer:
    """Visualizes calculated XBPM beam positions.

    This class handles visualization of position calculation results:
    - Nominal vs calculated positions on full grid
    - Closeup view of Region of Interest (ROI)
    - RMS position differences heatmap

    Can display results from either pairwise or cross-blade calculations,
    with or without suppression matrix corrections.

    Attributes:
        prm (BeamlinePrm): Parameters dataclass.
        title (str): Title for the visualization.
        fig (matplotlib.figure.Figure): Matplotlib figure object for
             current visualization.
    """

    def __init__(self,
                 prm: BeamlinePrm,
                 title: str = "",
                 titles: dict = None
                 ) -> None:
        """Initialize visualizer with parameters.

        Args:
            prm    : Parameters dataclass instance.
            title  : Legacy title prefix for plots.
            titles : Optional dictionary with explicit titles for keys
                'total', 'roi', and 'heatmap'.
        """
        self.prm    = prm
        self.title  = title
        self.titles = titles or {}
        self.fig    = None

        # Module logger
        self._logger = logging.getLogger(__name__)

    def plot_position_results(self,
                              roi        : ROISlice,
                              pos_nom    : Positions,
                              pos_calc   : Positions,
                              stat_roi   : RMSStatistics,
                              graph_type : str = "",
                              ) -> None:
        """Display full position results in 1x3 subplot layout.

        Args:
            roi      : Region of Interest slice.
            pos_nom  : Nominal positions.
            pos_calc : Calculated positions.
            stat_roi : RMS statistics for the ROI.
            graph_type: Type of graph to display (e.g., 'xbpm_pairwise_raw').
        """
        # Check dimensionality of the RMS statistics to determine layout.
        if stat_roi.diff_tot is None:
            is_1d = True
        else:
            is_1d = (stat_roi.diff_tot.ndim == 1 or
                     (stat_roi.diff_tot.ndim == 2 and
                      min(stat_roi.diff_tot.shape) == 1))
        if is_1d:
            gridspec = {'width_ratios': [1, 1, 0.1]}
        else:
            gridspec = None

        self.fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(18,6),
            constrained_layout=True,
            gridspec_kw=gridspec
        )
        (ax_all, ax_roi, ax_heat) = axes

        # Reduce vertical/horizontal padding between subplots
        # and figure edges for a tighter layout.
        try:
            engine = self.fig.get_layout_engine()
            if engine is not None and hasattr(engine, "set"):
                engine.set(
                    w_pad=0.06,   # space figure edge / axes, horizontal
                    h_pad=0.0,    # space figure edge / axes, vertical
                    wspace=0.06,  # space between axes, horizontal
                    hspace=0.0,   # space between axes, vertical
                )
            else:
                raise AttributeError("Layout engine missing or immutable")
        except Exception:
            # Log and continue if environment lacks this API
            self._logger.warning(
                "Layout engine padding not applied; falling back",
                exc_info=True,
            )

        # ROI slices.
        roi_h, roi_v  = roi.slice_h, roi.slice_v

        # Graph characteristics.
        _, calc_type, rort = graph_type.split('_')

        # Full grid view
        title_total    = _Title(
            beamline   = self.prm.beamline,
            graph_type = "xbpm_positions",
            calc_type  = calc_type,
            rort       = rort,
            ax_type    = "total",
            )
        self._plot_scaled_positions(
            ax_all,
            pos_nom.x,
            pos_nom.y,
            pos_calc.x,
            pos_calc.y,
            title = title_total
        )

        # ROI closeup
        title_roi      = _Title(
            beamline   = self.prm.beamline,
            graph_type = "xbpm_positions",
            calc_type  = calc_type,
            rort       = rort,
            ax_type    = "roi",
        )
        self._plot_scaled_positions(
            ax_roi,
            pos_nom.x[roi_v, roi_h],
            pos_nom.y[roi_v, roi_h],
            pos_calc.x[roi_v, roi_h],
            pos_calc.y[roi_v, roi_h],
            title = title_roi
        )

        # Difference heatmap
        title_heatmap  = _Title(
            beamline   = self.prm.beamline,
            graph_type = "xbpm_positions",
            calc_type  = calc_type,
            rort       = rort,
            ax_type    = "heatmap",
        )
        self._plot_position_differences(
            ax_heat,
            pos_nom.x[roi_v, roi_h],
            pos_nom.y[roi_v, roi_h],
            stat_roi.diff_tot,
            title = title_heatmap
        )

        return self.fig

    def save_figure(self, filename: str) -> None:
        """Save the figure to a file.

        Args:
            filename: Path to save the figure to.
        """
        if self.fig is not None:
            self.fig.savefig(filename, dpi=FIGDPI, bbox_inches='tight')
            logger.info("Figure saved to %s", filename)

    def _plot_scaled_positions(self,
                               ax: plt.Axes,
                               pos_nom_h: np.ndarray,
                               pos_nom_v: np.ndarray,
                               pos_h: np.ndarray,
                               pos_v: np.ndarray,
                               title: str
                               ) -> None:
        """Plot nominal vs calculated positions on given axis."""
        ax.set_title(title, pad=2)
        pos = ax.plot(pos_h, pos_v, 'bo')
        nom = ax.plot(pos_nom_h, pos_nom_v, 'r+')
        ax.set_xlabel(u"$x$ [$\\mu$m]", fontsize=14)
        ax.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)

        # Compute common limits to ensure equal aspect ratio with margin.
        # Filter non-finite values to avoid NaN/Inf axis-limit failures.
        all_h = np.concatenate([np.ravel(pos_h), np.ravel(pos_nom_h)])
        all_v = np.concatenate([np.ravel(pos_v), np.ravel(pos_nom_v)])
        all_h = all_h[np.isfinite(all_h)]
        all_v = all_v[np.isfinite(all_v)]

        if all_h.size == 0 or all_v.size == 0:
            logger.warning("No finite position data available for '%s'; "
                           "using default axis limits.", title)
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

    def _plot_position_differences(self,
                                   ax: plt.Axes,
                                   pos_nom_h : np.ndarray,
                                   pos_nom_v : np.ndarray,
                                   rmsroi    : np.ndarray,
                                   title     : str = ""
                                   ) -> None:
        """Plot position difference heatmap or scatter on given axis."""
        if rmsroi is None:
            ax.set_title(title, pad=2)
            ax.set_xlabel("")
            ax.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)
            ax.text(0.5, 0.5, "ROI unavailable",
                    ha='center', va='center', transform=ax.transAxes)
            ax.grid(False)
            return

        # Treat as 1-D if truly 1-D (shape = (n,)) or effectively 1-D (one
        # dimension is 1, like (1, n) or (n, 1)), or if one nominal axis is
        # constant (single-line sweep).

        h_const = np.nanmax(pos_nom_h) == np.nanmin(pos_nom_h)
        v_const = np.nanmax(pos_nom_v) == np.nanmin(pos_nom_v)
        is_1d = (rmsroi.ndim == 1 or
                 (rmsroi.ndim == 2 and min(rmsroi.shape) == 1) or
                 h_const or v_const)

        if is_1d:
            # 1D imshow: render as a thin band of square cells
            h_min = np.nanmin(pos_nom_h)
            h_max = np.nanmax(pos_nom_h)
            # h_center = (h_min + h_max) / 2

            color_vals = np.ravel(rmsroi).reshape(-1, 1)
            # extent = [h_center - 0.2, h_center + 0.2,
            extent = [0, 1, np.nanmin(pos_nom_v), np.nanmax(pos_nom_v)]
            aspect = 'auto'
            xlabel = ""

            # Make the single column visually wider
            ax.set_box_aspect(10)
            ax.set_anchor('C')
            ax.set_xticks([])
            fraction, pad = 0.4, 0.3

            im = ax.imshow(color_vals,
                           cmap='viridis', origin='lower',
                            aspect=aspect, extent=extent)
            cbar = self.fig.colorbar(im, ax=ax,
                                      fraction=fraction, pad=pad)
            cbar.set_label(u"RMS Difference [$\\mu$m]", fontsize=14)

        else:
            # 2D heatmap: use extent to map array indices to actual coordinates
            h_min = np.nanmin(pos_nom_h)
            h_max = np.nanmax(pos_nom_h)
            v_min = np.nanmin(pos_nom_v)
            v_max = np.nanmax(pos_nom_v)
            # extent = [left, right, bottom, top]
            extent = [h_min, h_max, v_min, v_max]

            # Calculate aspect ratio to maintain proper physical proportions.
            # Account for both physical extents and array shape to avoid
            # distortion when physical x and y ranges differ significantly.
            n_v, n_h = rmsroi.shape
            h_extent = h_max - h_min
            v_extent = v_max - v_min
            # aspect = (physical_y_per_pixel) / (physical_x_per_pixel)
            aspect = ((v_extent / n_v) / (h_extent / n_h)
                      if (h_extent > 0 and v_extent > 0) else 1)
            color_vals = rmsroi
            xlabel = u"$x$ [$\\mu$m]"
            fraction, pad = 0.04, 0.3

            im = ax.imshow(color_vals,
                           cmap='viridis', origin='lower',
                           aspect=aspect, extent=extent)
            cbar = self.fig.colorbar(im, ax=ax,
                                     fraction=0.046, pad=0.04)
            cbar.set_label(u"RMS Difference [$\\mu$m]", fontsize=14)

        # cbar = self.fig.colorbar(im, ax=ax,
        #                          fraction=0.4, pad=0.3)
        # cbar.set_label(u"RMS Difference [$\\mu$m]")
        ax.set_title(title, pad=2)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)
