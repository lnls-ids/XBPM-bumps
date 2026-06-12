"""Visualization classes for blade maps and positions."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging

from .parameters import Prm
from .constants import FIGDPI
from .config import Config

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


class BladeMapVisualizer:
    """Visualizes XBPM blade intensity maps.

    This class creates color maps showing the intensity (current) measured
    by each blade across the measurement grid.

    Attributes:
        data (dict): Measurement data dictionary.
        prm (Prm): Parameters dataclass.
    """

    def __init__(self, data: dict, prm: Prm):
        """Initialize visualizer with data and parameters.

        Args:
            data: Measurement data dictionary.
            prm: Parameters dataclass instance.
        """
        self.data = data
        self.prm = prm

    def show(self):
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
        from .processors import XBPMProcessor

        # Create temporary processor to parse blade data
        processor = XBPMProcessor(self.data, self.prm)
        blades, stddevs = processor.data_parse()
        to, ti, bi, bo = blades

        fig, rx = plt.subplots(2, 2, figsize=(8, 5))

        # Calculate extent for proper axis labels
        if (to.ndim < 2 or to.shape[0] <= 1 or to.shape[1] <= 1):
            extent = None
        else:
            alist = np.array(list(self.data.keys()))
            try:
                klist = np.unique(alist[:, 0])
                mlist = np.unique(alist[:, 1])
            except:  # noqa: E722
                # Some data are 1-D only
                mlist = np.unique(alist)
                klist = np.zeros(len(mlist))
            if klist.size == 0 or mlist.size == 0:
                extent = None
            else:
                minvalx, maxvalx = np.min(klist), np.max(klist)
                minvaly, maxvaly = np.min(mlist), np.max(mlist)
                extent = (minvalx, maxvalx, minvaly, maxvaly)

        quad = [[ti, to], [bi, bo]]
        names = [["TI", "TO"], ["BI", "BO"]]

        for idy in range(2):
            for idx in range(2):
                rx[idy][idx].imshow(quad[idy][idx], extent=extent)
                if extent is None:
                    rx[idy][idx].set_xlabel('')
                    rx[idy][idx].set_xticks([])
                else:
                    rx[idy][idx].set_xlabel(u"$x$ [$\\mu$rad]", fontsize=14)
                rx[idy][idx].set_ylabel(u"$y$ [$\\mu$rad]", fontsize=14)
                rx[idy][idx].set_title(names[idy][idx])

        fig.tight_layout(pad=0., w_pad=-17., h_pad=2.)

        if self.prm.outputfile:
            outfile = f"blade_map_{self.prm.beamline}.png"
            fig.savefig(outfile, dpi=FIGDPI)
            logger.info("Figure of blades' map saved to file %s", outfile)

        return fig

    @staticmethod
    def plot_from_hdf5(blade_grp):
        """Create blade map figure from HDF5 dataset.

        Args:
            blade_grp: HDF5 group containing blade_map data

        Returns:
            matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        # Load blade data and coordinates
        to = blade_grp['to'][:]
        ti = blade_grp['ti'][:]
        bi = blade_grp['bi'][:]
        bo = blade_grp['bo'][:]
        x_coords = blade_grp['x_coords'][:]
        y_coords = blade_grp['y_coords'][:]
        extent = (x_coords[0], x_coords[-1], y_coords[0], y_coords[-1])

        # Create 2x2 figure: [[TI, TO], [BI, BO]]
        fig, axes = plt.subplots(2, 2, figsize=(8, 5))
        quad = [[ti, to], [bi, bo]]
        names = [["TI", "TO"], ["BI", "BO"]]

        for row in range(2):
            for col in range(2):
                ax = axes[row][col]
                # Keep orientation consistent with live plotting path.
                ax.imshow(quad[row][col], extent=extent)
                ax.set_xlabel(u"$x$ [$\\mu$rad]", fontsize=14)
                ax.set_ylabel(u"$y$ [$\\mu$rad]", fontsize=14)
                ax.set_title(names[row][col])

        fig.tight_layout(pad=0., w_pad=-17., h_pad=2.)
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
        prm (Prm): Parameters dataclass.
        title (str): Title for the visualization.
        fig (matplotlib.figure.Figure): Matplotlib figure object for
             current visualization.
    """

    def __init__(self, prm: Prm, title: str = "", titles: dict = None):
        """Initialize visualizer with parameters.

        Args:
            prm: Parameters dataclass instance.
            title: Legacy title prefix for plots.
            titles: Optional dictionary with explicit titles for keys
                'total', 'roi', and 'heatmap'.
        """
        self.prm   = prm
        self.title = title
        self.titles = titles or {}
        self.fig   = None

        # Module logger
        self._logger = logging.getLogger(__name__)

    def show_position_results(self, pos_nom_h, pos_nom_v,
                              pos_h, pos_v, pos_roi_h, pos_roi_v,
                              pos_nom_h_roi, pos_nom_v_roi,
                              diff_roi, figsize=(18, 6)) -> None:
        """Display full position results in 1x3 subplot layout.

        Args:
            pos_nom_h: Nominal horizontal positions (full grid).
            pos_nom_v: Nominal vertical positions (full grid).
            pos_h: Calculated horizontal positions (full grid).
            pos_v: Calculated vertical positions (full grid).
            pos_roi_h: Calculated horizontal positions in ROI.
            pos_roi_v: Calculated vertical positions in ROI.
            pos_nom_h_roi: Nominal horizontal positions inside ROI.
            pos_nom_v_roi: Nominal vertical positions inside ROI.
            diff_roi: RMS position differences in ROI.
            figsize: Figure size as (width, height) tuple.
        """
        if diff_roi is None:
            is_1d = True
        else:
            is_1d = (diff_roi.ndim == 1 or
                     (diff_roi.ndim == 2 and min(diff_roi.shape) == 1))
        if is_1d:
            gridspec = {'width_ratios': [1, 1, 0.1]}
        else:
            gridspec = None

        self.fig, (ax_all, ax_close, ax_color) = plt.subplots(
            1, 3, figsize=figsize, constrained_layout=True,
            gridspec_kw=gridspec
        )

        # Reduce vertical/horizontal padding between subplots
        # and figure edges for a tighter layout.
        try:
            engine = self.fig.get_layout_engine()
            if engine is not None and hasattr(engine, "set"):
                engine.set(
                    w_pad=0.02,   # space figure edge / axes, horizontal
                    h_pad=0.0,    # space figure edge / axes, vertical
                    wspace=0.02,  # space between axes, horizontal
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

        if self.titles:
            title_total   = self.titles.get('total', self.title)
            title_roi     = self.titles.get('roi', self.title)
            title_heatmap = self.titles.get('heatmap', self.title)
        else:
            title_total   = f"XBPM @ {self.prm.beamline} : {self.title}"
            title_roi     = f"XBPM @ {self.prm.beamline} : {self.title} (ROI)"
            title_heatmap = f"XBPM @ {self.prm.beamline} : {self.title}"

        # Full grid view
        self._plot_scaled_positions(
            ax_all, pos_nom_h, pos_nom_v, pos_h, pos_v,
            title_total
        )

        # ROI closeup
        self._plot_scaled_positions(
            ax_close, pos_nom_h_roi, pos_nom_v_roi,
            pos_roi_h, pos_roi_v,
            title_roi
        )

        # Difference heatmap
        self._plot_position_differences(
            ax_color, diff_roi, pos_nom_h_roi, pos_nom_v_roi,
            title_heatmap
        )

        # constrained_layout handles spacing; avoid mixing with tight_layout

    def _plot_scaled_positions(self, ax, pos_nom_h, pos_nom_v,
                              pos_h, pos_v, title):
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

    def _plot_position_differences(self, ax, diffroi,
                                   pos_nom_h, pos_nom_v, title=""):
        """Plot position difference heatmap or scatter on given axis."""
        if diffroi is None:
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
        is_1d = (diffroi.ndim == 1 or
                 (diffroi.ndim == 2 and min(diffroi.shape) == 1) or
                 h_const or v_const)

        if is_1d:
            # 1D imshow: render as a thin band of square cells
            h_min = np.nanmin(pos_nom_h)
            h_max = np.nanmax(pos_nom_h)
            # h_center = (h_min + h_max) / 2

            color_vals = np.ravel(diffroi).reshape(-1, 1)
            # extent = [h_center - 0.2, h_center + 0.2,
            extent = [0, 1, np.nanmin(pos_nom_v), np.nanmax(pos_nom_v)]
            aspect = 'auto'
            xlabel = ""

            # Make the single column visually wider
            ax.set_box_aspect(10)
            ax.set_anchor('C')
            ax.set_xticks([])

            im = ax.imshow(color_vals,
                            cmap='viridis',
                            extent=extent,
                            aspect=aspect,
                            origin='lower')
            cbar = self.fig.colorbar(im, ax=ax,
                                      fraction=0.4, pad=0.3)
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
            n_v, n_h = diffroi.shape
            h_extent = h_max - h_min
            v_extent = v_max - v_min
            # aspect = (physical_y_per_pixel) / (physical_x_per_pixel)
            aspect = ((v_extent / n_v) / (h_extent / n_h)
                      if (h_extent > 0 and v_extent > 0) else 1)
            color_vals = diffroi
            xlabel = u"$x$ [$\\mu$m]"

            im = ax.imshow(color_vals,
                        cmap='viridis',
                        origin='lower',
                        aspect=aspect,
                        extent=extent)
            cbar = self.fig.colorbar(im, ax=ax,
                                     fraction=0.046, pad=0.04)
            cbar.set_label(u"RMS Difference [$\\mu$m]", fontsize=14)

        # cbar = self.fig.colorbar(im, ax=ax,
        #                          fraction=0.4, pad=0.3)
        # cbar.set_label(u"RMS Difference [$\\mu$m]")
        ax.set_title(title, pad=2)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)

    def save_figure(self, filename: str) -> None:
        """Save the figure to a file.

        Args:
            filename: Path to save the figure to.
        """
        if self.fig is not None:
            self.fig.savefig(filename, dpi=FIGDPI, bbox_inches='tight')
            logger.info("Figure saved to %s", filename)

    @staticmethod
    def plot_from_hdf5(pos_data, beamline: str = "",
                       rort: str = "", calc: str = "",
                       roi_bounds: dict = None,
                       roi_size: tuple = None):
        """Create position comparison figure from HDF5 dataset.

        Args:
            pos_data: HDF5 dataset with structured array containing
                position data.
            beamline: Optional beamline code (e.g. MNC1) for title templates.
            rort: Optional transform mode string (e.g. 'raw', 'Transf.').
            calc: Optional calculation type ('pairwise' or 'cross').

        Returns:
            matplotlib.figure.Figure with 3 subplots
        """
        import matplotlib.pyplot as plt

        if not hasattr(pos_data.dtype, 'names') or not pos_data.dtype.names:
            raise ValueError("pos_data is not a structured array")

        # Extract nominal and measured positions
        nom_x = pos_data['x_nom'][:]
        nom_y = pos_data['y_nom'][:]

        # Determine measured field names based on position type
        if 'x_raw' in pos_data.dtype.names:
            meas_x = pos_data['x_raw'][:]
            meas_y = pos_data['y_raw'][:]
        elif 'x_scaled' in pos_data.dtype.names:
            meas_x = pos_data['x_scaled'][:]
            meas_y = pos_data['y_scaled'][:]
        elif 'x' in pos_data.dtype.names:
            meas_x = pos_data['x'][:]
            meas_y = pos_data['y'][:]
        else:
            raise ValueError("Cannot find measured position fields in dataset")

        # Calculate differences
        diff_x = meas_x - nom_x
        diff_y = meas_y - nom_y
        diff_rms = np.sqrt(diff_x**2 + diff_y**2)

        # Create 3-subplot figure with wider proportions
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(20, 5), constrained_layout=True
        )

        beamline_ctx = beamline or pos_data.attrs.get('beamline', '')
        t_total = _Title('xbpm_positions', 'total',   beamline_ctx, rort, calc)
        t_roi   = _Title('xbpm_positions', 'roi',     beamline_ctx, rort, calc)
        t_heat  = _Title('xbpm_positions', 'heatmap', beamline_ctx, rort, calc)

        # Plot 1: All positions on full grid
        ax1.plot(meas_x, meas_y, 'bo', label='Calc.')
        ax1.plot(nom_x, nom_y, 'r+', label='Nom.', markersize=8)
        ax1.set_xlabel(u"$x$ [$\\mu$m]", fontsize=14)
        ax1.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)
        ax1.set_title(t_total or _Title('xbpm_positions', 'total'))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        PositionVisualizer._apply_equal_limits(
            ax1, meas_x, meas_y, nom_x, nom_y
        )

        # Reconstruct ROI and heatmap using index-based square-grid recovery
        # when possible. This matches live plotting behavior better than
        # coordinate masking for XBPM exports.
        roi_meas_x = np.array([])
        roi_meas_y = np.array([])
        roi_nom_x = np.array([])
        roi_nom_y = np.array([])
        roi_diff_2d = None
        roi_extent = None

        n_pts = int(len(nom_x))
        n_side = int(round(np.sqrt(n_pts)))
        use_square_grid = (n_pts > 0 and n_side * n_side == n_pts)

        if use_square_grid and (calc or rort):
            nom_x_2d = np.asarray(nom_x, dtype=float).reshape(n_side, n_side)
            nom_y_2d = np.asarray(nom_y, dtype=float).reshape(n_side, n_side)
            meas_x_2d = np.asarray(meas_x, dtype=float).reshape(n_side, n_side)
            meas_y_2d = np.asarray(meas_y, dtype=float).reshape(n_side, n_side)
            diff_2d_full = np.asarray(diff_rms, dtype=float).reshape(n_side,
                                                                     n_side)

            # Priority 1: use explicitly stored roi_size from HDF5 attrs.
            if roi_size is not None:
                roi_h, roi_v = roi_size
                roi_side = min(int(roi_h), int(roi_v), n_side)
            else:
                # Priority 2: infer from coordinate bounds.
                roi_side = PositionVisualizer._infer_roi_side_from_bounds(
                    nom_x_2d, nom_y_2d, roi_bounds
                )
            # Priority 3: fall back to half the grid.
            if roi_side is None or roi_side <= 0:
                roi_side = max(1, n_side // 2)

            sl = PositionVisualizer._center_slice(n_side, roi_side)

            roi_nom_x_2d = nom_x_2d[sl, sl]
            roi_nom_y_2d = nom_y_2d[sl, sl]
            roi_meas_x_2d = meas_x_2d[sl, sl]
            roi_meas_y_2d = meas_y_2d[sl, sl]
            roi_diff_2d = diff_2d_full[sl, sl]

            roi_nom_x = roi_nom_x_2d.ravel()
            roi_nom_y = roi_nom_y_2d.ravel()
            roi_meas_x = roi_meas_x_2d.ravel()
            roi_meas_y = roi_meas_y_2d.ravel()

            roi_extent = [
                float(np.nanmin(roi_nom_x_2d)),
                float(np.nanmax(roi_nom_x_2d)),
                float(np.nanmin(roi_nom_y_2d)),
                float(np.nanmax(roi_nom_y_2d)),
            ]
        else:
            roi_mask = PositionVisualizer._build_roi_mask(
                nom_x, nom_y, roi_bounds
            )
            roi_meas_x = meas_x[roi_mask]
            roi_meas_y = meas_y[roi_mask]
            roi_nom_x = nom_x[roi_mask]
            roi_nom_y = nom_y[roi_mask]

        if len(roi_meas_x) > 0:
            ax2.plot(roi_meas_x, roi_meas_y, 'bo', label='Calc.')
            ax2.plot(roi_nom_x, roi_nom_y, 'r+', label='Nom.', markersize=8)
        else:
            ax2.plot(meas_x, meas_y, 'bo', label='Calc.')
            ax2.plot(nom_x, nom_y, 'r+', label='Nom.', markersize=8)

        ax2.set_xlabel(u"$x$ [$\\mu$m]", fontsize=14)
        ax2.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)
        ax2.set_title(t_roi or _Title('xbpm_positions', 'roi'))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        PositionVisualizer._apply_equal_limits(
            ax2,
            roi_meas_x if len(roi_meas_x) > 0 else meas_x,
            roi_meas_y if len(roi_meas_y) > 0 else meas_y,
            roi_nom_x if len(roi_nom_x) > 0 else nom_x,
            roi_nom_y if len(roi_nom_y) > 0 else nom_y,
        )

        # Plot 3: RMS differences in ROI, matching live plots.
        if roi_diff_2d is not None and roi_extent is not None:
            im = ax3.imshow(
                roi_diff_2d,
                cmap='viridis',
                origin='lower',
                extent=roi_extent,
                aspect='auto',
            )
        else:
            hm_nom_x = roi_nom_x if len(roi_nom_x) > 0 else nom_x
            hm_nom_y = roi_nom_y if len(roi_nom_y) > 0 else nom_y
            hm_diff = diff_rms[roi_mask] if len(roi_meas_x) > 0 else diff_rms

            x_grid, y_grid, diff_2d = PositionVisualizer._grid_from_points(
                hm_nom_x, hm_nom_y, hm_diff
            )
            im = ax3.pcolormesh(x_grid, y_grid, diff_2d,
                                cmap='viridis', shading='auto')

        # No grid for heatmaps, and set aspect ratio to match data
        ax3.set_aspect('equal', adjustable='box')
        cbar = plt.colorbar(
            im, ax=ax3, label='RMS Difference [$\\mu$m]',
            fraction=0.046, pad=0.04
        )
        cbar.set_label('RMS Difference [$\\mu$m]', fontsize=14)
        ax3.set_xlabel(u"$x$ [$\\mu$m]", fontsize=14)
        ax3.set_ylabel(u"$y$ [$\\mu$m]", fontsize=14)
        ax3.set_title(t_heat or _Title('xbpm_positions', 'heatmap'))

        return fig

    @staticmethod
    def _build_roi_mask(nom_x, nom_y, roi_bounds=None):
        """Build ROI mask from saved bounds or central fallback."""
        if isinstance(roi_bounds, dict):
            try:
                x_min = float(roi_bounds['x_min'])
                x_max = float(roi_bounds['x_max'])
                y_min = float(roi_bounds['y_min'])
                y_max = float(roi_bounds['y_max'])
                return ((nom_x >= x_min) & (nom_x <= x_max) &
                        (nom_y >= y_min) & (nom_y <= y_max))
            except Exception:
                pass

        h_center = (np.min(nom_x) + np.max(nom_x)) / 2
        v_center = (np.min(nom_y) + np.max(nom_y)) / 2
        h_range = (np.max(nom_x) - np.min(nom_x)) / 4
        v_range = (np.max(nom_y) - np.min(nom_y)) / 4
        return ((np.abs(nom_x - h_center) <= h_range) &
                (np.abs(nom_y - v_center) <= v_range))

    @staticmethod
    def _apply_equal_limits(ax, pos_h, pos_v, nom_h, nom_v):
        """Apply the same centered equal-aspect limits used in live plots."""
        all_h = np.concatenate([np.ravel(pos_h), np.ravel(nom_h)])
        all_v = np.concatenate([np.ravel(pos_v), np.ravel(nom_v)])
        all_h = all_h[np.isfinite(all_h)]
        all_v = all_v[np.isfinite(all_v)]
        if all_h.size == 0 or all_v.size == 0:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', adjustable='box')
            return

        h_min, h_max = np.min(all_h), np.max(all_h)
        v_min, v_max = np.min(all_v), np.max(all_v)
        h_range = h_max - h_min if h_max > h_min else 1
        v_range = v_max - v_min if v_max > v_min else 1
        total_range = max(h_range, v_range) * 1.3
        h_center = (h_min + h_max) / 2
        v_center = (v_min + v_max) / 2
        ax.set_xlim(h_center - total_range / 2, h_center + total_range / 2)
        ax.set_ylim(v_center - total_range / 2, v_center + total_range / 2)
        ax.set_aspect('equal', adjustable='box')

    @staticmethod
    def _grid_from_points(nom_x, nom_y, values, rounding_digits=6):
        """Create dense 2D grid from point cloud with float-stable binning."""
        x_key = np.round(nom_x.astype(float), rounding_digits)
        y_key = np.round(nom_y.astype(float), rounding_digits)
        x_grid = np.unique(x_key)
        y_grid = np.unique(y_key)

        x_map = {val: idx for idx, val in enumerate(x_grid)}
        y_map = {val: idx for idx, val in enumerate(y_grid)}
        grid = np.full((len(y_grid), len(x_grid)), np.nan, dtype=float)

        for xx, yy, vv in zip(x_key, y_key, values):
            ix = x_map.get(xx)
            iy = y_map.get(yy)
            if ix is not None and iy is not None:
                grid[iy, ix] = vv

        return x_grid, y_grid, grid

    @staticmethod
    def _center_slice(n_side: int, roi_side: int) -> slice:
        """Return central square slice of size roi_side within n_side."""
        roi_side = int(max(1, min(roi_side, n_side)))
        start = max(0, (n_side - roi_side) // 2)
        end = min(n_side, start + roi_side)
        return slice(start, end)

    @staticmethod
    def _infer_roi_side_from_bounds(nom_x_2d, nom_y_2d, roi_bounds=None):
        """Infer square ROI side length from bounds and structured grid."""
        if not isinstance(roi_bounds, dict):
            return None
        try:
            x_min = float(roi_bounds['x_min'])
            x_max = float(roi_bounds['x_max'])
            y_min = float(roi_bounds['y_min'])
            y_max = float(roi_bounds['y_max'])
        except Exception:
            return None

        x_line = np.nanmedian(np.asarray(nom_x_2d, dtype=float), axis=0)
        y_line = np.nanmedian(np.asarray(nom_y_2d, dtype=float), axis=1)
        nx = int(np.count_nonzero((x_line >= x_min) & (x_line <= x_max)))
        ny = int(np.count_nonzero((y_line >= y_min) & (y_line <= y_max)))
        roi_side = min(nx, ny)
        return roi_side if roi_side > 0 else None


class SweepVisualizer:
    """Unified visualizer for central sweep analysis.

    Creates sweep plots from either:
    - Live analysis data (from processors)
    - HDF5 stored data (from readers)

    This eliminates redundancy between processors._central_sweeps_show()
    and readers._reconstruct_sweeps().
    """

    def plot_from_arrays(range_h, range_v, pos_h, pos_v,
                         fit_h=None, fit_v=None,
                         fit_h_line=None, fit_v_line=None,
                         xbpm_dist=1.0, figsize=(12, 5)):
        """Create sweep figure from numpy arrays (position reconstruction path).

        This is used when sweeps data is stored as pre-calculated positions
        (HDF5 or other sources). Formatting matches canonical plot_central_sweeps.

        Args:
            range_h: Horizontal sweep range
            range_v: Vertical sweep range
            pos_h: Calculated vertical positions at horizontal sweep (y_calc)
            pos_v: Calculated horizontal positions at vertical sweep (x_calc)
            fit_h: Fit coefficients for horizontal sweep [k, delta] or None
            fit_v: Fit coefficients for vertical sweep [k, delta] or None
            xbpm_dist: Distance scaling factor
            figsize: Figure size tuple

        Returns:
            matplotlib.figure.Figure
        """
        fig, (axh, axv) = plt.subplots(1, 2, figsize=figsize)

        # Horizontal sweep plot
        if range_h is not None and pos_h is not None:
            x_vals = range_h * xbpm_dist
            y_vals = pos_h * xbpm_dist

            axh.plot(x_vals, y_vals, 'o-', label="H sweep", zorder=2)

            if fit_h_line is not None:
                fit_line = fit_h_line * xbpm_dist
                axh.plot(x_vals, fit_line, '^-', label="H fit", zorder=3)
            elif fit_h is not None:
                fit_line = (fit_h[0] * range_h + fit_h[1]) * xbpm_dist
                axh.plot(x_vals, fit_line, '^-', label="H fit", zorder=3)

            axh.set_xlabel("$x$ [$\\mu$m]")
            axh.set_ylabel("$y$ [$\\mu$m]")
            axh.set_title(_Title('sweeps', 'h'))
            axh.grid(True)
            axh.legend()

        # Vertical sweep plot
        if range_v is not None and pos_v is not None:
            x_vals = pos_v * xbpm_dist
            y_vals = range_v * xbpm_dist

            axv.plot(x_vals, y_vals, 'o-', label="V sweep", zorder=2)

            if fit_v_line is not None:
                fit_line = fit_v_line * xbpm_dist
                axv.plot(fit_line, y_vals, '^-', label="V fit", zorder=3)
            elif fit_v is not None:
                fit_line = (fit_v[0] * range_v + fit_v[1]) * xbpm_dist
                axv.plot(fit_line, y_vals, '^-', label="V fit", zorder=3)

            axv.set_xlabel("$x$ [$\\mu$m]")
            axv.set_ylabel("$y$ [$\\mu$m]")
            axv.set_title(_Title('sweeps', 'v'))
            axv.grid(True)
            axv.legend()

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_from_hdf5(h_data, v_data, figsize=(12, 5)):
        """Create sweep figure from HDF5 datasets (reconstruction path).

        This reads pre-calculated sweep positions from HDF5 and plots them
        with the same formatting as canonical plot_central_sweeps.

        Args:
            h_data: HDF5 dataset for horizontal sweep (structured array)
            v_data: HDF5 dataset for vertical sweep (structured array)
            figsize: Figure size tuple

        Returns:
            matplotlib.figure.Figure
        """
        range_h = h_data['x_index'][:] if h_data is not None else None
        range_v = v_data['y_index'][:] if v_data is not None else None
        pos_h = h_data['y_calc'][:] if h_data is not None else None
        pos_v = v_data['x_calc'][:] if v_data is not None else None

        fit_h_line = (h_data['y_fit'][:] if (h_data is not None and
                      'y_fit' in h_data.dtype.names) else None)
        fit_v_line = (v_data['x_fit'][:] if (v_data is not None and
                      'x_fit' in v_data.dtype.names) else None)

        fit_h = None
        if h_data is not None:
            if fit_h_line is None and 'k' in h_data.attrs and 'delta' in h_data.attrs:
                fit_h = np.array([h_data.attrs['k'], h_data.attrs['delta']])
            elif fit_h_line is None and range_h is not None and 'y_fit' in h_data.dtype.names:
                try:
                    fit_h = np.polyfit(range_h, h_data['y_fit'][:], deg=1)
                except Exception:
                    fit_h = None

        fit_v = None
        if v_data is not None:
            if fit_v_line is None and 'k' in v_data.attrs and 'delta' in v_data.attrs:
                fit_v = np.array([v_data.attrs['k'], v_data.attrs['delta']])
            elif fit_v_line is None and range_v is not None and 'x_fit' in v_data.dtype.names:
                try:
                    fit_v = np.polyfit(range_v, v_data['x_fit'][:], deg=1)
                except Exception:
                    fit_v = None

        return SweepVisualizer.plot_from_arrays(
            range_h, range_v, pos_h, pos_v,
            fit_h=fit_h, fit_v=fit_v,
            fit_h_line=fit_h_line, fit_v_line=fit_v_line,
            xbpm_dist=1.0, figsize=figsize
        )


class BladeCurrentVisualizer:
    """Unified visualizer for blade current analysis.

    Creates blade current plots from either:
    - Live analysis data (from processors)
    - HDF5 stored data (from readers)

    This eliminates redundancy between processors.show_blades_at_center()
    and readers._reconstruct_blades_center().
    """

    @staticmethod
    def _plot_blade(ax, rng, y, yerr, marker, blade_name):
        if yerr is not None:
            container = ax.errorbar(
                rng, y, yerr=yerr, fmt=marker, label=blade_name, zorder=2
            )
            return container.lines[0] if container.lines else None
        else:
            line, = ax.plot(rng, y, marker, label=blade_name, zorder=2)
            return line

    @staticmethod
    def _fit_blade(rng, y, yerr, attrs, blade_name):
        k_attr = f'k_{blade_name}'
        d_attr = f'delta_{blade_name}'
        coef = None
        if attrs:
            k_val = attrs.get(k_attr)
            d_val = attrs.get(d_attr)
            if k_val is not None and d_val is not None:
                coef = (k_val, d_val)
        if coef is None:
            weights = None
            if ((yerr is not None and
                    np.all(np.isfinite(yerr)) and
                    np.all(yerr > 0))):
                weights = 1.0 / yerr
            coef = np.polyfit(rng, y, deg=1, w=weights)
        return coef

    @staticmethod
    def plot_from_dicts(blades_h, blades_v, range_h, range_v,
                        beamline="", figsize=(10, 5)):
        """Create blade current figure from in-memory dicts (live/processors path).

        Uses canonical plot_blades_center_from_dicts for consistent formatting.
        """
        return plot_blades_center_from_dicts(blades_h, blades_v, range_h,
                                             range_v, beamline)

    @staticmethod
    def plot_from_hdf5(h_data, v_data, figsize=(10, 5)):
        """Create blade current figure from HDF5 datasets.

        Extracts blade data and uses canonical plot_blades_center_from_dicts
        for consistent formatting across live and reconstruction paths.

        Args:
            h_data: HDF5 dataset for horizontal blades (structured array)
            v_data: HDF5 dataset for vertical blades (structured array)
            figsize: Figure size tuple

        Returns:
            matplotlib.figure.Figure
        """
        # Extract blade data from HDF5 into dicts matching live format
        blades_h = None
        range_h = None
        if h_data is not None:
            rng = h_data['x_index'][:]
            blades_h = {}
            for blade_name in ['to', 'ti', 'bi', 'bo']:
                vals = h_data[blade_name][:]
                err_name = f's_{blade_name}'
                errs = (h_data[err_name][:]
                        if err_name in h_data.dtype.names else None)
                if errs is not None:
                    blades_h[blade_name] = np.column_stack([vals, errs])
                else:
                    blades_h[blade_name] = vals
            range_h = rng

        blades_v = None
        range_v = None
        if v_data is not None:
            rng = v_data['y_index'][:]
            blades_v = {}
            for blade_name in ['to', 'ti', 'bi', 'bo']:
                vals = v_data[blade_name][:]
                err_name = f's_{blade_name}'
                errs = (v_data[err_name][:]
                        if err_name in v_data.dtype.names else None)
                if errs is not None:
                    blades_v[blade_name] = np.column_stack([vals, errs])
                else:
                    blades_v[blade_name] = vals
            range_v = rng

        # Use canonical plotting function
        return plot_blades_center_from_dicts(blades_h, blades_v, range_h,
                                             range_v, "")

    @staticmethod
    def _plot_side(ax, blades, rng, attrs, marker_map,
                   xlab_default, fit_style, side_label):
        if blades is None or rng is None:
            return
        for blade_name, marker in marker_map:
            arr = blades.get(blade_name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                y = arr[:, 0]
                yerr = arr[:, 1]
            else:
                y = arr
                yerr = None

            data_line = BladeCurrentVisualizer._plot_blade(
                ax, rng, y, yerr, marker, blade_name
            )

            # Fit only if variation exists
            if (np.any(np.isfinite(y)) and
                np.nanstd(y) > 0 and
                len(rng) > 1):
                try:
                    coef = (
                        BladeCurrentVisualizer._fit_blade(rng, y, yerr,
                                                        attrs, blade_name)
                    )
                    style = dict(fit_style)
                    if data_line is not None:
                        style['color'] = data_line.get_color()
                    ax.plot(rng, coef[0] * rng + coef[1],
                            label=f"{blade_name} fit", **style)
                except Exception:
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        "Blade fit failed for %s side, blade %s",
                        side_label,
                        blade_name,
                        exc_info=True,
                    )
                    pass

        ax.set_xlabel(xlab_default, fontsize=16)
        ax.set_ylabel('I', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.grid()
        ax.legend(fontsize=11)

    @staticmethod
    def _plot_blades_common(blades_h, blades_v, range_h, range_v,
                            attrs_h=None, attrs_v=None,
                            beamline="", figsize=(10, 5)):
        """Shared plotting path for blade currents (live and HDF5)."""
        # Keep fit lines consistent and visually overlaid on data.
        fit_style = {"linestyle": "--",
                 "linewidth": 2.0,
                 "alpha": 1.0,
                 "zorder": 6}

        fig, (axh, axv) = plt.subplots(1, 2, figsize=figsize)

        BladeCurrentVisualizer._plot_side(
            axh, blades_h, range_h, attrs_h,
            [('to', 'o-'), ('ti', 's-'), ('bi', 'd-'), ('bo', '^-')],
            (attrs_h or {}).get('xlabel_blades', 'x [μrad]'),
            fit_style, 'horizontal')
        BladeCurrentVisualizer._plot_side(
            axv, blades_v, range_v, attrs_v,
            [('to', 'o-'), ('ti', 's-'), ('bi', 'd-'), ('bo', 'v-')],
            (attrs_v or {}).get('xlabel_blades', 'y [μrad]'),
            fit_style, 'vertical')

        ylabel = (u"$I$ [# counts]" if beamline[:3] in ["MGN", "MNC"]
                 else u"$I$ [A]")
        axh.set_ylabel(ylabel, fontsize=14)
        axv.set_ylabel(ylabel, fontsize=14)
        axh.set_title(_Title('blades_at_sweeps', 'h'))
        axv.set_title(_Title('blades_at_sweeps', 'v'))

        fig.suptitle(
            _Title('blades_at_sweeps', 'suptitle'), fontsize=12, fontweight='bold'
        )
        fig.tight_layout()
        return fig


# ============================================================================
# Canonical plotting functions for live and HDF5 reconstruction paths
# ============================================================================

def plot_central_sweeps(range_h, range_v, blades_h, blades_v, xbpmdist):
    """Generate central sweep position plots (canonical version).

    This is the tuned plotting implementation used by both live analysis
    and HDF5 reconstruction. It encapsulates the exact visualization
    semantics for the "Positions along sweeps" tab.

    Args:
        range_h: Horizontal sweep range array.
        range_v: Vertical sweep range array.
        blades_h: Horizontal blades dict with 'to', 'ti', 'bi', 'bo' keys.
        blades_v: Vertical blades dict with 'to', 'ti', 'bi', 'bo' keys.
        xbpmdist: Scaling distance for physical units conversion.

    Returns:
        matplotlib.figure.Figure
    """
    # Calculate positions from blade data
    pos_ch_v = None
    fit_ch_v = None
    if blades_h is not None and len(range_h) > 1:
        to_ch = blades_h["to"]
        ti_ch = blades_h["ti"]
        bi_ch = blades_h["bi"]
        bo_ch = blades_h["bo"]

        pos_to_ti_v = (to_ch + ti_ch)
        pos_bi_bo_v = (bo_ch + bi_ch)
        pos_ch_v = ((pos_to_ti_v - pos_bi_bo_v) /
                    (pos_to_ti_v + pos_bi_bo_v))
        fit_ch_v = np.polyfit(range_h, pos_ch_v, deg=1)

    pos_cv_h = None
    fit_cv_h = None
    if blades_v is not None and len(range_v) > 1:
        to_cv = blades_v["to"]
        ti_cv = blades_v["ti"]
        bi_cv = blades_v["bi"]
        bo_cv = blades_v["bo"]

        pos_to_bo_h = (to_cv + bo_cv)
        pos_ti_bi_h = (ti_cv + bi_cv)
        pos_cv_h = ((pos_to_bo_h - pos_ti_bi_h) /
                    (pos_to_bo_h + pos_ti_bi_h))
        fit_cv_h = np.polyfit(range_v, pos_cv_h, deg=1)

    # Create figure
    fig, (axh, axv) = plt.subplots(1, 2, figsize=(12, 5))

    if fit_ch_v is not None:
        hline = ((fit_ch_v[0, 0] * range_h + fit_ch_v[1, 0])
                 * xbpmdist)
        axh.plot(range_h * xbpmdist,
                 pos_ch_v[:, 0] * xbpmdist, 'o-',
                 label="H sweep")
        axh.plot(range_h * xbpmdist, hline,
                 '^-', label="H fit")
        axh.set_xlabel("$x$ [$\\mu$m]")
        axh.set_ylabel("$y$ [$\\mu$m]")
        axh.set_title(_Title('sweeps', 'h'))
        ylim = (np.max(np.abs(hline + pos_ch_v[:, 0]
                              * xbpmdist)) * 1.1)
        axh.set_ylim(-ylim, ylim)
        axh.grid(True)
        axh.legend()

    if fit_cv_h is not None:
        vline = ((fit_cv_h[0, 0] * range_v + fit_cv_h[1, 0])
                 * xbpmdist)
        axv.plot(pos_cv_h[:, 0] * xbpmdist,
                 range_v * xbpmdist,
                 'o-', label="V sweep")
        axv.plot(vline, range_v * xbpmdist,
                 '^-', label="V fit")
        axv.set_xlabel("$x$ [$\\mu$m]")
        axv.set_ylabel("$y$ [$\\mu$m]")
        axv.set_title(_Title('sweeps', 'v'))
        axv.set_xlim((np.min(range_v) * 0.005 + fit_cv_h[1, 0])
                     * xbpmdist,
                     (np.max(range_v) * 0.005 + fit_cv_h[1, 0])
                     * xbpmdist)
        axv.grid(True)
        axv.legend()

    fig.tight_layout()
    return fig


def plot_blades_center_from_dicts(blades_h, blades_v, range_h, range_v,
                                   beamline=""):
    """Generate blade currents at center plots (canonical version).

    This is the tuned plotting implementation used by both live analysis
    and HDF5 reconstruction. It encapsulates the exact visualization
    semantics for the "Blades at sweeps" tab.

    Args:
        blades_h: Horizontal blades dict with 'to', 'ti', 'bi', 'bo' keys,
                  each mapping to (value, weight) pairs.
        blades_v: Vertical blades dict with 'to', 'ti', 'bi', 'bo' keys.
        range_h: Horizontal sweep range array.
        range_v: Vertical sweep range array.
        beamline: Beamline name for ylabel determination.

    Returns:
        matplotlib.figure.Figure or None if no blade data.
    """
    if blades_h is None and blades_v is None:
        return None

    fig, (axh, axv) = plt.subplots(1, 2, figsize=(10, 5))

    if blades_h is not None:
        for key, blval in blades_h.items():
            val = blval[:, 0]
            wval = blval[:, 1]
            weight = 1. / wval if not np.isinf(1. / wval).any() else None
            (acoef, bcoef) = np.polyfit(range_h, val, deg=1, w=weight)
            k = f"{key.upper()}"
            axh.errorbar(range_h, val, wval, fmt='o-', label=k,
                         zorder=1)
            axh.plot(range_h, range_h * acoef + bcoef,
                     "^-", label=f"{k} fit", zorder=2)

    if blades_v is not None:
        for key, blval in blades_v.items():
            val = blval[:, 0]
            wval = blval[:, 1]
            weight = 1. / wval if not np.isinf(1. / wval).any() else None
            (acoef, bcoef) = np.polyfit(range_v, val, deg=1, w=weight)
            k = f"{key.upper()}"
            axv.errorbar(range_v, val, wval, fmt='o-', label=k,
                         zorder=1)
            axv.plot(range_v, range_v * acoef + bcoef,
                     "^-", label=f"{k} fit", zorder=2)

    axh.set_title(_Title('blades_at_sweeps', 'h'))
    axv.set_title(_Title('blades_at_sweeps', 'v'))
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
