"""Visualization classes for blade maps and positions."""

import numpy as np
import matplotlib.pyplot as plt
import logging

from .parameters import Prm
from .constants import FIGDPI


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
        if to.shape[0] == 1 or to.shape[1] == 1:
            extent = None
        else:
            alist = np.array(list(self.data.keys()))
            klist = np.unique(alist[:, 0])
            mlist = np.unique(alist[:, 1])
            minvalx, maxvalx = np.min(klist), np.max(klist)
            minvaly, maxvaly = np.min(mlist), np.max(mlist)
            extent = (minvalx, maxvalx, minvaly, maxvaly)

        quad = [[ti, to], [bi, bo]]
        names = [["TI", "TO"], ["BI", "BO"]]

        for idy in range(2):
            for idx in range(2):
                rx[idy][idx].imshow(quad[idy][idx], extent=extent)
                rx[idy][idx].set_xlabel(u"$x$ [$\\mu$rad]")
                rx[idy][idx].set_ylabel(u"$y$ [$\\mu$rad]")
                rx[idy][idx].set_title(names[idy][idx])

        fig.tight_layout(pad=0., w_pad=-10., h_pad=2.)

        if self.prm.outputfile:
            outfile = f"blade_map_{self.prm.beamline}.png"
            fig.savefig(outfile, dpi=FIGDPI)
            logger.info("Figure of blades' map saved to file %s", outfile)

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

    def __init__(self, prm: Prm, title: str = ""):
        """Initialize visualizer with parameters.

        Args:
            prm: Parameters dataclass instance.
            title: Title prefix for plots (e.g., "Pairwise" or "Cross-blades").
        """
        self.prm = prm
        self.title = title
        self.fig = None

        # Module logger
        self._logger = logging.getLogger(__name__)

    def show_position_results(self, pos_nom_h, pos_nom_v,
                              pos_h, pos_v, pos_h_roi, pos_v_roi,
                              pos_nom_h_roi, pos_nom_v_roi,
                              diff_roi, figsize=(18, 6)) -> None:
        """Display full position results in 1x3 subplot layout.

        Args:
            pos_nom_h: Nominal horizontal positions (full grid).
            pos_nom_v: Nominal vertical positions (full grid).
            pos_h: Calculated horizontal positions (full grid).
            pos_v: Calculated vertical positions (full grid).
            pos_h_roi: Calculated horizontal positions in ROI.
            pos_v_roi: Calculated vertical positions in ROI.
            pos_nom_h_roi: Nominal horizontal positions in ROI.
            pos_nom_v_roi: Nominal vertical positions in ROI.
            diff_roi: RMS position differences in ROI.
            figsize: Figure size as (width, height) tuple.
        """
        self.fig, (ax_all, ax_close, ax_color) = plt.subplots(
            1, 3, figsize=figsize, constrained_layout=True
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

        # Full grid view
        self._plot_scaled_positions(
            ax_all, pos_nom_h, pos_nom_v, pos_h, pos_v,
            f"{self.title} @ {self.prm.beamline}"
        )

        # ROI closeup
        self._plot_scaled_positions(
            ax_close, pos_nom_h_roi, pos_nom_v_roi,
            pos_h_roi, pos_v_roi,
            f"{self.title} @ {self.prm.beamline} closeup"
        )

        # Difference heatmap
        self._plot_position_differences(
            ax_color, diff_roi, pos_nom_h_roi, pos_nom_v_roi,
            f"{self.title} @ {self.prm.beamline}"
        )

        # constrained_layout handles spacing; avoid mixing with tight_layout

    def _plot_scaled_positions(self, ax, pos_nom_h, pos_nom_v,
                              pos_h, pos_v, title):
        """Plot nominal vs calculated positions on given axis."""
        ax.set_title(title, pad=2)
        pos = ax.plot(pos_h, pos_v, 'bo')
        nom = ax.plot(pos_nom_h, pos_nom_v, 'r+')
        ax.set_xlabel(u"$x$ [$\\mu$m]")
        ax.set_ylabel(u"$y$ [$\\mu$m]")

        # Compute common limits to ensure equal aspect ratio with margin
        # Flatten all data to find overall min/max
        all_h = np.concatenate([np.ravel(pos_h), np.ravel(pos_nom_h)])
        all_v = np.concatenate([np.ravel(pos_v), np.ravel(pos_nom_v)])

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
            labels.append("Nominal")
        if len(pos) > 0:
            handles.append(pos[0])
            labels.append("Calculated")
        if handles:
            ax.legend(handles, labels)
        ax.grid()

    def _plot_position_differences(self, ax, diffroi, pos_nom_h=None,
                                   pos_nom_v=None, title=""):
        """Plot position difference heatmap or scatter on given axis."""
        if len(diffroi.shape) > 1:
            im = ax.imshow(diffroi, cmap='viridis')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(u"RMS differences (ROI)")
        else:
            # Ensure x, y, and color arrays have matching lengths
            if pos_nom_h is not None and pos_nom_v is not None:
                x = np.ravel(pos_nom_h)
                y = np.ravel(pos_nom_v)
            else:
                # Fallback to index-based positions
                n = int(np.size(diffroi))
                x = np.arange(n)
                y = np.zeros(n)

            cvals = np.ravel(diffroi)

            # Align lengths if shapes are inconsistent
            m = min(len(x), len(y), len(cvals))
            x, y, cvals = x[:m], y[:m], cvals[:m]

            scatter = ax.scatter(x, y, c=cvals, cmap='viridis', s=50)
            plt.colorbar(scatter, ax=ax, label='Difference [$\\mu$m]')

        ax.set_title(title, pad=2)
        ax.set_xlabel(u"$x$ [$\\mu$m]")
        ax.set_ylabel(u"$y$ [$\\mu$m]")

    def save_figure(self, filename: str) -> None:
        """Save the figure to a file.

        Args:
            filename: Path to save the figure to.
        """
        if self.fig is not None:
            self.fig.savefig(filename, dpi=FIGDPI, bbox_inches='tight')
            logger.info("Figure saved to %s", filename)
