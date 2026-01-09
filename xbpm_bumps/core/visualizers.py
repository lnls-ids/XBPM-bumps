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
                # Use the same orientation as live plots
                ax.imshow(quad[row][col], extent=extent, origin='upper')
                ax.set_xlabel(blade_grp.attrs.get('xlabel', 'x [μrad]'))
                ax.set_ylabel(blade_grp.attrs.get('ylabel', 'y [μrad]'))
                ax.set_title(names[row][col])

        fig.tight_layout(pad=0., w_pad=-10., h_pad=2.)
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

    @staticmethod
    def plot_from_hdf5(pos_data):
        """Create position comparison figure from HDF5 dataset.

        Args:
            pos_data: HDF5 dataset with structured array containing
                position data

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

        # Plot 1: All positions on full grid
        ax1.plot(meas_x, meas_y, 'bo', label='Calculated')
        ax1.plot(nom_x, nom_y, 'r+', label='Nominal', markersize=8)
        ax1.set_xlabel(pos_data.attrs.get('xlabel', 'x [μm]'))
        ax1.set_ylabel(pos_data.attrs.get('ylabel', 'y [μm]'))
        ax1.set_title('Full Grid')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Plot 2: ROI closeup (center 50% of data)
        h_center = (np.min(nom_x) + np.max(nom_x)) / 2
        v_center = (np.min(nom_y) + np.max(nom_y)) / 2
        h_range = (np.max(nom_x) - np.min(nom_x)) / 4
        v_range = (np.max(nom_y) - np.min(nom_y)) / 4

        roi_mask = (
            (np.abs(nom_x - h_center) <= h_range) &
            (np.abs(nom_y - v_center) <= v_range)
        )

        roi_meas_x = meas_x[roi_mask]
        roi_meas_y = meas_y[roi_mask]
        roi_nom_x = nom_x[roi_mask]
        roi_nom_y = nom_y[roi_mask]

        if len(roi_meas_x) > 0:
            ax2.plot(roi_meas_x, roi_meas_y, 'bo', label='Calculated')
            ax2.plot(roi_nom_x, roi_nom_y, 'r+', label='Nominal', markersize=8)
        else:
            ax2.plot(meas_x, meas_y, 'bo', label='Calculated')
            ax2.plot(nom_x, nom_y, 'r+', label='Nominal', markersize=8)

        ax2.set_xlabel(pos_data.attrs.get('xlabel', 'x [μm]'))
        ax2.set_ylabel(pos_data.attrs.get('ylabel', 'y [μm]'))
        ax2.set_title('ROI Closeup')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        # Plot 3: RMS differences as 2D heatmap
        try:
            n_unique_x = len(np.unique(nom_x))
            n_unique_y = len(np.unique(nom_y))
            if len(nom_x) == n_unique_x * n_unique_y:
                diff_2d = diff_rms.reshape(n_unique_y, n_unique_x)
                x_grid = np.unique(nom_x)
                y_grid = np.unique(nom_y)
                im = ax3.pcolormesh(
                    x_grid, y_grid, diff_2d, cmap='viridis', shading='auto'
                )
            else:
                raise ValueError("Data not gridded")
        except (ValueError, RuntimeError):
            im = ax3.scatter(
                nom_x, nom_y, c=diff_rms, cmap='viridis', s=50,
                edgecolors='k', linewidth=0.5
            )

        # No grid for heatmaps, and set aspect ratio to match data
        ax3.set_aspect('equal', adjustable='box')
        plt.colorbar(
            im, ax=ax3, label='RMS Difference [μm]',
            fraction=0.046, pad=0.04
        )
        ax3.set_xlabel(pos_data.attrs.get('xlabel', 'x [μm]'))
        ax3.set_ylabel(pos_data.attrs.get('ylabel', 'y [μm]'))
        ax3.set_title('Position Differences')

        # Extract title from attrs or use default
        pos_type = (
            pos_data.name.split('/')[-1]
            if hasattr(pos_data, 'name') else 'Position'
        )
        title_default = pos_type.replace('_', ' ').title() + ' Positions'
        title = pos_data.attrs.get('title', title_default)
        fig.suptitle(title, fontsize=12, fontweight='bold')

        return fig


class SweepVisualizer:
    """Unified visualizer for central sweep analysis.

    Creates sweep plots from either:
    - Live analysis data (from processors)
    - HDF5 stored data (from readers)

    This eliminates redundancy between processors._central_sweeps_show()
    and readers._reconstruct_sweeps().
    """

    @staticmethod
    def plot_from_arrays(range_h, range_v, pos_h, pos_v,
                         fit_h=None, fit_v=None,
                         xbpm_dist=1.0, figsize=(12, 5)):
        """Create sweep figure from numpy arrays.

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

            axh.plot(x_vals, y_vals, 'o-', label="H sweep")

            if fit_h is not None:
                fit_line = (fit_h[0] * range_h + fit_h[1]) * xbpm_dist
                axh.plot(x_vals, fit_line, '^-', label="H fit")

            axh.set_xlabel(u"$x$ [$\\mu$m]")
            axh.set_ylabel(u"$y$ [$\\mu$m]")
            axh.set_title("Horizontal Sweeps")
            axh.grid(True)
            axh.legend()

        # Vertical sweep plot
        if range_v is not None and pos_v is not None:
            x_vals = pos_v * xbpm_dist
            y_vals = range_v * xbpm_dist

            axv.plot(x_vals, y_vals, 'o-', label="V sweep")

            if fit_v is not None:
                fit_line = (fit_v[0] * range_v + fit_v[1]) * xbpm_dist
                axv.plot(fit_line, y_vals, '^-', label="V fit")

            axv.set_xlabel(u"$x$ [$\\mu$m]")
            axv.set_ylabel(u"$y$ [$\\mu$m]")
            axv.set_title("Vertical Sweeps")
            axv.grid(True)
            axv.legend()

        fig.suptitle(
            'Central Sweeps Analysis', fontsize=12, fontweight='bold'
        )
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_from_hdf5(h_data, v_data, figsize=(12, 5)):
        """Create sweep figure from HDF5 datasets.

        Args:
            h_data: HDF5 dataset for horizontal sweep (structured array)
            v_data: HDF5 dataset for vertical sweep (structured array)
            figsize: Figure size tuple

        Returns:
            matplotlib.figure.Figure
        """
        fig, (axh, axv) = plt.subplots(1, 2, figsize=figsize)

        # Horizontal sweep
        if h_data is not None:
            range_h = h_data['x_index'][:]
            pos_h = h_data['y_calc'][:]
            pos_h_err = (
                h_data['s_y_calc'][:]
                if 's_y_calc' in h_data.dtype.names else None
            )

            # Only use errorbar if errors are valid (positive, finite)
            if (pos_h_err is not None and np.all(pos_h_err >= 0)
                    and np.all(np.isfinite(pos_h_err))):
                axh.errorbar(
                    range_h, pos_h, yerr=pos_h_err,
                    fmt='o-', label="H sweep"
                )
            else:
                axh.plot(range_h, pos_h, 'o-', label="H sweep")

            # Plot fit line from stored column if available,
            # else calculate from attrs
            if 'y_fit' in h_data.dtype.names:
                fit_line = h_data['y_fit'][:]
                axh.plot(range_h, fit_line, '^-', label="H fit")
            elif 'k' in h_data.attrs and 'delta' in h_data.attrs:
                fit_line = (
                    h_data.attrs['k'] * range_h + h_data.attrs['delta']
                )
                axh.plot(range_h, fit_line, '^-', label="H fit")

            axh.set_xlabel(h_data.attrs.get('xlabel', 'x [μm]'))
            axh.set_ylabel(h_data.attrs.get('ylabel', 'y [μm]'))
            axh.set_title(h_data.attrs.get('title', 'Horizontal Sweeps'))
            axh.grid(True)
            axh.legend()

        # Vertical sweep
        if v_data is not None:
            range_v = v_data['y_index'][:]
            pos_v = v_data['x_calc'][:]
            pos_v_err = (
                v_data['s_x_calc'][:]
                if 's_x_calc' in v_data.dtype.names else None
            )

            # Only use errorbar if errors are valid (positive, finite)
            if (pos_v_err is not None and np.all(pos_v_err >= 0)
                    and np.all(np.isfinite(pos_v_err))):
                axv.errorbar(
                    pos_v, range_v, xerr=pos_v_err,
                    fmt='o-', label="V sweep"
                )
            else:
                axv.plot(pos_v, range_v, 'o-', label="V sweep")

            # Plot fit line from stored column if available,
            # else calculate from attrs
            if 'x_fit' in v_data.dtype.names:
                fit_line = v_data['x_fit'][:]
                axv.plot(fit_line, range_v, '^-', label="V fit")
            elif 'k' in v_data.attrs and 'delta' in v_data.attrs:
                fit_line = (
                    v_data.attrs['k'] * range_v + v_data.attrs['delta']
                )
                axv.plot(fit_line, range_v, '^-', label="V fit")

            axv.set_xlabel(v_data.attrs.get('xlabel', 'y [μm]'))
            axv.set_ylabel(v_data.attrs.get('ylabel', 'x [μm]'))
            axv.set_title(v_data.attrs.get('title', 'Vertical Sweeps'))
            axv.grid(True)
            axv.legend()

        fig.suptitle(
            'Central Sweeps Analysis', fontsize=12, fontweight='bold'
        )
        fig.tight_layout()
        return fig


class BladeCurrentVisualizer:
    """Unified visualizer for blade current analysis.

    Creates blade current plots from either:
    - Live analysis data (from processors)
    - HDF5 stored data (from readers)

    This eliminates redundancy between processors.show_blades_at_center()
    and readers._reconstruct_blades_center().
    """

    @staticmethod
    def plot_from_dicts(blades_h, blades_v, range_h, range_v,
                        beamline="", figsize=(10, 5)):
        """Create blade current figure from in-memory dicts (live analysis)."""
        return BladeCurrentVisualizer._plot_blades_common(
            blades_h, blades_v, range_h, range_v,
            attrs_h=None, attrs_v=None, beamline=beamline, figsize=figsize,
        )

    @staticmethod
    def plot_from_hdf5(h_data, v_data, figsize=(10, 5)):
        """Create blade current figure from HDF5 datasets.

        Args:
            h_data: HDF5 dataset for horizontal blades (structured array)
            v_data: HDF5 dataset for vertical blades (structured array)
            figsize: Figure size tuple

        Returns:
            matplotlib.figure.Figure
        """
        # Convert HDF5 datasets into blade dicts + attrs, then reuse common path
        blades_h = None
        blades_v = None
        attrs_h = None
        attrs_v = None

        if h_data is not None:
            attrs_h = dict(h_data.attrs)
            rng = h_data['x_index'][:]
            blades_h = {}
            for blade_name in ['to', 'ti', 'bi', 'bo']:
                vals = h_data[blade_name][:]
                err_name = f's_{blade_name}'
                errs = h_data[err_name][:] if err_name in h_data.dtype.names else None
                if errs is not None:
                    blades_h[blade_name] = np.column_stack([vals, errs])
                else:
                    blades_h[blade_name] = vals
            range_h = rng
        else:
            range_h = None

        if v_data is not None:
            attrs_v = dict(v_data.attrs)
            rng = v_data['y_index'][:]
            blades_v = {}
            for blade_name in ['to', 'ti', 'bi', 'bo']:
                vals = v_data[blade_name][:]
                err_name = f's_{blade_name}'
                errs = v_data[err_name][:] if err_name in v_data.dtype.names else None
                if errs is not None:
                    blades_v[blade_name] = np.column_stack([vals, errs])
                else:
                    blades_v[blade_name] = vals
            range_v = rng
        else:
            range_v = None

        return BladeCurrentVisualizer._plot_blades_common(
            blades_h, blades_v, range_h, range_v,
            attrs_h=attrs_h, attrs_v=attrs_v, beamline="", figsize=figsize,
        )

    @staticmethod
    def _plot_blades_common(blades_h, blades_v, range_h, range_v,
                            attrs_h=None, attrs_v=None,
                            beamline="", figsize=(10, 5)):
        """Shared plotting path for blade currents (live and HDF5)."""
        # Keep fit lines consistent (solid) for both live and HDF5 paths
        fit_style = {"linestyle": "-", "linewidth": 1.4, "alpha": 0.8, "zorder": 5}

        def _plot_side(ax, blades, rng, attrs, marker_map, xlab_default):
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

                if yerr is not None:
                    ax.errorbar(rng, y, yerr=yerr, fmt=marker, label=blade_name)
                else:
                    ax.plot(rng, y, marker, label=blade_name)

                # Fit only if variation exists
                if np.any(np.isfinite(y)) and np.nanstd(y) > 0 and len(rng) > 1:
                    try:
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
                            if yerr is not None and np.all(np.isfinite(yerr)) and np.all(yerr > 0):
                                weights = 1.0 / yerr
                            coef = np.polyfit(rng, y, deg=1, w=weights)

                        style = dict(fit_style)
                        ax.plot(rng, coef[0] * rng + coef[1], label=f"{blade_name} fit", **style)
                    except Exception:
                        pass

            ax.set_xlabel(xlab_default)
            ax.set_ylabel('I')
            ax.grid()
            ax.legend()

        fig, (axh, axv) = plt.subplots(1, 2, figsize=figsize)

        _plot_side(axh, blades_h, range_h, attrs_h,
                   [('to', 'o-'), ('ti', 's-'), ('bi', 'd-'), ('bo', '^-')],
                   (attrs_h or {}).get('xlabel_blades', 'x [μrad]'))
        _plot_side(axv, blades_v, range_v, attrs_v,
                   [('to', 'o-'), ('ti', 's-'), ('bi', 'd-'), ('bo', 'v-')],
                   (attrs_v or {}).get('xlabel_blades', 'y [μrad]'))

        ylabel = (u"$I$ [# counts]" if beamline[:3] in ["MGN", "MNC"]
                 else u"$I$ [A]")
        axh.set_ylabel(ylabel)
        axv.set_ylabel(ylabel)
        axh.set_title('Horizontal')
        axv.set_title('Vertical')

        fig.suptitle(
            'Blade Currents at Center', fontsize=12, fontweight='bold'
        )
        fig.tight_layout()
        return fig
