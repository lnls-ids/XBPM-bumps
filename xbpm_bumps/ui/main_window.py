"""Main window for XBPM analysis application."""

from typing import Callable
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QSplitter, QTabWidget,
    QStatusBar, QProgressBar, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui  import QFont
from PyQt5.QtGui  import QCloseEvent

import numpy as np
import logging
import os
# import traceback

from .widgets.parameter_panel import ParameterPanel
from .widgets.mpl_canvas      import MatplotlibCanvas
from .dialogs.beamline_dialog import BeamlineSelectionDialog
from .dialogs.help_dialog     import HelpDialog
from ..core.config            import Config
from ..core.constants         import FIGDPI
from ..core.reader_hdf5       import read_hdf5
from ..core.analysis_service  import AnalysisService
from ..core import data_structure as DStr

logger = logging.getLogger(__name__)


class XBPMMainWindow(QMainWindow):
    """Main application window for XBPM beam position analysis.

    Provides interface for:
    - Parameter configuration
    - Analysis execution
    - Progress monitoring
    - Result visualization
    """
    ANALYSIS_SECTION_TITLES = {
        'positions'       : 'Positions',
        'sweep_positions' : 'Sweep Positions',
        'blade_sweeps'    : 'Blades at Sweeps',
        'bpm'             : 'BPM',
    }

    BPM_STATS_DESCRIPTIONS = {
        'sigma_h'     : 'Horizontal RMS pos. difference ',
        'sigma_v'     : '  Vertical RMS pos. difference ',
        'sigma_total' : '     Total RMS pos. difference ',
        'diff_max_h'  : 'Max hor.  |x_meas - x_nom| [μm]',
        'diff_max_v'  : 'Max vert. |y_meas - y_nom| [μm]',
    }

    def __init__(self: "XBPMMainWindow") -> None:
        """Initialize the main window."""
        super().__init__()
        self.canvases          = {}
        self.beamlinedata      = None  # Canonical DataReader instance
        self.workbeamline      = None
        self.workdata          = None  # Effective BeamlineData instance
        self._last_inputfile   = ""
        self.results           = {}    # Single unified results storage
        self._last_roisize     = None
        self._analysis_running = False
        # self.grid_shape        : tuple[int, int] | None = None
        self.setup_ui()
        self.setWindowTitle("XBPM Calibration and Analysis Tool")
        # Wider default window to give canvases more horizontal room
        self.resize(1920, 1080)

    @pyqtSlot()
    def _on_run_clicked(self) -> None:
        if self.workdata is None:
            self.show_error("No data loaded", "Open an HDF5 file first.")
            return

        self._on_parameters_changed()  # synchronize final widget state
        self.set_analysis_running(True)

        try:
            analysis = AnalysisService.run(
                self.workdata,
                self.runtime_prm,
            )
        except Exception as exc:
            msg = f"{str(exc)}"   # + "\n(workdata: {type(self.workdata)})"
            self.show_error("Analysis failed", msg)
            return
        finally:
            self.set_analysis_running(False)

        self.analysis = analysis
        self.log_message("Analysis completed.")

    def setup_ui(self) -> None:
        """Initialize the main window layout."""
        # Central widget with splitter
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Create main splitter (left: controls, right: results)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: parameters and controls
        left_panel = self._create_control_panel()
        splitter.addWidget(left_panel)

        # Right panel: results tabs
        right_panel = self._create_results_panel()
        splitter.addWidget(right_panel)

        # Refresh analysis info when tabs change
        self.results_tabs.currentChanged.connect(self._on_tab_changed)

        # Set initial splitter sizes (25% controls, 75% results) for
        # wider canvases
        splitter.setSizes([400, 1200])

        # Status bar
        self._create_status_bar()

        # Menubar: File
        file_menu = self.menuBar().addMenu("File")
        # open_dir_action = file_menu.addAction("Open Directory…")
        # open_dir_action.triggered.connect(self._on_open_directory)

        open_hdf5_action = file_menu.addAction("Open HDF5 File…")
        open_hdf5_action.triggered.connect(self._on_open_hdf5)

        file_menu.addSeparator()
        export_hdf5_action = file_menu.addAction("Export to HDF5…")
        export_hdf5_action.triggered.connect(self._on_export_hdf5_clicked)

        # WARNING: must be implemented before enabling export to txt/png.
        export_action = file_menu.addAction("Export to txt/png…")
        export_action.setEnabled(False)
        # export_action.triggered.connect(self._on_export_clicked)

        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(self.close)

        # Menubar: Help
        help_menu = self.menuBar().addMenu("Help")
        help_action = help_menu.addAction("Help…")
        help_action.triggered.connect(self._on_help_clicked)

    def _create_control_panel(self) -> QWidget:
        """Create the left control panel with parameters, info, and buttons."""
        from PyQt5.QtWidgets import QScrollArea

        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Parameter input panel with scroll area
        self.param_panel = ParameterPanel()
        self.param_panel.parametersChanged.connect(
            self._on_parameters_changed
        )

        # Wrap parameter panel in scroll area to handle many widgets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.param_panel)
        layout.addWidget(scroll)

        # Analysis info box (read-only, compact)
        self.analysis_info = QTextEdit()
        self.analysis_info.setReadOnly(True)
        self.analysis_info.setMinimumHeight(220)
        self.analysis_info.setPlaceholderText(
            "Analysis info (scales, sweeps, BPM stats) will appear here."
        )
        layout.addWidget(self.analysis_info)

        # Control buttons
        button_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._on_run_clicked)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)

        self.quit_btn = QPushButton("Quit")
        self.quit_btn.setMinimumHeight(40)
        self.quit_btn.clicked.connect(self.close)

        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.quit_btn)
        layout.addLayout(button_layout)

        return panel

    def _create_results_panel(self) -> QWidget:
        """Create the right panel with tabs for different result views."""
        self.results_tabs = QTabWidget()

        # Console/Log tab
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Courier", 9))
        self.results_tabs.addTab(self.console, "Console")

        # Visualization tabs (ordered to match analysis options)
        bpm_tab, bpm_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(bpm_tab, "BPM")
        self.canvases["bpm"] = bpm_canvas

        blade_tab, blade_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(blade_tab, "Blade Map")
        self.canvases["blade"] = blade_canvas

        blades_center_tab, blades_center_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(blades_center_tab, "Blades at sweeps")
        self.canvases["blades_center"] = blades_center_canvas

        # Move the sweeps tab after blades_center
        sweep_tab, sweep_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(sweep_tab, "Positions along sweeps")
        self.canvases["sweeps"] = sweep_canvas

        xbpm_raw_pw_tab, xbpm_raw_pw_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_raw_pw_tab, "XBPM Δ/Σ raw")
        self.canvases["xbpm_raw_pairwise"] = xbpm_raw_pw_canvas

        xbpm_scaled_pw_tab, xbpm_scaled_pw_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_scaled_pw_tab, "XBPM Δ/Σ Sup. Mat.")
        self.canvases["xbpm_scaled_pairwise"] = xbpm_scaled_pw_canvas

        xbpm_raw_cr_tab, xbpm_raw_cr_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_raw_cr_tab, "XBPM part. Δ/Σ - raw")
        self.canvases["xbpm_raw_cross"] = xbpm_raw_cr_canvas

        xbpm_scaled_cr_tab, xbpm_scaled_cr_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_scaled_cr_tab, "XBPM part. Δ/Σ - LinTr")
        self.canvases["xbpm_scaled_cross"] = xbpm_scaled_cr_canvas

        return self.results_tabs

    @pyqtSlot(str)
    def log_message(self, message: str) -> None:
        """Append a message to the console log.

        Args:
            message: Text to append to console.
        """
        self.console.append(message)

    @pyqtSlot(str, str)
    def show_error(self,
                   title: str,
                   message: str
                   ) -> None:
        """Display error dialog.

        Args:
            title: Error dialog title.
            message: Error message text.
        """
        QMessageBox.critical(self, title, message)
        self.log_message(f"ERROR: {message}")

    @pyqtSlot()
    def _on_open_hdf5(self) -> None:
        """Open dialog to select HDF5 data file, read data and select beamline.

        (Routes through Analyzer for beamline selection.)
        """
        h5file, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 File",
            os.getcwd(),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )

        # Empty pathnames.
        if not h5file:
            return

        # Validate selected path and read data.
        try:
            self.runtime_prm, self.beamlinedata = read_hdf5(h5file)
        except OSError as exc:
            self.show_error(
                f"cannot open HDF5 file {h5file}:", f"\n{str(exc)}"
                )
            return

        # Store inputfile in parameter panel and update status bar
        self.param_panel.show_inputfile(h5file)
        self.status_bar.showMessage(f"Opened: {h5file}")

        # Select beamline and create effective links to
        # data and analysis objects.
        self._select_beamline()

        # Define links to effective beamline data. 
        self.workdata     = self.beamlinedata[self.workbeamline]
        self.analysis     = self.workdata.analysis
        self.beamline_prm : DStr.BeamlinePrm = self.workdata.prm

        # Update BPM distance.
        self.beamline_prm.bpmdist = Config.BPMDISTS.get(
            self.workbeamline[:3], None
            )

        # Calculate grid shape from nominal positions.
        nom_pos = self.workdata.raw_data.blade_avg.pos_nom
        self.grid_shape = (
            len(np.unique(nom_pos.y)),  # vertical dimension
            len(np.unique(nom_pos.x)),  # horizontal dimension
        )

        # Update parameter panel.
        self.param_panel.load_beamline_data(
            self.runtime_prm,
            self.beamline_prm,
            self.grid_shape,
        )

        self.log_message(
            f"Loading data from: {h5file} "
            f"(beamline: {self.workbeamline})"
        )

    def _select_beamline(self) -> str:
        """Centralized beamline selection: returns the chosen beamline."""
        # Set beamline list from dataset keys.
        self.beamlines = list(self.beamlinedata.keys())

        if len(self.beamlines) == 1:
            self.workbeamline = self.beamlines[0]
            self.log_message(f"Auto-selected beamline: {self.workbeamline}")
        else:
            dialog = BeamlineSelectionDialog(sorted(self.beamlines))
            if dialog.exec_() != dialog.Accepted:
                raise RuntimeError("Beamline selection cancelled by user.")
            self.workbeamline = dialog.get_selection()
            if not self.workbeamline:
                raise RuntimeError("No beamline selected.")
            self.log_message(f"Selected beamline: {self.workbeamline}")

        # Persist in parameter panel for future get_parameters() calls
        self.param_panel.set_beamline(self.workbeamline)

    def _update_xbpmdist_from_beamline(self) -> None:
        """Update the XBPM distance field from Config.XBPMDISTS."""
        try:
            dist = Config.XBPMDISTS.get(self.workbeamline)
            self.beamline_prm.xbpmdist = dist
            self.param_panel.xbpmdist_spin.setValue(float(dist))
            self.log_message(
                "XBPM distance set from beamline"
                f" {self.workbeamline}:"
                f" {dist:.3f} m"
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.log_message(f"Could not set XBPM distance: {exc}")

    @pyqtSlot()
    def _on_parameters_changed(self) -> None:
        """React to parameter changes; pre-select beamline on input set."""
        # If data was not imported yet.
        if self.workdata is None or self.grid_shape is None:
            return

        # Check and reread parameter set.
        params = self.param_panel.get_parameters()

        # Reset prm values from parameter panel.
        # Beamline parameters.
        bl_prm = self.beamline_prm
        bl_prm.xbpmdist     = params["xbpmdist"]
        bl_prm.skip         = params["skip"]
        bl_prm.scalepolydeg = params["scalepolydeg"]
        bl_prm.usebpmref    = params["usebpmref"]
        bl_prm.roi          = DStr.ROISlice.update(
            self.grid_shape,
            params["roisize"]
            )

        # Runtime parameters.
        rt_prm = self.runtime_prm
        rt_prm.show_bladecenter      = params["show_bladecenter"]
        rt_prm.show_blademap         = params["show_blademap"]
        rt_prm.show_bpmpositions     = params["show_bpmpositions"]
        rt_prm.show_centralsweep     = params["show_centralsweep"]
        rt_prm.show_xbpmpositionsraw = params["show_xbpmpositionsraw"]
        rt_prm.show_xbpmpositions    = params["show_xbpmpositions"]

    def _create_status_bar(self) -> None:
        """Create status bar with progress indicator."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.status_bar.showMessage("Ready")

    @pyqtSlot()
    def _on_export_clicked(self) -> None:
        """Handle Export button: write data/positions to user-chosen files."""
        try:
            # Ensure app is initialized and beamline selected
            if not hasattr(self, 'analyzer') or not self.analyzer.app:
                QMessageBox.warning(
                    self,
                    "Unavailable",
                    (
                        "Run analysis at least once"
                        " before exporting."
                    ),
                )
                return

            # Choose base filename prefix
            default_name = (
                f"xbpm_{self.workbeamline}.dat"
            )
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Choose export base filename",
                default_name,
                "Data Files (*.dat);;All Files (*)"
            )
            if not path:
                return

            # Strip extension to use as prefix
            prefix, _ext = os.path.splitext(path)

            params  = self.param_panel.get_parameters()
            results = getattr(self, '_last_results', {})
            exported_any = False

            # Always export suppression matrices (independent of checkboxes)
            # exported_any |= self._export_suppression_matrices(prefix, results)

            # # Always export BPM positions when they were computed
            # exported_any |= self._export_bpm_positions(prefix, results)

            # # Export XBPM data and figures for both raw and scaled scopes
            # exported_any |= self._export_xbpm_positions('raw', prefix,
            #                                             params, results)
            # exported_any |= self._export_xbpm_positions('scaled', prefix,
            #                                             params, results)

            # Export other analysis figures and data
            # exported_any |= self._export_other_figures(prefix, params, results)
            # exported_any |= self._export_analysis_info(prefix)

            if not exported_any:
                QMessageBox.information(
                    self,
                    "Nothing to export",
                    (
                        "Enable at least one analysis option "
                        "to export results."
                    ),
                )
                return

            self.log_message(f"Exported data using prefix: {prefix}")
            QMessageBox.information(
                self,
                "Export Complete",
                "Data and figures export finished.",
            )
        except Exception as exc:  # pragma: no cover
            self.show_error("Export Failed", str(exc))

    # Print quality export helper, for later reuse.
    def _save_figure_for_export(self, fig, path: str) -> None:
        """Save figure with print-friendly typography without changing UI use.

        The live app keeps its current figure sizing. Export applies a
        temporary font boost but preserves figure geometry so subplot spacing
        remains consistent with what is shown in the UI.
        """
        original_size = fig.get_size_inches().copy()
        axes = fig.get_axes()
        axis_state = []
        legend_state = []
        text_state = []

        width, height = original_size
        target_width = min(width, 10.0)
        target_height = (height * (target_width / width)
                 if width > 0
                 else height)
        fig.set_size_inches(target_width, target_height, forward=False)

        for ax in axes:
            axis_state.append({
                'title'  : ax.title.get_fontsize(),
                'xlabel' : ax.xaxis.label.get_fontsize(),
                'ylabel' : ax.yaxis.label.get_fontsize(),
                'xtick'  : ax.xaxis.get_ticklabels()[0].get_fontsize()
                            if ax.xaxis.get_ticklabels() else None,
                'ytick'  : ax.yaxis.get_ticklabels()[0].get_fontsize()
                            if ax.yaxis.get_ticklabels() else None,
            })

            # Keep former tuned proportions; only reduce title size a bit
            # to avoid collisions on 3-panel layouts.
            title_scale = 0.90
            title_min = 9
            title_max = 10
            title_text = ax.get_title() or ""
            if len(title_text) > 42:
                title_max = 9
            ax.title.set_fontsize(
                min(max(ax.title.get_fontsize() * title_scale, title_min),
                    title_max)
            )
            ax.xaxis.label.set_fontsize(
                max(ax.xaxis.label.get_fontsize() * 1.20, 12)
            )
            ax.yaxis.label.set_fontsize(
                max(ax.yaxis.label.get_fontsize() * 1.20, 12)
            )
            ax.tick_params(axis='both', which='major', labelsize=11)

            legend = ax.get_legend()
            if legend is not None:
                sizes = [text.get_fontsize() for text in legend.get_texts()]
                legend_state.append((legend, sizes))
                for text in legend.get_texts():
                    text.set_fontsize(max(text.get_fontsize() * 0.75, 9))

        for text in fig.findobj(match=lambda obj: hasattr(obj, 'get_text')):
            try:
                label = text.get_text()
            except Exception:  # noqa: S110
                continue
            if label == "RMS Difference [$\\mu$m]":
                text_state.append((text, text.get_fontsize()))
                text.set_fontsize(max(text.get_fontsize() * 0.8, 11))

        # Keep canvas geometry untouched to preserve constrained-layout
        # subplot spacing in exported PNGs (WYSIWYG with live tabs).
        fig.canvas.draw()
        fig.savefig(path, dpi=FIGDPI, bbox_inches='tight')

        for ax, state in zip(axes, axis_state):
            ax.title.set_fontsize(state['title'])
            ax.xaxis.label.set_fontsize(state['xlabel'])
            ax.yaxis.label.set_fontsize(state['ylabel'])
            if state['xtick'] is not None:
                ax.tick_params(axis='x', which='major', labelsize=state['xtick'])
            if state['ytick'] is not None:
                ax.tick_params(axis='y', which='major', labelsize=state['ytick'])

        for legend, sizes in legend_state:
            for text, size in zip(legend.get_texts(), sizes):
                text.set_fontsize(size)

        for text, size in text_state:
            text.set_fontsize(size)

        fig.set_size_inches(original_size[0], original_size[1], forward=False)

    @pyqtSlot()
    def _on_export_hdf5_clicked(self) -> None:
        """Export data to HDF5 file (with or without analysis results)."""
        # Ensure data is loaded (analysis is optional)
        if self.workdata is None:
            QMessageBox.warning(
                self,
                "No Data Loaded",
                (
                    "Please load data first.\n"
                    "Use 'Open HDF5 file' to load blade measurement data."
                ),
            )
            return

        try:
            default_name = (
                f"xbpm_{self.workbeamline}.h5"
            )
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export to HDF5",
                default_name,
                "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
            )
            if not path:
                return

            # Export using current data and last results
            from ..core.exporters import Exporter
            exporter = Exporter(self.workbeamline)

            # Include raw_data for complete re-analysis capability
            raw_data = getattr(self.workdata, 'raw_data', None)
            exporter.write_hdf5(
                path,
                self.workdata,
                self.analysis,
                include_figures=True,
                raw_data=raw_data
                )

            self.log_message(f"HDF5 export written: {path}")
            QMessageBox.information(
                self,
                "Export Complete",
                "Exported analysis and figures to HDF5.",
            )
        except Exception as exc:  # pragma: no cover
            self.show_error("Export to HDF5 Failed", str(exc))

    @pyqtSlot()
    def _on_help_clicked(self) -> None:
        """Open Help dialog with program guidance (non-blocking)."""
        try:
            if not hasattr(self, '_help_dialog') or self._help_dialog is None:
                self._help_dialog = HelpDialog(self)
            self._help_dialog.show()
            self._help_dialog.raise_()
            self._help_dialog.activateWindow()
            self.log_message("Help dialog opened")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to open Help dialog")
            self.show_error("Help", f"Could not open Help: {exc}")


    @pyqtSlot(bool)
    def set_analysis_running(self, running: bool) -> None:
        """Update UI state during analysis execution.

        Args:
            running: True if analysis is running, False otherwise.
        """
        self._analysis_running = running
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.param_panel.setEnabled(not running)
        self.progress_bar.setVisible(running)

        if running:
            self.status_bar.showMessage("Analysis running...")
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
        else:
            self.status_bar.showMessage("Ready")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)

    @pyqtSlot(str)
    def show_results_tab(self, tab_name: str) -> None:
        """Switch to a specific results tab.

        Args:
            tab_name: Name of the tab to show.
        """
        for i in range(self.results_tabs.count()):
            if self.results_tabs.tabText(i) == tab_name:
                self.results_tabs.setCurrentIndex(i)
                break

    def _create_canvas_tab(self) -> tuple[QWidget, MatplotlibCanvas]:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        canvas = MatplotlibCanvas()
        layout.addWidget(canvas)
        return widget, canvas

    def _tab_to_section(self, tab_text: str) -> str:
        text = (tab_text or "").lower()
        if 'blade map' in text:
            return 'none'
        if 'blades at' in text or 'blades at sweeps' in text:
            return 'blade_sweeps'
        if ('positions along sweep' in text or
            'positions along sweeps' in text):
            return 'sweep_positions'
        if 'sweep' in text:
            return 'sweeps'
        if 'xbpm' in text:
            return 'positions'
        if 'bpm' in text:
            return 'bpm'
        return ''

    def _tab_position_filter(self, tab_text: str):
        """Return (scope, label) filter for XBPM tabs or None."""
        text = (tab_text or "").lower()
        scope = None
        label = None

        if 'raw' in text:
            scope = 'raw'
        if 'scaled' in text or ' tr' in text:
            scope = 'scaled'
        if 'pair' in text:
            label = 'pair'
        if 'cross' in text or 'part.' in text:
            label = 'cross'

        # Pairwise tabs are named "XBPM Δ/Σ raw|Tr" and do not include
        # the literal word "pair".
        if 'xbpm' in text and label is None:
            label = 'pair'

        if scope or label:
            return scope, label
        return None

    def _format_analysis_info(self, active_tab: str) -> str:
        """Format analysis metadata for UI display.

        Delegates to helper methods to reduce complexity.

        Args:
            active_tab: Name of the currently active results tab.

        Returns:
            Formatted string for display in the read-only analysis info panel.
        """
        if not self.results:
            return "No analysis metadata available yet."

        active_section = self._tab_to_section(active_tab)
        pos_filter = self._tab_position_filter(active_tab)
        if active_section == 'none':
            return ""

        sections: dict[str, list[str]] = {}
        sections.update(self._format_scales_section(pos_filter))
        sections.update(self._format_sweeps_positions_section())
        sections.update(self._format_blades_section())
        sections.update(self._format_bpm_stats_section())

        # Add XBPM stats to positions section if present
        xbpm_stats_dict = self._format_xbpm_stats_section(active_tab)
        if xbpm_stats_dict:
            sections.setdefault('positions', []).extend(
                xbpm_stats_dict.get('xbpm', [])
                )

        supmat_lines = self._format_supmat_lines(active_tab)
        if supmat_lines:
            sections.setdefault('positions', []).extend(supmat_lines)

        return self._format_sections_output(sections, active_section)

    def _format_scales_section(self, pos_filter=None) -> dict[str, list[str]]:
        """Format scales (positions) metadata section.
        
        Inlines coefficient and error formatting for cleaner code.
        Supports both legacy (s_kx/s_dx) and current (skx/sdx) error key formats.

        Args:
            pos_filter: Optional tuple of (scope, label) to filter scales.

        Returns:
            Dictionary with 'positions' key containing formatted lines.
        """
        scale_lines: list[str] = []

        # Build unified scales view from both legacy and
        # current result layouts.
        scales: dict[str, dict] = {}

        if isinstance(self.results, dict):
            legacy_scales = self.results.get('scales')
            if isinstance(legacy_scales, dict):
                for scope in ('raw', 'scaled'):
                    block = legacy_scales.get(scope)
                    if isinstance(block, dict):
                        scales[scope] = block

            raw_full = self.results.get('positions_raw_full')
            if isinstance(raw_full, dict):
                raw_scales = raw_full.get('scales')
                if isinstance(raw_scales, dict):
                    scales['raw'] = raw_scales

            scaled_full = self.results.get('positions_scaled_full')
            if isinstance(scaled_full, dict):
                scaled_scales = scaled_full.get('scales')
                if isinstance(scaled_scales, dict):
                    scales['scaled'] = scaled_scales

        for scope in ('scaled', 'raw'):
            scope_block = scales.get(scope)
            if not isinstance(scope_block, dict):
                continue
            for label, coeffs in scope_block.items():
                if not isinstance(coeffs, dict):
                    continue
                if pos_filter:
                    filt_scope, filt_label = pos_filter
                    if filt_scope and scope != filt_scope:
                        continue
                    if filt_label and label != filt_label:
                        continue

                # Format coefficient lines inline
                lines_to_add: list[str] = []

                # Format qx/kx/dx pair
                coeffnames1 = (
                    ('qx', ('sqx', 's_qx')),
                    ('kx', ('skx', 's_kx')),
                    ('dx', ('sdx', 's_dx')),
                )
                line1 = self._format_coefficent_lines(coeffnames1, coeffs)

                # Format qy/ky/dy pair
                coeffnames2 = (
                    ('qy', ('sqy', 's_qy')),
                    ('ky', ('sky', 's_ky')),
                    ('dy', ('sdy', 's_dy')),
                )
                line2 = self._format_coefficent_lines(coeffnames2, coeffs)

                if line1:
                    lines_to_add.append(",\n".join(line1))
                if line2:
                    lines_to_add.append("\n" + ",\n".join(line2))

                if lines_to_add:
                    subject = Config.get_position_subject(scope, label)
                    scale_lines.append(f"  * {subject}:")
                    scale_lines.extend(lines_to_add)

        return {'positions': scale_lines} if scale_lines else {}

    def _format_coefficent_lines(self, coeffnames: tuple,
                                 coeffs: dict) -> list[str]:
        """Format coefficient lines for a given coefficient set.

        Args:
            coeffnames  : Tuple of polynomial coefficient strings.
            coeffs    : Dictionary of coefficient values.

        Returns:
            List of formatted coefficient lines.
        """
        line = []
        # Format k/d pair
        for key, err_keys in coeffnames:
            val = coeffs.get(key)
            if val is not None:
                err = None
                for ek in (err_keys
                           if isinstance(err_keys, (tuple, list))
                           else (err_keys,)):
                    if ek in coeffs and coeffs.get(ek) is not None:
                        err = coeffs.get(ek)
                        break
                try:
                    val_num = float(val)
                    if err is not None:
                        rel_err = float(err) / abs(val_num)
                        line.append(
                            f"{key:>8} = {val_num:8.3g}  "
                            f"({float(err):2.0e} : {rel_err:.2%})"
                            )
                    else:
                        line.append(f"{key:>8} =  {val_num:8.3g}")
                except Exception:
                    if err is not None:
                        line.append(f"{key:>8} =  {val} ({err})")
                    else:
                        line.append(f"{key:>8} =  {val}")
        return line

    def _format_sweeps_positions_section(self) -> dict[str, list[str]]:
        """Format sweeps positions (global fits) metadata section.
        
        Inlines fit entry formatting for cleaner code structure.
        """
        sweeps_pos_lines: list[str] = []
        sweeps = self.results.get('sweeps', {}) if isinstance(self.results, dict) else {}
        positions_meta = (sweeps.get('positions', {})
                         if isinstance(sweeps, dict) else {})

        for orient, label in (('horizontal', ' H '), ('vertical', ' V ')):
            fit = positions_meta.get(orient)
            if not isinstance(fit, dict):
                continue

            # Format fit entry inline
            lines_to_add: list[str] = []
            line1 = []
            line2 = []

            for key, bucket in (('k', line1), ('delta', line1),
                               ('s_k', line2), ('s_delta', line2)):
                if key in fit and fit[key] is not None:
                    try:
                        bucket.append(f"{key:>10} = {float(fit[key]):.4g}")
                    except Exception:
                        bucket.append(f"{key:>10} = {fit[key]}")

            if line1:
                lines_to_add.append("  " + ",  ".join(line1))
            if line2:
                lines_to_add.append("  " + ",  ".join(line2))

            if lines_to_add:
                sweeps_pos_lines.append(f"  * {label}:")
                sweeps_pos_lines.extend(lines_to_add)

        return ({'sweep_positions': sweeps_pos_lines}
                if sweeps_pos_lines else {})

    def _format_blades_section(self) -> dict[str, list[str]]:
        """Format blades-at-sweeps per-blade fits metadata section.
        
        Inlines blade fit entry formatting for cleaner code structure.
        """
        blades_lines: list[str] = []
        sweeps = self.results.get('sweeps', {}) if isinstance(self.results, dict) else {}
        blades_meta = (sweeps.get('blades', {})
                       if isinstance(sweeps, dict) else {})

        for orient, label in (('horizontal', 'H'), ('vertical', 'V')):
            bfits = blades_meta.get(orient)
            if not isinstance(bfits, dict):
                continue

            for blade, fit in bfits.items():
                if not isinstance(fit, dict):
                    continue

                # Format blade fit inline
                parts = []
                for key in ('k', 'delta'):
                    if key in fit and fit[key] is not None:
                        try:
                            parts.append(f"{key}={float(fit[key]):.4g}")
                        except Exception:
                            parts.append(f"{key}={fit[key]}")

                if parts:
                    blades_lines.append(f"  * {label} {blade}:")
                    blades_lines.append("   " + ", ".join(parts))

        return {'blade_sweeps': blades_lines} if blades_lines else {}

    def _format_bpm_stats_section(self) -> dict[str, list[str]]:
        """Format BPM statistics metadata section."""
        bpm_lines: list[str] = []
        bpm_stats = (self.results.get('bpm_stats', {})
                     if isinstance(self.results, dict) else {})

        if isinstance(bpm_stats, dict):
            bpm_lines.append(
                "  ROI size [lines x columns points] ="
                f" {self.beamline_prm.roi.sz_v} x {self.beamline_prm.roi.sz_h}"
            )
            bpm_lines.append("\n  Sigmas (all sites):")
            for key in ('sigma_h', 'sigma_v', 'sigma_total'):
                if key in bpm_stats:
                    entry = self.BPM_STATS_DESCRIPTIONS.get(key, key)
                    try:
                        bpm_lines.append(
                            f"  {entry:>17} = {float(bpm_stats[key]):.4g}"
                        )
                    except Exception:
                        bpm_lines.append(f"  {entry:>17} = {bpm_stats[key]}")

            bpm_lines.append("\n  Extremes (all sites):")
            for key in ('diff_max_h', 'diff_max_v'):
                if key in bpm_stats:
                    entry = self.BPM_STATS_DESCRIPTIONS.get(key, key)
                    try:
                        bpm_lines.append(
                            f"  {entry:>17} = {float(bpm_stats[key]):.4g}"
                        )
                    except Exception:
                        bpm_lines.append(f"  {entry:>17} = {bpm_stats[key]}")

            if bpm_stats.get('roi_available'):
                bpm_lines.append("\n  Sigmas (ROI):")
                roi_sig_h = bpm_stats.get('roi_sigma_h')
                roi_sig_v = bpm_stats.get('roi_sigma_v')
                roi_sig_t = bpm_stats.get('roi_sigma_total')
                roi_lines = [
                    f" {'ROI horizontal RMS':>17} = {float(roi_sig_h):.4g}",
                    f" {'ROI vertical RMS':>17} = {float(roi_sig_v):.4g}",
                    f" {'ROI total RMS':>17} = {float(roi_sig_t):.4g}"
                ]
                bpm_lines += roi_lines
        return {'bpm': bpm_lines}

    def _format_xbpm_stats_section(self,
                                   active_tab: str
                                   ) -> dict[str, list[str]]:
        """Format XBPM statistics metadata section.

        Only shows the relevant calculation type based on active tab:
        - Pairwise stats for tabs containing 'pair'
        - Cross-blade stats for tabs containing 'cross'
        """
        xbpm_lines: list[str] = []
        text = (active_tab or "").lower()

        # Determine which stats to display based on active tab
        xbpm_stats = None
        if 'raw' in text:
            xbpm_stats = (self.results.get('xbpm_stats_raw', {})
                          if isinstance(self.results, dict) else {})
        elif 'scaled' in text or ' tr' in text:
            xbpm_stats = (self.results.get('xbpm_stats_scaled', {})
                          if isinstance(self.results, dict) else {})
        else:
            xbpm_stats = {}

        if not isinstance(xbpm_stats, dict) or not xbpm_stats:
            return {}

        # Determine which calculation type to display based on tab name
        calc_type = None
        if 'pair' in text:
            calc_type = 'pairwise'
        elif 'cross' in text or 'part.' in text:
            calc_type = 'cross'

        # Default XBPM Δ/Σ raw|Tr tabs are pairwise.
        if calc_type is None and 'xbpm' in text:
            calc_type = 'pairwise'

        if not calc_type:
            return {}

        calc_stats = xbpm_stats.get(calc_type, {})
        if not isinstance(calc_stats, dict) or not calc_stats:
            return {}

        # Format statistics similar to BPM _std_dev_estimate print output
        # All statistics should always be present in calc_stats dictionary
        try:
            xbpm_lines.append("")
            xbpm_lines.append("  Sigmas (RMS differences):")
            xbpm_lines.append(f"     H = {float(calc_stats['sigma_h']):.4f}")
            xbpm_lines.append(f"     V = {float(calc_stats['sigma_v']):.4f}")
            xbpm_lines.append(
                f" total = {float(calc_stats['sigma_total']):.4f}"
            )

            xbpm_lines.append("")
            xbpm_lines.append("  Maximum difference:")
            xbpm_lines.append(
                f"     H = {float(calc_stats['diff_max_h']):.4f}"
            )
            xbpm_lines.append(
                f"     V = {float(calc_stats['diff_max_v']):.4f}"
            )

            xbpm_lines.append("")
            xbpm_lines.append("  Minimum difference:")
            xbpm_lines.append(
                f"     H = {float(calc_stats['diff_min_h']):.4f}"
            )
            xbpm_lines.append(
                f"     V = {float(calc_stats['diff_min_v']):.4f}"
            )
        except (KeyError, TypeError, ValueError):
            # If any key is missing or can't be converted, return empty
            return {}

        return {'xbpm': xbpm_lines} if xbpm_lines else {}

    def _format_supmat_lines(self, active_tab: str) -> list[str]:
        """Format suppression matrix lines for the active tab with uncertainties."""
        lines: list[str] = []
        text = (active_tab or "").lower()

        # Raw pairwise tab: show standard suppression matrix with uncertainties
        is_pairwise = ('pair' in text) or ('xbpm' in text and 'part.' not in text)
        if 'raw' in text and is_pairwise:
            supmat = self.results.get('supmat_standard')
            stddevmat = self.results.get('stddevmat_standard')
            if supmat is not None:
                lines.append("\n  ** Standard Suppression Matrix:")
                if stddevmat is not None:
                    lines.extend(self._format_matrix_with_uncertainties(supmat, stddevmat))
                else:
                    lines.extend(self._format_matrix(supmat))

        # Scaled pairwise tab: show calculated suppression matrix with uncertainties
        elif ('scaled' in text or ' tr' in text) and is_pairwise:
            supmat = self.results.get('supmat')
            stddevmat = self.results.get('stddevmat')
            if supmat is not None:
                lines.append("\n  ** Calculated Suppression Matrix:")
                if stddevmat is not None:
                    lines.extend(self._format_matrix_with_uncertainties(supmat, stddevmat))
                else:
                    lines.extend(self._format_matrix(supmat))

        return lines

    def _format_matrix(self, supmat) -> list[str]:
        """Pretty-print suppression matrix rows."""
        arr = np.asarray(supmat, dtype=float)
        return [
            "  " + " ".join(f"{val:8.4f}" for val in row)
            for row in arr
        ]

    def _format_matrix_with_uncertainties(self, supmat, stddevmat) -> list[str]:
        """Pretty-print suppression matrix rows with standard deviations.
        
        Args:
            supmat: Suppression matrix (4x4 array)
            stddevmat: Standard deviation matrix (4x4 array)
            
        Returns:
            List of formatted strings showing value (±uncertainty) format
        """
        arr = np.asarray(supmat, dtype=float)
        std = np.asarray(stddevmat, dtype=float)
        lines = []
        for ii, row in enumerate(arr):
            row_parts = []
            for jj, val in enumerate(row):
                uncertainty = std[ii, jj] if ii < std.shape[0] and jj < std.shape[1] else 0.0
                if uncertainty > 0:
                    row_parts.append(f"{val:8.2f} ({uncertainty:1.0e})")
                else:
                    row_parts.append(f"{val:8.2f} (0.0)")
            lines.append("  " + "  ".join(row_parts))
        return lines

    def _format_sections_output(self, sections: dict[str, list[str]],
                                active_section: str) -> str:
        """Format all sections into final output string."""
        ordered_sections = (
            [active_section] if active_section else list(sections.keys())
        )

        lines: list[str] = []
        for name in ordered_sections:
            content = sections.get(name)
            if not content:
                continue
            title = self.ANALYSIS_SECTION_TITLES.get(name, name.replace('_', ' ').title())
            lines.append(f"\n** {title}:")
            lines.extend(content)
            lines.append("")

        if lines and lines[-1] == "":
            lines.pop()

        if lines:
            return "\n".join(lines)

        if active_section:
            return "No metadata for this tab."
        return "No analysis metadata available yet."

    def _refresh_analysis_info(self, tab_index: int = None) -> None:
        """Update the analysis info panel based on the current tab.
        
        Args:
            tab_index: Optional index of the tab to refresh.
                       If None, uses the current tab index.
        """
        # If UI not fully built yet, skip
        if not hasattr(self, 'analysis_info') or self.analysis_info is None:
            return

        try:
            current_tab = (
                self.results_tabs.tabText(tab_index)
                if tab_index is not None else
                self.results_tabs.tabText(self.results_tabs.currentIndex())
            )
        except Exception:
            current_tab = ""
        text = self._format_analysis_info(current_tab)
        self.analysis_info.setText(text)

    @pyqtSlot(int)
    def _on_tab_changed(self, index: int):
        """Update analysis info when the active tab changes."""
        self._refresh_analysis_info(index)

    def _embed_figure(self, canvas: MatplotlibCanvas, source_fig):
        """Embed entire figure by replacing canvas figure.

        Args:
            canvas: Target MatplotlibCanvas widget.
            source_fig: Source matplotlib figure with content.
        """
        try:
            import matplotlib.pyplot as plt

            # Properly close old figure to prevent matplotlib state leaks
            if canvas.figure and canvas.figure != source_fig:
                try:
                    plt.close(canvas.figure)
                except Exception:  # pragma: no cover - defensive
                    logger.warning(
                        "Failed to close previous figure during embed",
                        exc_info=True,
                    )

            # Recreate canvas and toolbar for the new figure to keep
            # interactivity working
            layout = canvas.layout()
            if canvas.toolbar is not None:
                layout.removeWidget(canvas.toolbar)
                canvas.toolbar.setParent(None)
            if canvas.canvas is not None:
                layout.removeWidget(canvas.canvas)
                canvas.canvas.setParent(None)

            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg,
                NavigationToolbar2QT
                )

            canvas.figure = source_fig
            canvas.canvas = FigureCanvasQTAgg(canvas.figure)
            canvas.toolbar = NavigationToolbar2QT(canvas.canvas, canvas)

            layout.addWidget(canvas.toolbar)
            layout.addWidget(canvas.canvas)

            # Set figure DPI to match canvas DPI for proper scaling
            dpi = canvas.canvas.figure.dpi
            if dpi is None:
                dpi = 100
            canvas.figure.set_dpi(dpi)

            # Get canvas widget size and set figure size accordingly
            canvas_width = canvas.canvas.width()
            canvas_height = canvas.canvas.height()
            if canvas_width > 1 and canvas_height > 1:
                figsize_w = canvas_width / dpi
                figsize_h = canvas_height / dpi
                canvas.figure.set_size_inches(figsize_w, figsize_h)

            # Respect original figure layout (tight/constrained)
            # without overriding

            # Redraw
            canvas.canvas.draw_idle()
        except Exception as exc:  # pragma: no cover - defensive
            # Fallback: show error message
            try:
                canvas.ax.clear()
                canvas.ax.text(
                    0.5, 0.5, f"Figure embed error: {exc}",
                    ha='center', va='center',
                    transform=canvas.ax.transAxes,
                )
                canvas.canvas.draw_idle()
            except Exception:  # noqa: BLE001
                # Log fallback failure for debugging
                logger.exception("Fallback figure embed failed")

    def _show_figure_in_window(self, fig, title: str):
        """Display matplotlib figure in a separate popup window.

        Args:
            fig: Matplotlib figure object.
            title: Window title.
        """
        try:
            # Create popup window
            popup = QMainWindow()
            popup.setWindowTitle(title)
            popup.resize(1400, 700)  # Wider to maintain aspect ratio

            # Create canvas and embed figure
            canvas_widget = QWidget()
            layout = QVBoxLayout(canvas_widget)
            canvas = MatplotlibCanvas()
            layout.addWidget(canvas)
            popup.setCentralWidget(canvas_widget)

            # Embed figure
            self._embed_figure(canvas, fig)

            # Show window (non-blocking)
            popup.show()
            popup.raise_()
            popup.activateWindow()

            # Keep reference to prevent garbage collection
            if not hasattr(self, '_detail_windows'):
                self._detail_windows = []
            self._detail_windows.append(popup)

            logger.info("Displayed detail figure: %s", title)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to show detail figure %s", title)
            self.log_message(f"Error displaying {title}: {exc}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Clean up worker thread and detail windows on close."""
        # Close all detail windows
        if hasattr(self, '_detail_windows'):
            for window in self._detail_windows:
                try:
                    window.close()
                except Exception:  # pragma: no cover
                    logger.exception("Failed to close detail window")
        event.accept()
