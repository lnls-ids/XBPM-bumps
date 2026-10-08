"""Main window for XBPM analysis application."""

# from typing import Callable, Optional
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QSplitter, QTabWidget,
    QStatusBar, QProgressBar, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui  import QFont
from PyQt5.QtGui  import QCloseEvent

import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.figure import Figure

import numpy as np
import logging
import os

# import traceback

from .widgets.parameter_panel import ParameterPanel
from .widgets.mpl_canvas      import MatplotlibCanvas
from .dialogs.beamline_dialog import BeamlineSelectionDialog
from .dialogs.help_dialog     import HelpDialog
from ..core                   import data_structure as DStr
from ..core.analysis_info     import format_analysis_info
from ..core.config            import Config
from ..core.analysis_service  import AnalysisService
from ..core.reader_hdf5       import read_hdf5
from ..core.visualizers       import render_data_analysis

logger = logging.getLogger(__name__)


class XBPMMainWindow(QMainWindow):
    """Main application window for XBPM beam position analysis.

    Provides interface for:
    - Parameter configuration
    - Analysis execution
    - Progress monitoring
    - Result visualization
    """

    def __init__(self: "XBPMMainWindow") -> None:
        """Initialize the main window."""
        super().__init__()
        self.canvases          = {}
        self.beamlinedata      = None  # Canonical DataReader instance
        self.workbeamline      = None
        self.workdata          = None  # Effective BeamlineData instance
        self._last_inputfile   = ""
        self._last_roisize     = None
        self._analysis_running = False

        self.analysis          = None
        self._tab_info         = {}

        self.setup_ui()
        self.resize(1920, 1080)
        self._refresh_analysis_info()
        self.setWindowTitle("XBPM Calibration and Analysis Tool")

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
            msg = f"{str(exc)}"
            self.show_error("Run Analysis failed", msg)
            return
        finally:
            self.set_analysis_running(False)

        self.analysis = analysis
        self._build_tab_info()
        self._refresh_analysis_info()
        self.log_message("Analysis completed.")

        # Render every populated tab via the single orchestrator.
        figures = render_data_analysis(
            self.pos_nom,
            analysis,
            self.beamline_prm,
            )
        for key, fig in figures.items():
            self._embed_figure(self.canvases[key], fig)

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
        self.console._info_key = None

        # Visualization tabs (ordered to match analysis options)
        bpm_tab, bpm_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(bpm_tab, "BPM")
        self.canvases["bpm"] = bpm_canvas
        bpm_tab._info_key = "bpm"

        blade_tab, blade_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(blade_tab, "Blade Map")
        self.canvases["blade_map"] = blade_canvas
        blade_tab._info_key = "blade_map"

        blades_center_tab, blades_center_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(blades_center_tab, "Blades at sweeps")
        self.canvases["blade_central_sweeps"] = blades_center_canvas
        blades_center_tab._info_key = "blade_central_sweeps"

        xbpm_raw_pw_tab, xbpm_raw_pw_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_raw_pw_tab, "XBPM Δ/Σ raw")
        self.canvases["xbpm_pairwise_raw"] = xbpm_raw_pw_canvas
        xbpm_raw_pw_tab._info_key = "xbpm_pairwise_raw"

        xbpm_scaled_pw_tab, xbpm_scaled_pw_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_scaled_pw_tab, "XBPM Δ/Σ Sup. Mat.")
        self.canvases["xbpm_pairwise_trn"] = xbpm_scaled_pw_canvas
        xbpm_scaled_pw_tab._info_key = "xbpm_pairwise_trn"

        sweep_tab, sweep_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(sweep_tab, "Positions central sweeps")
        self.canvases["position_central_sweeps"] = sweep_canvas
        sweep_tab._info_key = "position_central_sweeps"

        xbpm_raw_cr_tab, xbpm_raw_cr_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_raw_cr_tab, "XBPM part. Δ/Σ - raw")
        self.canvases["xbpm_cross_raw"] = xbpm_raw_cr_canvas
        xbpm_raw_cr_tab._info_key = "xbpm_cross_raw"

        xbpm_scaled_cr_tab, xbpm_scaled_cr_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_scaled_cr_tab, "XBPM part. Δ/Σ - LinTr")
        self.canvases["xbpm_cross_trn"] = xbpm_scaled_cr_canvas
        xbpm_scaled_cr_tab._info_key = "xbpm_cross_trn"

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
        self.beamline_prm = self.workdata.prm
        self.pos_nom      = self.workdata.raw_data.blade_avg.pos_nom
        self._build_tab_info()

        # Update BPM distance.
        self.beamline_prm.bpmdist = Config.BPMDISTS.get(
            self.workbeamline[:3], None
            )

        # Calculate grid shape from nominal positions.
        self.nom_pos = self.workdata.raw_data.blade_avg.pos_nom
        self.grid_shape = (
            len(np.unique(self.nom_pos.y)),  # vertical dimension
            len(np.unique(self.nom_pos.x)),  # horizontal dimension
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
        bl_prm              = self.beamline_prm
        bl_prm.xbpmdist     = params["xbpmdist"]
        bl_prm.skip         = params["skip"]
        bl_prm.scalepolydeg = params["scalepolydeg"]
        bl_prm.usebpmref    = params["usebpmref"]
        bl_prm.roislice     = DStr.ROISlice.update(
            self.grid_shape,
            params["roisize"]
            )

        # Runtime parameters.
        rt_prm                       = self.runtime_prm
        rt_prm.show_blademap         = params["show_blademap"]
        rt_prm.show_bpmpositions     = params["show_bpmpositions"]
        rt_prm.show_centralsweep     = params["show_centralsweep"]
        rt_prm.show_xbpmpositions    = params["show_xbpmpositions"]
        rt_prm.slice_by_roi          = params["slice_by_roi"]
        # rt_prm.show_bladecenter      = params["show_bladecenter"]
        # rt_prm.show_xbpmpositionsraw = params["show_xbpmpositionsraw"]

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

    def _build_tab_info(self) -> None:
        """Place holder."""
        self._tab_info = format_analysis_info(self.analysis)

    def _refresh_analysis_info(self) -> None:
        """Update the analysis info panel based on the current tab."""
        # If analysis not available yet, skip.
        if self.analysis is None:
            self.analysis_info.setText("No analysis available yet.")
            return

        key =  getattr(self.results_tabs.currentWidget(), "_info_key", None)
        self.analysis_info.setText(self._tab_info.get(key, ""))

    @pyqtSlot(int)
    def _on_tab_changed(self, index: int):
        """Update analysis info when the active tab changes."""
        self._refresh_analysis_info()

    def _embed_figure(self,
                      canvas: MatplotlibCanvas,
                      source_fig: "matplotlib.figure.Figure"
                      ) -> None:
        """Embed entire figure by replacing canvas figure.

        Args:
            canvas: Target MatplotlibCanvas widget.
            source_fig: Source matplotlib figure with content.
        """
        try:
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
