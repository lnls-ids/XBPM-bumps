"""Main window for XBPM analysis application."""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QSplitter, QTabWidget,
    QStatusBar, QProgressBar, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer
from PyQt5.QtGui import QFont
import numpy as np
import logging
import os

from .widgets.parameter_panel import ParameterPanel
from .widgets.mpl_canvas import MatplotlibCanvas
from .dialogs.beamline_dialog import BeamlineSelectionDialog
from .dialogs.help_dialog import HelpDialog
from .analyzer import XBPMAnalyzer
from ..core.config import Config
from ..core.constants import FIGDPI


logger = logging.getLogger(__name__)


class XBPMMainWindow(QMainWindow):
    """Main application window for XBPM beam position analysis.

    Provides interface for:
    - Parameter configuration
    - Analysis execution
    - Progress monitoring
    - Result visualization
    """
    # Signals
    # Emits parameters when Run is clicked
    analysisRequested = pyqtSignal(dict)  # noqa: N815

    ANALYSIS_SECTION_TITLES = {
        'positions': 'Positions',
        'sweep_positions': 'Sweep Positions',
        'blades_sweeps': 'Blades at Sweeps',
        'bpm': 'BPM',
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
        from ..core.parameters import Prm, ParameterBuilder
        self.builder         = ParameterBuilder()
        self.prm             = self.builder.prm or Prm()  # Persistent
        self.canvases        = {}
        self.reader          = None  # Canonical DataReader instance
        self.rawdata         = None  # Canonical rawdata
        self._last_workdir   = ""
        self._last_meta      = {}
        self._last_hdf5_meta = {}
        self._last_roisize   = None
        self._analysis_running = False
        self._roi_rerun_timer = QTimer(self)
        self._roi_rerun_timer.setSingleShot(True)
        self._roi_rerun_timer.timeout.connect(self._on_run_clicked)
        self.setup_ui()
        self.setup_worker_thread()
        self.setWindowTitle("XBPM Beam Position Analysis")
        # Wider default window to give canvases more horizontal room
        self.resize(1600, 900)

    def _read_data_and_select_beamline(self, workdir: str) -> str:
        """Centralized beamline selection: returns the chosen beamline."""
        from ..core.readers import DataReader
        import os
        # Always instantiate and read DataReader for each new workdir
        self.prm.workdir = workdir
        self.reader = DataReader(self.prm, self.builder)
        self.reader.read()
        self.rawdata = self.reader.rawdata

        # Extract beamlines using HDF5 logic if file is HDF5
        beamlines = []
        if (os.path.isfile(workdir) and
            workdir.lower().endswith(('.h5', '.hdf5'))):
            from xbpm_bumps.core.reader_hdf5 import HDF5DataReader
            with HDF5DataReader(workdir) as reader:
                reader.load_all()
                beamlines = reader.beamlines
        else:
            from xbpm_bumps.core.reader_pickle import extract_beamlines
            beamlines = extract_beamlines(self.rawdata)

        if not beamlines:
            raise RuntimeError("No beamlines found in data.")

        if len(beamlines) == 1:
            chosen = beamlines[0]
            self.log_message(f"Auto-selected beamline: {chosen}")
        else:
            dialog = BeamlineSelectionDialog(sorted(beamlines))
            if dialog.exec_() != dialog.Accepted:
                raise RuntimeError("Beamline selection cancelled by user.")
            chosen = dialog.get_selection()
            if not chosen:
                raise RuntimeError("No beamline selected.")
            self.log_message(f"Selected beamline: {chosen}")
        # Persist in parameter panel for future get_parameters() calls
        if hasattr(self, 'param_panel'):
            self.param_panel.set_beamline(chosen)

        # Set ROI defaults to the maximum available points in each direction.
        # This keeps defaults aligned with the actual loaded measurement grid.
        try:
            self.prm.beamline = chosen
            fetched = self.reader._blades_fetch()
            data = fetched[0] if isinstance(fetched, tuple) else fetched
            keys = np.array(list(data.keys())) if data else np.array([])
            if keys.size > 0 and keys.ndim == 2 and keys.shape[1] >= 2:
                n_h = np.unique(keys[:, 0]).shape[0]
                n_v = np.unique(keys[:, 1]).shape[0]
                self.param_panel.set_roi_defaults_from_grid(n_h, n_v)
                self.log_message(
                    "ROI defaults set from loaded grid:"
                    f" H={n_h}, V={n_v}"
                )
        except Exception as exc:  # pragma: no cover - defensive
            self.log_message(f"Could not set ROI defaults from grid: {exc}")

        self._update_xbpmdist_from_beamline(chosen)
        return chosen

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
        open_dir_action = file_menu.addAction("Open Directory…")
        open_dir_action.triggered.connect(self._on_open_directory)

        open_hdf5_action = file_menu.addAction("Import HDF5 File…")
        open_hdf5_action.triggered.connect(self._on_open_hdf5)

        file_menu.addSeparator()
        export_action = file_menu.addAction("Export…")
        export_action.triggered.connect(self._on_export_clicked)

        export_hdf5_action = file_menu.addAction("Export to HDF5…")
        export_hdf5_action.triggered.connect(self._on_export_hdf5_clicked)

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
        self.results_tabs.addTab(xbpm_scaled_pw_tab, "XBPM Δ/Σ Tr")
        self.canvases["xbpm_scaled_pairwise"] = xbpm_scaled_pw_canvas

        xbpm_raw_cr_tab, xbpm_raw_cr_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_raw_cr_tab, "XBPM part. Δ/Σ - raw")
        self.canvases["xbpm_raw_cross"] = xbpm_raw_cr_canvas

        xbpm_scaled_cr_tab, xbpm_scaled_cr_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_scaled_cr_tab, "XBPM part. Δ/Σ - Tr")
        self.canvases["xbpm_scaled_cross"] = xbpm_scaled_cr_canvas

        return self.results_tabs

    def _on_parameters_changed(self) -> None:
        """React to parameter changes; pre-select beamline on workdir set."""
        params = self.param_panel.get_parameters()
        workdir = params.get('workdir') or ""
        if workdir and workdir != self._last_workdir:
            self._last_workdir = workdir
            # self._preselected_beamline removed

    def _schedule_roi_rerun(self) -> None:
        """Debounce ROI changes and re-run analysis if data is loaded."""
        if self._roi_rerun_timer.isActive():
            self._roi_rerun_timer.stop()
        self._roi_rerun_timer.start(400)

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

    def _prompt_beamline_selection(self, workdir: str) -> None:
        """Attempt beamline selection immediately after workdir change."""
        try:
            self.prm.workdir = workdir
            # DataReader instantiation and reading now handled by
            # _read_data_and_select_beamline
            # Handle beamline result
            if self.reader.prm.beamline:
                self.log_message(
                    f"Beamline selected: {self.reader.prm.beamline}"
                )
                self._update_xbpmdist_from_beamline(self.reader.prm.beamline)
        except Exception as exc:  # pragma: no cover - defensive
            self.log_message(f"Beamline preselection failed: {exc}")

    def _create_beamline_selector(self) -> callable:
        """Create beamline selector function for UI dialog."""
        def selector(bls):
            if len(bls) == 1:
                return bls[0]
            dialog = BeamlineSelectionDialog(sorted(bls))
            if dialog.exec_() == dialog.Accepted:
                return dialog.get_selection()
            return None
        return selector

    def _log_captured_output(self, log_capture) -> None:
        """Log captured stdout/stderr output."""
        output = log_capture.getvalue()
        if output:
            for line in output.split('\n'):
                if line.strip():
                    self.log_message(line)

    def _update_xbpmdist_from_beamline(self, beamline: str) -> None:
        """Update the XBPM distance field from Config.XBPMDISTS."""
        try:
            if not beamline:
                return
            dist = Config.XBPMDISTS.get(beamline)
            if dist is not None and hasattr(self, 'param_panel'):
                self.param_panel.xbpmdist_spin.setValue(float(dist))
                self.log_message(
                    f"XBPM distance set from beamline {beamline}: {dist:.3f} m"
                )
        except Exception as exc:  # pragma: no cover - defensive
            self.log_message(f"Could not set XBPM distance: {exc}")

    # def _handle_fallback_beamline_selection(self, reader,
    #                                         workdir: str) -> None:
    #     """Handle beamline selection when initial read doesn't set it."""
    #     import os

    #     beamlines = []
    #     if (os.path.isfile(workdir) and getattr(reader, "rawdata", None)):
    #         from xbpm_bumps.core.reader_pickle import extract_beamlines
    #         beamlines = extract_beamlines(reader.rawdata)

    #     if len(beamlines) == 1:
    #         self.log_message(f"Auto-selected beamline: {beamlines[0]}")
    #         self._update_xbpmdist_from_beamline(beamlines[0])

    def setup_worker_thread(self) -> None:
        """Initialize worker thread for analysis execution."""
        self.worker_thread = QThread()
        # Analyzer will be re-instantiated with canonical Prm and builder
        # before each run
        # Temporary builder for thread setup; will be replaced before each run
        self.analyzer = XBPMAnalyzer(
            self.prm,
            self.builder,
            self.reader,
            self.rawdata
        )
        self.analyzer.moveToThread(self.worker_thread)

        self.analysisRequested.connect(
            self._run_analysis_with_canonical_params
            )
        self.stop_btn.clicked.connect(self.analyzer.stop_analysis)

        self.analyzer.analysisStarted.connect(self._on_analysis_started)
        self.analyzer.analysisProgress.connect(self._on_analysis_progress)
        self.analyzer.analysisComplete.connect(self._on_analysis_complete)
        self.analyzer.analysisError.connect(self._on_analysis_error)
        self.analyzer.logMessage.connect(self.log_message)
        self.worker_thread.start()

    def _run_analysis_with_canonical_params(self, params: dict):
        """Build and enrich canonical ParameterBuilder/Prm, run analysis."""
        # Set canonical Prm fields directly from params
        for k, v in params.items():
            if hasattr(self.prm, k):
                setattr(self.prm, k, v)

        # Use canonical rawdata for parameter enrichment
        self.builder.rawdata = self.rawdata
        self.builder._add_beamline_parameters()

        # Update persistent Prm reference
        self.prm = self.builder.prm

        # Always convert rawdata to expected dict format for analysis
        analysis_data = self.reader._blades_fetch()
        self.analyzer = XBPMAnalyzer(self.prm, self.builder,
                                     self.reader, self.rawdata)
        self.analyzer.moveToThread(self.worker_thread)
        self.analyzer.app = None  # Reset to force re-init

        def set_app_reader_data():
            if self.analyzer.app is not None:
                self.analyzer.app.reader = self.reader
                self.analyzer.app.data = analysis_data
        self.analyzer.analysisStarted.connect(set_app_reader_data)
        self.stop_btn.clicked.connect(self.analyzer.stop_analysis)
        self.analyzer.analysisStarted.connect(self._on_analysis_started)
        self.analyzer.analysisProgress.connect(self._on_analysis_progress)
        self.analyzer.analysisComplete.connect(self._on_analysis_complete)
        self.analyzer.analysisError.connect(self._on_analysis_error)
        self.analyzer.logMessage.connect(self.log_message)
        self.analyzer.run_analysis()

    @pyqtSlot(list)
    def _on_beamline_selection_request(self, beamlines: list):
        """Show beamline selection dialog on the UI thread."""
        dialog = BeamlineSelectionDialog(sorted(beamlines))
        choice = ""
        if dialog.exec_() == dialog.Accepted:
            choice = dialog.get_selection() or ""
        self.analyzer.beamlineSelected.emit(choice)

    @pyqtSlot()
    def _on_analysis_started(self) -> None:
        """Handle analysis started signal."""
        self.set_analysis_running(True)
        self.log_message("\n" + "=" * 60)
        self.log_message("Analysis started")
        self.log_message("=" * 60)

    @pyqtSlot(str)
    def _on_analysis_progress(self, message: str) -> None:
        """Handle analysis progress update."""
        self.status_bar.showMessage(message)
        self.log_message(f"[PROGRESS] {message}")

    @pyqtSlot(dict)
    def _on_analysis_complete(self, results: dict) -> None:
        """Handle analysis completion."""
        self._last_results = results
        self.set_analysis_running(False)
        self.log_message("=" * 60)
        self.log_message("Analysis completed successfully!")
        self.log_message("=" * 60 + "\n")

        self._update_canvases(results)
        self._last_meta = self._collect_meta_from_results(results)
        self._refresh_analysis_info()

        self.status_bar.showMessage("Analysis complete", 5000)

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
                f"xbpm_{self.analyzer.app.prm.beamline}.dat"
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
            exported_any |= self._export_suppression_matrices(prefix, results)

            # Always export BPM positions when they were computed
            exported_any |= self._export_bpm_positions(prefix, results)

            # Export XBPM data and figures
            exported_any |= self._export_xbpm_raw(prefix, params, results)
            exported_any |= self._export_xbpm_scaled(prefix, params, results)

            # Export other analysis figures and data
            exported_any |= self._export_other_figures(prefix, params, results)
            exported_any |= self._export_analysis_info(prefix)

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

    @pyqtSlot()
    def _on_export_hdf5_clicked(self) -> None:
        """Export data to HDF5 file (with or without analysis results)."""
        try:
            # Ensure data is loaded (analysis is optional)
            if (not hasattr(self, 'analyzer') or not self.analyzer.app
                or not hasattr(self.analyzer.app, 'data')):
                QMessageBox.warning(
                    self,
                    "No Data Loaded",
                    (
                        "Please load data first.\n"
                        "Use 'Open Directory' to load blade measurement data."
                    ),
                )
                return

            default_name = (
                f"xbpm_{self.analyzer.app.prm.beamline or 'export'}.h5"
            )
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export to HDF5",
                default_name,
                "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
            )
            if not path:
                return

            results = getattr(self, '_last_results', {}) or {}

            # Export using current data and last results
            from ..core.exporters import Exporter
            exporter = Exporter(self.analyzer.app.prm)

            # Include rawdata for complete re-analysis capability
            rawdata = getattr(self.analyzer.app.reader, 'rawdata', None)
            exporter.write_hdf5(path, self.analyzer.app.data, results,
                                include_figures=True, rawdata=rawdata)

            self.log_message(f"HDF5 export written: {path}")
            QMessageBox.information(
                self,
                "Export Complete",
                "Exported analysis and figures to HDF5.",
            )
        except Exception as exc:  # pragma: no cover
            self.show_error("Export to HDF5 Failed", str(exc))

    @pyqtSlot()
    def _on_open_directory(self) -> None:
        """Open dialog to select working directory with pickle files."""
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Select Working Directory")
        dialog.setDirectory(os.getcwd())
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, False)
        # Show all file types since we're just selecting a directory container
        dialog.setNameFilter("All Files (*)")

        # Connect to directory change to show current path
        dialog.directoryEntered.connect(
            lambda path: dialog.setWindowTitle(
                f"Select Working Directory - Current: {path}"
            )
        )

        if dialog.exec() == QFileDialog.Accepted:
            path = dialog.directory().absolutePath()
            # Ensure path is set in parameter panel and visible in field
            self.param_panel.set_workdir(path)
            self.status_bar.showMessage(f"Working Directory: {path}")

            # Automatically load data (without running analysis)
            self._load_data_from_directory()

    @pyqtSlot()
    def _on_open_hdf5(self) -> None:
        """Open dialog to select HDF5 data file.

        (Routes through Analyzer for beamline selection.)
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 File",
            os.getcwd(),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )

        if path:
            # Store workdir in parameter panel and update status bar
            self.param_panel.set_workdir(path)
            self.status_bar.showMessage(f"Opened: {path}")

            # Use Analyzer workflow for loading HDF5.
            # (Ensures beamline selection.)
            params = self.param_panel.get_parameters()
            if not self._validate_workdir(params):
                return

            # Ensure beamline is selected using canonical workflow
            workdir = params.get('workdir')
            try:
                chosen_beamline = self._read_data_and_select_beamline(workdir)
            except Exception as exc:
                self.show_error("Beamline Selection Failed", str(exc))
                return
            self.log_message(
                f"Loading data from: {workdir} "
                f"(beamline: {chosen_beamline})"
            )
            self.analyzer.reader = self.reader
            self.analyzer.rawdata = self.rawdata
            self.analyzer.load_data_only()
            # Assign analysis metadata for UI display
            self._last_hdf5_meta = getattr(self.reader, 'analysis_meta', {})
            self._last_meta = self._last_hdf5_meta

            # Automatically load and display figures after import
            self._on_load_hdf5_figures(workdir)

    def _on_load_hdf5_figures(self, hdf5_path: str) -> None:
        """Load figures from HDF5 file and display them.

        Args:
            hdf5_path: Path to the HDF5 file.
        """
        try:
            self.prm.workdir = hdf5_path
            from xbpm_bumps.core.reader_hdf5 import HDF5FigureReconstructor
            reconstructor = HDF5FigureReconstructor(hdf5_path)
            figures = reconstructor.load_figures()
            if not figures:
                self.log_message("No figures found in HDF5 file")
                return
            self._display_hdf5_figures(figures)
            self._refresh_analysis_info()
        except Exception as e:
            self.log_message(f"Error loading figures from HDF5: {e}")

    def _display_hdf5_figures(self, figures: dict) -> None:
        """Display loaded HDF5 figures in canvases or separate windows.

        Parameters
        ----------
        figures : dict
            Dictionary of reconstructed figures from HDF5.
        """
        # Figures that go in main window canvases
        inline_specs = [
            ('blade_figure', 'blade', "Loaded blade map from HDF5"),
            ('sweeps_figure', 'sweeps',
             "Loaded sweeps figure from HDF5"),
            ('blades_center_figure', 'blades_center',
             "Loaded blades center from HDF5"),
            ('bpm_figure', 'bpm', "Loaded BPM positions from HDF5"),
        ]

        # Figures that need separate windows or tabs (position grids)
        popup_specs = [
            ('xbpm_raw_pairwise_figure', 'xbpm_raw_pairwise',
               "XBPM Raw Δ/Σ"),
              ('xbpm_raw_cross_figure', 'xbpm_raw_cross',
               "XBPM Raw Partial Δ/Σ"),
            ('xbpm_scaled_pairwise_figure', 'xbpm_scaled_pairwise',
               "XBPM Transf. Δ/Σ"),
            ('xbpm_scaled_cross_figure', 'xbpm_scaled_cross',
               "XBPM Transf. Partial Δ/Σ"),
        ]

        # Display inline figures in main window
        for fig_key, canvas_key, msg in inline_specs:
            if fig_key in figures:
                canvas = self.canvases.get(canvas_key)
                if canvas:
                    self._embed_figure(canvas, figures[fig_key])
                    self.log_message(msg)

        # Display position grid figures in tabs
        for fig_key, canvas_key, title in popup_specs:
            if fig_key in figures:
                canvas = self.canvases.get(canvas_key)
                if canvas:
                    self._embed_figure(canvas, figures[fig_key])
                    self.log_message(f"Loaded {title} in tab")

    def _show_figure_window(self, fig, title: str) -> None:
        """Display matplotlib figure in a separate window.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to display.
        title : str
            Window title.
        """
        try:
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas
            )
            from PyQt5.QtWidgets import QDialog, QVBoxLayout

            dialog = QDialog(self)
            dialog.setWindowTitle(title)
            dialog.resize(1400, 600)

            layout = QVBoxLayout()
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            dialog.setLayout(layout)

            dialog.show()
        except Exception as e:
            self.log_message(f"Error displaying {title}: {e}")

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

    def _export_suppression_matrices(self, prefix: str,
                                      results: dict) -> bool:
        """Export both suppression matrices unconditionally.

        Writes:
          * ``<prefix>_supmat_standard.dat``  – fixed 1/-1 matrix
          * ``<prefix>_supmat_calculated.dat`` – matrix fitted from slopes

        Args:
            prefix: Export filename prefix.
            results: Results dict from last analysis run.

        Returns:
            True if at least one matrix was written.
        """
        from ..core.processors import XBPMProcessor as XPROC
        from ..core.exporters import Exporter

        exporter = Exporter(self.analyzer.app.prm)
        outdir = os.path.dirname(prefix) or '.'
        wrote_any = False

        # --- Standard (1/-1) suppression matrix ---
        supmat_std, _ = XPROC.standard_suppression_matrix()
        std_path = os.path.join(
            outdir,
            f"xbpm_supmat_standard_{self.analyzer.app.prm.beamline}.dat"
        )
        exporter.write_supmat(supmat_std, write_file=True, outpath=std_path)
        logger.info("Standard suppression matrix saved to %s", std_path)
        wrote_any = True

        # --- Calculated (slope-fitted) suppression matrix ---
        supmat_calc = results.get('supmat')
        if supmat_calc is None:
            # Compute directly from processor if analysis was run
            try:
                supmat_calc = self.analyzer.app.processor.suppression_matrix(
                    showmatrix=False, nosuppress=False
                )
            except Exception as exc:
                logger.warning("Could not compute calculated supmat: %s", exc)

        if supmat_calc is not None:
            calc_path = os.path.join(
                outdir,
                f"xbpm_supmat_calculated_{self.analyzer.app.prm.beamline}.dat"
            )
            exporter.write_supmat(supmat_calc, write_file=True,
                                  outpath=calc_path)
            logger.info("Calculated suppression matrix saved to %s",
                        calc_path)

        return wrote_any

    def _write_position_info_file(self, path: str, calc_type: str, sup: str,
                                    scales: dict, stats: dict) -> None:
        """Write scale coefficients and statistics to a labeled text file."""
        beamline = self.analyzer.app.prm.beamline
        with open(path, 'w') as fp:
            fp.write("# Scaling coefficients and statistics\n")
            fp.write(
                f"# Beamline: {beamline}"
                f" | Type: positions / {calc_type} / {sup}\n"
            )
            fp.write(f"kx           = {float(scales.get('kx', 1)):.6f}\n")
            fp.write(f"skx          = {float(scales.get('skx', 0)):.6f}\n")
            fp.write(f"dx           = {float(scales.get('dx', 0)):.6f}\n")
            fp.write(f"sdx          = {float(scales.get('sdx', 0)):.6f}\n")
            fp.write(f"ky           = {float(scales.get('ky', 1)):.6f}\n")
            fp.write(f"sky          = {float(scales.get('sky', 0)):.6f}\n")
            fp.write(f"dy           = {float(scales.get('dy', 0)):.6f}\n")
            fp.write(f"sdy          = {float(scales.get('sdy', 0)):.6f}\n")
            if stats:
                fp.write("u\n# Position differences statistics ($\\mu$m)\n")
                fp.write("RMS H diff.     = "
                         f"{float(stats.get('sigma_h', 0)):.6f}  um\n")
                fp.write("RMS V diff.     = "
                         f"{float(stats.get('sigma_v', 0)):.6f}  um\n")
                fp.write("RMS Total diff. = "
                         f"{float(stats.get('sigma_total', 0)):.6f}  um\n")
                fp.write("Max RMS H diff. = "
                         f"{float(stats.get('diff_max_h', 0)):.6f}  um\n")
                fp.write("Max RMS V diff. = "
                         f"{float(stats.get('diff_max_v', 0)):.6f}  um\n")
                fp.write("Min RMS H diff. = "
                         f"{float(stats.get('diff_min_h', 0)):.6f}  um\n")
                fp.write("Min RMS V diff. = "
                         f"{float(stats.get('diff_min_v', 0)):.6f}  um\n")

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

    def _export_xbpm_raw(self, prefix: str,
                         params: dict,
                         results: dict) -> bool:
        """Export raw XBPM positions and figures.

        Args:
            prefix: Export filename prefix.
            params: Parameter dictionary from UI.
            results: Last analysis results dictionary.

        Returns:
            True if export occurred, False otherwise.
        """
        if not params.get('xbpmpositionsraw'):
            return False

        from ..core.exporters import Exporter

        exporter = Exporter(self.analyzer.app.prm)
        result_raw = results.get('positions_raw_full')
        if not result_raw:
            logger.warning("Raw XBPM export skipped: no cached analysis data.")
            return False

        exporter.data_dump_with_prefix(
            prefix,
            self.analyzer.app.data,
            result_raw['positions'],
            sup="raw",
        )

        # Save figures
        pairwise_fig = (result_raw.get('pairwise_figure') or
                        results.get('xbpm_raw_pairwise_figure'))
        if pairwise_fig is not None:
            fig_pair = os.path.join(
                os.path.dirname(prefix),
                f"xbpm_positions_pair_raw_{self.analyzer.app.prm.beamline}.png"
            )
            self._save_figure_for_export(pairwise_fig, fig_pair)
            logger.info("Pairwise figure saved to %s", fig_pair)

        cross_fig = (result_raw.get('cross_figure') or
                     results.get('xbpm_raw_cross_figure'))
        if cross_fig is not None:
            fig_cross = os.path.join(
                os.path.dirname(prefix),
                f"xbpm_positions_cross_raw_{self.analyzer.app.prm.beamline}.png"
            )
            self._save_figure_for_export(cross_fig, fig_cross)
            logger.info("Cross figure saved to %s", fig_cross)

        # Export scaling factors and statistics
        scales = result_raw.get('scales') or {}
        stats = result_raw.get('xbpm_stats') or {}
        bl = self.analyzer.app.prm.beamline
        for calc_type, stats_key in (('pair', 'pairwise'), ('cross', 'cross')):
            calc_scales = scales.get(calc_type)
            if calc_scales:
                info_path = os.path.join(
                    os.path.dirname(prefix),
                    f"xbpm_positions_{calc_type}_raw_{bl}_info.dat"
                )
                self._write_position_info_file(
                    info_path, calc_type, 'raw',
                    calc_scales, stats.get(stats_key) or {}
                )
                logger.info("Info file saved to %s", info_path)

        return True

    def _export_xbpm_scaled(self, prefix: str,
                            params: dict,
                            results: dict) -> bool:
        """Export scaled XBPM positions and figures.

        Args:
            prefix: Export filename prefix.
            params: Parameter dictionary from UI.
            results: Last analysis results dictionary.

        Returns:
            True if export occurred, False otherwise.
        """
        if not params.get('xbpmpositions'):
            return False

        from ..core.exporters import Exporter

        exporter = Exporter(self.analyzer.app.prm)
        result_scaled = results.get('positions_scaled_full')
        if not result_scaled:
            logger.warning("Scaled XBPM export skipped: no cached analysis data.")
            return False

        exporter.data_dump_with_prefix(
            prefix,
            self.analyzer.app.data,
            result_scaled['positions'],
            sup="scaled",
        )

        # Save figures
        pairwise_fig = (result_scaled.get('pairwise_figure') or
                        results.get('xbpm_scaled_pairwise_figure'))
        if pairwise_fig is not None:
            fig_pair = os.path.join(
                os.path.dirname(prefix),
                f"xbpm_positions_pair_scaled_{self.analyzer.app.prm.beamline}.png"
            )
            self._save_figure_for_export(pairwise_fig, fig_pair)
            logger.info("Pairwise figure saved to %s", fig_pair)

        cross_fig = (result_scaled.get('cross_figure') or
                     results.get('xbpm_scaled_cross_figure'))
        if cross_fig is not None:
            fig_cross = os.path.join(
                os.path.dirname(prefix),
                f"xbpm_positions_cross_scaled_{self.analyzer.app.prm.beamline}.png"
            )
            self._save_figure_for_export(cross_fig, fig_cross)
            logger.info("Cross figure saved to %s", fig_cross)

        # Export scaling factors and statistics
        scales = result_scaled.get('scales') or {}
        stats = result_scaled.get('xbpm_stats') or {}
        bl = self.analyzer.app.prm.beamline
        for calc_type, stats_key in (('pair', 'pairwise'), ('cross', 'cross')):
            calc_scales = scales.get(calc_type)
            if calc_scales:
                info_path = os.path.join(
                    os.path.dirname(prefix),
                    f"xbpm_positions_{calc_type}_scaled_{bl}_info.dat"
                )
                self._write_position_info_file(
                    info_path, calc_type, 'scaled',
                    calc_scales, stats.get(stats_key) or {}
                )
                logger.info("Info file saved to %s", info_path)

        return True

    def _export_bpm_positions(self, prefix: str, results: dict) -> bool:
        """Export BPM positions to a text file (always when data is available).

        Args:
            prefix: Export filename prefix (directory is derived from it).
            results: Last analysis results dictionary.

        Returns:
            True if export occurred, False otherwise.
        """
        bpm_data = results.get('positions', {}).get('bpm')
        if not bpm_data:
            return False
        measured = bpm_data.get('measured')
        nominal = bpm_data.get('nominal')
        if measured is None or nominal is None:
            return False

        from ..core.exporters import Exporter
        exporter = Exporter(self.analyzer.app.prm)
        exporter.data_dump_bpm(measured, nominal, prefix=prefix)
        return True

    def _export_analysis_info(self, prefix: str) -> bool:
        """Export analysis info text shown in the read-only panel."""
        if not hasattr(self, 'analysis_info') or self.analysis_info is None:
            return False

        text = (self.analysis_info.toPlainText() or "").strip()
        if not text:
            return False

        info_path = os.path.join(
            os.path.dirname(prefix),
            f"xbpm_analysis_info_{self.analyzer.app.prm.beamline}.txt"
        )
        with open(info_path, 'w') as fp:
            fp.write(text + "\n")
        logger.info("Analysis info saved to %s", info_path)
        return True

    def _export_other_figures(self, prefix: str, params: dict,
                              results: dict) -> bool:
        """Export analysis figures (blade map, sweeps, etc) and sweeps data."""
        exported = self._export_simple_figures(prefix, params, results)
        if params.get('centralsweep'):
            exported |= self._export_central_sweeps(prefix, results)
        return exported

    def _export_simple_figures(self, prefix: str, params: dict,
                               results: dict) -> bool:
        """Save blade-map, blades-at-center, and BPM figures."""
        exported = False
        beamline = self.analyzer.app.prm.beamline
        figure_exports = [
            ('showblademap', 'blade_figure',
             f'xbpm_blademap_{beamline}.png'),
            ('showbladescenter', 'blades_center_figure',
             f'xbpm_blades_center_{beamline}.png'),
        ]
        for option_key, result_key, filename in figure_exports:
            if not params.get(option_key):
                continue
            fig = results.get(result_key)
            if fig is None:
                continue
            fig_path = os.path.join(os.path.dirname(prefix), filename)
            self._save_figure_for_export(fig, fig_path)
            logger.info("Figure saved to %s", fig_path)
            exported = True
        # BPM positions: always export the figure when it was computed
        bpm_fig = results.get('bpm_figure')
        if bpm_fig is not None:
            fig_path = os.path.join(
                os.path.dirname(prefix),
                f'xbpm_bpm_positions_{beamline}.png'
            )
            self._save_figure_for_export(bpm_fig, fig_path)
            logger.info("Figure saved to %s", fig_path)
            exported = True
        return exported

    def _export_central_sweeps(self, prefix: str, results: dict) -> bool:
        """Save central-sweeps figure and blade-current .dat files."""
        exported = False
        beamline = self.analyzer.app.prm.beamline
        outdir = os.path.dirname(prefix)

        fig = results.get('sweeps_figure')
        if fig is not None:
            fig_path = os.path.join(
                outdir, f'xbpm_central_sweeps_{beamline}.png'
            )
            self._save_figure_for_export(fig, fig_path)
            logger.info("Figure saved to %s", fig_path)
            exported = True

        sweeps_data = results.get('sweeps_data')
        if not sweeps_data or len(sweeps_data) < 4:
            if sweeps_data:
                logger.warning("Sweeps data missing expected arrays")
            return exported

        range_h, range_v, blades_h, blades_v = sweeps_data[:4]
        try:
            exported |= self._save_sweep_dat(
                outdir, f'xbpm_central_sweeps_horizontal_{beamline}.dat',
                range_h, blades_h, "range_h"
            )
            exported |= self._save_sweep_dat(
                outdir, f'xbpm_central_sweeps_vertical_{beamline}.dat',
                range_v, blades_v, "range_v"
            )
        except Exception:
            logger.exception("Failed to save central sweeps data")
        return exported

    @staticmethod
    def _save_sweep_dat(outdir: str, filename: str, axis_range,
                        blades, range_label: str) -> bool:
        """Write a single sweep axis to a .dat file; return True if written."""
        if not isinstance(blades, dict):
            return False
        arrays = [blades.get(k) for k in ("to", "ti", "bi", "bo")]
        if not all(arr is not None for arr in arrays):
            return False
        path = os.path.join(outdir, filename)
        np.savetxt(
            path,
            np.column_stack([axis_range, *arrays]),
            header=f"{range_label} to ti bi bo",
            fmt="%.6f",
        )
        logger.info("Sweeps data saved to %s", path)
        return True

    @pyqtSlot(str, str)
    def _on_analysis_error(self, title: str, message: str):
        """Handle analysis error.

        Args:
            title: Error dialog title.
            message: Error message.
        """
        self.set_analysis_running(False)
        self.show_error(title, message)

    def _load_data_from_directory(self) -> None:
        """Load data from the selected directory without running analysis.

        This allows users to export raw data to HDF5 without analysis.
        """
        params = self.param_panel.get_parameters()
        if not self._validate_workdir(params):
            return
        # Centralized data reading and beamline selection
        chosen_beamline = (
            self._read_data_and_select_beamline(params['workdir'])
            )
        self.log_message(f"Loading data from: {params['workdir']}"
                         f" (beamline: {chosen_beamline})")
        # Set canonical Prm fields directly from params
        for k, v in params.items():
            if hasattr(self.prm, k):
                setattr(self.prm, k, v)
        self.prm.beamline = chosen_beamline  # Set beamline programmatically
        # Use canonical rawdata for parameter enrichment
        self.builder.rawdata = self.rawdata
        self.builder._add_beamline_parameters()
        # Update persistent Prm reference
        self.prm = self.builder.prm
        # Re-instantiate analyzer with canonical Prm and builder
        self.analyzer = XBPMAnalyzer(self.prm, self.builder,
                                     self.reader, self.rawdata)
        self.analyzer.moveToThread(self.worker_thread)
        # Reconnect signals
        self.analyzer.analysisStarted.connect(self._on_analysis_started)
        self.analyzer.analysisProgress.connect(self._on_analysis_progress)
        self.analyzer.analysisComplete.connect(self._on_analysis_complete)
        self.analyzer.analysisError.connect(self._on_analysis_error)
        self.analyzer.logMessage.connect(self.log_message)
        # Use load_data_only instead of run_analysis
        self.analyzer.load_data_only()

    @pyqtSlot()
    def _on_run_clicked(self) -> None:
        """Handle Run Analysis button click."""
        params = self.param_panel.get_parameters()
        if not self._validate_workdir(params):
            return
        # Use already selected beamline
        self.log_message("Starting analysis with workdir:"
                         f" {params['workdir']}"
                         f" (beamline: {self.prm.beamline})")
        self.analysisRequested.emit(params)

    def _validate_workdir(self, params: dict) -> bool:
        """Ensure workdir is provided."""
        if params.get('workdir'):
            return True
        QMessageBox.warning(
            self,
            "Missing Input",
            "Please select a working directory or data file."
        )
        return False

    @pyqtSlot(str)
    def log_message(self, message: str) -> None:
        """Append a message to the console log.

        Args:
            message: Text to append to console.
        """
        self.console.append(message)

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

    @pyqtSlot(str, str)
    def show_error(self, title: str, message: str) -> None:
        """Display error dialog.

        Args:
            title: Error dialog title.
            message: Error message text.
        """
        QMessageBox.critical(self, title, message)
        self.log_message(f"ERROR: {message}")

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

    def _update_canvases(self, results: dict) -> None:
        """Render available results into canvases."""
        for canvas in self.canvases.values():
            canvas.clear()

        positions = (results.get('positions', {})
                     if isinstance(results, dict) else {})

        def _render_position(canvas_key, figure_key, position_key, title):
            canvas = self.canvases.get(canvas_key)
            if canvas is None:
                return
            fig = results.get(figure_key)
            if fig is not None:
                self._embed_figure(canvas, fig)
                return
            pos_data = positions.get(position_key)
            if pos_data is not None:
                self._plot_positions(canvas, pos_data, title=title)

        def _render_figure(canvas_key, figure_key):
            canvas = self.canvases.get(canvas_key)
            fig = results.get(figure_key)
            if canvas is not None and fig is not None:
                self._embed_figure(canvas, fig)

        position_specs = [
            ("xbpm_raw_pairwise", "xbpm_raw_pairwise_figure", "xbpm_raw",
             "XBPM Raw Pairwise Positions"),
            ("xbpm_scaled_pairwise", "xbpm_scaled_pairwise_figure",
             "xbpm_scaled", "XBPM Scaled Pairwise Positions"),
            ("bpm", "bpm_figure", "bpm", "BPM Positions"),
        ]

        figure_specs = [
            ("xbpm_raw_cross", "xbpm_raw_cross_figure"),
            ("xbpm_scaled_cross", "xbpm_scaled_cross_figure"),
            ("blade", "blade_figure"),
            ("sweeps", "sweeps_figure"),
            ("blades_center", "blades_center_figure"),
        ]

        for canvas_key, fig_key, pos_key, title in position_specs:
            _render_position(canvas_key, fig_key, pos_key, title)

        for canvas_key, fig_key in figure_specs:
            _render_figure(canvas_key, fig_key)

    def _collect_meta_from_results(self, results: dict) -> dict:
        """Extract analysis metadata from results for UI display."""
        if not isinstance(results, dict):
            return {}

        meta: dict = {}
        self._collect_scales_meta(results, meta)
        self._collect_bpm_stats_meta(results, meta)
        self._collect_sweeps_meta_from_results(results, meta)
        self._collect_supmat_meta(results, meta)
        self._collect_xbpm_stats_meta(results, meta)
        return meta

    def _collect_scales_meta(self, results: dict, meta: dict) -> None:
        """Extract scaling coefficients from results."""
        def _pick_scales(block):
            if not isinstance(block, dict):
                return None
            return block.get('scales') or None

        scales = {}
        raw_scales = _pick_scales(results.get('positions_raw_full'))
        scaled_scales = _pick_scales(results.get('positions_scaled_full'))
        if raw_scales:
            scales['raw'] = raw_scales
        if scaled_scales:
            scales['scaled'] = scaled_scales
        if scales:
            meta['scales'] = scales

    def _collect_bpm_stats_meta(self, results: dict, meta: dict) -> None:
        """Extract BPM statistics from results."""
        bpm_stats = results.get('bpm_stats')
        if bpm_stats:
            meta['bpm_stats'] = bpm_stats

    def _collect_sweeps_meta_from_results(self, results: dict,
                                          meta: dict) -> None:
        """Extract sweeps metadata from results."""
        sweeps_meta = self._extract_sweeps_meta(results.get('sweeps_data'))
        if sweeps_meta:
            meta['sweeps'] = sweeps_meta

    def _collect_supmat_meta(self, results: dict, meta: dict) -> None:
        """Extract suppression matrices from results."""
        if 'supmat_standard' in results:
            meta['supmat_standard'] = results.get('supmat_standard')
        if 'supmat' in results:
            meta['supmat'] = results.get('supmat')

    def _collect_xbpm_stats_meta(self, results: dict, meta: dict) -> None:
        """Extract XBPM statistics from results."""
        if 'xbpm_stats_raw' in results:
            meta['xbpm_stats_raw'] = results.get('xbpm_stats_raw')
        if 'xbpm_stats_scaled' in results:
            meta['xbpm_stats_scaled'] = results.get('xbpm_stats_scaled')

    def _extract_sweeps_meta(self, sweeps_data: list) -> dict:
        """Extract sweep metadata: positions fits and per-blade fits.
        
        Args:
            sweeps_data (list): List containing sweep data arrays.

        Returns:
            dict: Dictionary containing sweep metadata or empty dict if     extraction fails.
        """
        if not sweeps_data or len(sweeps_data) < 8:
            return {}

        try:
            (range_h, range_v, blades_h, blades_v, fit_h,
             fit_v) = sweeps_data[0:2] + sweeps_data[2:4] + sweeps_data[6:8]
        except Exception:
            return {}

        meta = {}

        positions = self._extract_sweeps_positions(fit_h, fit_v)
        if positions:
            meta['positions'] = positions

        blades = self._extract_sweeps_blades(blades_h, blades_v,
                                             range_h, range_v)
        if blades:
            meta['blades'] = blades

        return meta

    def _extract_sweeps_positions(self, fit_h : np.ndarray,
                                  fit_v : np.ndarray) -> dict:
        """Extract global fit positions from horizontal and vertical fits."""
        positions = {}
        h_meta = self._parse_fit(fit_h)
        if h_meta:
            positions['horizontal'] = h_meta
        v_meta = self._parse_fit(fit_v)
        if v_meta:
            positions['vertical'] = v_meta
        return positions

    def _parse_fit(self, fit : np.ndarray) -> dict:
        """Parse fit parameters into k and delta values."""
        try:
            # Fit can be either [k, delta] or [[k, s_k], [delta, s_delta]].
            if hasattr(fit, "shape") and fit.shape == (2, 2):
                return {
                    'k'       : float(fit[0][0]),
                    'delta'   : float(fit[1][0]),
                    's_k'     : float(fit[0][1]),
                    's_delta' : float(fit[1][1]),
                }

            k_val = (float(fit[0][0])
                     if hasattr(fit, "__len__") else float(fit[0]))
            delta_val = (float(fit[1][0])
                         if hasattr(fit, "__len__") else float(fit[1]))
            return {'k': k_val, 'delta': delta_val}
        except Exception:
            return None

    def _extract_sweeps_blades(self, blades_h: dict, blades_v: dict,
                               range_h: np.ndarray,
                               range_v: np.ndarray) -> dict:
        """Extract per-blade fits from blade data."""
        blades = {}
        h_blades = self._fit_blades(blades_h, range_h)
        if h_blades:
            blades['horizontal'] = h_blades
        v_blades = self._fit_blades(blades_v, range_v)
        if v_blades:
            blades['vertical'] = v_blades
        return blades

    def _fit_blades(self, blades_dict : dict,
                    axis_range : np.ndarray) -> dict:
        """Fit blade data to extract k and delta for each blade.
        
        Args:
            blades_dict (dict): Dictionary containing blade data.
            axis_range (np.ndarray): Array representing the axis range.

        Returns:
            dict: Dictionary containing fit parameters for each blade or None if fitting fails.
        """
        if not isinstance(blades_dict, dict) or axis_range is None:
            return None
        fits = {}
        for blade in ('to', 'ti', 'bi', 'bo'):
            arr = blades_dict.get(blade)
            if arr is None:
                continue
            y = (arr[:, 0]
                 if hasattr(arr, 'ndim') and arr.ndim == 2
                 else arr)
            try:
                coef = np.polyfit(axis_range, y, deg=1)
                fits[blade] = {
                    'k'     : float(coef[0]),
                    'delta' : float(coef[1])
                    }
            except Exception as exc:
                logger.debug("Failed to fit blade %s: %s", blade, exc)
                continue
        return fits or None

    def _tab_to_section(self, tab_text: str) -> str:
        text = (tab_text or "").lower()
        if 'blade map' in text:
            return 'none'
        if 'blades at' in text or 'blades at sweeps' in text:
            return 'blades_sweeps'
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

    def _format_analysis_info(self, meta: dict, active_tab: str) -> str:
        """Format analysis metadata for UI display.

        Delegates to helper methods to reduce complexity.
        """
        if not meta:
            return "No analysis metadata available yet."

        active_section = self._tab_to_section(active_tab)
        pos_filter = self._tab_position_filter(active_tab)
        if active_section == 'none':
            return ""

        sections: dict[str, list[str]] = {}
        sections.update(self._format_scales_section(meta, pos_filter))
        sections.update(self._format_sweeps_positions_section(meta))
        sections.update(self._format_blades_section(meta))
        sections.update(self._format_bpm_stats_section(meta))

        # Add XBPM stats to positions section if present
        xbpm_stats_dict = self._format_xbpm_stats_section(meta, active_tab)
        if xbpm_stats_dict:
            sections.setdefault('positions', []).extend(
                xbpm_stats_dict.get('xbpm', [])
                )

        supmat_lines = self._format_supmat_lines(meta, active_tab)
        if supmat_lines:
            sections.setdefault('positions', []).extend(supmat_lines)

        return self._format_sections_output(sections, active_section)

    def _format_scales_section(self, meta: dict,
                               pos_filter=None) -> dict[str, list[str]]:
        """Format scales (positions) metadata section."""
        scale_lines: list[str] = []
        scales = meta.get('scales', {}) if isinstance(meta, dict) else {}

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

                lines_to_add = self._format_scale_entry(coeffs)
                if lines_to_add:
                    subject = Config.get_position_subject(scope, label)
                    scale_lines.append(f"  * {subject}:")
                    scale_lines.extend(lines_to_add)

        return {'positions': scale_lines} if scale_lines else {}

    def _format_scale_entry(self, coeffs: dict) -> list[str]:
        """Format individual scale coefficient entry."""
        lines_to_add: list[str] = []
        line1 = []
        line2 = []

        for key, err_keys in (
            ('kx', ('skx', 's_kx')),
            ('dx', ('sdx', 's_dx')),
        ):
            item = self._format_value_with_optional_error(coeffs, key, err_keys)
            if item:
                line1.append(item)

        for key, err_keys in (
            ('ky', ('sky', 's_ky')),
            ('dy', ('sdy', 's_dy')),
        ):
            item = self._format_value_with_optional_error(coeffs, key, err_keys)
            if item:
                line2.append(item)

        if line1:
            lines_to_add.append("   " + ", ".join(line1))
        if line2:
            lines_to_add.append("   " + ", ".join(line2))

        return lines_to_add

    @staticmethod
    def _format_value_with_optional_error(coeffs: dict,
                                          key: str,
                                          err_keys) -> str:
        """Format coeff as value or value and error when available."""
        val = coeffs.get(key)
        if val is None:
            return ""

        # Accept both legacy keys (s_kx/s_dx/...) and current keys
        # (skx/sdx/...) to support old and new result payloads.
        if isinstance(err_keys, (tuple, list)):
            err = None
            for ek in err_keys:
                if ek in coeffs and coeffs.get(ek) is not None:
                    err = coeffs.get(ek)
                    break
        else:
            err = coeffs.get(err_keys)

        try:
            val_num = float(val)
            if err is not None:
                return f"{key:>10} = {val_num:.6f} ({float(err):.3g})"
            return f"{key:>10} = {val_num:.6f}"
        except Exception:
            if err is not None:
                return f"{key:>10} = {val} ({err})"
            return f"{key:>10} = {val}"

    def _format_sweeps_positions_section(
            self, meta: dict) -> dict[str, list[str]]:
        """Format sweeps positions (global fits) metadata section."""
        sweeps_pos_lines: list[str] = []
        sweeps = meta.get('sweeps', {}) if isinstance(meta, dict) else {}
        positions_meta = (sweeps.get('positions', {})
                         if isinstance(sweeps, dict) else {})

        for orient, label in (('horizontal', ' H '), ('vertical', ' V ')):
            fit = positions_meta.get(orient)
            if not isinstance(fit, dict):
                continue

            lines_to_add = self._format_sweeps_fit_entry(fit)
            if lines_to_add:
                sweeps_pos_lines.append(f"  * {label}:")
                sweeps_pos_lines.extend(lines_to_add)

        return ({'sweep_positions': sweeps_pos_lines}
                if sweeps_pos_lines else {})

    def _format_sweeps_fit_entry(self, fit: dict) -> list[str]:
        """Format sweeps fit entry for a single orientation."""
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

        return lines_to_add

    def _format_blades_section(self, meta: dict) -> dict[str, list[str]]:
        """Format blades-at-sweeps per-blade fits metadata section."""
        blades_lines: list[str] = []
        sweeps = meta.get('sweeps', {}) if isinstance(meta, dict) else {}
        blades_meta = (sweeps.get('blades', {})
                       if isinstance(sweeps, dict) else {})

        for orient, label in (('horizontal', 'H'), ('vertical', 'V')):
            bfits = blades_meta.get(orient)
            if not isinstance(bfits, dict):
                continue

            for blade, fit in bfits.items():
                if not isinstance(fit, dict):
                    continue

                parts = self._format_blade_fit_entry(fit)
                if parts:
                    blades_lines.append(f"  * {label} {blade}:")
                    blades_lines.append("   " + ", ".join(parts))

        return {'blades_sweeps': blades_lines} if blades_lines else {}

    def _format_blade_fit_entry(self, fit: dict) -> list[str]:
        """Format blade fit entry for a single blade."""
        parts = []
        for key in ('k', 'delta'):
            if key in fit and fit[key] is not None:
                try:
                    parts.append(f"{key}={float(fit[key]):.4g}")
                except Exception:
                    parts.append(f"{key}={fit[key]}")
        return parts

    def _format_bpm_stats_section(self, meta: dict) -> dict[str, list[str]]:
        """Format BPM statistics metadata section."""
        bpm_lines: list[str] = []
        bpm_stats = meta.get('bpm_stats', {}) if isinstance(meta, dict) else {}

        if isinstance(bpm_stats, dict):
            roi_size = getattr(self.prm, 'roisize', None)
            if isinstance(roi_size, (list, tuple)) and len(roi_size) >= 2:
                try:
                    bpm_lines.append(
                        f"  ROI size [H x V points] = {int(roi_size[0])} x {int(roi_size[1])}"
                    )
                    bpm_lines.append("")
                except Exception:
                    pass

            bpm_lines.append("  Sigmas (all sites):")
            for key in ('sigma_h', 'sigma_v', 'sigma_total'):
                if key in bpm_stats:
                    entry = self.BPM_STATS_DESCRIPTIONS.get(key, key)
                    try:
                        bpm_lines.append(
                            f"  {entry:>17} = {float(bpm_stats[key]):.4g}"
                        )
                    except Exception:
                        bpm_lines.append(f"  {entry:>17} = {bpm_stats[key]}")

            bpm_lines.append("")
            bpm_lines.append("")

            bpm_lines.append("  Extremes (all sites):")
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
                roi_sigma_h = bpm_stats.get('roi_sigma_h')
                roi_sigma_v = bpm_stats.get('roi_sigma_v')
                roi_sigma_t = bpm_stats.get('roi_sigma_total')
                if (roi_sigma_h is not None or
                    roi_sigma_v is not None or
                    roi_sigma_t is not None):
                    bpm_lines.append("")
                    bpm_lines.append("  Sigmas (ROI):")
                    if roi_sigma_h is not None:
                        bpm_lines.append(
                            f"  {'ROI horizontal RMS':>17} = {float(roi_sigma_h):.4g}"
                        )
                    if roi_sigma_v is not None:
                        bpm_lines.append(
                            f"  {'ROI vertical RMS':>17} = {float(roi_sigma_v):.4g}"
                        )
                    if roi_sigma_t is not None:
                        bpm_lines.append(
                            f"  {'ROI total RMS':>17} = {float(roi_sigma_t):.4g}"
                        )

        return {'bpm': bpm_lines} if bpm_lines else {}

    def _format_xbpm_stats_section(self, meta: dict,
                                   active_tab: str) -> dict[str, list[str]]:
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
            xbpm_stats = (meta.get('xbpm_stats_raw', {})
                          if isinstance(meta, dict) else {})
        elif 'scaled' in text or ' tr' in text:
            xbpm_stats = (meta.get('xbpm_stats_scaled', {})
                          if isinstance(meta, dict) else {})
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

    def _format_supmat_lines(self, meta: dict, active_tab: str) -> list[str]:
        """Format suppression matrix lines for the active tab."""
        lines: list[str] = []
        text = (active_tab or "").lower()

        # Raw pairwise tab: show standard suppression matrix
        is_pairwise = ('pair' in text) or ('xbpm' in text and 'part.' not in text)
        if 'raw' in text and is_pairwise:
            supmat = meta.get('supmat_standard')
            if supmat is not None:
                lines.append("\n  ** Standard Suppression Matrix:")
                lines.extend(self._format_matrix(supmat))

        # Scaled pairwise tab: show calculated suppression matrix
        elif ('scaled' in text or ' tr' in text) and is_pairwise:
            supmat = meta.get('supmat')
            if supmat is not None:
                lines.append("\n  ** Calculated Suppression Matrix:")
                lines.extend(self._format_matrix(supmat))

        return lines

    def _format_matrix(self, supmat) -> list[str]:
        """Pretty-print suppression matrix rows."""
        try:
            arr = np.asarray(supmat, dtype=float)
        except Exception:
            return [f"  {supmat}"]

        if arr.ndim != 2 or arr.size == 0:
            return [f"  {arr}"]

        return [
            "  " + " ".join(f"{val:8.4f}" for val in row)
            for row in arr
        ]

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

    def _refresh_analysis_info(self, tab_index=None):
        # If UI not fully built yet, skip
        if not hasattr(self, 'analysis_info') or self.analysis_info is None:
            return

        meta = self._last_meta or self._last_hdf5_meta or {}
        try:
            current_tab = (
                self.results_tabs.tabText(tab_index)
                if tab_index is not None else
                self.results_tabs.tabText(self.results_tabs.currentIndex())
            )
        except Exception:
            current_tab = ""
        text = self._format_analysis_info(meta, current_tab)
        self.analysis_info.setText(text)

    @pyqtSlot(int)
    def _on_tab_changed(self, index: int):
        """Update analysis info when the active tab changes."""
        self._refresh_analysis_info(index)

    def _plot_positions(self, canvas: MatplotlibCanvas, data, title: str):
        """Plot x/y positions; support optional nominal overlay."""
        try:
            if isinstance(data, dict):
                measured = data.get('measured')
                nominal = data.get('nominal')
            else:
                measured = data
                nominal = None

            if measured is not None:
                arr = np.array(measured)
            else:
                arr = None

            if arr is not None and arr.ndim == 2 and arr.shape[1] >= 2:
                canvas.ax.scatter(arr[:, 0], arr[:, 1], s=12,
                                  alpha=0.8, label='Measured')
                if nominal is not None:
                    nom = np.array(nominal)
                    if nom.ndim == 2 and nom.shape[1] >= 2:
                        canvas.ax.scatter(
                            nom[:, 0], nom[:, 1], s=24, marker='+',
                            color='red', label='Nominal'
                        )
                canvas.ax.set_xlabel('X')
                canvas.ax.set_ylabel('Y')
                canvas.ax.set_title(title)
                canvas.ax.grid(True, alpha=0.3)
                if nominal is not None:
                    canvas.ax.legend()
            else:
                canvas.ax.text(
                    0.5, 0.5, "Unsupported data shape",
                    ha='center', va='center',
                    transform=canvas.ax.transAxes)
        except Exception as exc:  # pragma: no cover - defensive
            canvas.ax.text(
                0.5, 0.5, f"Plot error: {exc}",
                ha='center', va='center',
                transform=canvas.ax.transAxes)
        canvas.canvas.draw_idle()

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

    def _show_detail_figures(self, results: dict):
        """Display detailed XBPM position figures in popup windows.

        Opens auxiliary windows for pairwise and cross-blade calculations
        if available.
        """
        figures_to_show = [
            ('xbpm_raw_pairwise_figure',
             'XBPM Raw - Δ/Σ Positions'),
            ('xbpm_raw_cross_figure',
             'XBPM Raw - Partial Δ/Σ Positions'),
            ('xbpm_scaled_pairwise_figure',
             'XBPM Scaled - Δ/Σ Positions'),
            ('xbpm_scaled_cross_figure',
             'XBPM Scaled - Partial Δ/Σ Positions'),
        ]

        found_any = False
        for fig_key, title in figures_to_show:
            if fig_key in results and results[fig_key] is not None:
                self.log_message(f"Opening detail figure: {title}")
                self._show_figure_in_window(results[fig_key], title)
                found_any = True
            else:
                logger.debug("Figure not found or None: %s", fig_key)

        if not found_any:
            logger.debug("No detail figures available in results")

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

    def closeEvent(self, event):  # noqa: N802
        """Clean up worker thread and detail windows on close."""
        # Close all detail windows
        if hasattr(self, '_detail_windows'):
            for window in self._detail_windows:
                try:
                    window.close()
                except Exception:  # pragma: no cover
                    logger.exception("Failed to close detail window")

        # Clean up worker thread
        if hasattr(self, 'worker_thread'):
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()
