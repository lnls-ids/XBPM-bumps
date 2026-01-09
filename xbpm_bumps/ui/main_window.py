"""Main window for XBPM analysis application."""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QSplitter, QTabWidget,
    QStatusBar, QProgressBar, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
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

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.canvases = {}
        self._last_workdir = ""
        self._preselected_beamline = None
        self._last_meta = {}
        self._last_hdf5_meta = {}
        self.setup_ui()
        self.setup_worker_thread()
        self.setWindowTitle("XBPM Beam Position Analysis")
        # Wider default window to give canvases more horizontal room
        self.resize(1600, 900)

    def setup_ui(self):
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
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Parameter input panel
        self.param_panel = ParameterPanel()
        self.param_panel.parametersChanged.connect(
            self._on_parameters_changed
        )
        layout.addWidget(self.param_panel)

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
        self.results_tabs.addTab(bpm_tab, "BPM Positions")
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
        self.results_tabs.addTab(xbpm_raw_pw_tab, "XBPM Raw Pairwise")
        self.canvases["xbpm_raw_pairwise"] = xbpm_raw_pw_canvas

        xbpm_scaled_pw_tab, xbpm_scaled_pw_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_scaled_pw_tab, "XBPM Scaled Pairwise")
        self.canvases["xbpm_scaled_pairwise"] = xbpm_scaled_pw_canvas

        xbpm_raw_cr_tab, xbpm_raw_cr_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_raw_cr_tab, "XBPM Raw Cross")
        self.canvases["xbpm_raw_cross"] = xbpm_raw_cr_canvas

        xbpm_scaled_cr_tab, xbpm_scaled_cr_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_scaled_cr_tab, "XBPM Scaled Cross")
        self.canvases["xbpm_scaled_cross"] = xbpm_scaled_cr_canvas

        return self.results_tabs

    def _on_parameters_changed(self):
        """React to parameter changes; pre-select beamline on workdir set."""
        params = self.param_panel.get_parameters()
        workdir = params.get('workdir') or ""
        if not workdir or workdir == self._last_workdir:
            return

        self._last_workdir = workdir
        self._preselected_beamline = None
        self._prompt_beamline_selection(workdir)

    def _create_status_bar(self):
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
            from ..core.readers import DataReader
            from ..core.parameters import Prm
            import sys
            from io import StringIO

            reader = DataReader(Prm(workdir=workdir))

            # Capture stdout/stderr during read
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            log_capture = StringIO()
            try:
                sys.stdout = log_capture
                sys.stderr = log_capture

                # Perform actual read to populate rawdata and beamline
                data = reader.read(
                    beamline_selector=self._create_beamline_selector()
                )
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Log captured output
            self._log_captured_output(log_capture)

            # Handle beamline result
            if reader.prm.beamline:
                self._preselected_beamline = reader.prm.beamline
                self.log_message(
                    f"Preselected beamline: {reader.prm.beamline}"
                )
                # Update XBPM distance based on selected beamline
                self._update_xbpmdist_from_beamline(reader.prm.beamline)
            else:
                self._handle_fallback_beamline_selection(reader, workdir, data)

        except Exception as exc:  # pragma: no cover - defensive
            self.log_message(f"Beamline preselection failed: {exc}")

    def _create_beamline_selector(self):
        """Create beamline selector function for UI dialog."""
        def selector(bls):
            if len(bls) == 1:
                return bls[0]
            dialog = BeamlineSelectionDialog(sorted(bls))
            if dialog.exec_() == dialog.Accepted:
                return dialog.get_selection()
            return None
        return selector

    def _log_captured_output(self, log_capture):
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

    def _handle_fallback_beamline_selection(self, reader, workdir, data):
        """Handle beamline selection when initial read doesn't set it."""
        import os

        beamlines = []
        if (os.path.isfile(workdir) and getattr(reader, "rawdata", None)):
            beamlines = reader._extract_beamlines_fallback(reader.rawdata)
        elif data:
            beamlines = reader._extract_beamlines(reader.rawdata)

        if len(beamlines) == 1:
            self._preselected_beamline = beamlines[0]
            self.log_message(f"Auto-selected beamline: {beamlines[0]}")
            self._update_xbpmdist_from_beamline(beamlines[0])

    def setup_worker_thread(self):
        """Initialize worker thread for analysis execution."""
        self.worker_thread = QThread()
        self.analyzer = XBPMAnalyzer()
        self.analyzer.moveToThread(self.worker_thread)

        self.analysisRequested.connect(self.analyzer.run_analysis)
        self.stop_btn.clicked.connect(self.analyzer.stop_analysis)

        self.analyzer.analysisStarted.connect(self._on_analysis_started)
        self.analyzer.analysisProgress.connect(self._on_analysis_progress)
        self.analyzer.analysisComplete.connect(self._on_analysis_complete)
        self.analyzer.analysisError.connect(self._on_analysis_error)
        self.analyzer.logMessage.connect(self.log_message)

        self.analyzer.beamlineSelectionNeeded.connect(
            self._on_beamline_selection_request
        )

        self.worker_thread.start()

    @pyqtSlot(list)
    def _on_beamline_selection_request(self, beamlines: list):
        """Show beamline selection dialog on the UI thread."""
        dialog = BeamlineSelectionDialog(sorted(beamlines))
        choice = ""
        if dialog.exec_() == dialog.Accepted:
            choice = dialog.get_selection() or ""
        self.analyzer.beamlineSelected.emit(choice)

    @pyqtSlot()
    def _on_analysis_started(self):
        """Handle analysis started signal."""
        self.set_analysis_running(True)
        self.log_message("=" * 60)
        self.log_message("Analysis started")
        self.log_message("=" * 60)

    @pyqtSlot(str)
    def _on_analysis_progress(self, message: str):
        """Handle analysis progress update."""
        self.status_bar.showMessage(message)
        self.log_message(f"[PROGRESS] {message}")

    @pyqtSlot(dict)
    def _on_analysis_complete(self, results: dict):
        """Handle analysis completion."""
        self._last_results = results
        self.set_analysis_running(False)
        self.log_message("=" * 60)
        self.log_message("Analysis completed successfully!")
        self.log_message("=" * 60)

        self._update_canvases(results)
        self._last_meta = self._collect_meta_from_results(results)
        self._refresh_analysis_info()

        self.status_bar.showMessage("Analysis complete", 5000)

    def _on_export_clicked(self):
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
                f"xbpm_export_{self.analyzer.app.prm.beamline}.dat"
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

            params = self.param_panel.get_parameters()
            results = getattr(self, '_last_results', {})
            exported_any = False

            # Export XBPM data and figures
            exported_any |= self._export_xbpm_raw(prefix, params)
            exported_any |= self._export_xbpm_scaled(prefix, params)

            # Export other analysis figures and data
            exported_any |= self._export_other_figures(prefix, params, results)

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
    def _on_export_hdf5_clicked(self):
        """Export full analysis package to an HDF5 file."""
        try:
            # Ensure app is initialized and we have results
            if not hasattr(self, 'analyzer') or not self.analyzer.app:
                QMessageBox.warning(
                    self,
                    "Unavailable",
                    (
                        "Run analysis at least once "
                        "before exporting to HDF5."
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
    @pyqtSlot()
    def _on_open_directory(self):
        """Open dialog to select working directory with pickle files."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            os.getcwd(),
        )

        if path:
            # Store workdir in parameter panel and update status bar
            self.param_panel.set_workdir(path)
            self.status_bar.showMessage(f"Opened: {path}")
            self._on_parameters_changed()

    @pyqtSlot()
    def _on_open_hdf5(self):
        """Open dialog to select HDF5 data file."""
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
            self._on_parameters_changed()

            # Try to load and display figures from HDF5
            self._on_load_hdf5_figures(path)

    def _on_load_hdf5_figures(self, hdf5_path: str) -> None:
        """Load figures from HDF5 file and display them.

        Args:
            hdf5_path: Path to the HDF5 file.
        """
        try:
            from ..core.readers import DataReader
            from ..core.parameters import Prm

            prm = Prm(workdir=hdf5_path)
            reader = DataReader(prm)
            figures = reader.load_figures_from_hdf5(hdf5_path)
            # Preserve analysis metadata (scales, BPM stats, etc.) for reuse
            self._last_hdf5_meta = getattr(reader, 'analysis_meta', {})
            self._last_meta = self._last_hdf5_meta

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
             "XBPM Raw Pairwise"),
            ('xbpm_raw_cross_figure', 'xbpm_raw_cross', "XBPM Raw Cross"),
            ('xbpm_scaled_pairwise_figure', 'xbpm_scaled_pairwise',
             "XBPM Scaled Pairwise"),
            ('xbpm_scaled_cross_figure', 'xbpm_scaled_cross',
             "XBPM Scaled Cross"),
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
    def _on_help_clicked(self):
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

    def _export_xbpm_raw(self, prefix: str, params: dict) -> bool:
        """Export raw XBPM positions and figures.

        Args:
            prefix: Export filename prefix.
            params: Parameter dictionary from UI.

        Returns:
            True if export occurred, False otherwise.
        """
        if not params.get('xbpmpositionsraw'):
            return False

        from ..core.exporters import Exporter

        exporter = Exporter(self.analyzer.app.prm)
        processor = self.analyzer.app.processor

        result_raw = processor.calculate_raw_positions(showmatrix=True)
        exporter.data_dump_with_prefix(
            prefix,
            self.analyzer.app.data,
            result_raw['positions'],
            sup="raw",
        )

        # Save figures
        if result_raw.get('pairwise_figure'):
            fig_pair = os.path.join(
                os.path.dirname(prefix),
                f"{os.path.basename(prefix)}_pair_raw.png"
            )
            result_raw['pairwise_figure'].savefig(
                fig_pair, dpi=FIGDPI, bbox_inches='tight'
            )
            logger.info("Pairwise figure saved to %s", fig_pair)

        if result_raw.get('cross_figure'):
            fig_cross = os.path.join(
                os.path.dirname(prefix),
                f"{os.path.basename(prefix)}_cross_raw.png"
            )
            result_raw['cross_figure'].savefig(
                fig_cross, dpi=FIGDPI, bbox_inches='tight'
            )
            logger.info("Cross figure saved to %s", fig_cross)

        # Export scaling factors if available
        scales = result_raw.get('scales') or {}
        pair_scales = scales.get('pair')
        if pair_scales:
            scale_path = os.path.join(
                os.path.dirname(prefix),
                f"{os.path.basename(prefix)}_pair_raw_scales.dat"
            )
            with open(scale_path, 'w') as fp:
                fp.write("kx ky dx dy\n")
                fp.write(
                    f"{pair_scales.get('kx', 0):.6f} "
                    f"{pair_scales.get('ky', 0):.6f} "
                    f"{pair_scales.get('dx', 0):.6f} "
                    f"{pair_scales.get('dy', 0):.6f}\n"
                )
            logger.info("Pairwise raw scales saved to %s", scale_path)

        return True

    def _export_xbpm_scaled(self, prefix: str, params: dict) -> bool:
        """Export scaled XBPM positions and figures.

        Args:
            prefix: Export filename prefix.
            params: Parameter dictionary from UI.

        Returns:
            True if export occurred, False otherwise.
        """
        if not params.get('xbpmpositions'):
            return False

        from ..core.exporters import Exporter

        exporter = Exporter(self.analyzer.app.prm)
        processor = self.analyzer.app.processor

        result_scaled = processor.calculate_scaled_positions(showmatrix=True)
        exporter.data_dump_with_prefix(
            prefix,
            self.analyzer.app.data,
            result_scaled['positions'],
            sup="scaled",
        )

        # Save figures
        if result_scaled.get('pairwise_figure'):
            fig_pair = os.path.join(
                os.path.dirname(prefix),
                f"{os.path.basename(prefix)}_pair_scaled.png"
            )
            result_scaled['pairwise_figure'].savefig(
                fig_pair, dpi=FIGDPI, bbox_inches='tight'
            )
            logger.info("Pairwise figure saved to %s", fig_pair)

        if result_scaled.get('cross_figure'):
            fig_cross = os.path.join(
                os.path.dirname(prefix),
                f"{os.path.basename(prefix)}_cross_scaled.png"
            )
            result_scaled['cross_figure'].savefig(
                fig_cross, dpi=FIGDPI, bbox_inches='tight'
            )
            logger.info("Cross figure saved to %s", fig_cross)

        # Export scaling factors if available
        scales = result_scaled.get('scales') or {}
        pair_scales = scales.get('pair')
        cross_scales = scales.get('cross')

        if pair_scales:
            scale_path = os.path.join(
                os.path.dirname(prefix),
                f"{os.path.basename(prefix)}_pair_scaled_scales.dat"
            )
            with open(scale_path, 'w') as fp:
                fp.write("kx ky dx dy\n")
                fp.write(
                    f"{pair_scales.get('kx', 0):.6f} "
                    f"{pair_scales.get('ky', 0):.6f} "
                    f"{pair_scales.get('dx', 0):.6f} "
                    f"{pair_scales.get('dy', 0):.6f}\n"
                )
            logger.info("Pairwise scaled scales saved to %s", scale_path)

        if cross_scales:
            scale_path = os.path.join(
                os.path.dirname(prefix),
                f"{os.path.basename(prefix)}_cross_scaled_scales.dat"
            )
            with open(scale_path, 'w') as fp:
                fp.write("kx ky dx dy\n")
                fp.write(
                    f"{cross_scales.get('kx', 0):.6f} "
                    f"{cross_scales.get('ky', 0):.6f} "
                    f"{cross_scales.get('dx', 0):.6f} "
                    f"{cross_scales.get('dy', 0):.6f}\n"
                )
            logger.info("Cross scaled scales saved to %s", scale_path)

        return True

    def _export_other_figures(self, prefix: str, params: dict,
                              results: dict) -> bool:
        """Export analysis figures (blade map, sweeps, etc) and sweeps data.

        Args:
            prefix: Export filename prefix.
            params: Parameter dictionary from UI.
            results: Results dictionary from last analysis.

        Returns:
            True if any export occurred, False otherwise.
        """
        exported = False

        # Blade map and blades-at-center figures
        figure_exports = [
            ('showblademap', 'blade_figure', 'blade_map.png'),
            ('showbladescenter', 'blades_center_figure',
             'blades_center.png'),
        ]

        for option_key, result_key, filename in figure_exports:
            if params.get(option_key) and result_key in results:
                fig = results.get(result_key)
                if fig is not None:
                    fig_path = os.path.join(
                        os.path.dirname(prefix), filename
                    )
                    fig.savefig(fig_path, dpi=FIGDPI, bbox_inches='tight')
                    logger.info("Figure saved to %s", fig_path)
                    exported = True

        # Central sweeps: export figure and data when available
        if params.get('centralsweep'):
            if ('sweeps_figure' in results and
                results['sweeps_figure'] is not None):
                fig_path = os.path.join(
                    os.path.dirname(prefix), 'central_sweeps.png'
                )
                results['sweeps_figure'].savefig(
                    fig_path, dpi=FIGDPI, bbox_inches='tight'
                )
                logger.info("Figure saved to %s", fig_path)
                exported = True

            sweeps_data = results.get('sweeps_data')
            if sweeps_data:
                range_h, range_v, blades_h, blades_v = sweeps_data
                try:
                    # blades_h and blades_v are dicts keyed by blade labels
                    h_arrays = [blades_h.get(k)
                                for k in ("to", "ti", "bi", "bo")]
                    v_arrays = [blades_v.get(k)
                                for k in ("to", "ti", "bi", "bo")]

                    if all(arr is not None for arr in h_arrays):
                        h_out = os.path.join(
                            os.path.dirname(prefix),
                            'central_sweeps_horizontal.dat'
                        )
                        h_cols = np.column_stack([range_h, *h_arrays])
                        np.savetxt(
                            h_out, h_cols,
                            header="range_h to ti bi bo",
                            fmt="%.6f"
                        )
                        logger.info("Horizontal sweeps data saved to %s",
                                    h_out)
                        exported = True

                    if all(arr is not None for arr in v_arrays):
                        v_out = os.path.join(
                            os.path.dirname(prefix),
                            'central_sweeps_vertical.dat'
                        )
                        v_cols = np.column_stack([range_v, *v_arrays])
                        np.savetxt(
                            v_out, v_cols,
                            header="range_v to ti bi bo",
                            fmt="%.6f"
                        )
                        logger.info("Vertical sweeps data saved to %s", v_out)
                        exported = True
                except Exception:
                    logger.exception("Failed to save central sweeps data")

        return exported

    @pyqtSlot(str, str)
    def _on_analysis_error(self, title: str, message: str):
        """Handle analysis error.

        Args:
            title: Error dialog title.
            message: Error message.
        """
        self.set_analysis_running(False)
        self.show_error(title, message)

    @pyqtSlot()
    def _on_run_clicked(self):
        """Handle Run Analysis button click."""
        params = self.param_panel.get_parameters()
        if not self._validate_workdir(params):
            return

        params = self._ensure_beamline(params)
        self.log_message(
            "Starting analysis with workdir: "
            f"{params['workdir']}"
        )
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

    def _ensure_beamline(self, params: dict) -> dict:
        """Apply preselected beamline or prompt selection if needed."""
        try:
            from ..core.readers import DataReader
            from ..core.parameters import Prm
            import os

            reader = DataReader(Prm(workdir=params['workdir']))

            # Use any preselected beamline from workdir change
            if self._preselected_beamline:
                params['beamline'] = self._preselected_beamline

            if params.get('beamline'):
                return params

            beamlines = []
            if os.path.isfile(params['workdir']):
                # For files, skip heavy parsing here;
                # defer to analyzer if needed
                beamlines = reader._extract_beamlines_from_header()
            else:
                rawdata = reader._get_pickle_data()
                beamlines = reader._extract_beamlines(rawdata)
                if not beamlines:
                    beamlines = reader._extract_beamlines_fallback(rawdata)

            if len(beamlines) == 1:
                params['beamline'] = beamlines[0]
                self._preselected_beamline = beamlines[0]
                self.log_message(f"Auto-selected beamline: {beamlines[0]}")
                # Update XBPM distance based on auto-selected beamline
                self._update_xbpmdist_from_beamline(beamlines[0])
            elif len(beamlines) > 1:
                if not self._prompt_beamline_dialog(beamlines, params):
                    return params
        except Exception as exc:  # pragma: no cover - defensive
            self.log_message("Warning: Could not pre-extract beamlines:"
                             f" {exc}")

        return params

    def _prompt_beamline_dialog(self, beamlines: list, params: dict) -> bool:
        """Show beamline dialog; return False if user cancels or no choice."""
        dialog = BeamlineSelectionDialog(sorted(beamlines))
        if dialog.exec_() != dialog.Accepted:
            return False

        selected = dialog.get_selection()
        if selected:
            params['beamline'] = selected
            self._preselected_beamline = selected
            self.log_message(f"Selected beamline: {selected}")
            # Update XBPM distance based on user-selected beamline
            self._update_xbpmdist_from_beamline(selected)
            return True

        QMessageBox.warning(self, "No Selection", "Please select a beamline.")
        return False

    def log_message(self, message: str):
        """Append a message to the console log.

        Args:
            message: Text to append to console.
        """
        self.console.append(message)

    def set_analysis_running(self, running: bool):
        """Update UI state during analysis execution.

        Args:
            running: True if analysis is running, False otherwise.
        """
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

    def show_error(self, title: str, message: str):
        """Display error dialog.

        Args:
            title: Error dialog title.
            message: Error message text.
        """
        QMessageBox.critical(self, title, message)
        self.log_message(f"ERROR: {message}")

    def show_results_tab(self, tab_name: str):
        """Switch to a specific results tab.

        Args:
            tab_name: Name of the tab to show.
        """
        for i in range(self.results_tabs.count()):
            if self.results_tabs.tabText(i) == tab_name:
                self.results_tabs.setCurrentIndex(i)
                break

    def _create_canvas_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        canvas = MatplotlibCanvas()
        layout.addWidget(canvas)
        return widget, canvas

    def _update_canvases(self, results: dict):
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
        meta: dict = {}
        if not isinstance(results, dict):
            return meta

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

        bpm_stats = results.get('bpm_stats')
        if bpm_stats:
            meta['bpm_stats'] = bpm_stats

        sweeps_meta = self._extract_sweeps_meta(results.get('sweeps_data'))
        if sweeps_meta:
            meta['sweeps'] = sweeps_meta

        if 'supmat_standard' in results:
            meta['supmat_standard'] = results.get('supmat_standard')
        if 'supmat' in results:
            meta['supmat'] = results.get('supmat')

        return meta

    def _extract_sweeps_meta(self, sweeps_data):
        """Extract sweep metadata: positions fits and per-blade fits."""
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

    def _extract_sweeps_positions(self, fit_h, fit_v) -> dict:
        """Extract global fit positions from horizontal and vertical fits."""
        positions = {}
        h_meta = self._parse_fit(fit_h)
        if h_meta:
            positions['horizontal'] = h_meta
        v_meta = self._parse_fit(fit_v)
        if v_meta:
            positions['vertical'] = v_meta
        return positions

    def _parse_fit(self, fit):
        """Parse fit parameters into k and delta values."""
        try:
            k_val = (float(fit[0][0])
                     if hasattr(fit, "__len__") else float(fit[0]))
            delta_val = (float(fit[1][0])
                         if hasattr(fit, "__len__") else float(fit[1]))
            return {'k': k_val, 'delta': delta_val}
        except Exception:
            return None

    def _extract_sweeps_blades(self, blades_h, blades_v, range_h,
                              range_v) -> dict:
        """Extract per-blade fits from blade data."""
        blades = {}
        h_blades = self._fit_blades(blades_h, range_h)
        if h_blades:
            blades['horizontal'] = h_blades
        v_blades = self._fit_blades(blades_v, range_v)
        if v_blades:
            blades['vertical'] = v_blades
        return blades

    def _fit_blades(self, blades_dict, axis_range):
        """Fit blade data to extract k and delta for each blade."""
        if not isinstance(blades_dict, dict) or axis_range is None:
            return None
        fits = {}
        for blade in ('to', 'ti', 'bi', 'bo'):
            arr = blades_dict.get(blade)
            if arr is None:
                continue
            y = arr[:, 0] if hasattr(arr, 'ndim') and arr.ndim == 2 else arr
            try:
                coef = np.polyfit(axis_range, y, deg=1)
                fits[blade] = {
                    'k': float(coef[0]),
                    'delta': float(coef[1])
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
        if 'scaled' in text:
            scope = 'scaled'

        if 'pair' in text:
            label = 'pair'
        if 'cross' in text:
            label = 'cross'

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
                    scale_lines.append(f"  * {scope} {label}:")
                    scale_lines.extend(lines_to_add)

        return {'positions': scale_lines} if scale_lines else {}

    def _format_scale_entry(self, coeffs: dict) -> list[str]:
        """Format individual scale coefficient entry."""
        lines_to_add: list[str] = []
        line1 = []
        line2 = []

        for key in ('kx', 'dx'):
            val = coeffs.get(key)
            if val is not None:
                try:
                    line1.append(f"{key:>10} = {float(val):.4g}")
                except Exception:
                    line1.append(f"{key:>10} = {val}")

        for key in ('ky', 'dy'):
            val = coeffs.get(key)
            if val is not None:
                try:
                    line2.append(f"{key:>10} = {float(val):.4g}")
                except Exception:
                    line2.append(f"{key:>10} = {val}")

        if line1:
            lines_to_add.append("   " + ", ".join(line1))
        if line2:
            lines_to_add.append("   " + ", ".join(line2))

        return lines_to_add

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
                sweeps_pos_lines.append(f"\n  * {label}:")
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
                    blades_lines.append(f"\n  * {label} {blade}:")
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
            for key in ('sigma_h', 'sigma_v', 'sigma_total',
                       'diff_max_h', 'diff_max_v'):
                if key in bpm_stats:
                    try:
                        bpm_lines.append(
                            f"  {key:>17}  =  {float(bpm_stats[key]):.4g}"
                        )
                    except Exception:
                        bpm_lines.append(f"  {key:>17}  =  {bpm_stats[key]}")

        return {'bpm': bpm_lines} if bpm_lines else {}

    def _format_supmat_lines(self, meta: dict, active_tab: str) -> list[str]:
        """Format suppression matrix lines for the active tab."""
        lines: list[str] = []
        text = (active_tab or "").lower()

        # Raw pairwise tab: show standard suppression matrix
        if 'raw' in text and 'pair' in text:
            supmat = meta.get('supmat_standard')
            if supmat is not None:
                lines.append("\n  ** Standard Suppression Matrix:")
                lines.extend(self._format_matrix(supmat))

        # Scaled pairwise tab: show calculated suppression matrix
        elif 'scaled' in text and 'pair' in text:
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
            prefix = "** " if name == active_section else ""
            lines.append(f"{prefix}{name.capitalize()}:")
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
            ('xbpm_raw_pairwise_figure', 'XBPM Raw - Pairwise Positions'),
            ('xbpm_raw_cross_figure', 'XBPM Raw - Cross-blades Positions'),
            ('xbpm_scaled_pairwise_figure',
             'XBPM Scaled - Pairwise Positions'),
            ('xbpm_scaled_cross_figure',
             'XBPM Scaled - Cross-blades Positions'),
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
