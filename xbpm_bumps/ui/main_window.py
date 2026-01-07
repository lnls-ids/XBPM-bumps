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
        self.setup_ui()
        self.setup_worker_thread()
        self.setWindowTitle("XBPM Beam Position Analysis")
        self.resize(1200, 800)

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

        # Set initial splitter sizes (30% controls, 70% results)
        splitter.setSizes([360, 840])

        # Status bar
        self._create_status_bar()

        # Menubar: File
        file_menu = self.menuBar().addMenu("File")
        open_action = file_menu.addAction("Open…")
        open_action.triggered.connect(self._on_open_workdir)

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
        """Create the left control panel with parameters and buttons."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Parameter input panel
        self.param_panel = ParameterPanel()
        self.param_panel.parametersChanged.connect(
            self._on_parameters_changed
        )
        layout.addWidget(self.param_panel)

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

        sweep_tab, sweep_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(sweep_tab, "Central Sweeps")
        self.canvases["sweeps"] = sweep_canvas

        blades_center_tab, blades_center_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(blades_center_tab, "Blades at Center")
        self.canvases["blades_center"] = blades_center_canvas

        xbpm_raw_tab, xbpm_raw_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_raw_tab, "XBPM Raw")
        self.canvases["xbpm_raw"] = xbpm_raw_canvas

        xbpm_scaled_tab, xbpm_scaled_canvas = self._create_canvas_tab()
        self.results_tabs.addTab(xbpm_scaled_tab, "XBPM Scaled")
        self.canvases["xbpm_scaled"] = xbpm_scaled_canvas

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
        """Update the XBPM distance field from Config.XBPMDISTS.

        Args:
            beamline: Selected beamline code (e.g., 'MNC1').
        """
        try:
            if not beamline:
                return
            dist = Config.XBPMDISTS.get(beamline)
            if dist is not None and hasattr(self, 'param_panel'):
                # Update UI spinbox to reflect auto value from beamline
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
        if (os.path.isfile(workdir) and
            getattr(reader, "rawdata", None)):
            beamlines = reader._extract_beamlines_fallback(
                reader.rawdata
            )
        elif data:
            beamlines = reader._extract_beamlines(reader.rawdata)

        if len(beamlines) == 1:
            self._preselected_beamline = beamlines[0]
            self.log_message(
                f"Auto-selected beamline: {beamlines[0]}"
            )
            # Update XBPM distance based on auto-selected beamline
            self._update_xbpmdist_from_beamline(beamlines[0])

    def setup_worker_thread(self):
        """Initialize worker thread for analysis execution."""
        # Create worker thread
        self.worker_thread = QThread()

        # Create analyzer and move to thread
        self.analyzer = XBPMAnalyzer()
        self.analyzer.moveToThread(self.worker_thread)

        # Connect signals
        self.analysisRequested.connect(self.analyzer.run_analysis)
        self.stop_btn.clicked.connect(self.analyzer.stop_analysis)

        self.analyzer.analysisStarted.connect(self._on_analysis_started)
        self.analyzer.analysisProgress.connect(self._on_analysis_progress)
        self.analyzer.analysisComplete.connect(self._on_analysis_complete)
        self.analyzer.analysisError.connect(self._on_analysis_error)
        self.analyzer.logMessage.connect(self.log_message)

        # Beamline selection happens on UI thread
        self.analyzer.beamlineSelectionNeeded.connect(
            self._on_beamline_selection_request
        )

        # Start thread
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
        """Handle analysis progress update.

        Args:
            message: Progress message.
        """
        self.status_bar.showMessage(message)
        self.log_message(f"[PROGRESS] {message}")

    @pyqtSlot(dict)
    def _on_analysis_complete(self, results: dict):
        """Handle analysis completion.

        Args:
            results: Dictionary with analysis results.
        """
        # Keep last results for potential export
        self._last_results = results
        self.set_analysis_running(False)
        self.log_message("=" * 60)
        self.log_message("Analysis completed successfully!")
        self.log_message("=" * 60)

        self._update_canvases(results)
        self._show_detail_figures(results)

        # TODO: Populate result tabs with visualization widgets
        # For now, just log what we have
        if 'positions' in results:
            for key in results['positions']:
                self.log_message(f"Generated positions: {key}")

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
            exporter.write_hdf5(path, self.analyzer.app.data, results,
                                include_figures=True)

            self.log_message(f"HDF5 export written: {path}")
            QMessageBox.information(
                self,
                "Export Complete",
                "Exported analysis and figures to HDF5.",
            )
        except Exception as exc:  # pragma: no cover
            self.show_error("Export to HDF5 Failed", str(exc))

    @pyqtSlot()
    def _on_open_workdir(self):
        """Open dialog to select working directory or data file."""
        # Prefer directory; fall back to file selection
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            os.getcwd(),
        )

        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Data File",
                os.getcwd(),
                "Data Files (*.dat *.txt *.h5 *.hdf5);;All Files (*)",
            )

        if path:
            # Store workdir in parameter panel and update status bar
            self.param_panel.set_workdir(path)
            self.status_bar.showMessage(f"Opened: {path}")
            self._on_parameters_changed()

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

        # XBPM raw positions
        xbpm_raw_canvas = self.canvases.get("xbpm_raw")
        xbpm_raw = positions.get('xbpm_raw')
        if xbpm_raw_canvas is not None and xbpm_raw is not None:
            self._plot_positions(xbpm_raw_canvas, xbpm_raw,
                               title="XBPM Raw Positions")

        # XBPM scaled positions
        xbpm_scaled_canvas = self.canvases.get("xbpm_scaled")
        xbpm_scaled = positions.get('xbpm_scaled')
        if xbpm_scaled_canvas is not None and xbpm_scaled is not None:
            self._plot_positions(xbpm_scaled_canvas, xbpm_scaled,
                               title="XBPM Scaled Positions")

        # BPM positions
        bpm_canvas = self.canvases.get("bpm")
        bpm_data = positions.get('bpm')
        if bpm_canvas is not None and bpm_data is not None:
            self._plot_positions(
                bpm_canvas, bpm_data, title="BPM Positions"
            )

        # Blade map - embed entire figure with 2x2 subplots
        blade_canvas = self.canvases.get("blade")
        if blade_canvas is not None and results.get('blade_figure'):
            self._embed_figure(blade_canvas, results['blade_figure'])

        # Central sweeps - blade current plots
        sweep_canvas = self.canvases.get("sweeps")
        if sweep_canvas is not None and results.get('blades_center_figure'):
            self._embed_figure(sweep_canvas, results['blades_center_figure'])

        # Blades at center - position plots
        blades_center_canvas = self.canvases.get("blades_center")
        if blades_center_canvas is not None and results.get('sweeps_figure'):
            self._embed_figure(blades_center_canvas, results['sweeps_figure'])

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

            # Replace canvas figure references
            canvas.figure = source_fig
            canvas.canvas.figure = source_fig

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
