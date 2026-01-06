"""Main window for XBPM analysis application."""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QSplitter, QTabWidget,
    QStatusBar, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QFont

from .widgets.parameter_panel import ParameterPanel
from .analyzer import XBPMAnalyzer


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

    def _create_control_panel(self) -> QWidget:
        """Create the left control panel with parameters and buttons."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Parameter input panel
        self.param_panel = ParameterPanel()
        layout.addWidget(self.param_panel)

        # Control buttons
        button_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._on_run_clicked)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)

        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.stop_btn)
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

        # Placeholder tabs for visualizations (to be populated later)
        self.results_tabs.addTab(QWidget(), "XBPM Positions")
        self.results_tabs.addTab(QWidget(), "BPM Positions")
        self.results_tabs.addTab(QWidget(), "Blade Map")
        self.results_tabs.addTab(QWidget(), "Central Sweeps")

        return self.results_tabs

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

        # Start thread
        self.worker_thread.start()

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
        self.set_analysis_running(False)
        self.log_message("=" * 60)
        self.log_message("Analysis completed successfully!")
        self.log_message("=" * 60)

        # TODO: Populate result tabs with visualization widgets
        # For now, just log what we have
        if 'positions' in results:
            for key in results['positions']:
                self.log_message(f"Generated positions: {key}")

        self.status_bar.showMessage("Analysis complete", 5000)

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

        # Validate workdir is provided
        if not params.get('workdir'):
            QMessageBox.warning(
                self,
                "Missing Input",
                "Please select a working directory or data file."
            )
            return

        # Emit signal with parameters
        self.log_message("Starting analysis with workdir:"
                         f" {params['workdir']}")
        self.analysisRequested.emit(params)

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

    def closeEvent(self, event):  # noqa: N802
        """Clean up worker thread on window close."""
        if hasattr(self, 'worker_thread'):
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()
