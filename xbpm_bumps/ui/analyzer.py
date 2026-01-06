"""Qt-friendly wrapper for XBPM analysis core."""

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from typing import Optional, Dict, Any
import sys
from io import StringIO
import traceback

from ..core.app import XBPMApp
from ..core.parameters import Prm, ParameterBuilder
from ..core.readers import DataReader
from ..core.processors import XBPMProcessor
from ..core.exporters import Exporter


class XBPMAnalyzer(QObject):
    """Qt wrapper for XBPMApp with signals for async execution.

    This class wraps the core XBPMApp to make it Qt-friendly and thread-safe.
    Emits signals for progress updates, results, and errors.
    """

    # Signals
    analysisStarted = pyqtSignal()        # noqa: N815
    analysisProgress = pyqtSignal(str)    # noqa: N815
    analysisComplete = pyqtSignal(dict)   # noqa: N815
    analysisError = pyqtSignal(str, str)  # noqa: N815
    logMessage = pyqtSignal(str)          # noqa: N815

    def __init__(self, parent=None):
        """Initialize the analyzer.

        Args:
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.app: Optional[XBPMApp] = None
        self.prm: Optional[Prm] = None
        self._should_stop = False

    @pyqtSlot(dict)
    def run_analysis(self, params: Dict[str, Any]):
        """Execute XBPM analysis with given parameters.

        This method is designed to be called from a worker thread.

        Args:
            params: Dictionary of analysis parameters from ParameterPanel.
        """
        self._should_stop = False
        self.analysisStarted.emit()

        try:
            # Convert params dict to command-line style arguments
            argv = self._params_to_argv(params)
            self.logMessage.emit(f"Running with args: {' '.join(argv)}")

            # Create and configure app
            self.app = XBPMApp()

            # Capture stdout/stderr for logging
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            log_capture = StringIO()

            try:
                sys.stdout = log_capture
                sys.stderr = log_capture

                # Build parameters
                self.analysisProgress.emit("Building parameters...")
                self.app.builder = ParameterBuilder()
                self.app.prm = self.app.builder.from_cli(argv)
                self.prm = self.app.prm

                if self._should_stop:
                    return

                # Read data
                self.analysisProgress.emit("Reading data...")
                self.app.reader = DataReader(self.app.prm, self.app.builder)
                self.app.data = self.app.reader.read()

                # Emit captured output
                output = log_capture.getvalue()
                if output:
                    for line in output.split('\n'):
                        if line.strip():
                            self.logMessage.emit(line)

                if self._should_stop:
                    return

                # Initialize processor and exporter
                self.analysisProgress.emit("Initializing processor...")
                self.app.processor = XBPMProcessor(self.app.data, self.app.prm)
                self.app.exporter = Exporter(self.app.prm)

                # Run analysis steps
                results = self._run_analysis_steps()

                # Emit completion with results
                self.analysisComplete.emit(results)

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

                # Emit any remaining captured output
                output = log_capture.getvalue()
                if output:
                    for line in output.split('\n'):
                        if line.strip():
                            self.logMessage.emit(line)

        except Exception as e:
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.analysisError.emit("Analysis Failed", error_msg)

    def _params_to_argv(self, params: Dict[str, Any]) -> list:
        """Convert parameter dictionary to command-line argument list.

        Args:
            params: Dictionary from ParameterPanel.get_parameters()

        Returns:
            List of command-line style arguments.
        """
        argv = []

        # Required workdir
        if params.get('workdir'):
            argv.extend(['-w', params['workdir']])

        # Optional numeric parameters
        if params.get('xbpmdist') is not None:
            argv.extend(['-d', str(params['xbpmdist'])])
        if params.get('gridstep') is not None:
            argv.extend(['-g', str(params['gridstep'])])
        if params.get('skip', 0) > 0:
            argv.extend(['-k', str(params['skip'])])

        # Boolean flags - map parameter name to flag
        flags = {
            'xbpmpositions'   : '-x',
            'xbpmpositionsraw': '-r',
            'xbpmfrombpm'     : '-b',
            'showblademap'    : '-m',
            'centralsweep'    : '-c',
            'showbladescenter': '-s',
            'outputfile'      : '-o',
        }
        for param, flag in flags.items():
            if params.get(param):
                argv.append(flag)

        return argv

    def _run_analysis_steps(self) -> Dict[str, Any]:
        """Execute individual analysis steps and collect results.

        Returns:
            Dictionary with analysis results.
        """
        results = {
            'prm': self.app.prm,
            'data': self.app.data,
            'positions': {},
        }

        # Define analysis steps as a list of tuples: (condition, method)
        steps = [
            (self.app.prm.xbpmfrombpm, self._step_bpm_positions),
            (self.app.prm.showblademap, self._step_blade_map),
            (self.app.prm.centralsweep, self._step_central_sweeps),
            (self.app.prm.showbladescenter, self._step_blades_center),
            (self.app.prm.xbpmpositionsraw, self._step_xbpm_raw),
            (self.app.prm.xbpmpositions, self._step_xbpm_scaled),
        ]

        for condition, step_method in steps:
            if self._should_stop:
                return results
            if condition:
                step_method(results)

        self.analysisProgress.emit("Analysis complete!")
        return results

    def _step_bpm_positions(self, results: dict):
        """Calculate BPM positions."""
        self.analysisProgress.emit("Calculating BPM positions...")
        from ..core.processors import BPMProcessor
        raw = (
            self.app.reader.rawdata
            if self.app.reader.rawdata is not None
            else self.app.data
        )
        bpm_processor = BPMProcessor(raw, self.app.prm)
        bpm_positions = bpm_processor.calculate_positions()
        results['positions']['bpm'] = bpm_positions

    def _step_blade_map(self, results: dict):
        """Generate blade map."""
        self.analysisProgress.emit("Generating blade map...")
        results['showblademap'] = True

    def _step_central_sweeps(self, results: dict):
        """Analyze central sweeps."""
        self.analysisProgress.emit("Analyzing central sweeps...")
        supmat = self.app.processor.analyze_central_sweeps(show=False)
        results['supmat'] = supmat

    def _step_blades_center(self, results: dict):  # noqa: ARG002
        """Analyze blades at center."""
        self.analysisProgress.emit("Analyzing blades at center...")
        # Handled by processor

    def _step_xbpm_raw(self, results: dict):
        """Calculate raw XBPM positions."""
        self.analysisProgress.emit("Calculating raw XBPM positions...")
        positions = self.app.processor.calculate_raw_positions(
            showmatrix=True
        )
        results['positions']['xbpm_raw'] = positions

        if self.app.prm.outputfile:
            self.app.exporter.data_dump(self.app.data, positions, sup="raw")

    def _step_xbpm_scaled(self, results: dict):
        """Calculate scaled XBPM positions."""
        self.analysisProgress.emit("Calculating scaled XBPM positions...")
        positions = self.app.processor.calculate_scaled_positions(
            showmatrix=True
        )
        results['positions']['xbpm_scaled'] = positions

        if self.app.prm.outputfile:
            self.app.exporter.data_dump(self.app.data, positions, sup="scaled")

    @pyqtSlot()
    def stop_analysis(self):
        """Request analysis to stop at next checkpoint."""
        self._should_stop = True
        self.logMessage.emit("Stop requested...")
