"""Qt-friendly wrapper for XBPM analysis core."""

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QEventLoop
from typing import Optional, Dict, Any
import sys
from io import StringIO
import traceback
import numpy as np

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

    # UI-thread beamline selection
    beamlineSelectionNeeded = pyqtSignal(list)  # noqa: N815
    beamlineSelected = pyqtSignal(str)          # noqa: N815

    def __init__(self, prm, parent=None):
        """Initialize the analyzer.

        Args:
            prm: Required persistent Prm instance to use.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self.app: Optional[XBPMApp] = None
        self.prm: Prm = prm
        self._should_stop = False
        self._preselected_beamline: Optional[str] = None

    @pyqtSlot(dict)
    def load_data_only(self, params: Dict[str, Any]):
        """Load data without running analysis.

        This allows exporting raw data to HDF5 without analysis results.

        Args:
            params: Parameters dictionary from UI.
        """
        try:
            # Convert params dict to command-line style arguments
            argv = self._params_to_argv(params)
            self.logMessage.emit(f"Loading data with args: {' '.join(argv)}")

            # Create and configure app
            self.app = XBPMApp()
            self._preselected_beamline = params.get('beamline')
            # Use persistent Prm if provided
            if self.prm is not None:
                self.app.prm = self.prm

            # Capture stdout/stderr for logging
            log_capture = StringIO()
            old_stdout = sys.stdout
            old_stderr = sys.stderr

            try:
                sys.stdout = log_capture
                sys.stderr = log_capture

                self._setup_app_and_read_data(
                    argv, old_stdout, old_stderr, log_capture
                )

                # After reading, enforce UI beamline selection if multiple
                self._maybe_select_beamline(
                    old_stdout, old_stderr, log_capture
                )

                # Set pre-selected beamline after reading if still unset
                if self._preselected_beamline and not self.app.prm.beamline:
                    self._set_beamline(self._preselected_beamline)

                self.logMessage.emit(
                    "Data loaded successfully (no analysis run)"
                    )

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
            self.analysisError.emit("Data Load Failed", error_msg)

    def _setup_app_and_read_data(self, argv: list,
                                  old_stdout, old_stderr, log_capture):
        """Setup app, build params, asks beamline to DataReader, and read data.

        Args:
            argv: Command-line arguments.
            old_stdout: Original stdout before capture.
            old_stderr: Original stderr before capture.
            log_capture: StringIO for logging.
        """
        self.analysisProgress.emit("Building parameters...")
        self.app.builder = ParameterBuilder()
        self.app.prm = self.prm
        self.app.builder.from_cli(argv)

        # Ask DataReader for available beamlines
        temp_reader = DataReader(self.prm, self.app.builder)
        beamlines = temp_reader.get_available_beamlines()
        beamlines = [str(b) for b in beamlines]

        # If multiple beamlines and none selected, prompt user for
        # string selection
        if not self.prm.beamline and beamlines and len(beamlines) > 1:
            selected = self._select_beamline_on_ui(
                beamlines, old_stdout, old_stderr, log_capture
            )
            chosen = selected or beamlines[0]
            self.prm.beamline = chosen

        # Now, with beamline set, read the data (DataReader will validate)
        self.analysisProgress.emit("Reading data...")
        self.app.reader = DataReader(self.prm, self.app.builder)

        def beamline_selector_ui(beamlines):
            beamlines = [str(b) for b in beamlines]
            return self._select_beamline_on_ui(
                beamlines, old_stdout, old_stderr, log_capture
            )

        # DEBUG
        print("[DEBUG] Calling DataReader.read with beamline_selector_ui"
              f"{beamlines}")
        # DEBUG

        self.app.data = self.app.reader.read(
            beamline_selector=beamline_selector_ui
            )

        # Emit captured output
        output = log_capture.getvalue()
        if output:
            for line in output.split('\n'):
                if line.strip():
                    self.logMessage.emit(line)

    def _initialize_and_run_analysis(self):
        """Initialize processor/exporter and run analysis steps."""
        # Initialize processor and exporter
        self.analysisProgress.emit("Initializing processor...")
        self.app.processor = XBPMProcessor(self.app.data, self.app.prm)
        self.app.exporter = Exporter(self.app.prm)

        # Run analysis steps
        results = self._run_analysis_steps()

        # DEBUG: Print results and Prm state before emitting
        import pprint
        print("[DEBUG] Analysis results:")
        pprint.pprint(results)
        print("[DEBUG] Prm state:")
        pprint.pprint(vars(self.app.prm)
                      if hasattr(self.app, 'prm') else self.app.prm)

        # Emit completion with results
        self.analysisComplete.emit(results)

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
            self._preselected_beamline = params.get('beamline')

            # Always build a fresh Prm from current UI parameters
            self.app.builder = ParameterBuilder()
            new_prm = self.app.builder.from_cli(argv)
            self.app.prm = new_prm
            self.prm = new_prm  # Update persistent reference as well
            # Capture stdout/stderr for logging
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            log_capture = StringIO()

            try:
                sys.stdout = log_capture
                sys.stderr = log_capture

                # If beamline was pre-selected in params (main thread),
                # remember it to apply after from_cli
                if self._preselected_beamline:
                    if self.app.prm is None:
                        if self.prm is not None:
                            self.app.prm = self.prm
                        # else: do not create a new Prm, rely on
                        # main window to provide
                    self.app.prm.beamline = self._preselected_beamline

                self._setup_app_and_read_data(
                    argv, old_stdout, old_stderr, log_capture
                )

                # After reading, enforce UI beamline selection if multiple
                self._maybe_select_beamline(
                    old_stdout, old_stderr, log_capture
                )

                if self._should_stop:
                    return

                # Set pre-selected beamline after reading if still unset
                if self._preselected_beamline and not self.app.prm.beamline:
                    self._set_beamline(self._preselected_beamline)

                if self._should_stop:
                    return

                self._initialize_and_run_analysis()

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

    # No longer needed: all beamline state is in self.prm

    def _select_beamline_on_ui(self, beamlines, old_stdout,
                               old_stderr, log_capture):
        """Request beamline choice on UI thread and wait for result."""
        loop = QEventLoop()
        choice_holder = {"choice": None}

        def on_selected(choice: str):
            choice_holder["choice"] = choice
            loop.quit()

        self.beamlineSelected.connect(on_selected)
        try:
            # Temporarily restore stdout/stderr so UI prints (if any) go there
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.beamlineSelectionNeeded.emit(sorted(beamlines))
            loop.exec_()
        finally:
            sys.stdout = log_capture
            sys.stderr = log_capture
            self.beamlineSelected.disconnect(on_selected)

        return choice_holder["choice"]

    def _maybe_select_beamline(
        self, old_stdout, old_stderr, log_capture
    ) -> None:
        """Handle beamline selection if multiple are detected.

        Args:
            old_stdout: Original stdout before capture.
            old_stderr: Original stderr before capture.
            log_capture: StringIO capturing stdout/stderr.
        """
        # If beamline already determined, skip selection
        if self.prm.beamline:
            return

        beamlines = self.app.reader.get_available_beamlines()
        if len(beamlines) <= 1:
            return

        selected = self._select_beamline_on_ui(
            beamlines, old_stdout, old_stderr, log_capture
        )

        # If user cancels, default to first to avoid terminal prompt
        chosen = selected or beamlines[0]
        self.prm.beamline = chosen
        rawdata = getattr(self.app.reader, "rawdata", None) or self.app.data
        self.app.builder.enrich_from_data(
            rawdata, selected_beamline=chosen
        )

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
        measured, nominal = bpm_processor.calculate_positions()
        stats = getattr(bpm_processor, 'last_stats', None)
        results['positions']['bpm'] = {
            'measured': measured,
            'nominal': nominal,
        }
        if stats is not None:
            results['bpm_stats'] = stats

    def _step_blade_map(self, results: dict):
        """Generate blade map."""
        self.analysisProgress.emit("Generating blade map...")
        from ..core.visualizers import BladeMapVisualizer
        visualizer = BladeMapVisualizer(self.app.data, self.app.prm)
        fig = visualizer.show()  # Generate and capture figure
        results['blade_figure'] = fig

    def _step_central_sweeps(self, results: dict):
        """Analyze central sweeps and generate position plots."""
        self.analysisProgress.emit("Analyzing central sweeps...")

        # Analyze and generate figure
        (range_h, range_v, blades_h, blades_v, pos_h, pos_v,
         fit_h, fit_v) = self.app.processor.analyze_central_sweeps(show=False)

        # Recreate the sweep visualization
        if range_h is not None and range_v is not None:
            fig = self._generate_sweep_figure(range_h, range_v,
                                              blades_h, blades_v)
            results['sweeps_figure'] = fig
            results['sweeps_data'] = (
                range_h, range_v, blades_h, blades_v,
                pos_h, pos_v, fit_h, fit_v
            )

        # Store calculated suppression matrix (not used in UI display)
        # UI gets matrices from raw/scaled results instead
        supmat = self.app.processor.suppression_matrix()
        results['supmat'] = supmat

    def _step_blades_center(self, results: dict):
        """Analyze blades at center and generate blade current plots."""
        self.analysisProgress.emit("Analyzing blades at center...")

        # Generate the blades at center figure
        fig = self._generate_blades_center_figure()
        if fig is not None:
            results['blades_center_figure'] = fig

            # Enrich Prm with data-derived fields (section, bpmdist, etc.)
            rawdata = (getattr(self.app.reader, "rawdata", None) or
                       self.app.data)
            self.app.builder.enrich_from_data(
                rawdata, selected_beamline=self.prm.beamline
            )

    def _extract_position_coordinates(self, pos_list):
        """Extract coordinates from position result dictionary.

        Args:
            pos_list: List of position dictionaries from calculate_*_positions.

        Returns:
            Array of [x, y] coordinates from first position dict (pair).
        """
        if not pos_list or len(pos_list) == 0:
            return None
        pos_dict = pos_list[0]  # Get pairwise positions (first in list)
        if not pos_dict:
            return None
        try:
            coords = []
            for (_x, _y), (pos_x, pos_y) in pos_dict.items():
                coords.append([pos_x, pos_y])
            return np.array(coords) if coords else None
        except (ValueError, TypeError):
            return None

    def _extract_measured_and_nominal(self, pos_list):
        """Extract both measured and nominal coordinates from positions.

        Args:
            pos_list: List of position dictionaries from calculate_*_positions.
                      Each dict has keys=(agx, agy) nominal and values=[x,y]
                      measured coordinates.

        Returns:
            Tuple of (measured_coords, nominal_coords) arrays.
        """
        measured = None
        nominal = None

        if not pos_list or len(pos_list) == 0:
            return measured, nominal

        # Extract from first position dict (pairwise positions)
        pos_dict = pos_list[0]
        if not pos_dict:
            return measured, nominal

        try:
            measured_coords = []
            nominal_coords = []
            for (nom_x, nom_y), (meas_x, meas_y) in pos_dict.items():
                measured_coords.append([meas_x, meas_y])
                nominal_coords.append([nom_x, nom_y])

            if measured_coords:
                measured = np.array(measured_coords)
            if nominal_coords:
                nominal = np.array(nominal_coords)
        except (ValueError, TypeError):
            pass

        return measured, nominal

    def _step_xbpm_raw(self, results: dict):
        """Calculate raw XBPM positions."""
        self.analysisProgress.emit("Calculating raw XBPM positions...")
        result_data = self.app.processor.calculate_raw_positions(
            showmatrix=True
        )
        # Handle both dict and list returns for backward compatibility
        if isinstance(result_data, dict):
            # New format: positions is [pairwise_dict, cross_dict]
            positions = result_data.get('positions', [])
            pairwise_fig = result_data.get('pairwise_figure')
            cross_fig = result_data.get('cross_figure')
            # Store the full dict including positions list for HDF5 export
            results['positions_raw_full'] = result_data
        else:
            # Old format: positions is list [pairwise_dict, cross_dict]
            positions = result_data
            pairwise_fig = None
            cross_fig = None
            # Wrap in dict format for HDF5 export
            results['positions_raw_full'] = {'positions': positions}

        # Extract pairwise for backward compatibility display
        measured, nominal = self._extract_measured_and_nominal(positions)
        results['positions']['xbpm_raw'] = {
            'measured': measured,
            'nominal': nominal,
        }
        # Capture standard matrix (1/-1 pattern) from raw calculation
        supmat_std = (result_data.get('supmat')
                      if isinstance(result_data, dict) else None)
        if supmat_std is not None:
            results['supmat_standard'] = supmat_std
        if pairwise_fig:
            results['xbpm_raw_pairwise_figure'] = pairwise_fig
            self.logMessage.emit("Captured raw pairwise figure")
        if cross_fig:
            results['xbpm_raw_cross_figure'] = cross_fig
            self.logMessage.emit("Captured raw cross-blade figure")

        if self.app.prm.outputfile:
            self.app.exporter.data_dump(self.app.data, positions, sup="raw")

    def _step_xbpm_scaled(self, results: dict):
        """Calculate scaled XBPM positions."""
        self.analysisProgress.emit("Calculating scaled XBPM positions...")
        result_data = self.app.processor.calculate_scaled_positions(
            showmatrix=True
        )
        # Handle both dict and list returns for backward compatibility
        if isinstance(result_data, dict):
            # New format: positions is [pairwise_dict, cross_dict]
            positions = result_data.get('positions', [])
            pairwise_fig = result_data.get('pairwise_figure')
            cross_fig = result_data.get('cross_figure')
            # Store the full dict including positions list for HDF5 export
            results['positions_scaled_full'] = result_data
        else:
            # Old format: positions is list [pairwise_dict, cross_dict]
            positions = result_data
            pairwise_fig = None
            cross_fig = None
            # Wrap in dict format for HDF5 export
            results['positions_scaled_full'] = {'positions': positions}

        # Extract pairwise for backward compatibility display
        measured, nominal = self._extract_measured_and_nominal(positions)
        results['positions']['xbpm_scaled'] = {
            'measured': measured,
            'nominal': nominal,
        }
        # Capture calculated matrix (from slopes) from scaled calculation
        supmat_calc = (result_data.get('supmat')
                       if isinstance(result_data, dict) else None)
        if supmat_calc is not None:
            results['supmat'] = supmat_calc
        if pairwise_fig:
            results['xbpm_scaled_pairwise_figure'] = pairwise_fig
            self.logMessage.emit("Captured scaled pairwise figure")
        if cross_fig:
            results['xbpm_scaled_cross_figure'] = cross_fig
            self.logMessage.emit("Captured scaled cross-blade figure")

        if self.app.prm.outputfile:
            self.app.exporter.data_dump(self.app.data, positions, sup="scaled")

    @pyqtSlot()
    def stop_analysis(self):
        """Request analysis to stop at next checkpoint."""
        self._should_stop = True
        self.logMessage.emit("Stop requested...")

    def _generate_sweep_figure(self, range_h, range_v, blades_h, blades_v):
        """Generate central sweep position plots."""
        import matplotlib.pyplot as plt
        import numpy as np

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
                     * self.app.prm.xbpmdist)
            axh.plot(range_h * self.app.prm.xbpmdist, hline,
                     '^-', label="H fit")
            axh.plot(range_h * self.app.prm.xbpmdist,
                    pos_ch_v[:, 0] * self.app.prm.xbpmdist, 'o-',
                    label="H sweep")
            axh.set_xlabel("$x$ [$\\mu$m]")
            axh.set_ylabel("$y$ [$\\mu$m]")
            axh.set_title("Central Horizontal Sweeps")
            ylim = (np.max(np.abs(hline + pos_ch_v[:, 0]
                                  * self.app.prm.xbpmdist)) * 1.1)
            axh.set_ylim(-ylim, ylim)
            axh.grid(True)
            axh.legend()

        if fit_cv_h is not None:
            vline = ((fit_cv_h[0, 0] * range_v + fit_cv_h[1, 0])
                     * self.app.prm.xbpmdist)
            axv.plot(pos_cv_h[:, 0] * self.app.prm.xbpmdist,
                     range_v * self.app.prm.xbpmdist,
                     'o-', label="V sweep")
            axv.plot(vline, range_v * self.app.prm.xbpmdist,
                     '^-', label="V fit")
            axv.set_xlabel("$x$ [$\\mu$m]")
            axv.set_ylabel("$y$ [$\\mu$m]")
            axv.set_title("Central Vertical Sweeps")
            axv.set_xlim((np.min(range_v) * 0.005 + fit_cv_h[1, 0])
                         * self.app.prm.xbpmdist,
                         (np.max(range_v) * 0.005 + fit_cv_h[1, 0])
                         * self.app.prm.xbpmdist)
            axv.grid(True)
            axv.legend()

        fig.tight_layout()
        return fig

    def _generate_blades_center_figure(self):
        """Generate blade currents at center plots."""
        import matplotlib.pyplot as plt
        import numpy as np

        # Ensure we have sweep data
        if (self.app.processor.range_h is None or
            self.app.processor.range_v is None):
            self.app.processor.analyze_central_sweeps(show=False)

        blades_h = self.app.processor.blades_h
        blades_v = self.app.processor.blades_v
        range_h = self.app.processor.range_h
        range_v = self.app.processor.range_v

        if blades_h is None and blades_v is None:
            return None

        fig, (axh, axv) = plt.subplots(1, 2, figsize=(10, 5))

        if blades_h is not None:
            for key, blval in blades_h.items():
                val = blval[:, 0]
                wval = blval[:, 1]
                weight = 1. / wval if not np.isinf(1. / wval).any() else None
                (acoef, bcoef) = np.polyfit(range_h, val, deg=1, w=weight)
                axh.plot(range_h, range_h * acoef + bcoef, "o-",
                         label=f"{key} fit")
                axh.errorbar(range_h, val, wval, fmt='^-', label=key)

        if blades_v is not None:
            for key, blval in blades_v.items():
                val = blval[:, 0]
                wval = blval[:, 1]
                weight = 1. / wval if not np.isinf(1. / wval).any() else None
                (acoef, bcoef) = np.polyfit(range_v, val, deg=1, w=weight)
                axv.plot(range_v, range_v * acoef + bcoef, "o-",
                         label=f"{key} fit")
                axv.errorbar(range_v, val, wval, fmt='^-', label=key)

        axh.set_title("Horizontal")
        axv.set_title("Vertical")
        axh.legend()
        axv.legend()
        axh.grid()
        axv.grid()
        axh.set_xlabel("$x$ $\\mu$rad")
        axv.set_xlabel("$y$ $\\mu$rad")

        ylabel = ("$I$ [# counts]" if self.app.prm.beamline[:3]
                  in ["MGN", "MNC"] else "$I$ [A]")
        axh.set_ylabel(ylabel)
        axv.set_ylabel(ylabel)
        fig.tight_layout()

        return fig
