"""Qt-friendly wrapper for XBPM analysis core."""

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from typing import Optional, Dict, Any
import sys
from io import StringIO
import traceback
import numpy as np

from ..core.app import XBPMApp
from ..core.parameters import Prm
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

    # (No longer needed: beamline selection is centralized)

    def __init__(self, prm, builder, reader, rawdata, parent=None):
        """Initialize the analyzer.

        Args:
            prm:     persistent Prm instance to use.
            builder: ParameterBuilder instance (enriched, canonical).
            reader:  Canonical DataReader instance.
            rawdata: Canonical rawdata.
            parent:  Optional parent QObject.
        """
        super().__init__(parent)
        self.app: Optional[XBPMApp] = None
        self.prm: Prm = prm
        self.builder = builder
        self.reader = reader
        self.rawdata = rawdata
        self._should_stop = False
        # self._preselected_beamline: Optional[str] = None

    @pyqtSlot(dict)
    def load_data_only(self):
        """Load data without running analysis.

        This allows exporting raw data to HDF5 without analysis results.
        """
        try:
            # The main window/controller must have already set up Prm and
            # builder
            self.app         = XBPMApp()
            self.app.prm     = self.prm
            self.app.builder = self.builder
            self.app.reader  = self.reader
            self.app.data, self.app.rawblades = self.reader._blades_fetch()

            # Capture stdout/stderr for logging
            log_capture = StringIO()
            old_stdout = sys.stdout
            old_stderr = sys.stderr

            try:
                sys.stdout = log_capture
                sys.stderr = log_capture

                # Data loading logic (if needed) can go here
                # (No CLI parsing or builder instantiation)

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

    def _initialize_and_run_analysis(self):
        """Initialize processor/exporter and run analysis steps."""
        # Initialize processor and exporter
        self.analysisProgress.emit("Initializing processor...")
        self.app.processor = XBPMProcessor(self.app.data, self.app.prm)
        self.app.exporter = Exporter(self.app.prm)

        # Run analysis steps
        results = self._run_analysis_steps()

        # Emit completion with results
        self.analysisComplete.emit(results)

    @pyqtSlot(dict)
    def run_analysis(self):
        """Execute XBPM analysis with given parameters.

        This method is designed to be called from a worker thread.

        Args:
            params: Dictionary of analysis parameters from ParameterPanel.
        """
        try:
            # The main window/controller must have already set up Prm and
            # builder
            self.app = XBPMApp()
            self.app.prm = self.prm
            self.app.builder = self.builder
            self.app.reader = self.reader
            self.app.data, self.app.rawblades = self.reader._blades_fetch()
            # Beamline is now always set in canonical Prm

            # Capture stdout/stderr for logging
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            log_capture = StringIO()

            try:
                sys.stdout = log_capture
                sys.stderr = log_capture

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
            'usebpmref'       : '--bpmref',
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
            'rawblades': self.app.rawblades,
            'positions': {},
        }

        # Define analysis steps as a list of tuples: (condition, method)
        steps = [
            (self.app.prm.xbpmfrombpm,      self._step_bpm_positions),
            (self.app.prm.showblademap,     self._step_blade_map),
            (self.app.prm.centralsweep,     self._step_central_sweeps),
            (self.app.prm.showbladescenter, self._step_blades_center),
            (self.app.prm.xbpmpositionsraw, self._step_xbpm_raw),
            (self.app.prm.xbpmpositions,    self._step_xbpm_scaled),
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

        if self.app.prm.section is None:
            print("### ERROR: section not defined for BPM analysis. Skipping.")
            return

        raw = (
            self.app.reader.rawdata
            if self.app.reader.rawdata is not None
            else self.app.data
        )
        if raw is None:
            print("### ERROR: no raw data available for BPM analysis."
                  " Skipping.")
            return

        bpm_processor = BPMProcessor(raw, self.app.prm)
        measured, nominal = bpm_processor.calculate_positions()
        stats = getattr(bpm_processor, 'last_stats', None)
        results['positions']['bpm'] = {
            'measured': measured,
            'nominal': nominal,
        }
        results['bpm_reference'] = {
            'xpos': getattr(bpm_processor, 'xpos', None),
            'ypos': getattr(bpm_processor, 'ypos', None),
            'xnom': getattr(bpm_processor, 'xnom', None),
            'ynom': getattr(bpm_processor, 'ynom', None),
        }
        if stats is not None:
            results['bpm_stats'] = stats
        results['bpm_figure'] = bpm_processor.fig

    def _compute_nominal_grid(self):
        """Compute nominal position grid from beam position pair.

        Returns:
            Tuple (pos_nom_h, pos_nom_v) - nominal position grids scaled
            by XBPM distance.
        """
        supmat, _ = self.app.processor.suppression_matrix(
                showmatrix=False, nosuppress=True
            )
        pair = self.app.processor.beam_position_pair(supmat)
        pos_nom_h, pos_nom_v, _, _ = (
            self.app.processor.position_dict_parse(pair)
        )
        pos_nom_h *= self.app.prm.xbpmdist
        pos_nom_v *= self.app.prm.xbpmdist
        return pos_nom_h, pos_nom_v

    def _resolve_reference_positions(self, results: dict):
        """Resolve reference positions for XBPM analysis.

        Uses the usebpmref parameter to determine whether to use BPM
        measured positions or nominal grid as the reference.

        Returns:
            Tuple (pos_nom_h, pos_nom_v) - reference position grids.
        """
        if not self.app.prm.usebpmref:
            # Use nominal grid from beam position pair
            return self._compute_nominal_grid()

        # Use BPM measured positions as reference
        bpm_ref = results.get('bpm_reference')
        if bpm_ref is None:
            self._step_bpm_positions(results)
            bpm_ref = results.get('bpm_reference')

        ref_x = bpm_ref.get('xpos')
        ref_y = bpm_ref.get('ypos')
        return ref_x, ref_y

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
            self.app.builder.rawdata = rawdata
            self.app.builder._add_beamline_parameters()

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
        pos_nom_h, pos_nom_v = self._resolve_reference_positions(results)
        result_data = self.app.processor.xbpm_position_calculation(
            pos_nom_h, pos_nom_v,
            nosuppress=True,
            showmatrix=True,
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
        # Capture XBPM statistics
        xbpm_stats = (result_data.get('xbpm_stats')
                      if isinstance(result_data, dict) else None)
        if xbpm_stats is not None:
            results['xbpm_stats_raw'] = xbpm_stats
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
        pos_nom_h, pos_nom_v = self._resolve_reference_positions(results)
        result_data = self.app.processor.xbpm_position_calculation(
            pos_nom_h, pos_nom_v,
            nosuppress=False,
            showmatrix=True,
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
        # Capture calculated standard deviation matrix from scaled calculation
        stddevmat_calc = (result_data.get('stddevmat')
                          if isinstance(result_data, dict) else None)
        if stddevmat_calc is not None:
            results['stddevmat'] = stddevmat_calc
        # Capture XBPM statistics
        xbpm_stats = (result_data.get('xbpm_stats')
                      if isinstance(result_data, dict) else None)
        if xbpm_stats is not None:
            results['xbpm_stats_scaled'] = xbpm_stats
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
        """Generate central sweep position plots.

        Delegates to the canonical visualizer implementation.
        """
        from ..core.visualizers import plot_central_sweeps
        return plot_central_sweeps(range_h, range_v, blades_h, blades_v,
                                   self.app.prm.xbpmdist)

    def _generate_blades_center_figure(self):
        """Generate blade currents at center plots.

        Delegates to the canonical visualizer implementation.
        """
        from ..core.visualizers import plot_blades_center_from_dicts

        # Ensure we have sweep data
        if (self.app.processor.range_h is None or
            self.app.processor.range_v is None):
            self.app.processor.analyze_central_sweeps(show=False)

        blades_h = self.app.processor.blades_h
        blades_v = self.app.processor.blades_v
        range_h = self.app.processor.range_h
        range_v = self.app.processor.range_v

        return plot_blades_center_from_dicts(blades_h, blades_v, range_h,
                                             range_v, self.app.prm.beamline)
