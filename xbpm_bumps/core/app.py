"""XBPM Application orchestrator."""

import matplotlib.pyplot as plt
from typing import Optional

from .exporters   import Exporter
from .parameters  import ParameterBuilder
from .processors  import XBPMProcessor, BPMProcessor
from .readers     import DataReader
from .visualizers import BladeMapVisualizer


class XBPMApp:
    """Top-level orchestrator for XBPM analysis workflow."""

    def __init__(self) -> None:
        """Initialize app with empty state; components created during run()."""
        self.builder       : Optional[ParameterBuilder] = None
        self.reader        : Optional[DataReader]       = None
        self.processor     : Optional[XBPMProcessor]    = None
        self.exporter      : Optional[Exporter]         = None
        self.bpm_reference : Optional[tuple]            = None
        self.data = None
        self.prm  = None

    def run(self, argv: list[str], builder: ParameterBuilder) -> None:
        """'Parse args, load data, run analyses, and render outputs.

        Args:
            argv    : command-line arguments (list of strings).
            builder : canonical ParameterBuilder instance
        """
        self.builder   = builder
        self.prm       = self.builder.from_cli(argv)
        self.reader    = DataReader(self.prm, self.builder)
        self.reader.read_data()
        self.data, self.rawblades = self.reader._blades_fetch()

        self.processor = XBPMProcessor(self.data, self.prm)
        self.exporter  = Exporter(self.prm)

        self._bpm_positions()
        self._show_blade_map()
        self._central_sweeps()
        self._show_blades_at_center()
        self._xbpm_positions()

        plt.show()

    @staticmethod
    def cli_prompt(beamlines : list[str]) -> str:
        """Prompt the user to select a beamline from the CLI."""
        print("Available beamlines:")
        for i, b in enumerate(beamlines):
            print(f"  {i+1}: {b}")
        while True:
            try:
                choice = int(input("Select beamline by number: ")) - 1
                if 0 <= choice < len(beamlines):
                    return beamlines[choice]
                else:
                    print("Please enter a number between"
                          f" 1 and {len(beamlines)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def _bpm_positions(self) -> None:
        """Calculate and optionally export BPM positions if requested."""
        if self.prm.section is None:
            print("### ERROR: section not defined for BPM analysis."
                  " Skipping.")
            return
        raw = (
            self.reader.rawdata
            if self.reader.rawdata is not None
            else self.data
            )
        if raw is None:
            print("### ERROR: no raw data available for BPM analysis."
                  " Skipping.")
            return
        bpmproc = BPMProcessor(raw, self.prm)
        measured, nominal = bpmproc.calculate_positions()
        self.bpm_reference = (bpmproc.xpos, bpmproc.ypos)
        if (self.prm.outputfile and measured is not None and
            nominal is not None):
            self.exporter.data_dump_bpm(measured, nominal)

    def _compute_nominal_grid(self) -> tuple:
        """Compute nominal position grid from beam position pair.

        Returns:
            Tuple (pos_nom_h, pos_nom_v) - nominal position grids scaled
            by XBPM distance.
        """
        supmat, _ = self.processor.suppression_matrix(
                showmatrix=False, nosuppress=True
            )
        pair = self.processor.beam_position_pair(supmat)
        pos_nom_h, pos_nom_v, _, _ = (
            self.processor.position_dict_parse(pair)
        )
        pos_nom_h *= self.prm.xbpmdist
        pos_nom_v *= self.prm.xbpmdist
        return pos_nom_h, pos_nom_v

    def _resolve_reference_positions(self) -> Optional[tuple]:
        """Resolve reference positions for XBPM analysis.

        Uses the usebpmref parameter to determine whether to use BPM
        measured positions or nominal grid as the reference.

        Returns:
            Tuple (pos_nom_h, pos_nom_v) - reference position grids.
        """
        if not self.prm.usebpmref:
            # Use nominal grid from beam position pair
            return self._compute_nominal_grid()

        # Use BPM measured positions as reference
        if self.bpm_reference is not None:
            return self.bpm_reference
        if self.prm.section is None:
            print("### ERROR: section not defined for BPM analysis. Skipping.")
            return None
        raw = (self.reader.rawdata
               if self.reader.rawdata is not None
               else self.data)
        if raw is None:
            print("### ERROR: no raw data available for BPM analysis."
                  " Skipping.")
            return None
        bpmproc = BPMProcessor(raw, self.prm)
        bpmproc.calculate_positions()
        self.bpm_reference = (bpmproc.xpos, bpmproc.ypos)

    def _show_blade_map(self) -> None:
        if not self.prm.showblademap:
            return
        blade_map = BladeMapVisualizer(self.data, self.prm)
        blade_map.show()

    def _central_sweeps(self) -> None:
        if not self.prm.centralsweep:
            return
        self.processor.analyze_central_sweeps(show=True)

    def _show_blades_at_center(self) -> None:
        if not self.prm.showbladescenter:
            return
        self.processor.show_blades_at_center()

    def _xbpm_positions(self) -> None:
        self._resolve_reference_positions()
        pos_nom_h, pos_nom_v = self.bpm_reference
        if self.prm.xbpmpositionsraw:
            positions = self.processor.xbpm_position_calculation(
                pos_nom_h, pos_nom_v,
                nosuppress=True,
                showmatrix=True,
            )
            if self.prm.outputfile:
                self.exporter.data_dump(self.data, positions['positions'],
                                        sup="raw")
                supmat = positions.get('supmat')
                if supmat is not None:
                    self.exporter.write_supmat(
                        supmat,
                        write_file=True,
                        stddevmat=positions.get('stddevmat'),
                    )

        if self.prm.xbpmpositions:
            positions = self.processor.xbpm_position_calculation(
                pos_nom_h, pos_nom_v,
                nosuppress=False,
                showmatrix=True,
            )
            if self.prm.outputfile:
                self.exporter.data_dump(self.data, positions['positions'],
                                        sup="scaled")
                supmat = positions.get('supmat')
                if supmat is not None:
                    self.exporter.write_supmat(
                        supmat,
                        write_file=True,
                        stddevmat=positions.get('stddevmat'),
                    )
