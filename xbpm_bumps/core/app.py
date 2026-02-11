"""XBPM Application orchestrator."""

import matplotlib.pyplot as plt
from typing import Optional

from .parameters import Prm, ParameterBuilder
from .readers import DataReader
from .processors import XBPMProcessor, BPMProcessor
from .visualizers import BladeMapVisualizer
from .exporters import Exporter


class XBPMApp:
    """Top-level orchestrator for XBPM analysis workflow."""

    def __init__(self):
        """Initialize app with empty state; components created during run()."""
        self.prm: Optional[Prm] = None
        self.builder: Optional[ParameterBuilder] = None
        self.reader: Optional[DataReader] = None
        self.data = None
        self.processor: Optional[XBPMProcessor] = None
        self.exporter: Optional[Exporter] = None
        self._bpm_reference = None

    def run(self, argv, builder) -> None:
        """'Parse args, load data, run analyses, and render outputs.

        Args:
            argv: Command-line arguments (list of strings).
            builder: Canonical ParameterBuilder instance (must be provided)
        """
        self.builder = builder
        self.prm = self.builder.from_cli(argv)
        self.reader = DataReader(self.prm, self.builder)
        self.reader.read()
        self.data = self.reader._blades_fetch()
        self.processor = XBPMProcessor(self.data, self.prm)
        self.exporter = Exporter(self.prm)

        self._maybe_bpm_positions()
        self._maybe_show_blade_map()
        self._maybe_central_sweeps()
        self._maybe_show_blades_at_center()
        self._maybe_xbpm_positions()

        plt.show()

    @staticmethod
    def cli_prompt(beamlines):
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

    def _maybe_bpm_positions(self) -> None:
        if not self.prm.xbpmfrombpm:
            return
        raw = (self.reader.rawdata
               if self.reader.rawdata is not None
               else self.data)
        bpm_processor = BPMProcessor(raw, self.prm)
        bpm_processor.calculate_positions()

    def _compute_nominal_grid(self):
        """Compute nominal position grid from beam position pair.

        Returns:
            Tuple (pos_nom_h, pos_nom_v) - nominal position grids scaled
            by XBPM distance.
        """
        pair = self.processor.beam_position_pair(
            self.processor.suppression_matrix(
                showmatrix=False, nosuppress=True
            )
        )
        pos_nom_h, pos_nom_v, _, _ = (
            self.processor.position_dict_parse(pair)
        )
        pos_nom_h *= self.prm.xbpmdist
        pos_nom_v *= self.prm.xbpmdist
        return pos_nom_h, pos_nom_v

    def _resolve_reference_positions(self):
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
        if self._bpm_reference is not None:
            return self._bpm_reference
        raw = (self.reader.rawdata
               if self.reader.rawdata is not None
               else self.data)
        bpm_processor = BPMProcessor(raw, self.prm)
        bpm_processor.calculate_positions()
        self._bpm_reference = (bpm_processor.xpos, bpm_processor.ypos)
        return self._bpm_reference

    def _maybe_show_blade_map(self) -> None:
        if not self.prm.showblademap:
            return
        blade_map = BladeMapVisualizer(self.data, self.prm)
        blade_map.show()

    def _maybe_central_sweeps(self) -> None:
        if not self.prm.centralsweep:
            return
        self.processor.analyze_central_sweeps(show=True)

    def _maybe_show_blades_at_center(self) -> None:
        if not self.prm.showbladescenter:
            return
        self.processor.show_blades_at_center()

    def _maybe_xbpm_positions(self) -> None:
        pos_nom_h, pos_nom_v = self._resolve_reference_positions()
        if self.prm.xbpmpositionsraw:
            positions = self.processor.calculate_raw_positions(
                pos_nom_h, pos_nom_v,
                showmatrix=True,
            )
            if self.prm.outputfile:
                self.exporter.data_dump(self.data, positions, sup="raw")

        if self.prm.xbpmpositions:
            positions = self.processor.calculate_scaled_positions(
                pos_nom_h, pos_nom_v,
                showmatrix=True,
            )
            if self.prm.outputfile:
                self.exporter.data_dump(self.data, positions, sup="scaled")
