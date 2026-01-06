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

    def run(self, argv=None) -> None:
        """Parse args, load data, run analyses, and render outputs."""
        self.builder = ParameterBuilder()
        self.prm = self.builder.from_cli(argv)
        self.reader = DataReader(self.prm, self.builder)
        self.data = self.reader.read()
        self.processor = XBPMProcessor(self.data, self.prm)
        self.exporter = Exporter(self.prm)

        self._maybe_bpm_positions()
        self._maybe_show_blade_map()
        self._maybe_central_sweeps()
        self._maybe_show_blades_at_center()
        self._maybe_xbpm_positions()

        plt.show()

    def _maybe_bpm_positions(self) -> None:
        if not self.prm.xbpmfrombpm:
            return
        raw = (self.reader.rawdata
               if self.reader.rawdata is not None
               else self.data)
        bpm_processor = BPMProcessor(raw, self.prm)
        bpm_processor.calculate_positions()

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
        if self.prm.xbpmpositionsraw:
            positions = self.processor.calculate_raw_positions(showmatrix=True)
            if self.prm.outputfile:
                self.exporter.data_dump(self.data, positions, sup="raw")

        if self.prm.xbpmpositions:
            positions = self.processor.calculate_scaled_positions(
                showmatrix=True
                )
            if self.prm.outputfile:
                self.exporter.data_dump(self.data, positions, sup="scaled")
