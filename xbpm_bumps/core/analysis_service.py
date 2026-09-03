"""Application-level orchestration of one XBPM analysis run."""

from .            import data_structure as DStr
from .processors import XBPMProcessor
from .processors import BPMProcessor


class AnalysisService:
    """Run selected calculations and return one typed analysis result."""

    @staticmethod
    def run(
        workdata: DStr.BeamlineData,
        runtime_prm: DStr.Prm,
        ) -> DStr.DataAnalysis:
        # Initialize the analysis result container.
        analysis = DStr.DataAnalysis(beamline_prm=workdata.prm)

        # Create a processor instance to perform the calculations.
        xprocessor = XBPMProcessor(
            beamlinedata=workdata,
            beamline_prm=workdata.prm,
            runtime_prm=runtime_prm,
            analysis=analysis,
        )

        # BPM tab.
        if runtime_prm.show_bpmpositions:
            bprocessor = BPMProcessor(
                raw_data=workdata.raw_data,
                prm_bml=workdata.prm,
                )
            analysis.bpm = bprocessor.calculate_positions()

        # Blade map.
        if runtime_prm.show_blademap:
            analysis.blademap = DStr.BladeMap(
                prm=workdata.prm,
                blades=workdata.raw_data.blade_avg.blades,
                pos=workdata.raw_data.blade_avg.pos_nom
            )

        # Blades at center are necessary for the suppression matrix.
        if runtime_prm.show_bladecenter:
            analysis.bladecenter = xprocessor.analyze_blade_centers()

        # Central sweeps are needed for linear transformation of
        # partial Delta/Sigma calculations.
        needs_sweeps = (
            runtime_prm.show_centralsweep
            or runtime_prm.show_bladecenter
            or runtime_prm.show_xbpmpositionsraw
            or runtime_prm.show_xbpmpositions
        )
        if needs_sweeps:
            analysis.centralsweeps = xprocessor.analyze_central_sweeps()

        # XBPM positions calculation.
        if (
            runtime_prm.show_xbpmpositionsraw
            or runtime_prm.show_xbpmpositions
            ):
            analysis.positions = xprocessor.xbpm_position_calculation()

        return analysis
