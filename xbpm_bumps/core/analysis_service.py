"""Application-level orchestration of one XBPM analysis run."""

from .            import data_structure as DStr
from .processors import XBPMProcessor


class AnalysisService:
    """Run selected calculations and return one typed analysis result."""

    @staticmethod
    def run(
        workdata: DStr.BeamlineData,
        runtime_prm: DStr.Prm,
    ) -> DStr.DataAnalysis:
        analysis = DStr.DataAnalysis(prm=workdata.prm)

        processor = XBPMProcessor(
            beamlinedata=workdata,
            prm_bml=workdata.prm,
            prm_gen=runtime_prm,
            analysis=analysis,
        )

        needs_sweeps = (
            runtime_prm.show_centralsweep
            or runtime_prm.show_bladecenter
            or runtime_prm.show_xbpmpositionsraw
            or runtime_prm.show_xbpmpositions
        )
        if needs_sweeps:
            analysis.centralsweeps = processor.analyze_central_sweeps()

        if (
            runtime_prm.show_xbpmpositionsraw
            or runtime_prm.show_xbpmpositions
        ):
            analysis.positions = processor.xbpm_position_calculation()

        return analysis
