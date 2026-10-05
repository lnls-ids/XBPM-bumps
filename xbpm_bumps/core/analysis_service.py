"""Application-level orchestration of one XBPM analysis run."""

from .           import data_structure as DStr
from .processors import XBPMProcessor
from .processors import BPMProcessor


class AnalysisService:
    """Run selected calculations and return one typed analysis result."""

    @staticmethod
    def run(
        beamlinedata: DStr.BeamlineData,
        runtime_prm: DStr.GenPrm,
        ) -> DStr.DataAnalysis:
        # Initialize the analysis result container.
        analysis = DStr.DataAnalysis(beamline_prm = beamlinedata.prm)

        # BPM tab.
        if runtime_prm.show_bpmpositions:
            bprocessor   = BPMProcessor(
                raw_data = beamlinedata.raw_data,
                prm_bml  = beamlinedata.prm,
                )
            analysis.bpm = bprocessor.bpmanalysis

        # Create a processor instance to perform the calculations.
        try:
            xprocessor = XBPMProcessor(
                beamlinedata = beamlinedata,
                beamline_prm = beamlinedata.prm,
                runtime_prm  = runtime_prm,
                analysis     = analysis,
            )
        except Exception as e:
            print(f"Error initializing XBPMProcessor: {e}")
            raise

        # Blade map.
        if runtime_prm.show_blademap:
            analysis.blademap = DStr.BladeMap(
                prm    = beamlinedata.prm,
                blades = beamlinedata.raw_data.blade_avg.blades,
                pos    = beamlinedata.raw_data.blade_avg.pos_nom
            )

        # Blades at center are necessary for the suppression matrix.
        if runtime_prm.show_centralsweep:
            analysis.bladecenter = xprocessor.analyze_central_sweep_blades()

        # Central sweeps are needed for linear transformation of
        # partial Delta/Sigma calculations.
        needs_sweeps = (
            runtime_prm.show_centralsweep
            or runtime_prm.show_xbpmpositions
        )
        if needs_sweeps:
            analysis.centralsweeps = (
                xprocessor.analyze_central_sweep_positions()
                )

        # XBPM positions calculation.
        if runtime_prm.show_xbpmpositions:
            analysis.positions = xprocessor.xbpm_position_calculation()

        return analysis
