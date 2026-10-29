"""Application-level orchestration of one XBPM analysis run."""

from .           import data_structure as DStr
from .processors import XBPMProcessor
from .processors import BPMProcessor


class AnalysisService:
    """Run selected calculations and return one typed analysis result."""

    @staticmethod
    def run(
        beamlinedatum : DStr.BeamlineDatum,
        runtime_prm  : DStr.GenPrm,
        ) -> DStr.DataAnalysis:
        # Initialize the analysis result container.
        analysis = DStr.DataAnalysis()

        # BPM tab.
        if runtime_prm.show_bpmpositions:
            bprocessor   = BPMProcessor(
                raw_data = beamlinedatum.raw_data,
                prm_bml  = beamlinedatum.prm,
                )
            analysis.bpm = bprocessor.bpmanalysis

        # Create a processor instance to perform the calculations.
        try:
            xprocessor = XBPMProcessor(
                beamlinedata = beamlinedatum,
                beamline_prm = beamlinedatum.prm,
                runtime_prm  = runtime_prm,
                analysis     = analysis,
            )
        except Exception as e:
            print(f"Error initializing XBPMProcessor: {e}")
            raise

        # Blade map.
        if runtime_prm.show_blademap:
            analysis.blademap = DStr.BladeAvgData(
                prm     = {"beamline" : beamlinedatum.prm.beamline},
                blades  = beamlinedatum.raw_data.blade_avg.blades,
                pos_nom = beamlinedatum.raw_data.blade_avg.pos_nom 
            )

        # Blades at center are necessary for the suppression matrix.
        if runtime_prm.show_centralsweep:
            analysis.bladecenter = xprocessor.analyze_central_sweep_blades(
                slice_by_roi = runtime_prm.slice_by_roi
            )

        # XBPM positions calculation.
        if runtime_prm.show_xbpmpositions:
            analysis.xpositions = xprocessor.xbpm_position_calculation()

        return analysis
