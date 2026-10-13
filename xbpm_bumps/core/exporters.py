"""HDF5 backend for XBPM DataReader."""

import h5py
import logging
from matplotlib.figure import Figure
# import numpy as np
# import sys
import matplotlib

from .config import Config as Cfg
from . import data_structure as DStr


def export_hdf5(
    filepath     : str,
    runtime_prm  : DStr.GenPrm,
    beamlinedata : dict[str, DStr.BeamlineData],
    figures      : dict[str, Figure] | None = None,
    ) -> None:
    """Write runtime parameters, per-beamline raw/analysis data and figures.
    
    Args:
        filepath     : Path to the output HDF5 file.
        runtime_prm  : Runtime parameters to be written to the root attributes.
        beamlinedata : Dictionary containing per-beamline raw and analysis data.
        figures      : Optional dictionary of figures to be written under the
                        /figures group.
    """
    with h5py.File(filepath, "w") as hf:
        # root attrs; read_hdf5() reads these
        runtime_prm.to_hdf5(hf)

        # /<beamline>/{attrs,raw_data,analysis}
        for beamline, bldata in beamlinedata.items():
            bldata.to_hdf5(hf)

        # /figures/<tab key>
        if figures:
            # write_figures(hf.require_group("figures"), figures)
            pass


class Exporter:
    @staticmethod
    def to_hdf5(
        filepath     : str,
        runtime_prm  : DStr.GenPrm,
        beamlinedata : dict[str, DStr.BeamlineData],
        figures      : dict[str, Figure] | None = None,
        ) -> None:
        export_hdf5(filepath, runtime_prm, beamlinedata, figures)
        return None
