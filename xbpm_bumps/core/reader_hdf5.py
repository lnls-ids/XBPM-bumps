"""HDF5 backend for XBPM DataReader."""

import h5py
from .config import Config as Cfg
from . import data_structure as DStr

def read_hdf5(filepath: str) -> tuple[DStr.GenPrm,
                                      dict[str, DStr.BeamlineDatum]]:
    beamlinedata ={}
    with h5py.File(filepath, 'r') as hf:
        runtime_prm = DStr.GenPrm.from_hdf5(
            dset_grp=hf,
            inputfile=filepath
        )

        for beamline, bldata in hf.items():
            if beamline not in Cfg.BLADEMAP.keys():
                print(f" WARNING: Unknown beamline '{beamline}'"
                        " defined in HDF5 file. Skipping.")
                continue

            # Assemble the extracted data in the raw_data dictionary.
            beamlinedata[beamline] = DStr.BeamlineDatum.from_hdf5(bldata)
    return runtime_prm, beamlinedata
