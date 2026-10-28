"""HDF5 backend for XBPM DataReader."""

import os

import h5py
from matplotlib.figure import Figure

from . import data_structure as DStr
from . import constants      as Cts

def export_hdf5(
    filepath     : str,
    runtime_prm  : DStr.GenPrm,
    beamlinedata : dict[str, DStr.BeamlineDatum],
    ) -> None:
    """Write runtime parameters, per-beamline raw/analysis data and figures.
    
    Args:
        filepath     : Path to the output HDF5 file.
        runtime_prm  : Runtime parameters to be written to the root attributes.
        beamlinedata : Dictionary containing per-beamline raw and analysis data.
    """
    with h5py.File(filepath, "w") as hf:
        # root attrs; read_hdf5() reads these
        runtime_prm.to_hdf5(hf)

        # /<beamline>/{attrs,raw_data,analysis}
        for beamline, bldatum in beamlinedata.items():
            # Create a group for the beamline data.
            bldatum.to_hdf5(hf.require_group(beamline))


def export_plain(
    dirpath  : str,
    tab_info : dict[str, str],
    figures  : dict[str, Figure] | None = None,
    ) -> None:
    # Ensure the output directory exists.
    os.makedirs(dirpath, exist_ok=True)

    if figures is not None:
        for key, fig in figures.items():
            # Export the figure to a PNG file.
            figname = _filename(key, dirpath, prefix="fig", ext="png")
            fig.savefig(
                figname,
                dpi = Cts.FIGDPI,
                bbox_inches = "tight"
                )
    else:
        print("WARNING: when exporting figures, no objects were provided.")

    if tab_info is not None:
        for key, info_data in tab_info.items():
            if info_data is None or info_data.strip() == "":
                print(f"WARNING: tab '{key}' has no information to export.")
                continue

            # Export the plain information to a text file.
            txtname = _filename(key, dirpath, prefix="info", ext="txt")
            _write_plain_txt(txtname, info_data)
    else:
        print("WARNING: when exporting text table information,"
              " no objects were provided.")


def _write_plain_txt(
    path       : str,
    info_data  : str | None,
    ) -> None:
    """Write plain information to a text file.

    Args:
        path      : Path to the output text file.
        info_data : The plain information to write.
    """
    # Export plain information to a text file
    if info_data is not None:
        with open(path, "w") as f:
            f.write(info_data)


def _filename(
        key     : str,
        path    : str = "",
        prefix  : str = "file",
        ext     : str = "png",
        ) -> str:
    """Generate a filename for a figure based on the base path and key."""
    if not path:
        return f"{prefix}_{key}.{ext}"
    return os.path.join(path, f"{prefix}_{key}.{ext}")
