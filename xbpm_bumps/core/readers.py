"""Data reading from files and pickle directories."""

import os
import logging
import numpy as np

# Reading trusted local experiment data; HDF5 path will replace this

from .config import Config
from .parameters import Prm, ParameterBuilder

logger = logging.getLogger(__name__)


class DataReader:
    """Handles reading XBPM data from files or pickle directories.

    This class provides a unified interface for reading XBPM measurement
    data from two different sources:
    - Text files with optional header metadata
    - Directories containing pickle files

    Attributes:
        prm (Prm): Parameters dataclass containing configuration
            and runtime data.
        data (dict): Dictionary containing parsed measurement data.
    """
    def __init__(self, prm: Prm, builder: ParameterBuilder):
        """Initialize DataReader with parameters.

        Args:
            prm (Prm): Parameters dataclass instance.
            builder: Canonical ParameterBuilder instance to share state.
        """
        self.prm     = prm
        self.builder = builder
        # self.data is deprecated; use self.rawdata for canonical data
        self.rawdata = None
        self._hdf5_path = None
        self.analysis_meta = {}

    def read(self):
        """Read data from working directory or file using backend modules.

        Returns:
            The canonical rawdata (list of tuples) as a convenience.
            Canonical access is via self.rawdata.
        """
        from .reader_pickle import read_pickle_dir
        from .reader_text import read_text_file

        path = self.prm.workdir
        if os.path.isfile(path):
            lower = (path or '').lower()
            if lower.endswith('.h5') or lower.endswith('.hdf5'):
                from .reader_hdf5 import HDF5DataReader
                with HDF5DataReader(path) as reader:
                    reader.load_all()
                    self.rawdata = reader.rawdata
                    self.measured_data = reader.measured_data
                    self.analysis_meta = reader.get_analysis_meta()
            else:
                self.rawdata = read_text_file(path)
                self.analysis_meta = {}
        else:
            self.rawdata = read_pickle_dir(path)
            self.analysis_meta = {}

        # DEBUG
        # print("\n\n #### DEBUG (DataReader.read): ####\n")
        # print(f" rawdata type: {type(self.rawdata)}")
        # print(f" rawdata [0]: {self.rawdata[0] if self.rawdata else 'None'}")
        # print("\n ########## END DEBUG DataReader.read ##########\n\n")
        # END DEBUG

        return self.rawdata

    def _infer_gridstep_from_grid(self, grid_x, grid_y):
        """Infer gridstep from HDF5 grid datasets when missing."""
        if (getattr(self.prm, 'gridstep', None) not in (None, 0)
            or grid_x is None or grid_y is None):
            return
        try:
            dx = np.min(np.diff(np.sort(grid_x))) if grid_x.size > 1 else None
            dy = np.min(np.diff(np.sort(grid_y))) if grid_y.size > 1 else None
            steps = [v for v in (dx, dy) if v not in (None, 0)]
            if steps:
                self.prm.gridstep = float(min(steps))
        except Exception:
            logger.warning(
                "Failed to infer gridstep from HDF5 grid", exc_info=True
            )

    def _configure_xbpmdist(self) -> None:
        """Configure XBPM distance from source."""
        if self.prm.xbpmdist is None:
            try:
                self.prm.xbpmdist = Config.XBPMDISTS[self.prm.beamline]
            except Exception:
                self.prm.xbpmdist = 1.0
            print("\n WARNING: distance from source to XBPM not provided."
                  f" Using default value: {self.prm.xbpmdist} m")

    # def _infer_gridstep(self) -> None:
    #     """Infer grid step from data keys if not explicitly provided."""
    #     try:
    #         xs = np.unique([k[0] for k in self.data.keys()])
    #         ys = np.unique([k[1] for k in self.data.keys()])
    #         dx = np.min(np.diff(np.sort(xs))) if xs.size > 1 else None
    #         dy = np.min(np.diff(np.sort(ys))) if ys.size > 1 else None
    #         steps = [v for v in (dx, dy) if v not in (None, 0)]
    #         if steps:
    #             self.prm.gridstep = float(min(steps))
    #     except Exception as err:
    #         print(f"\nWARNING: could not infer grid step from data: {err}.")

    def _print_summary(self) -> None:
        """Print a summary of the loaded data and parameters."""
        bl = self.prm.beamline or "N/A"
        blname = Config.BEAMLINENAME[bl[:3]]
        xbpm_str = f"{self.prm.xbpmdist} m" if self.prm.xbpmdist else "N/A"
        bpm_str  = f"{self.prm.bpmdist} m" if self.prm.bpmdist else "N/A"

        print(f"""
### Working beamline:\t   {blname} ({bl})
### Storage ring current : {self.prm.current}
### Grid step:             {self.prm.gridstep}
### Distance source-XBPM : {xbpm_str}
### Distance between BPMs: {bpm_str}
### Gap or phase:          {self.prm.phaseorgap}
"""
)

    def _blades_fetch(self) -> dict:
        """Retrieve each blade's data and average over their values."""
        data = dict()
        beamline = self.prm.beamline

        for idx, dt in enumerate(self.rawdata):
            try:
                xbpm = dt[0][beamline]
                vals = list()
                for blade in Config.BLADEMAP[beamline].values():
                    av, sd = self._blade_average(xbpm[f'{blade}_val'])
                    vals.append((av, sd))
                bpm_x = dt[2]['agx']
                bpm_y = dt[2]['agy']

                # Debug: print types and values
                print(f"[DEBUG] Entry {idx}: agx={bpm_x} ({type(bpm_x)}), agy={bpm_y} ({type(bpm_y)})")
                # Validate both are floats or ints and not None
                if (bpm_x is None or bpm_y is None or
                    not isinstance(bpm_x, (float, int)) or
                    not isinstance(bpm_y, (float, int))):
                    print(f"[WARNING] Skipping entry {idx}: Invalid agx/agy: agx={bpm_x}, agy={bpm_y}")
                    continue
                # END Debug
                data[(float(bpm_x), float(bpm_y))] = np.array(vals)

            except Exception as err:
                print("\n WARNING: when fetching blades' values and averaging:"
                      f" {err}\n")
        return data

    def _blade_average(self, blade):
        """Calculate the average of blades' values for current beamline."""
        if self.prm.beamline in ["MGN", "MNC"]:
            return np.average(blade), np.std(blade)

        vals = np.array([
            vv * Config.AMPSUB[un] for vv, un in blade
        ])
        return np.average(vals), np.std(vals)
