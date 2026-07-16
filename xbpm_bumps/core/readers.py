"""Data reading from files and pickle directories."""

from copy import deepcopy
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
        self.prm              = prm
        self.builder          = builder
        self.rawdata          = None
        self._hdf5_path       = None
        self.analysis_meta    = {}
        self._blades_cache    = {}
        self._rawblades_cache = {}

    def read_data(self) -> list:
        """Read data from working directory or file using backend modules.

        Returns:
            The canonical rawdata (list of tuples) as a convenience.
            Canonical access is via self.rawdata.
        """
        path = self.prm.workdir
        if os.path.isfile(path):
            lower = (path or '').lower()

            # If the file is HDF5, use the HDF5DataReader to load data.
            if lower.endswith('.h5') or lower.endswith('.hdf5'):
                from .reader_hdf5 import HDF5DataReader
                with HDF5DataReader(path) as rdr:
                    rdr.load_all()
                    self.rawdata       = rdr.rawdata
                    self.measured_data = rdr.measured_data
                    self.analysis_meta = rdr.get_analysis_meta()
            else:
                # If the file is a text file, use the text reader to load data.
                from .reader_text   import read_text_file
                self.rawdata       = read_text_file(path)
                self.analysis_meta = {}
        else:
            # If the path is a directory, use the pickle reader to load data.
            from .reader_pickle import read_pickle_dir
            self.rawdata       = read_pickle_dir(path)
            self.analysis_meta = {}

        self._blades_cache    = {}
        self._rawblades_cache = {}
        return self.rawdata

    def _configure_xbpmdist(self) -> None:
        """Configure XBPM distance from source."""
        if self.prm.xbpmdist is None:
            try:
                self.prm.xbpmdist = Config.XBPMDISTS[self.prm.beamline]
            except Exception:
                self.prm.xbpmdist = 1.0
            print("\n WARNING: distance from source to XBPM not provided."
                  f" Using default value: {self.prm.xbpmdist} m")

    def _print_summary(self) -> None:
        """Print a summary of the loaded data and parameters."""
        bl = self.prm.beamline or "N/A"
        blname = Config.BEAMLINENAME[bl[:3]]
        xbpm_str = f"{self.prm.xbpmdist} m" if self.prm.xbpmdist else "N/A"
        bpm_str  = f"{self.prm.bpmdist} m" if self.prm.bpmdist else "N/A"

        print(f"""\n
### Working beamline      : {blname} ({bl})
### Storage ring current  : {self.prm.current}
### Grid step             : {self.prm.gridstep}
### Distance source-XBPM  : {xbpm_str}
### Distance between BPMs : {bpm_str}
### Gap or phase          : {self.prm.phaseorgap}
\n"""
)

    def _blades_fetch(self) -> dict:
        """Retrieve each blade's data and average over their values.

        The fetched data is cached for each beamline and stored in
        self._blades_cache, as average and std dev, and in
        self._rawblades_cache, as a dictionary of arrays of each blade's
        raw values.

        Returns:
            Tuple of two dictionaries:
            - data: {(bpm_x, bpm_y): np.array([(av, sd), ...])}
            - rawblades: {(bpm_x, bpm_y): {'A': array, 'B': array, ...}}
        """
        beamline = self.prm.beamline

        # If cached for current beamline, return from cache.
        cached    = self._blades_cache.get(beamline)
        rawcached = self._rawblades_cache.get(beamline)
        if cached is not None and rawcached is not None:
            return cached, rawcached

        data = dict()
        rawblades = dict()
        for dt in self.rawdata:
            try:
                xbpm = dt[0][beamline]
                vals = list()
                rawblade = dict()
                for blade in Config.BLADEMAP[beamline].values():
                    blade_vals = xbpm.get(f'{blade}_val')
                    av, sd, rawblade[blade] = self._blade_average(blade_vals)
                    vals.append((av, sd))
                bpm_x = float(dt[2]['agx'])
                bpm_y = float(dt[2]['agy'])

                data[(bpm_x, bpm_y)] = np.array(vals)
                rawblades[(bpm_x, bpm_y)] = deepcopy(rawblade)

            except Exception as err:
                print("\n WARNING: when fetching blades' values and averaging:"
                      f" {err}\n")
        self._blades_cache[beamline]    = data
        self._rawblades_cache[beamline] = rawblades
        return data, rawblades

    def _blade_average(self, blade: list) -> tuple:
        """Calculate the average of blades' values for current beamline."""
        if self.prm.beamline in ["MGN", "MNC"]:
            return np.average(blade), np.std(blade), blade

        vals = np.array([
            vv * Config.AMPSUB[un] for vv, un in blade
        ])
        return np.average(vals), np.std(vals), vals
