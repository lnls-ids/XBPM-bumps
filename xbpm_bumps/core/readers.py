"""Data reading from files and pickle directories."""

import os
import logging
import numpy as np
from typing import Optional

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
    def __init__(self, prm: Prm, builder: Optional[ParameterBuilder] = None):
        """Initialize DataReader with parameters.

        Args:
            prm (Prm): Parameters dataclass instance.
            builder: Optional ParameterBuilder instance to share state.
        """
        self.prm     = prm
        self.builder = builder or ParameterBuilder(prm)
        self.data    = {}
        self.rawdata = None
        self._hdf5_path = None
        self.analysis_meta = {}

    def get_available_beamlines(self):
        """Extract available beamlines from rawdata or file header."""
        if hasattr(self, 'rawdata') and self.rawdata is not None:
            return self._extract_beamlines(self.rawdata)
        # Otherwise, try to peek into the file (HDF5 or text)
        # For HDF5, open file and list /raw_data keys
        if hasattr(self.prm, 'inputfile') and self.prm.inputfile:
            import h5py
            fname = self.prm.inputfile
            if os.path.isfile(fname) and fname.endswith(('.h5', '.hdf5')):
                try:
                    with h5py.File(fname, 'r') as h5f:
                        if 'raw_data' in h5f:
                            return list(h5f['raw_data'].keys())
                except Exception as exc:
                    logger.exception(
                        "Exception occurred while reading available beamlines"
                        " from HDF5: %s", exc)
        # Fallback: return empty list
        return []

    def _extract_beamlines(self, rawdata):
        """Extract unique beamlines from raw data headers."""
        try:
            beamlines = set()
            for record in rawdata or []:
                if not (isinstance(record, (list, tuple)) and len(record) > 0):
                    continue
                header = record[0]

                if isinstance(header, dict):
                    # Beamlines are dict keys (e.g., 'MNC1', 'MNC2', 'CAT')
                    for key in header.keys():
                        if isinstance(key, str):
                            beamlines.add(key)
                elif isinstance(header, (list, tuple)):
                    for item in header:
                        if isinstance(item, str):
                            beamlines.add(item)
            return list(beamlines)
        except Exception:
            return []

    def read(self, beamline_selector=None) -> dict:
        """Read data from working directory or file using backend modules.

        Returns:
            dict: Parsed measurement data dictionary.
        """
        from .reader_hdf5 import read_hdf5, read_hdf5_analysis_meta
        from .reader_pickle import read_pickle_dir
        from .reader_text import read_text_file

        path = self.prm.workdir
        if os.path.isfile(path):
            lower = (path or '').lower()
            if lower.endswith('.h5') or lower.endswith('.hdf5'):
                self.rawdata = read_hdf5(path, beamline_selector)
                self.analysis_meta = read_hdf5_analysis_meta(path)
            else:
                self.rawdata = read_text_file(path)
                self.analysis_meta = {}
        else:
            self.rawdata = read_pickle_dir(path)
            self.analysis_meta = {}
        return self.data

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

    def _handle_beamline_selection_file(self, beamline_selector) -> None:
        """Handle beamline selection logic for file-based input."""
        # If beamline still unknown, try selector to avoid terminal prompt
        if beamline_selector and not self.prm.beamline:
            beamlines = self._extract_beamlines_from_header()
            if not beamlines:
                beamlines = self._extract_beamlines_fallback(self.rawdata)

            if len(beamlines) == 1:
                chosen = beamlines[0]
                self.prm.beamline = chosen
                self.builder.prm.beamline = chosen
            elif beamlines:
                selected = beamline_selector(beamlines)
                # If user cancels, pick first to avoid terminal prompt
                chosen = selected or beamlines[0]
                self.prm.beamline = chosen
                self.builder.prm.beamline = chosen

    def _configure_xbpmdist(self) -> None:
        """Configure XBPM distance from source."""
        if self.prm.xbpmdist is None:
            try:
                self.prm.xbpmdist = Config.XBPMDISTS[self.prm.beamline]
            except Exception:
                self.prm.xbpmdist = 1.0
            print("\n WARNING: distance from source to XBPM not provided."
                  f" Using default value: {self.prm.xbpmdist} m")

    def _infer_gridstep(self) -> None:
        """Infer grid step from data keys if not explicitly provided."""
        try:
            xs = np.unique([k[0] for k in self.data.keys()])
            ys = np.unique([k[1] for k in self.data.keys()])
            dx = np.min(np.diff(np.sort(xs))) if xs.size > 1 else None
            dy = np.min(np.diff(np.sort(ys))) if ys.size > 1 else None
            steps = [v for v in (dx, dy) if v not in (None, 0)]
            if steps:
                self.prm.gridstep = float(min(steps))
        except Exception as err:
            print(f"\nWARNING: could not infer grid step from data: {err}.")

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

        for dt in self.rawdata:
            try:
                xbpm = dt[0][beamline]
                vals = list()
                for blade in Config.BLADEMAP[beamline].values():
                    av, sd = self._blade_average(xbpm[f'{blade}_val'])
                    vals.append((av, sd))
                bpm_x = dt[2]['agx']
                bpm_y = dt[2]['agy']
                data[bpm_x, bpm_y] = np.array(vals)
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
