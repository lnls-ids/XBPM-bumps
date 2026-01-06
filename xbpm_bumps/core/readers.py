"""Data reading from files and pickle directories."""

import os
import re
import pickle
import numpy as np
from typing import Optional

from .config import Config
from .parameters import Prm, ParameterBuilder


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

    def read(self) -> dict:
        """Read data from working directory or file.

        Automatically determines whether to read from a text file or
        pickle directory based on the workdir path.

        Returns:
            dict: Parsed measurement data dictionary.
        """
        if os.path.isfile(self.prm.workdir):
            self._read_from_file()
        else:
            self._read_from_directory()

        self._print_summary()
        return self.data

    def _read_from_file(self) -> None:
        """Read data from a text file with optional header metadata."""
        self.data = {}

        with open(self.prm.workdir, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                if self._is_empty(line):
                    continue
                if self._is_header(line):
                    self._parse_header_line(line)
                    continue
                self._parse_data_line(line)

        if self.prm.xbpmdist is None:
            try:
                self.prm.xbpmdist = Config.XBPMDISTS[self.prm.beamline]
            except Exception:
                self.prm.xbpmdist = 1.0
            print("\n WARNING: distance from source to XBPM not provided."
                  f" Using default value: {self.prm.xbpmdist} m")

        self._infer_gridstep()

    def _read_from_directory(self) -> None:
        """Read data from pickle files in a directory."""
        self.rawdata = self._get_pickle_data()[self.prm.skip:]
        self.prm = self.builder.enrich_from_data(self.rawdata)
        self.data = self._blades_fetch()

    def _get_pickle_data(self) -> list:
        """Open pickle files in directory and extract data.

        Returns:
            list: All data collected from pickle files in working directory.
        """
        allfiles = os.listdir(self.prm.workdir)
        picklefiles = [pf for pf in allfiles if pf.endswith("pickle")]

        if len(picklefiles) == 0:
            print(f"No pickle files found in directory '{self.prm.workdir}'."
                  "Aborting.")
            sys.exit(0)

        sfiles = sorted(picklefiles,
                        key=lambda name: self.builder.lastfield(name, '_'))
        rawdata = []
        for file in sfiles:
            with open(self.prm.workdir + "/" + file, 'rb') as df:
                rawdata.append(pickle.load(df))  # noqa: S301

        return rawdata

    @staticmethod
    def _is_empty(line: str) -> bool:
        """Check if a line is empty or contains only whitespace."""
        return bool(re.match(r'^\s*$', line))

    @staticmethod
    def _is_header(line: str) -> bool:
        """Check if a line is a header/comment line (starts with #)."""
        return bool(re.match(r'^\s*#', line))

    def _parse_header_line(self, line: str) -> None:
        """Extract metadata from a header line.

        Expected format: # Key: value
        """
        cline = line.strip('#')
        # Split only on the first ':' to avoid accidental extra splits
        kprm, vprm = cline.split(':', 1)
        key = kprm.strip().lower()
        val = vprm.strip()

        # Helper to extract a numeric value from strings like "199.86 mA"
        def _extract_float(text: str) -> float:
            m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
                          text.strip())
            if not m:
                raise ValueError(f"Could not parse float from '{text}'")
            return float(m.group(0))

        if re.search(r'current', key, re.IGNORECASE):
            self.prm.current = _extract_float(val)
        elif re.search(r'beamline', key, re.IGNORECASE):
            blval = val
            # Prefer code inside parentheses, e.g., "Caterete (CAT)" -> "CAT"
            m = re.search(r"\(([^)]+)\)", blval)
            if m:
                self.prm.beamline = m.group(1).strip().upper()
            else:
                # Try to find the beamline code by looking up the full name
                try:
                    self.prm.beamline = self.builder.found_key(
                        Config.BEAMLINENAME, blval.strip()
                        )
                except StopIteration:
                    # Fallback: if exact match not found, keep original
                    self.prm.beamline = blval
        elif re.search(r'gap', key, re.IGNORECASE):
            self.prm.phaseorgap = val
        elif re.search(r'xbpm', key, re.IGNORECASE):
            self.prm.xbpmdist = _extract_float(val)
        elif re.search(r'inter bpm', key, re.IGNORECASE):
            self.prm.bpmdist = _extract_float(val)
        else:
            self.prm[key] = val

    def _parse_data_line(self, line: str) -> None:
        """Parse a data line and add it to the data dictionary.

        Expected format: nom_x nom_y to err_to ti err_ti bi err_bi bo err_bo
        """
        parts = line.strip().split()
        key = (float(parts[0]), float(parts[1]))
        self.data[key] = np.array([
            [float(parts[ii]), float(parts[ii + 1])]
            for ii in range(2, len(parts), 2)
        ])

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