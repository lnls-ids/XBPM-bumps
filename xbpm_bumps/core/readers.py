"""Data reading from files and pickle directories."""

import os
import re
import sys
import logging
import numpy as np
from typing import Optional

# Reading trusted local experiment data; HDF5 path will replace this
import pickle  # noqa: S403

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

    def read(self, beamline_selector=None) -> dict:
        """Read data from working directory or file.

        Automatically determines whether to read from a text file or
        pickle directory based on the workdir path.

        Returns:
            dict: Parsed measurement data dictionary.
        """
        if os.path.isfile(self.prm.workdir):
            # Detect HDF5 file by extension
            lower = (self.prm.workdir or '').lower()
            if lower.endswith('.h5') or lower.endswith('.hdf5'):
                self._read_from_hdf5()
            else:
                self._read_from_file(beamline_selector)
        else:
            self._read_from_directory(beamline_selector)

        self._print_summary()
        return self.data

    def _read_from_file(self, beamline_selector=None) -> None:
        """Read data from a text file with optional header metadata."""
        self.data = {}
        self.rawdata = []
        header_meta = {}

        with open(self.prm.workdir, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                if self._is_empty(line):
                    continue
                if self._is_header(line):
                    self._parse_header_line(line)
                    continue
                self._parse_data_line(line)

        # Capture minimal header info for beamline extraction fallback
        if self.prm.beamline:
            header_meta['beamline'] = self.prm.beamline
        if header_meta:
            self.rawdata = [(header_meta, self.data)]

        # Handle beamline selection and xbpmdist configuration
        self._handle_beamline_selection_file(beamline_selector)
        self._configure_xbpmdist()
        self._infer_gridstep()

    def _read_from_hdf5(self) -> None:
        """Read data and parameters from an HDF5 file."""
        try:
            import h5py  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "h5py is required to read HDF5 files. Please install it"
            ) from exc

        self.data = {}
        self.rawdata = None
        with h5py.File(self.prm.workdir, 'r') as h5:
            self._load_hdf5_parameters(h5)
            grid_x, grid_y, pairs, pairs_nom = self._load_hdf5_grid(
                h5, None, None
            )
            self._load_hdf5_data(h5, grid_x, grid_y, pairs_nom=pairs_nom)
            self._infer_gridstep_from_grid(grid_x, grid_y)
            self._fallback_beamline_from_meta(h5)

    def _load_hdf5_parameters(self, h5file):
        """Load parameter attributes from HDF5 into prm."""
        if 'parameters' in h5file:
            gprm = h5file['parameters']
            for k, v in gprm.attrs.items():
                try:
                    setattr(self.prm, k, v)
                except Exception:
                    logger.warning(
                        "Skipping parameter attr %s from HDF5", k,
                        exc_info=True,
                    )
        return None, None

    def _load_hdf5_grid(self, h5file, grid_x, grid_y):
        """Load grid x/y arrays and pair lists from HDF5 if present."""
        pairs = None
        pairs_nom = None
        if 'grid' not in h5file:
            return grid_x, grid_y, pairs, pairs_nom

        ggrid = h5file['grid']
        grid_x = np.array(ggrid.get('x')) if 'x' in ggrid else None
        grid_y = np.array(ggrid.get('y')) if 'y' in ggrid else None
        if 'pairs' in ggrid:
            pairs = np.array(ggrid['pairs'])
        if 'pairs_nom' in ggrid:
            pairs_nom = np.array(ggrid['pairs_nom'])
        # Expose for callers if needed
        self.grid_pairs = pairs
        self.grid_pairs_nom = pairs_nom
        return grid_x, grid_y, pairs, pairs_nom

    def _load_hdf5_data(self, h5file, grid_x, grid_y, pairs_nom=None):
        """Load blades dataset and reconstruct data dict.

        Preference order for keys:
        1) pairs_nom (nominal positions) when present
        2) grid_x/grid_y fallback
        """
        blades = self._get_blades_array(h5file)
        if blades is None:
            return

        ny, nx = blades.shape[:2]
        # Prefer by-nominal pairs if present and sized correctly
        if pairs_nom is not None and pairs_nom.shape == (ny * nx, 2):
            self._populate_data_from_pairs_nom(blades, pairs_nom, ny, nx)
            return

        # Else require grid_x/y to reconstruct
        self._populate_data_from_grid(blades, grid_x, grid_y, ny, nx)

    def _get_blades_array(self, h5file) -> Optional[np.ndarray]:
        """Extract and validate blades array from HDF5."""
        if 'data' not in h5file or 'blades' not in h5file['data']:
            return None
        blades = np.array(h5file['data']['blades'])
        return blades if blades.size else None

    def _populate_data_from_pairs_nom(
        self, blades: np.ndarray, pairs_nom: np.ndarray, ny: int, nx: int
    ) -> None:
        """Populate self.data using nominal position pairs."""
        flat = blades.reshape(ny * nx, 4, 2)
        # Use strict=True in Python 3.10+; noqa ZIP002 for Python <3.10
        if sys.version_info >= (3, 10):
            for (xn, yn), arr in zip(pairs_nom, flat, strict=True):
                self.data[(float(xn), float(yn))] = arr
        else:
            for (xn, yn), arr in zip(pairs_nom, flat):  # noqa
                self.data[(float(xn), float(yn))] = arr

    def _populate_data_from_grid(
        self, blades: np.ndarray, grid_x: Optional[np.ndarray],
        grid_y: Optional[np.ndarray], ny: int, nx: int
    ) -> None:
        """Populate self.data using grid x/y arrays."""
        if grid_x is None or grid_y is None:
            logger.warning("Missing grid axes; cannot reconstruct data map")
            return
        if ny != grid_y.shape[0] or nx != grid_x.shape[0]:
            logger.warning(
                "Blades dataset shape (%s, %s) does not match grid (%s, %s)",
                ny, nx, grid_y.shape[0], grid_x.shape[0],
            )
            return
        for ii in range(ny):
            for jj in range(nx):
                y = grid_y[::-1][ii]
                x = grid_x[jj]
                self.data[(float(x), float(y))] = blades[ii, jj]

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

    def _fallback_beamline_from_meta(self, h5file):
        """Populate beamline from meta group if still unset."""
        if self.prm.beamline or 'meta' not in h5file:
            return
        bl = h5file['meta'].attrs.get('beamline')
        if isinstance(bl, (str, bytes)):
            self.prm.beamline = bl.decode() if isinstance(bl, bytes) else bl

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

    def _read_from_directory(self, beamline_selector=None) -> None:
        """Read data from pickle files in a directory."""
        self.rawdata = self._get_pickle_data()[self.prm.skip:]

        # If beamline not set, try provided selector to avoid terminal prompt
        if beamline_selector and not self.prm.beamline:
            beamlines = self._extract_beamlines(self.rawdata)
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

        # Enrich parameters (will skip prompt if beamline already set)
        self.prm = self.builder.enrich_from_data(
            self.rawdata,
            selected_beamline=self.prm.beamline,
            beamline_selector=beamline_selector,
        )
        self.data = self._blades_fetch()

    @staticmethod
    def _extract_beamlines(rawdata):
        """Extract unique beamlines from raw data list."""
        try:
            beamlines = set()
            for record in rawdata:
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

    @staticmethod
    def _extract_beamlines_fallback(rawdata):
        """Fallback: gather all header keys/values as beamline candidates."""
        try:
            beamlines = set()
            for record in rawdata:
                if not (isinstance(record, (list, tuple)) and len(record) > 0):
                    continue
                header = record[0]
                if isinstance(header, dict):
                    # Try to get keys first (primary beamline identifiers)
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

    def _extract_beamlines_from_header(self):
        """Extract beamline from already parsed header if present."""
        return [self.prm.beamline] if self.prm.beamline else []

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
