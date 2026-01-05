#!/usr/bin/python3

"""Extract XBPM's data and calculate the beam position.

This program extract data from pickle files containing the measurements
of XBPM blades' currents of a few beamlines (CAT, CNB, MNC, MGN) and
calculates the respective positions based on pairwise and cross-blade
formulas.

The data is treated with linear transformations first, to correct for
distortions promoted by different gains in each blade. Firstly, gains
(or their reciprocal, 1/G, which is the suppression) are estimated by
analyzing the slope of the line formed by central points in vertical
and horizontal sweeping. The supression is applied to the set of points
to correct the grid, then a linear scaling is calculated to set distances
in micrometers.

Usage:
    xbpm_bumps.py [OPTION] [VALUE]

where options are:
  -h : this help

Input parameters:
  -w <directory> : the directory with measured data
  -d <distance>  : distance from source to XBPM (optional; if not given,
                   the standard ones from Sirius are used)
  -g <step>      : step between neighbour sites in the grid, default = 4.0
  -k <number>    : initial data to be skipped, default = 0

Output parameters (information to be shown):
  -b  : positions calculated from BPM data
  -c  : show each blade's response at central line sweeps
  -m  : the blade map (to check blades' positions)
  -r  : positions calculated from XBPM data without suppression
  -s  : anaysis of blades' behaviour by sweeping through the center
        of canvas. Fit lines to data.
  -x  : positions calculated from XBPM data

"""

import argparse
from dataclasses import dataclass
from typing import Optional, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pickle                 # noqa: S403
import os
import re
import sys
from copy import deepcopy
import unicodedata


HELP_DESCRIPTION = (
"""
This program extract data from a file or a directory with the
measurements of XBPM blades' currents of a few beamlines
(CAT, CNB, MNC, MGN) and calculates the respective positions
based on pairwise and cross-blade formulas.

If data is read from a text file, it  may have a header with parameters,
starting with '#', like:
# SR current: 300.0
# Beamline: CAT
# Phase/Gap: 6.0
# XBPM distance: 15.74
# Inter BPM distance: 6.175495

The data lines must have the format:

nom_x nom_y to err_to ti err_ti bi err_bi bo err_bo

where nom_x and nom_y are the horizontal and vertical nominal
positions of the beam,
to, ti, bi, bo are the blade currents and
err_to, err_ti, err_bi, err_bo are the respective errors.

If data is read from a directory, it must contain pickle files with
data saved.

The data is treated with linear transformations first, to correct for
distortions promoted by different gains in each blade. Firstly, gains
(or their reciprocal, 1/G, which is the suppression) are estimated by
analyzing the slope of the line formed by central points in vertical
and horizontal sweeping. The supression is applied to the set of points
to correct the grid, then a linear scaling is calculated to set distances
in micrometers.
""")


FILE_EXTENSION = ".pickle"    # Data file type.
GRIDSTEP = 2                  # Default grid step.
STD_ROI_SIZE = 2              # Default range for ROI.
FIGDPI = 300                  # Figure dpi saving parameter.


def found_key(my_dict: dict, target_value: str) -> str:
    """Find key in dictionary by its value.

    Args:
        my_dict: Dictionary to search.
        target_value: The value to find.

    Returns:
        The key corresponding to target_value.

    Raises:
        StopIteration: If target_value is not found.
    """
    return next(key
                for key, value in my_dict.items()
                if value == target_value)


@dataclass
class Prm:
    """Typed container for command-line and runtime parameters.

    This class implements __getitem__/__setitem__ so existing code that
    uses prm["key"] remains compatible while also providing attribute
    access (prm.key).
    """
    showblademap     : bool                = False
    centralsweep     : bool                = False
    showbladescenter : bool                = False
    xbpmpositions    : bool                = False
    xbpmfrombpm      : bool                = False
    xbpmpositionsraw : bool                = False
    outputfile       : Optional[str]       = None
    xbpmdist         : Optional[float]     = None
    workdir          : Optional[str]       = None
    skip             : int                 = 0
    gridstep         : float               = GRIDSTEP
    maxradangle      : float               = 20.0

    # runtime-filled fields
    beamline         : str                 = ""
    current          : Optional[Any]       = None
    phaseorgap       : Optional[Any]       = None
    bpmdist          : Optional[float]     = None
    section          : Optional[str]       = None
    blademap         : Optional[Any]       = None
    nroi             : Optional[List[int]] = None

    def __getitem__(self, key: str):
        """Dictionary-style access (prm['key']) for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Allow setting attributes via prm['key'] = value."""
        setattr(self, key, value)


class ParameterBuilder:
    """Builds and enriches Prm parameters from CLI and data sources.

    Consolidates all parameter-related logic:
    - CLI argument parsing
    - Beamline identification from data
    - Grid step inference
    - Data-derived parameter enrichment
    """

    @staticmethod
    def from_cli(argv=None) -> Prm:
        """Parse command-line arguments and build initial Prm instance.

        Args:
            argv: List of command-line arguments. If None, uses sys.argv.
        """
        args = ParameterBuilder._parse_args(argv)
        return Prm(
            showblademap     = bool(args.showblademap),
            centralsweep     = bool(args.centralsweep),
            showbladescenter = bool(args.showbladescenter),
            xbpmpositions    = bool(args.xbpmpositions),
            xbpmfrombpm      = bool(args.xbpmfrombpm),
            xbpmpositionsraw = bool(args.xbpmpositionsraw),
            outputfile       = bool(args.outputfile),
            xbpmdist         = args.xbpmdist,
            workdir          = args.workdir,
            skip             = int(args.skip),
            gridstep         = float(args.gridstep),
            maxradangle      = 20.0,
            beamline         = "",
        )

    @staticmethod
    def enrich_from_data(prm: Prm, rawdata: list) -> Prm:
        """Enrich Prm with other data-derived values.

        Adds beamline-specific parameters and infers grid step if needed.

        Args:
            prm: Initial Prm instance.
            rawdata: Raw data list from measurements.

        Returns:
            prm: Enriched Prm instance.
        """
        if not prm.beamline:
            prm.beamline = ParameterBuilder._identify_beamline(rawdata)
        ParameterBuilder._add_beamline_parameters(prm, rawdata)
        return prm

    @staticmethod
    def _parse_args(argv=None):
        """Parse command-line arguments using argparse.

        Args:
            argv: List of command-line arguments. If None, uses sys.argv.
        """
        parser = argparse.ArgumentParser(
            prog="xbpm_bumps",
            description="Extract XBPM's data and calculate the beam position.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f"{HELP_DESCRIPTION}\n")

        # Boolean flags
        parser.add_argument(
            '-b', action='store_true', dest='xbpmfrombpm',
            help='Show positions calculated from BPM data'
        )
        parser.add_argument(
            '-c', action='store_true', dest='showbladescenter',
            help="Show each blade's response at central line sweeps",
        )
        parser.add_argument(
            '-m', action='store_true', dest='showblademap',
            help="Show blades' map (to check blades' positions)"
        )
        parser.add_argument(
            '-s', action='store_true', dest='centralsweep',
            help='Positions when sweeping through center'
        )
        parser.add_argument(
            '-r', action='store_true', dest='xbpmpositionsraw',
            help='Positions calculated from XBPM data without suppression',
        )
        parser.add_argument(
            '-x', action='store_true', dest='xbpmpositions',
            help='Positions calculated from XBPM data'
        )

        # Options with values
        parser.add_argument(
            '-d', '--xbpmdist', type=float,
            help='Distance from source to XBPM [m]'
        )
        parser.add_argument(
            '-g', '--gridstep', type=float, default=GRIDSTEP,
            help=("""Step between neighbour sites in the grid.
                Usually inferred from data, but might be provided
                in some cases."""
            ),
        )
        parser.add_argument(
            '-k', '--skip', type=int, default=0,
            help='Initial data to be skipped (default=0)'
        )
        parser.add_argument(
            '-o', '--outputfile', action="store_true", dest="outputfile",
            help='Dump data to output file'
        )
        parser.add_argument(
            '-w', '--workdir', type=str, required=True,
            help='Working directory _or_ file with measured data'
        )

        return parser.parse_args(argv)

    @staticmethod
    def _identify_beamline(rawdata: list) -> str:
        """Identify and select beamline from raw data."""
        beamlines = sorted(list(set([
            dic for dt in rawdata for dic in dt[0]
            ])))

        if len(beamlines) > 1:
            print("\nWARNING: found these beamlines: " +
                  ", ".join(beamlines))
            print(" Which one should I work on?")
            for ii, bl in enumerate(beamlines):
                print(f" {ii + 1} - {bl}")

            opt = None
            while opt is None:
                try:
                    opt = int(input(" Pick your option: "))
                    if opt not in list(range(1, 1 + len(beamlines))):
                        opt = None
                        raise Exception
                except Exception:
                    print(' Invalid option.')
                    continue
        else:
            opt = 1

        beamline = beamlines[opt - 1]
        if beamline not in BLADEMAP.keys():
            print(f" ERROR: beamline {beamline} not defined in blade maps.")
            print(" Defined blade maps are:"
                  f" {', '.join(BLADEMAP.keys())}.")
            print("\n Please, check your data. Aborting.")
            sys.exit(0)

        return beamline

    @staticmethod
    def _add_beamline_parameters(prm: Prm, rawdata: list) -> None:
        """Add beamline-specific parameters to prm."""
        prm.current = rawdata[0][2]["current"]

        try:
            prm.phaseorgap = rawdata[0][1][prm.beamline[:3].lower()]
            print(f"### Phase / Gap ({prm.beamline})   :\t"
                  f" {prm.phaseorgap}")
        except Exception:
            print(f"\n WARNING: no phase/gap defined for {prm.beamline}.")

        prm.bpmdist  = BPMDISTS[prm.beamline[:3]]
        prm.section  = SECTIONS[prm.beamline[:3]]
        prm.blademap = BLADEMAP[prm.beamline[:3]]

        if prm.xbpmdist is None:
            prm.xbpmdist = XBPMDISTS[prm.beamline]
            print(f"\n WARNING: distance from source to {prm.beamline}'s XBPM "
                  f" set to {prm.xbpmdist:.3f} m (beamline default).")

        prm.gridstep = ParameterBuilder._infer_gridstep(rawdata)

    @staticmethod
    def _infer_gridstep(rawdata: list) -> float:
        """Calculate grid step from data."""
        agx = [rawdata[ii][2]['agx'] for ii in range(len(rawdata))]
        agy = [rawdata[ii][2]['agy'] for ii in range(len(rawdata))]

        xset = list(set(agx))
        gridstepx = 0 if len(xset) == 1 else np.abs(xset[1] - xset[0])
        yset = list(set(agy))
        gridstepy = 0 if len(yset) == 1 else np.abs(yset[1] - yset[0])

        if gridstepx != gridstepy:
            print(f"\n WARNING: horizontal grid step ({gridstepx})"
                  f" differs from vertical grid step ({gridstepy})."
                  "\n I'll try it with the smaller value, if not zero.")

        if gridstepx < gridstepy and gridstepx != 0:
            return gridstepx
        elif gridstepy != 0:
            return gridstepy

        print(" ERROR: I could not infer the grid step size. "
              " Please, rerun and provide the value manually,"
              " with option -g. Aborting.")
        sys.exit(0)

    @staticmethod
    def lastfield(name: str, fld: str = ' ') -> str:
        """Get the last part of string separated by given character."""
        return name.split(fld)[-1]


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

    def __init__(self, prm: Prm):
        """Initialize DataReader with parameters.

        Args:
            prm (Prm): Parameters dataclass instance.
        """
        self.prm = prm
        self.data = {}
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
                self.prm.xbpmdist = XBPMDISTS[self.prm.beamline]
            except Exception:
                self.prm.xbpmdist = 1.0
            print("\n WARNING: distance from source to XBPM not provided."
                  f" Using default value: {self.prm.xbpmdist} m")

        self._infer_gridstep()

    def _read_from_directory(self) -> None:
        """Read data from pickle files in a directory."""
        self.rawdata = self._get_pickle_data()[self.prm.skip:]
        self.prm = ParameterBuilder.enrich_from_data(self.prm, self.rawdata)
        self.data = blades_fetch(self.rawdata, self.prm.beamline)

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
                        key=lambda name: ParameterBuilder.lastfield(name, '_'))
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
                    self.prm.beamline = found_key(BEAMLINENAME, blval.strip())
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
        blname = BEAMLINENAME[bl[:3]]
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


class XBPMProcessor:
    """Processes XBPM data to calculate beam positions.

    This class handles all calculation logic for XBPM position analysis:
    - Central sweep analysis to determine suppression matrices
    - Pairwise and cross-blade position calculations
    - Raw (no suppression) and scaled (with suppression) positions
    - Blade behavior analysis at central positions

    Attributes:
        data (dict): Measurement data dictionary.
        prm (Prm): Parameters dataclass.
        range_h (np.ndarray): Horizontal sweep range.
        range_v (np.ndarray): Vertical sweep range.
        blades_h (dict): Blade measurements along horizontal center line.
        blades_v (dict): Blade measurements along vertical center line.
        suppression_matrix_val: Calculated suppression matrix.
    """

    def __init__(self, data: dict, prm: Prm):
        """Initialize processor with data and parameters.

        Args:
            data: Measurement data dictionary.
            prm: Parameters dataclass instance.
        """
        self.data = data
        self.prm = prm
        self.range_h = None
        self.range_v = None
        self.blades_h = None
        self.blades_v = None
        self.suppression_matrix_val = None

    def analyze_central_sweeps(self, show: bool = False) -> tuple:
        """Analyze blade behavior at central sweep positions.

        Examines blade measurements along central horizontal and vertical
        lines to understand blade response and calculate suppression factors.

        Args:
            show: Whether to display sweep plots.

        Returns:
            Tuple of (range_h, range_v, blades_h, blades_v).
        """
        keys = np.array(list(self.data.keys()))
        self.range_h = np.unique(keys[:, 0])
        self.range_v = np.unique(keys[:, 1])

        # Run through central horizontal line if data is not just a point
        if len(self.range_h) > 1:
            (pos_ch_v, fit_ch_v,
             self.blades_h) = self._central_sweep_horizontal()
        else:
            pos_ch_v = np.zeros(len(self.range_h))
            fit_ch_v, self.blades_h = None, None

        # Run through central vertical line if data is not just a point
        if len(self.range_v) > 1:
            pos_cv_h, fit_cv_h, self.blades_v = self._central_sweep_vertical()
        else:
            pos_cv_h = np.zeros(len(self.range_v))
            fit_cv_h, self.blades_v = None, None

        if show:
            self._central_sweeps_show(pos_ch_v, fit_ch_v, pos_cv_h, fit_cv_h)

        return (self.range_h, self.range_v, self.blades_h, self.blades_v)

    def _central_sweep_horizontal(self) -> tuple:
        """Extract blade measurements along horizontal center line."""
        try:
            to_ch = np.array([self.data[jj, 0][0] for jj in self.range_h])
            ti_ch = np.array([self.data[jj, 0][1] for jj in self.range_h])
            bi_ch = np.array([self.data[jj, 0][2] for jj in self.range_h])
            bo_ch = np.array([self.data[jj, 0][3] for jj in self.range_h])
            blades_h = {"to": to_ch, "ti": ti_ch, "bi": bi_ch, "bo": bo_ch}
        except Exception as err:
            print("\n WARNING: horizontal sweeping interrupted,"
                  f" data grid may be incomplete: {err}")
            blades = {bl: np.array([[1., 0] for _ in self.range_h])
                      for bl in ["to", "ti", "bi", "bo"]}
            return None, None, blades

        pos_to_ti_v = (to_ch + ti_ch)
        pos_bi_bo_v = (bo_ch + bi_ch)
        pos_ch_v = (pos_to_ti_v - pos_bi_bo_v) / (pos_to_ti_v + pos_bi_bo_v)
        fit_ch_v = np.polyfit(self.range_h, pos_ch_v, deg=1)

        return pos_ch_v, fit_ch_v, blades_h

    def _central_sweep_vertical(self) -> tuple:
        """Extract blade measurements along vertical center line."""
        try:
            to_cv = np.array([self.data[0, jj][0] for jj in self.range_v])
            ti_cv = np.array([self.data[0, jj][1] for jj in self.range_v])
            bi_cv = np.array([self.data[0, jj][2] for jj in self.range_v])
            bo_cv = np.array([self.data[0, jj][3] for jj in self.range_v])
            blades_v = {"to": to_cv, "ti": ti_cv, "bi": bi_cv, "bo": bo_cv}
        except Exception as err:
            print("\n WARNING: vertical sweeping interrupted,"
                  f" data grid may be incomplete: {err}")
            blades = {bl: np.array([[1., 0] for _ in self.range_v])
                      for bl in ["to", "ti", "bi", "bo"]}
            return None, None, blades

        pos_to_bo_h = (to_cv + bo_cv)
        pos_ti_bi_h = (ti_cv + bi_cv)
        pos_cv_h = (pos_to_bo_h - pos_ti_bi_h) / (pos_to_bo_h + pos_ti_bi_h)
        fit_cv_h = np.polyfit(self.range_v, pos_cv_h, deg=1)

        return pos_cv_h, fit_cv_h, blades_v

    def _central_sweeps_show(self, pos_ch_v, fit_ch_v, pos_cv_h, fit_cv_h):
        """Plot results from fittings on central sweeps."""
        fig, (axh, axv) = plt.subplots(1, 2, figsize=(12, 5))

        if fit_ch_v is not None:
            hline = ((fit_ch_v[0, 0] * self.range_h + fit_ch_v[1, 0])
                     * self.prm.xbpmdist)
            axh.plot(self.range_h * self.prm.xbpmdist, hline,
                     '^-', label="H fit")
            axh.plot(self.range_h * self.prm.xbpmdist,
                    pos_ch_v[:, 0] * self.prm.xbpmdist, 'o-', label="H sweep")
            axh.set_xlabel(u"$x$ [$\\mu$m]")
            axh.set_ylabel(u"$y$ [$\\mu$m]")
            axh.set_title("Central Horizontal Sweeps")
            ylim = (np.max(np.abs(hline + pos_ch_v[:, 0]
                                  * self.prm.xbpmdist)) * 1.1)
            axh.set_ylim(-ylim, ylim)
            axh.grid(True)
            axh.legend()

        if fit_cv_h is not None:
            vline = ((fit_cv_h[0, 0] * self.range_v + fit_cv_h[1, 0])
                     * self.prm.xbpmdist)
            axv.plot(pos_cv_h[:, 0] * self.prm.xbpmdist,
                     self.range_v * self.prm.xbpmdist,
                     'o-', label="V sweep")
            axv.plot(vline, self.range_v * self.prm.xbpmdist,
                     '^-', label="V fit")
            axv.set_xlabel(u"$x$ [$\\mu$m]")
            axv.set_ylabel(u"$y$ [$\\mu$m]")
            axv.set_title("Central Vertical Sweeps")
            axv.set_xlim((np.min(self.range_v) * 0.005 + fit_cv_h[1, 0])
                         * self.prm.xbpmdist,
                         (np.max(self.range_v) * 0.005 + fit_cv_h[1, 0])
                         * self.prm.xbpmdist)
            axv.grid(True)
            axv.legend()

        if self.prm.outputfile:
            outfile = f"xbpm_sweeps_{self.prm.beamline}.png"
            fig.savefig(outfile, dpi=FIGDPI)
            print(f" Figure of central sweeps saved to file {outfile}.\n")

    def show_blades_at_center(self) -> None:
        """Display blade measurements along central sweeping points."""
        # Ensure we have sweep data
        if self.range_h is None or self.range_v is None:
            self.analyze_central_sweeps(show=False)

        if self.blades_h is None and self.blades_v is None:
            print("\n WARNING: could not retrieve blades' currents,"
                  " maybe there is insufficient data."
                  " Skipping central analysis.")
            return

        fig, (axh, axv) = plt.subplots(1, 2, figsize=(10, 5))

        if self.blades_h is not None:
            for key, blval in self.blades_h.items():
                val = blval[:, 0]
                wval = blval[:, 1]
                weight = 1. / wval if not np.isinf(1. / wval).any else None
                (acoef, bcoef) = np.polyfit(self.range_h, val, deg=1, w=weight)
                axh.plot(self.range_h, self.range_h * acoef + bcoef, "o-",
                         label=f"{key} fit")
                axh.errorbar(self.range_h, val, wval, fmt='^-', label=key)

        if self.blades_v is not None:
            for key, blval in self.blades_v.items():
                val = blval[:, 0]
                wval = blval[:, 1]
                weight = 1. / wval if not np.isinf(1. / wval).any else None
                (acoef, bcoef) = np.polyfit(self.range_v, val, deg=1, w=weight)
                axv.plot(self.range_v, self.range_v * acoef + bcoef, "o-",
                         label=f"{key} fit")
                axv.errorbar(self.range_v, val, wval, fmt='^-', label=key)

        axh.set_title("Horizontal")
        axv.set_title("Vertical")
        axh.legend()
        axv.legend()
        axh.grid()
        axv.grid()
        axh.set_xlabel(u"$x$ $\\mu$rad")
        axv.set_xlabel(u"$y$ $\\mu$rad")

        # DEBUG
        print(f" SHOW BLADES AT CENTER: {self.prm.beamline}")
        # DEBUG

        ylabel = (u"$I$ [# counts]" if self.prm.beamline[:3]
                  in ["MGN", "MNC"] else u"$I$ [A]")
        axh.set_ylabel(ylabel)
        axv.set_ylabel(ylabel)
        fig.tight_layout()

        if self.prm.outputfile:
            outfile = f"central_sweep_{self.prm.beamline}.png"
            fig.savefig(outfile, dpi=FIGDPI)
            print("\n Figure of blades behaviour at central sweeps"
                  f" saved to file {outfile}.\n")

    def calculate_scaled_positions(self, showmatrix: bool = True) -> list:
        """Calculate positions with suppression matrix correction.

        Applies suppression matrices to correct for blade gain variations
        and scales results to physical distances.

        Args:
            showmatrix: If True, display blade behavior matrices.

        Returns:
            List of [pairwise_positions_dict, cross_positions_dict].
        """
        return self.calculate_positions(
            rtitle="scaled XBPM positions",
            nosuppress=False,
            showmatrix=showmatrix
        )

    def calculate_raw_positions(self, showmatrix: bool = True) -> list:
        """Calculate positions without suppression (raw)."""
        return self.calculate_positions(
            rtitle="raw XBPM positions",
            nosuppress=True,
            showmatrix=showmatrix
        )

    def calculate_positions(self, rtitle: str,
                             nosuppress: bool,
                             showmatrix: bool = True) -> list:
        """Orchestrate XBPM position calculations and visualization.

        Ensures central sweeps are analyzed, then delegates to
        `xbpm_position_calc` to compute positions and plot results.
        """
        # Ensure sweep data is available
        if (self.range_h is None or self.range_v is None or
                self.blades_h is None or self.blades_v is None):
            self.analyze_central_sweeps(show=False)

        return xbpm_position_calc(
            self.data, self.prm,
            self.range_h, self.range_v,
            self.blades_h, self.blades_v,
            rtitle=rtitle,
            nosuppress=nosuppress,
            showmatrix=showmatrix,
        )


class BPMProcessor:
    """Processes BPM-only data to estimate XBPM positions.

    Encapsulates the legacy BPM calculation flow so orchestration can
    call a single entry point instead of standalone functions.
    """

    def __init__(self, rawdata, prm: Prm):
        """Store raw BPM/XBPM dataset and parameters for later processing."""
        self.rawdata = rawdata
        self.prm = prm

    def calculate_positions(self):
        """Calculate and plot XBPM positions derived from BPM data."""
        if self.prm.section is None:
            print("### ERROR: no section defined for the beamline in data set."
                  "\n### Cannot proceed with BPM data analysis. Skipping.")
            return

        if self.rawdata is None:
            print("### ERROR: no raw BPM data available."
                  "\n### Skipping BPM analysis.")
            return

        fig, ax = plt.subplots()

        sector_idx = self._sector_index()
        tangents = self._tangents_calc(sector_idx)

        print("# Distance between BPMs            ="
              f" {self.prm.bpmdist:8.4f}  m\n"
              "# Distance between source and XBPM ="
              f" {self.prm.xbpmdist:8.4f} m\n")

        xbpm_pos = self._positions_from_tangents(tangents, self.prm.xbpmdist)

        xpos, ypos, xnom, ynom = [], [], [], []
        for key, val in xbpm_pos.items():
            xnom.append(key[0])
            ynom.append(key[1])
            xpos.append(val[0])
            ypos.append(val[1])

        self._std_dev_estimate(xnom, ynom, xpos, ypos)

        ax.plot(xpos, np.array(ypos), 'bo', label="measured")
        ax.plot(xnom, ynom, 'r+', label="nominal")

        ax.set_xlabel("$x$ [$\\mu$m]")  # noqa: W605
        ax.set_ylabel("$y$ [$\\mu$m]")  # noqa: W605
        ax.set_title(f"Beam positions @ {self.prm.beamline} from BPM values")

        lim = np.max(np.abs(xnom + ynom)) * 1.7
        ax.axis("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.legend()
        ax.grid()

        if self.prm.outputfile:
            outfile = f"bpm_positions_{self.prm.beamline}.png"
            fig.savefig(outfile, dpi=FIGDPI)
            print(" Figure of positions calculated by BPM measurements "
                  f"saved to file {outfile}.\n")

    def _sector_index(self) -> int:
        sector = int(self.prm.section.split(':')[1][:2])
        return 8 * (sector - 1) - 1

    def _tangents_calc(self, idx):
        """Calculate tangents of beam angles between neighbour BPMs."""
        nextidx = idx + 1
        offset_x_sect, offset_y_sect = 0, 0
        offset_x_next, offset_y_next = 0, 0
        offsetfound = False

        for dt in self.rawdata:
            if dt[2]['agx'] == 0 and dt[2]['agy'] == 0:
                offset_x_sect = dt[2]['orbx'][idx]
                offset_y_sect = dt[2]['orby'][idx]
                offset_x_next = dt[2]['orbx'][nextidx]
                offset_y_next = dt[2]['orby'][nextidx]
                offsetfound = True
                break

        if not offsetfound:
            (offset_x_sect, offset_x_next,
             offset_y_sect, offset_y_next) = self._offset_search(idx)

        tangents = dict()
        for dt in self.rawdata:
            tx = (((dt[2]['orbx'][nextidx] - offset_x_next)) -
                  (dt[2]['orbx'][idx]     - offset_x_sect)) / self.prm.bpmdist
            ty = (((dt[2]['orby'][nextidx] - offset_y_next)) -
                  (dt[2]['orby'][idx]     - offset_y_sect)) / self.prm.bpmdist
            agx, agy = dt[2]['agx'], dt[2]['agy']
            tangents[agx, agy] = np.array([tx, ty])
        return tangents

    def _offset_search(self, idx):
        """Extrapolate offsets when reference orbit is missing."""
        nextidx = idx + 1
        agx    = np.array([dt[2]['agx']           for dt in self.rawdata])
        orbx   = np.array([dt[2]['orbx'][idx]     for dt in self.rawdata])
        n_orbx = np.array([dt[2]['orbx'][nextidx] for dt in self.rawdata])

        agxmax = np.max(agx)
        agxmin = np.min(agx)

        osx = np.array(sorted(list(set(orbx))))
        oxmin, oxmax = osx[0], osx[-1]
        offset_x_sect = (oxmin * agxmax - oxmax * agxmin) / (agxmax - agxmin)

        onx = np.array(sorted(list(set(n_orbx))))
        onxmin, onxmax = onx[0], onx[-1]
        offset_x_next = (onxmin * agxmax - onxmax * agxmin) / (agxmax - agxmin)

        agy    = np.array([dt[2]['agy']           for dt in self.rawdata])
        orby   = np.array([dt[2]['orby'][idx]     for dt in self.rawdata])
        n_orby = np.array([dt[2]['orby'][nextidx] for dt in self.rawdata])

        agymax = np.max(agy)
        agymin = np.min(agy)

        osy = np.array(sorted(list(set(orby))))
        oymin, oymax = osy[0], osy[-1]
        offset_y_sect = (oymin * agymax - oymax * agymin) / (agymax - agymin)

        ony = np.array(sorted(list(set(n_orby))))
        onymin, onymax = ony[0], ony[-1]
        offset_y_next = (onymin * agymax - onymax * agymin) / (agymax - agymin)

        return (offset_x_sect, offset_x_next, offset_y_sect, offset_y_next)

    def _positions_from_tangents(self, tangents, xbpm_dist):
        """Calculate beam positions from tangents at BPMs."""
        positions = dict()
        for key, tg in tangents.items():
            newkey = (key[0] * xbpm_dist, key[1] * xbpm_dist)
            positions[newkey] = tg * xbpm_dist
        return positions

    def _std_dev_estimate(self, xnom, ynom, xpos, ypos):
        """Estimate RMS deviations between measured and nominal positions."""
        np_x_nom = np.array(xnom)
        np_y_nom = np.array(ynom)
        np_x_pos = np.array(xpos)
        np_y_pos = np.array(ypos)

        nfh = np_x_nom.shape[0]
        nfv = np_y_nom.shape[0]
        diff_h = np.abs(np_x_nom.ravel() - np_x_pos.ravel())
        diff_h_max = np.max(diff_h)
        sig2_h = np.sum(diff_h**2) / nfh

        diff_v = np.abs(np_y_nom.ravel() - np_y_pos.ravel())
        diff_v_max = np.max(diff_v)
        sig2_v = np.sum(diff_v**2) / nfv

        print("Sigmas:\n"
              f"   (all sites)     H = {np.sqrt(sig2_h):.4f}\n"
              f"   (all sites)     V = {np.sqrt(sig2_v):.4f},\n"
              f"   (all sites) total = {np.sqrt(sig2_h + sig2_v):.4f}\n"
              "\n  Maximum difference:\n"
              f"   (all sites) H = {diff_h_max:.4f}\n"
              f"   (all sites) V = {diff_v_max:.4f},\n")

        nsh_x, nsh_y = len(set(xnom)), len(set(ynom))
        nmax = nsh_x * nsh_y

        if nmax > nfh or nmax > nfv:
            print("\n WARNING: sweeping looks incomplete, no ROI was defined. "
                  " (Maybe just one line swept?)")
            return

        frh, uptoh = int(nsh_x / 2 - 2), int(nsh_x / 2 + 2)
        frv, uptov = int(nsh_y / 2 - 2), int(nsh_y / 2 + 2)

        if nsh_x == 1 or nsh_y == 1:
            np_x_nom_cut = np_x_nom.reshape(nsh_x, nsh_y)[0, frv:uptov]
            np_y_nom_cut = np_y_nom.reshape(nsh_x, nsh_y)[0, frv:uptov]
            np_x_pos_cut = np_x_pos.reshape(nsh_x, nsh_y)[0, frv:uptov]
            np_y_pos_cut = np_y_pos.reshape(nsh_x, nsh_y)[0, frv:uptov]
        else:
            np_x_nom_cut = np_x_nom.reshape(nsh_x, nsh_y)[frv:uptov, frh:uptoh]
            np_y_nom_cut = np_y_nom.reshape(nsh_x, nsh_y)[frv:uptov, frh:uptoh]
            np_x_pos_cut = np_x_pos.reshape(nsh_x, nsh_y)[frv:uptov, frh:uptoh]
            np_y_pos_cut = np_y_pos.reshape(nsh_x, nsh_y)[frv:uptov, frh:uptoh]

        sig2_v_roi = np.sum((np_y_nom_cut.ravel() -
                             np_y_pos_cut.ravel())**2) / nfv
        sig2_h_roi = np.sum((np_x_nom_cut.ravel() -
                             np_x_pos_cut.ravel())**2) / nfh

        print("  Differences in ROI\n"
              f"   (x in [{np.min(np_x_nom_cut)}, {np.max(np_x_nom_cut)}];"
              f"  y in [{np.min(np_y_nom_cut)}, {np.max(np_y_nom_cut)}])\n"
              f"       H = {np.sqrt(sig2_h_roi):.4f}\n"
              f"       V = {np.sqrt(sig2_v_roi):.4f},\n"
              f"   total = {np.sqrt(sig2_h_roi + sig2_v_roi):.4f}")


class BladeMapVisualizer:
    """Visualizes XBPM blade intensity maps.

    This class creates color maps showing the intensity (current) measured
    by each blade across the measurement grid.

    Attributes:
        data (dict): Measurement data dictionary.
        prm (Prm): Parameters dataclass.
    """

    def __init__(self, data: dict, prm: Prm):
        """Initialize visualizer with data and parameters.

        Args:
            data: Measurement data dictionary.
            prm: Parameters dataclass instance.
        """
        self.data = data
        self.prm = prm

    def show(self) -> None:
        """Display blade intensity maps for all four blades.

        Creates a 2x2 subplot figure showing heatmaps for:
        - Top-Inner (TI), Top-Outer (TO)
        - Bottom-Inner (BI), Bottom-Outer (BO)

        Arranges blades in quadrants:
        [TI  TO]
        [BI  BO]
        """
        blades, stddevs = data_parse(self.data)
        to, ti, bi, bo = blades

        fig, rx = plt.subplots(2, 2, figsize=(8, 5))

        # Calculate extent for proper axis labels
        if to.shape[0] == 1 or to.shape[1] == 1:
            extent = None
        else:
            alist = np.array(list(self.data.keys()))
            klist = np.unique(alist[:, 0])
            mlist = np.unique(alist[:, 1])
            minvalx, maxvalx = np.min(klist), np.max(klist)
            minvaly, maxvaly = np.min(mlist), np.max(mlist)
            extent = (minvalx, maxvalx, minvaly, maxvaly)

        quad = [[ti, to], [bi, bo]]
        names = [["TI", "TO"], ["BI", "BO"]]

        for idy in range(2):
            for idx in range(2):
                rx[idy][idx].imshow(quad[idy][idx], extent=extent)
                rx[idy][idx].set_xlabel(u"$x$ [$\\mu$rad]")
                rx[idy][idx].set_ylabel(u"$y$ [$\\mu$rad]")
                rx[idy][idx].set_title(names[idy][idx])

        fig.tight_layout(pad=0., w_pad=-15., h_pad=2.)

        if self.prm.outputfile:
            outfile = f"blade_map_{self.prm.beamline}.png"
            fig.savefig(outfile, dpi=FIGDPI)
            print(f" Figure of blades' map saved to file {outfile}.\n")


class PositionVisualizer:
    """Visualizes calculated XBPM beam positions.

    This class handles visualization of position calculation results:
    - Nominal vs calculated positions on full grid
    - Closeup view of Region of Interest (ROI)
    - RMS position differences heatmap

    Can display results from either pairwise or cross-blade calculations,
    with or without suppression matrix corrections.

    Attributes:
        prm (Prm): Parameters dataclass.
        title (str): Title for the visualization.
        fig (matplotlib.figure.Figure): Matplotlib figure object for
             current visualization.
    """

    def __init__(self, prm: Prm, title: str = ""):
        """Initialize visualizer with parameters.

        Args:
            prm: Parameters dataclass instance.
            title: Title prefix for plots (e.g., "Pairwise" or "Cross-blades").
        """
        self.prm = prm
        self.title = title
        self.fig = None

    def show_position_results(self, pos_nom_h, pos_nom_v,
                         pos_h, pos_v, pos_h_roi, pos_v_roi,
                         pos_nom_h_roi, pos_nom_v_roi,
                         diff_roi, figsize=(18, 6)) -> None:
        """Display full position results in 1x3 subplot layout.

        Args:
            pos_nom_h: Nominal horizontal positions (full grid).
            pos_nom_v: Nominal vertical positions (full grid).
            pos_h: Calculated horizontal positions (full grid).
            pos_v: Calculated vertical positions (full grid).
            pos_h_roi: Calculated horizontal positions in ROI.
            pos_v_roi: Calculated vertical positions in ROI.
            pos_nom_h_roi: Nominal horizontal positions in ROI.
            pos_nom_v_roi: Nominal vertical positions in ROI.
            diff_roi: RMS position differences in ROI.
            figsize: Figure size as (width, height) tuple.
        """
        self.fig, (ax_all, ax_close, ax_color) = plt.subplots(1, 3,
                                                              figsize=figsize)

        # Full grid view
        self._plot_scaled_positions(
            ax_all, pos_nom_h, pos_nom_v, pos_h, pos_v,
            f"{self.title} @ {self.prm.beamline}"
        )

        # ROI closeup
        self._plot_scaled_positions(
            ax_close, pos_nom_h_roi, pos_nom_v_roi,
            pos_h_roi, pos_v_roi,
            f"{self.title} @ {self.prm.beamline} closeup"
        )

        # Difference heatmap
        self._plot_position_differences(
            ax_color, diff_roi, pos_nom_h_roi, pos_nom_v_roi,
            f"{self.title} @ {self.prm.beamline}"
        )

        self.fig.tight_layout()

    def _plot_scaled_positions(self, ax, pos_nom_h, pos_nom_v,
                              pos_h, pos_v, title):
        """Plot nominal vs calculated positions on given axis."""
        ax.set_title(title)
        pos = ax.plot(pos_h, pos_v, 'bo')
        nom = ax.plot(pos_nom_h, pos_nom_v, 'r+')
        ax.set_xlabel(u"$x$ [$\\mu$m]")
        ax.set_ylabel(u"$y$ [$\\mu$m]")
        ax.axis('equal')
        handles, labels = [], []
        if len(nom) > 0:
            handles.append(nom[0])
            labels.append("Nominal")
        if len(pos) > 0:
            handles.append(pos[0])
            labels.append("Calculated")
        if handles:
            ax.legend(handles, labels)
        ax.grid()

    def _plot_position_differences(self, ax, diffroi, pos_nom_h=None,
                                   pos_nom_v=None, title=""):
        """Plot position difference heatmap or scatter on given axis."""
        if len(diffroi.shape) > 1:
            im = ax.imshow(diffroi, cmap='viridis')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(u"RMS differences (ROI)")
        else:
            # Ensure x, y, and color arrays have matching lengths
            if pos_nom_h is not None and pos_nom_v is not None:
                x = np.ravel(pos_nom_h)
                y = np.ravel(pos_nom_v)
            else:
                # Fallback to index-based positions
                n = int(np.size(diffroi))
                x = np.arange(n)
                y = np.zeros(n)

            cvals = np.ravel(diffroi)

            # Align lengths if shapes are inconsistent
            m = min(len(x), len(y), len(cvals))
            x, y, cvals = x[:m], y[:m], cvals[:m]

            scatter = ax.scatter(x, y, c=cvals, cmap='viridis', s=50)
            plt.colorbar(scatter, ax=ax, label='Difference [$\\mu$m]')

        ax.set_title(title)
        ax.set_xlabel(u"$x$ [$\\mu$m]")
        ax.set_ylabel(u"$y$ [$\\mu$m]")

    def save_figure(self, filename: str) -> None:
        """Save the figure to a file.

        Args:
            filename: Path to save the figure to.
        """
        if self.fig is not None:
            self.fig.savefig(filename, dpi=FIGDPI, bbox_inches='tight')
            print(f" Figure saved to {filename}")


# Power relative to Ampere subunits.
AMPSUB = {
    0    : 1.0,    # no unit defined.
    "mA" : 1e-3,   # mili
    "uA" : 1e-6,   # micro
    "nA" : 1e-9,   # nano
    "pA" : 1e-12,  # pico
    "fA" : 1e-15,  # femto
    "aA" : 1e-18,  # atto
}

# Map of blades positions in each XBPM.
# TO, TI, BO, BI : top/bottom, in/out, relative to the storage ring;
# A, B, C, D : names of respective P.V.s
BLADEMAP = {
    "MNC"  : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
    "MNC1" : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
    "MNC2" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
    "CAT"  : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
    "CAT1" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},

    # ## To be checked: ## #
    # "CAT2": {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
    "CNB"  : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
    "CNB1" : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
    "CNB2" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
    "MGN"  : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
    "MGN1" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
    "MGN2" : {"TO": 'B', "TI": 'C', "BI": 'A', "BO": 'D'},
    "SIMUL": {"TO": 'A', "TI": 'B', "BI": 'C', "BO": 'D'},
}

# The XBPM beamlines.
BEAMLINENAME = {
    "CAT": "Caterete",
    "CNB": "Carnauba",
    "MGN": "Mogno",
    "MNC": "Manaca",
    "N/A": "Not defined",
}

# Distances between two adjacent BPMs around the source of bump at each line.
BPMDISTS = {
    "CAT": 6.175495,
    "CNB": 6.175495,
    "MGN": 2.2769999999999015,
    "MNC": 7.035495,
}

# Distance from source (its center) to XBPM at each beamline.
# Obtained from comissioning reports.
XBPMDISTS = {
    "CAT":  15.740,
    "CAT1": 15.740,
    "CAT2": 19.590,
    "CNB": 15.740,
    "CNB1": 15.740,
    "CNB2": 19.590,
    "MGN1": 10.237,
    "MGN2": 16.167,
    "MNC1": 15.740,
    "MNC2": 19.590,
}

# Sections of the ring for each beamline.
SECTIONS = {
    "CAT": "subsec:07SP",
    "CNB": "subsec:06SB",
    "MGN": "subsec:10BC",
    "MNC": "subsec:09SA"
}



# ## Select data.

def blades_fetch(rawdata, beamline):
    """Retrieve each blade's data and average over their values.

    Obs.: A map of the blade positions must be provided for each beamline.

    Args:
        rawdata (list) : acquired data from bpm and xbpm measurements.
        beamline (str) : beamline to be analysed.

    Returns:
        data (dict) : averaged and std dev values from blades' measured
            data; bpm h and v angular positions (in urad), taken as 'real'
            positions, work as indices.
    """
    data = dict()

    for dt in rawdata:
        try:
            xbpm = dt[0][beamline]
            vals = list()
            for blade in BLADEMAP[beamline].values():
                # Average and std dev over measured values of current blade.
                av, sd = blade_average(xbpm[f'{blade}_val'], beamline)
                vals.append((av, sd))
            bpm_x = dt[2]['agx']
            bpm_y = dt[2]['agy']
            data[bpm_x, bpm_y] = np.array(vals)
        except Exception as err:
            print("\n WARNING: when fetching blades' values and averaging:"
                  f" {err}\n")
    return data


def blade_average(blade, beamline):
    """Calculate the average of blades' values.

    Args:
        blade (numpy array) : raw data measured by given blade.
        beamline (str) : beamline identifier to define the type of data
            for averaging.

    Returns: averaged measured value for the blade and its standard deviation.
    """
    # Decide the the type of data to be average over.
    # If data is given in arbitrary units.
    if beamline in ["MGN", "MNC"]:
        return np.average(blade), np.std(blade)

    # If data is in subunits, convert to Amperes.
    vals = np.array([
        vv * AMPSUB[un] for vv, un in blade
    ])
    return np.average(vals), np.std(vals)


def data_parse(data):
    """Extract each blade's data from whole data.

    Args:
        data (dict): keys are (x, y) grid positions, values are blades'
            measurements with errors (order: to, ti, bi, bo).

    Returns:
        to, ti, bi, bo (numpy arrays): the values read by each blade.

    Note: the keys in data are the positions in the coordinate system, but the
        indexing of the numpy arrrays to, ti, bi, bo are the conventional ones,
        so the [0, 0] index corresponds to the left-upmost position etc.
    """
    dk = np.array(list(data.keys()))

    nh = np.unique(dk[:, 0])
    nv = np.unique(dk[:, 1])
    ngrid = (nv.shape[0], nh.shape[0])
    to,  ti  = np.zeros(ngrid), np.zeros(ngrid)
    bo,  bi  = np.zeros(ngrid), np.zeros(ngrid)
    sto, sti = np.zeros(ngrid), np.zeros(ngrid)
    sbo, sbi = np.zeros(ngrid), np.zeros(ngrid)

    for ii, nl in enumerate(nv):
        for jj, nc in enumerate(nh):
            key = (nc, nl)
            ilin = ngrid[0] - ii - 1
            icol = jj

            # Check whether data ends prematurely.
            if key not in data.keys():
                break

            try:
                to[ilin, icol]  = data[key][0, 0]
                ti[ilin, icol]  = data[key][1, 0]
                bi[ilin, icol]  = data[key][2, 0]
                bo[ilin, icol]  = data[key][3, 0]

                sto[ilin, icol] = data[key][0, 1]
                sti[ilin, icol] = data[key][1, 1]
                sbi[ilin, icol] = data[key][2, 1]
                sbo[ilin, icol] = data[key][3, 1]
            except Exception as err:
                print(f"\n WARNING, when trying to parse blade data: {err}"
                      f"\n nominal position: {err},"
                      f" array index: {ilin}, {icol}"
                      "\n Maybe data grid is incomplete?")

    return [to, ti, bi, bo], [sto, sti, sbi, sbo]


# ## Beam position from XBPM data.


def xbpm_position_calc(data, prm, range_h, range_v, blades_h, blades_v,
                       rtitle="", nosuppress=False, showmatrix=True):
    """Calculate positions from blades' measured data.

    Notation:
      'cr' means 'cross-blade' calculation;
      'pair' means 'pairwise-blade' calculation.

    Args:
        data    (dict)  : nominal positions and respective data from blades.
        prm     (dict)  : all parameters of the data and the analysis.
        range_h (numpy array) : horizontal range swept by the beam
        range_v (numpy array) : vertical range swept by the beam
        blades_h (dict) : measured data from each blade at the horizontal
                central line
        blades_v (dict) : measured data from each blade at the vertical
                central line
        rtitle (str) : title to be shown in the plots.
        nosuppress (boolean) : do not apply suppression matrix if True.
        showmatrix (boolean) : show blades' behaviour at center or not.

    Returns:
        pos_pair_h/_v (list) : the pairwise calculated positions
        pos_cr_h/_v (list)   : the cross-blade calculated positions.
    """
    # Parse data. blades and stddevs are lists with the order:
    # to, ti, bi, bo.
    blades, stddevs = data_parse(data)

    # Pairwise-blades calculation.
    supmat = suppression_matrix(range_h, range_v,
                                blades_h, blades_v, prm,
                                showmatrix=showmatrix, nosuppress=nosuppress)

    pos_pair  = beam_position_pair(data, supmat)
    (pos_nom_h, pos_nom_v,
     pos_pair_h, pos_pair_v) = position_dict_parse(pos_pair, prm.gridstep)

    # Adjust to real distance.
    pos_nom_h *= prm.xbpmdist
    pos_nom_v *= prm.xbpmdist

    # Indices of central ranges for scaling - ROI
    # (data keys are the nominal positions).
    keys = np.array(list(data.keys()))
    range_h = np.unique(keys[:, 0])
    range_v = np.unique(keys[:, 1])
    halfh = int(range_h.shape[0] / 2)
    halfv = int(range_v.shape[0] / 2)
    abh = STD_ROI_SIZE if range_h[-1] > STD_ROI_SIZE else halfh
    frh, uph = halfh - abh, halfh + abh + 1
    abv = STD_ROI_SIZE if range_v[-1] > STD_ROI_SIZE else halfv
    frv, upv = halfv - abv, halfv + abv + 1

    # ROI: nominal.
    if pos_nom_h.shape[0] == 1 or pos_nom_v.shape[0] == 1:
        pos_nom_h_roi = pos_nom_h[0, frv:upv]
        pos_nom_v_roi = pos_nom_v[0, frv:upv]
    elif pos_nom_h.shape[1] == 1 or pos_nom_v.shape[1] == 1:
        pos_nom_h_roi = pos_nom_h[frv:upv, 0]
        pos_nom_v_roi = pos_nom_v[frv:upv, 0]
    else:
        pos_nom_h_roi  = pos_nom_h[frh:uph, frv:upv]
        pos_nom_v_roi  = pos_nom_v[frh:uph, frv:upv]

    # ### Pairwise-blades calculation.
    # ROI.
    if pos_nom_h.shape[0] == 1 or pos_nom_v.shape[0] == 1:
        pos_pair_h_roi = pos_pair_h[0, frv:upv]
        pos_pair_v_roi = pos_pair_v[0, frv:upv]
    elif pos_nom_h.shape[1] == 1 or pos_nom_v.shape[1] == 1:
        pos_pair_h_roi = pos_pair_h[frv:upv, 0]
        pos_pair_v_roi = pos_pair_v[frv:upv, 0]
    else:
        pos_pair_h_roi = pos_pair_h[frh:uph, frv:upv]
        pos_pair_v_roi = pos_pair_v[frh:uph, frv:upv]

    # Scaling coefficients, pairwise calculation.
    (kxp, deltaxp,
     kyp, deltayp) = scaling_fit(pos_pair_h_roi, pos_pair_v_roi,
                                 pos_nom_h_roi, pos_nom_v_roi, "Pairwise")
    pos_pair_h_scaled = kxp * pos_pair_h + deltaxp
    pos_pair_v_scaled = kyp * pos_pair_v + deltayp
    # ROI scaled positions.
    pos_pair_h_roi_scaled = kxp * pos_pair_h_roi + deltaxp
    pos_pair_v_roi_scaled = kyp * pos_pair_v_roi + deltayp

    # Calculate differences for pairwise
    diffx2  = (pos_pair_h_roi_scaled - pos_nom_h_roi) ** 2
    diffy2  = (pos_pair_v_roi_scaled - pos_nom_v_roi) ** 2
    diffpairroi = np.sqrt(diffx2 + diffy2)

    # Visualize pairwise positions
    pair_visualizer = PositionVisualizer(prm, f"Pairwise {rtitle}")
    pair_visualizer.show_position_results(
        pos_nom_h, pos_nom_v,
        pos_pair_h_scaled, pos_pair_v_scaled,
        pos_pair_h_roi_scaled, pos_pair_v_roi_scaled,
        pos_nom_h_roi, pos_nom_v_roi,
        diffpairroi
    )

    # ### Cross-blades calculation.
    pos_cr_h, pos_cr_v = beam_position_cross(blades)

    # ROI: crossing blades.
    if pos_nom_h.shape[0] == 1 or pos_nom_v.shape[0] == 1:
        pos_cr_h_roi  = pos_cr_h[0, frv:upv]
        pos_cr_v_roi  = pos_cr_v[0, frv:upv]
    elif pos_nom_h.shape[1] == 1 or pos_nom_v.shape[1] == 1:
        pos_cr_h_roi  = pos_cr_h[frv:upv, 0]
        pos_cr_v_roi  = pos_cr_v[frv:upv, 0]
    else:
        pos_cr_h_roi  = pos_cr_h[frh:uph, frv:upv]
        pos_cr_v_roi  = pos_cr_v[frh:uph, frv:upv]

    # Scaling coefficients, cross-blades calculation.
    (kxc, deltaxc,
     kyc, deltayc) = scaling_fit(pos_cr_h_roi, pos_cr_v_roi,
                                 pos_nom_h_roi, pos_nom_v_roi, "Cross")
    pos_cr_h_scaled = kxc * pos_cr_h + deltaxc
    pos_cr_v_scaled = kyc * pos_cr_v + deltayc
    # ROI scaled positions.
    pos_cr_h_roi_scaled = kxc * pos_cr_h_roi + deltaxc
    pos_cr_v_roi_scaled = kyc * pos_cr_v_roi + deltayc

    diffx2  = (pos_cr_h_roi_scaled - pos_nom_h_roi) ** 2
    diffy2  = (pos_cr_v_roi_scaled - pos_nom_v_roi) ** 2
    diffcrroi = np.sqrt(diffx2 + diffy2)

    # Visualize cross-blade positions
    cross_visualizer = PositionVisualizer(prm, f"Cross-blades {rtitle}")
    cross_visualizer.show_position_results(
        pos_nom_h, pos_nom_v,
        pos_cr_h_scaled, pos_cr_v_scaled,
        pos_cr_h_roi_scaled, pos_cr_v_roi_scaled,
        pos_nom_h_roi, pos_nom_v_roi,
        diffcrroi
    )

    if prm.outputfile:
        # Save images to current working directory when -o is set
        outdir = '.'
        sup = "raw" if nosuppress else "scaled"
        bl = prm.beamline

        # Save pairwise figure
        outfile_p = os.path.join(outdir, f"xbpm_pair_pos_{sup}_{bl}.png")
        pair_visualizer.save_figure(outfile_p)

        # Save cross-blade figure
        outfile_c = os.path.join(outdir, f"xbpm_cross_pos_{sup}_{bl}.png")
        cross_visualizer.save_figure(outfile_c)

    # Return calculated values.
    scaled_pos_pair = dict()
    scaled_pos_cros = dict()
    for ii, lin in enumerate(pos_nom_h):
        for jj, xx in enumerate(lin):
            yy = pos_nom_v[ii, jj]
            scaled_pos_pair[xx, yy] = [
                pos_pair_h_scaled[ii, jj],
                pos_pair_v_scaled[ii, jj]
                ]
            #
            scaled_pos_cros[xx, yy] = [
                pos_cr_h_scaled[ii, jj],
                pos_cr_v_scaled[ii, jj]
                ]
    return [scaled_pos_pair, scaled_pos_cros]


# Pairwise blades calculation.


def suppression_matrix(range_h, range_v, blades_h, blades_v, prm,
                       showmatrix=False, nosuppress=False):
    """Calculate the suppression matrix.

    Args:
        range_h (numpy array)  : horizontal coordinates to be analysed.
        range_v (numpy array)  : vertical coordinates to be analysed.
        blades_h (numpy array) : data recorded by blade's measurements at
            horizontal central line.
        blades_v (numpy array) : data recorded by blade's measurements at
            vertical central line.
        prm (dict) : general parameters of the system.

        showmatrix (bool) : show blades' curves.
        nosuppress (bool) : return a matrix where gains are set to 1.

    Returns:
        the suppression matrix (numpy array)
    """
    if nosuppress:
        # Set all suppressions / gains to 1.
        pch = np.ones(8).reshape(4, 2)
        pcv = np.ones(8).reshape(4, 2)
    else:
        # Linear fittings to each blade's data through
        # horizontal and vertical central lines.
        pch = central_line_fit(blades_h, range_h, 'h')
        pcv = central_line_fit(blades_v, range_v, 'v')

    # Normalize by the first blade and define suppression as 1/m.
    if len(range_h) > 1:
        pch = pch[0] / np.abs(pch)
    else:
        pch = np.ones(8).reshape(4, 2)
    #
    if len(range_v) > 1:
        pcv = pcv[0] / np.abs(pcv)
    else:
        pcv = np.ones(8).reshape(4, 2)

    # Signs are defined according to the blade positions for pairwise
    # calculations.
    supmat = np.array([
        [pcv[0, 0], -pcv[1, 0], -pcv[2, 0],  pcv[3, 0]],   # noqa: E241
        [pcv[0, 0],  pcv[1, 0],  pcv[2, 0],  pcv[3, 0]],   # noqa: E241
        [pch[0, 0],  pch[1, 0], -pch[2, 0], -pch[3, 0]],   # noqa: E241
        [pch[0, 0],  pch[1, 0],  pch[2, 0],  pch[3, 0]],   # noqa: E241
    ])

    if showmatrix:
        # blades_center_show(range_h, range_v, blades_h, blades_v,
        #                    np.array(pch), np.array(pcv))
        print("\n Suppression matrix:")
        for lin in supmat:
            for col in lin:
                print(f" {col:12.6f}", end='')
            print()
        print()

    with open(f"supmat_{prm.beamline}.dat", 'w') as fs:
        for lin in supmat:
            for col in lin:
                fs.write(f" {col:12.6f}")
            fs.write("\n")

    return supmat


def central_line_fit(blades, range_vals, direction):
    """Linear fittings to each blade's data through central line."""
    if blades is None:
        dr = 'horizontal' if direction == 'h' else 'vertical'
        print(f"\n WARNING: (central_line_fit) "
              f"{dr} blades' values not defined."
              " Seetting fitting values to [1, 0].")
        return np.array([[1, 0] for _ in range(4)])

    pc = list()
    for blade in blades.values():
        weight = 1. / blade[:, 1]

        # Ill-defined weights: use standard.
        if np.isinf(weight).any():
            weight = None

        pc.append(np.polyfit(range_vals, blade[:, 0], deg=1, w=weight))
    pc = np.array(pc)

    # Set elements to 1 if fitting is unsuccessful.
    if np.isinf(pc).any() or (pc == 0).any():
        pc = np.array([[1, 0] for _ in range(4)])

    return pc


def beam_position_pair(data, supmat):
    """Calculate beam position from blades' currents.

    Args:
        data (dict) : fetched data from bpm and xbpm (blades) measurements.
        supmat (numpy array): supression matrix.

    Returns:
        positions (numpy array) : h and v calculated positions for
            bpm and xbpm.
    """
    positions = dict()
    for pos, bld in data.items():
        dsps = supmat @ bld[:, 0]  # .T
        # Position is calculated as delta over sigma.
        positions[pos] = np.array([dsps[0] / dsps[1], dsps[2] / dsps[3]])

    # zero_origin(positions)
    return positions


def zero_origin(positions):
    """Subtract the values at the center of the grid to correct offset.

    Args:
        positions (dict) : calculated values of positions, indexed by nominal
        values at the grid.
    """
    zero = deepcopy(positions[0, 0])
    for val in positions.values():
        val -= zero


def position_dict_parse(data, gridstep):
    """Parse data from XBPM dictionary (scaled from BPM positions).

    Args:
        data (dict) :  calculated beam positions indexed by their respective
                       nominal positions.
        gridstep (float) : Step between neighbour sites in the grid.

    Returns:
        xbpm_nom_h, xbpm_nom_v (numpy array) : beam nominal positions.
        xbpm_meas_h, xbpm_meas_v (numpy array) : beam calculated positions
            from measured blades' currents.
    """
    gridlist = np.array(list(data.keys()))
    gsh_lin = len(np.unique(gridlist[:, 1]))
    gsh_col = len(np.unique(gridlist[:, 0]))

    xbpm_nom_h  = np.zeros((gsh_lin, gsh_col))
    xbpm_nom_v  = np.zeros((gsh_lin, gsh_col))
    xbpm_meas_h = np.zeros((gsh_lin, gsh_col))
    xbpm_meas_v = np.zeros((gsh_lin, gsh_col))

    minval_h = np.min(gridlist[:, 0])
    maxval_v = np.max(gridlist[:, 1])
    for key, val in data.items():
        col = int((key[0] - minval_h) / gridstep)
        lin = int((maxval_v - key[1]) / gridstep)

        try:
            xbpm_nom_h[lin, col]  = key[0]
            xbpm_nom_v[lin, col]  = key[1]
            xbpm_meas_h[lin, col] = val[0]
            xbpm_meas_v[lin, col] = val[1]
        except Exception as err:
            print(f"\n WARNING: failed when parsing positions dictionary:"
                  f"\n{err}\n lin, col = {lin}, {col}, key = {key}")
            continue

    return (xbpm_nom_h, xbpm_nom_v, xbpm_meas_h, xbpm_meas_v)


# Cross-blades calculation.

def beam_position_cross(blades):
    """Calculate beam position from blades' currents.

    Args:
        blades (list) : averaged values measured for each blade.
        prm (dict) : general parameters of the analysis.

    Returns:
        pos_h, pos_v (numpy array): calculated positions by crossing
            differences.
    """
    to, ti, bi, bo = blades
    v1 = (to - bi) / (to + bi)
    v2 = (ti - bo) / (ti + bo)
    hpos = (v1 - v2)
    vpos = (v1 + v2)
    return [hpos, vpos]


def scaling_fit(pos_h, pos_v, nom_h, nom_v, calctype=""):
    """Calculate scaling coefficients from fitted positions.

    Args:
        pos_h (numpy array) : calculated horizontal positions.
        pos_v (numpy array) : calculated vertical positions.
        nom_h (numpy array) : nominal horizontal positions.
        nom_v (numpy array) : nominal vertical positions.
        calctype (str) : type of calculation (for printing purposes).

    Returns:
        kx, ky, deltax, deltay (float) : scaling coefficients.
    """
    print(f"\n#### {calctype} blades:")

    # Clean up expurious numbers for fitting.
    hfinitemask = np.isfinite(pos_h)
    ph_cln = pos_h[hfinitemask]
    nh_cln = nom_h[hfinitemask]
    #
    vfinitemask = np.isfinite(pos_v)
    pv_cln = pos_v[vfinitemask]
    nv_cln = nom_v[vfinitemask]

    # Linear fit for scaling.
    kx, deltax = 1., 0.
    # Check if horizontal range spans through whole grid.
    if len(set(nom_h.ravel())) > 1:
        try:
            kx, deltax = np.polyfit(ph_cln, nh_cln, deg=1)
        except Exception as err:
            print(f"\n WARNING: when calculating horizontal scaling"
                  f" coefficients:\n{err}\n Setting to default values.")

    ky, deltay = 1., 0.
    # Check if vertical range spans through whole grid.
    if len(set(nom_v.ravel())) > 1:
        try:
            ky, deltay = np.polyfit(pv_cln, nv_cln, deg=1)
        except Exception as err:
            print(f"\n WARNING: when calculating vertical scaling"
                  f" coefficients:\n{err}\n Setting to default values.")

    print(f"kx = {kx:12.4f},   deltax = {deltax:12.4f}")
    print(f"ky = {ky:12.4f},   deltay = {deltay:12.4f}\n")
    return kx, deltax, ky, deltay


# ## Dump XBPM's selected and averaged data to file.


def data_dump(data, positions, prm, sup=""):
    """Dump data to file.

    Args:
        data (dict) : calculated beam positions at the grid indexed by
            their nominal position values.

        positions (list) : list of four arrays with positions calculated,
            pairwise blades hor/vert, cross-blades hor/vert.
        prm (dict) : general parameters of the analysis.
        sup (str) : data was rescaled with suppression matrix or not.

    """
    outfile = f"xbpm_blades_values_{prm.beamline}.dat"
    print(f"\n Writing out data to file {outfile} ...", end='')
    with open(outfile, 'w') as df:
        for key, val in data.items():
            df.write(f"{key[0]}  {key[1]}")
            for vv in val:
                df.write(f"  {vv[0]} {vv[1]}")
            df.write("\n")

    pos_pair, pos_cr = positions

    outfilep = f"xbpm_positions_pair_{sup}_{prm.beamline}.dat"
    print("\n Writing out pairwise blade calculated positions to file"
          f" {outfilep} ...", end='')
    with open(outfilep, 'w') as fp:
        for key, val in pos_pair.items():
            fp.write(f"{key[0]}  {key[1]}")
            fp.write(f"  {val[0]} {val[1]}\n")

    outfilec = f"xbpm_positions_cross_{sup}_{prm.beamline}.dat"
    print("\n Writing out cross-blade calculated positions to file"
          f" {outfilec} ...", end='')
    with open(outfilec, 'w') as fc:
        for key, val in pos_cr.items():
            fc.write(f"{key[0]}  {key[1]}")
            fc.write(f"  {val[0]} {val[1]}\n")

    print("done.\n")


# ## Main function.

def main():
    """Main entry point for XBPM data analysis."""
    # Read command line options.
    prm = ParameterBuilder.from_cli()

    # Read data from working directory or file.
    reader = DataReader(prm)
    data = reader.read()

    # Initialize processor for XBPM calculations
    processor = XBPMProcessor(data, prm)

    # Show results demanded by command line options.

    # Beam position at XBPM calculated from BPM data solely.
    # The sector is selected from 'section' parameter.
    if prm.xbpmfrombpm:
        raw = reader.rawdata if reader.rawdata is not None else data
        bpm_processor = BPMProcessor(raw, prm)
        bpm_processor.calculate_positions()

    # Dictionary with measured data from blades for each nominal position.
    if prm.showblademap:
        blade_map = BladeMapVisualizer(data, prm)
        blade_map.show()

    # Show central sweeping results.
    if prm.centralsweep:
        processor.analyze_central_sweeps(show=True)

    # Calculate beam position from XBPM data.
    if prm.showbladescenter:
        processor.show_blades_at_center()

    # Calculated positions with simple matrix.
    if prm.xbpmpositionsraw:
        positions = processor.calculate_raw_positions(showmatrix=True)
        # Dump data to file.
        if prm.outputfile:
            data_dump(data, positions, prm, sup="raw")

    # Calculated positions with suppression matrix.
    if prm.xbpmpositions:
        positions = processor.calculate_scaled_positions(showmatrix=True)
        # Dump data to file.
        if prm.outputfile:
            data_dump(data, positions, prm, sup="scaled")

    plt.show()


if __name__ == "__main__":
    main()
    print("\n\n Done.")
