"""Parameter handling and CLI parsing."""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional, Any, List

import numpy as np

# Import DataReader for canonical _extract_beamlines
# from xbpm_bumps.core.readers import DataReader

from .config import Config
from .constants import GRIDSTEP, HELP_DESCRIPTION, ROI_SIZE_H, ROI_SIZE_V


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
    showbpmpositions : bool                = False
    xbpmpositionsraw : bool                = False
    usebpmref        : bool                = False
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
    scalepolydeg     : Optional[int]       = 1
    section          : Optional[str]       = None
    blademap         : Optional[Any]       = None
    roisize          : Optional[List[int]] = field(
        default_factory=lambda: [ROI_SIZE_H, ROI_SIZE_V]
    )

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

    def __init__(self, prm: Optional[Prm] = None):
        """Initialize builder with optional pre-existing parameters."""
        self.prm:     Prm = prm if prm is not None else Prm()
        self.rawdata: Optional[list] = None

    def from_cli(self, argv=None) -> Prm:
        """Parse command-line arguments and build initial Prm instance.

        Args:
            argv: List of command-line arguments. If None, uses sys.argv.
        """
        args = self._parse_args(argv)
        self.prm = Prm(
            showblademap     = bool(args.showblademap),
            centralsweep     = bool(args.centralsweep),
            showbladescenter = bool(args.showbladescenter),
            xbpmpositions    = bool(args.xbpmpositions),
            showbpmpositions = bool(args.showbpmpositions),
            xbpmpositionsraw = bool(args.xbpmpositionsraw),
            usebpmref        = bool(args.usebpmref),
            outputfile       = bool(args.outputfile),
            xbpmdist         = args.xbpmdist,
            scalepolydeg     = args.scalepolydeg,
            workdir          = args.workdir,
            skip             = int(args.skip),
            gridstep         = float(args.gridstep),
            maxradangle      = 20.0,
            beamline         = "",
            roisize          = [
                args.roi_h if args.roi_h is not None else ROI_SIZE_H,
                args.roi_v if args.roi_v is not None else ROI_SIZE_V,
            ],
        )
        return self.prm

    def _parse_args(self,
                    argv: Optional[list[str]] = None) -> argparse.Namespace:
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
            '-b', action='store_true', dest='showbpmpositions',
            help='Show BPM positions panel in GUI'
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
        parser.add_argument(
            '--bpmref', action='store_true', dest='usebpmref',
            help='Use BPM measured positions as nominal reference'
        )

        # Options with values
        parser.add_argument(
            '-d', '--xbpmdist', type=float,
            help='Distance from source to XBPM [m]'
        )
        parser.add_argument(
            '-p', '--scalepolydeg', type=int, default=1,
            help='Scaling polynomial degree'
        )
        parser.add_argument(
            '-g', '--gridstep', type=float, default=GRIDSTEP,
            help=("""Step between neighbour sites in the grid.
                Usually inferred from data, but might be provided
                in some cases."""
            ),
        )
        parser.add_argument(
            '--roi-h', type=int, default=None,
            help='ROI horizontal size (number of sites)'
        )
        parser.add_argument(
            '--roi-v', type=int, default=None,
            help='ROI vertical size (number of sites)'
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

    def add_beamline_parameters(self) -> None:
        """Add beamline-specific parameters to prm."""
        self.prm.current = self.rawdata[0][2]["current"]

        try:
            self.prm.phaseorgap = self.rawdata[0][1][
                self.prm.beamline[:3].lower()
                ]
            # print(f"### Phase / Gap ({self.prm.beamline})   :\t"
            #       f" {self.prm.phaseorgap}")
        except Exception:
            print(f"\n WARNING: no phase/gap defined for {self.prm.beamline}.")

        self.prm.bpmdist  = Config.BPMDISTS[self.prm.beamline[:3]]
        self.prm.section  = Config.SECTIONS[self.prm.beamline[:3]]
        self.prm.blademap = Config.BLADEMAP[self.prm.beamline]

        if self.prm.xbpmdist is None:
            self.prm.xbpmdist = Config.XBPMDISTS[self.prm.beamline]
            print("\n WARNING: distance from source to"
                  f" {self.prm.beamline}'s XBPM set to"
                  f" {self.prm.xbpmdist:.3f} m (beamline default).")

        self.prm.gridstep = self._infer_gridstep()

    def _infer_gridstep(self) -> float:
        """Calculate grid step from data."""
        agx = [self.rawdata[ii][2]['agx'] for ii in range(len(self.rawdata))]
        agy = [self.rawdata[ii][2]['agy'] for ii in range(len(self.rawdata))]

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

    def lastfield(self, name: str, fld: str = ' ') -> str:
        """Get the last part of string separated by given character."""
        return name.split(fld)[-1]

    def found_key(self, my_dict: dict, target_value: str) -> str:
        """Find key in dictionary by its value."""
        return next(
            key for key, value in my_dict.items() if value == target_value
        )
