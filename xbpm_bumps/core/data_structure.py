"""Parameter handling and CLI parsing."""

# import sys
from dataclasses import dataclass, field
from typing import Optional, Any, List

import numpy as np

# Import DataReader for canonical _extract_beamlines
# from xbpm_bumps.core.readers import DataReader

from .config import Config
from .constants import ROI_SIZE_H, ROI_SIZE_V


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
    # gridstep         : float               = GRIDSTEP
    maxradangle      : float               = 20.0

    # runtime-filled fields
    beamline         : str                 = ""
    sr_current       : Optional[float]     = None
    phaseorgap       : Optional[dict]      = None
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


@dataclass
class Positions:
    """Container for calculated XBPM positions."""
    x : np.ndarray
    y : np.ndarray

    @classmethod
    def from_hdf5_group(cls, h5group) -> "Positions":
        """Create a Positions instance from x and y arrays."""
        if "x" not in h5group or "y" not in h5group:
            raise ValueError(
                " ERROR while reading XBPM positions from HDF5 file:\n"
                " Missing 'x' or 'y' dataset in HDF5 group."
            )
        return cls(x=h5group["x"][:], y=h5group["y"][:])


@dataclass
class Blades:
    """Container for blade current data and associated metadata.
    
    to, ti, bi, bo: measured currents for top in, top out, bottom in
                    and bottom out blades
    sto, sti, sbi, sbo: standard deviations of the respective currents
    """
    to  : np.ndarray = field(default_factory=lambda: np.array([]))
    ti  : np.ndarray = field(default_factory=lambda: np.array([]))
    bi  : np.ndarray = field(default_factory=lambda: np.array([]))
    bo  : np.ndarray = field(default_factory=lambda: np.array([]))
    sto : np.ndarray = field(default_factory=lambda: np.array([]))
    sti : np.ndarray = field(default_factory=lambda: np.array([]))
    sbi : np.ndarray = field(default_factory=lambda: np.array([]))
    sbo : np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_hdf5_group(cls, h5group) -> "Blades":
        """Create a Blades instance from an HDF5 group."""
        blades = ['to', 'ti', 'bi', 'bo', 'sto', 'sti', 'sbi', 'sbo']
        for blade in blades:
            if blade not in h5group:
                raise ValueError(
                    " ERROR while reading Average Blade Currents from HDF5"
                    f" file:\n Missing '{blade}' dataset in HDF5 group.")

        return cls(
            to  = h5group["to"][:],
            ti  = h5group["ti"][:],
            bi  = h5group["bi"][:],
            bo  = h5group["bo"][:],
            sto = h5group["sto"][:],
            sti = h5group["sti"][:],
            sbi = h5group["sbi"][:],
            sbo = h5group["sbo"][:],
        )


@dataclass
class BladeAvgData:
    """Container for averaged blade current data and associated metadata."""
    prm    : Optional[dict]      = None
    nom    : Optional[Positions] = None
    blades : Optional[Blades]    = None

    @classmethod
    def from_hdf5_group(cls, h5group) -> "BladeAvgData":
        """Create a BladeAvgData instance from an HDF5 group."""
        # Extract metadata attributes.
        prm = {key : val for key, val in h5group.attrs.items()}

        nom    = Positions.from_hdf5_group(
            h5group["nom"]) if "nom" in h5group else None

        blades = Blades.from_hdf5_group(h5group["blades"]) if "blades" in h5group else None

        return cls(prm=prm, nom=nom, blades=blades)

@dataclass
class BladeVals:
    """Container for one blade raw data and associated metadata.
    
    val        : measured currents for the blade
    range      : measurement range for the blade
    saturation : saturation levels for the blade
    """
    val        : np.ndarray = field(default_factory=lambda: np.array([]))
    range      : np.ndarray = field(default_factory=lambda: np.array([]))
    saturation : np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_hdf5_group(cls, h5group) -> "BladeVals":
        """Create a BladeVals instance from an HDF5 group."""
        required_fields = ['val', 'range', 'saturation']
        for fld in required_fields:
            if fld not in h5group:
                raise ValueError(
                    f" ERROR while reading BladeVals from HDF5 file:\n"
                    f" Missing '{fld}' dataset in HDF5 group."
                )

        return cls(
            val        = h5group["val"][:],
            range      = h5group["range"][:],
            saturation = h5group["saturation"][:]
        )


@dataclass
class BPMData:
    desc : str
    pos  : Positions

    @classmethod
    def from_hdf5_group(cls, h5group) -> "BPMData":
        """Create a BPMData instance from an HDF5 group."""
        if "desc" not in h5group.attrs:
            raise ValueError(
                " ERROR while reading BPMData from HDF5 file:\n"
                " Missing 'desc' attribute in HDF5 group."
            )
        if "pos" not in h5group:
            raise ValueError(
                " ERROR while reading BPMData from HDF5 file:\n"
                " Missing 'pos' dataset in HDF5 group."
            )

        desc = h5group.attrs["desc"]
        pos  = Positions.from_hdf5_group(h5group)
        return cls(desc=desc, pos=pos)


@dataclass
class BladeRawData:
    """Container for all blades' raw data and associated metadata.
    
    A, B, C, D: BladeVals for each blade
    """
    A : BladeVals = field(
        default_factory=lambda: BladeVals(
        val=np.array([]), range=np.array([]), saturation=np.array([])
        ))
    B : BladeVals = field(
        default_factory=lambda: BladeVals(
        val=np.array([]), range=np.array([]), saturation=np.array([])
        ))
    C : BladeVals = field(
        default_factory=lambda: BladeVals(
        val=np.array([]), range=np.array([]), saturation=np.array([])
        ))
    D : BladeVals = field(
        default_factory=lambda: BladeVals(
        val=np.array([]), range=np.array([]), saturation=np.array([])
        ))

    @classmethod
    def from_hdf5_group(cls, h5group) -> "BladeRawData":
        """Create a BladeRawData instance from an HDF5 group."""
        blades = ['A', 'B', 'C', 'D']
        for blade in blades:
            if blade not in h5group:
                raise ValueError(
                    " ERROR while reading 'Blade Raw Data' from HDF5 file:\n"
                    f" Missing '{blade}' dataset in HDF5 group."
                )

        return cls(
            A = BladeVals.from_hdf5_group(h5group["A"]),
            B = BladeVals.from_hdf5_group(h5group["B"]),
            C = BladeVals.from_hdf5_group(h5group["C"]),
            D = BladeVals.from_hdf5_group(h5group["D"])
        )


@dataclass
class SweepData:
    """Container for sweep data and associated metadata.
    
    prm   : parameters of the sweep
    bpm   : BPM registered orbit at the time of the sweep
    blades: BladeRawData
    """
    prm    : Optional[dict]         = None
    bpm    : Optional[BPMData]      = None
    blades : Optional[BladeRawData] = None

    @classmethod
    def from_hdf5_group(cls, h5group) -> "SweepData":
        """Create a SweepData instance from an HDF5 group."""
        try:
            # Sweep metadata.
            prms = {k : v for k, v in h5group.attrs.items()}

            # BPM dataset.
            bpm = BPMData.from_hdf5_group(h5group['bpm_data'])

            # Read raw data.
            raw = BladeRawData.from_hdf5_group(h5group["raw_data"])

            # Instantiate SweepData with parameters, BPM data, and raw data.
            return cls(prm=prms, bpm=bpm, blades=raw)
        except Exception as err:
            raise ValueError(
                "### ERROR while reading 'Sweep Data' from HDF5 file:\n"
                f" {err}"
            )


@dataclass
class BladeMap:
    """Container for blade current data and associated metadata.

    Attributes:
        blades: Blades (measured currents)
        coords: Horizontal and vertical positions which define the grid of
                measurements.
    """
    blades: Blades
    coords: Positions


@dataclass
class CentralSweeps:
    """Container for central sweep data.
    
    index: variable coordinate
    fix: fixed coordinate
    blades: Blades
    """
    index  : np.ndarray
    fix    : np.ndarray
    blades : Blades


@dataclass
class Scales:
    """Container for scaling factors.
    
    q: quadratic coefficient
    k: linear coefficient
    d: constant offset
    s: standard deviation of the respective coefficient
    """
    qx  : float
    sqx : float
    kx  : float
    skx : float
    dx  : float
    sdx : float
    qy  : float
    sqy : float
    ky  : float
    sky : float
    dy  : float
    sdy : float


@dataclass
class SupressionMatrix:
    """Container for suppression matrix data.
    
    matrix: 4x4 numpy array representing the suppression matrix
    """
    standard : np.ndarray = field(init=False)
    stddev   : np.ndarray = field(init=False)
    calc     : Optional[np.ndarray] = None
    optim    : Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Extract standard matrix and std dev matrix from Config."""
        self.standard, self.stddev = Config.standard_suppression_matrix()


@dataclass
class DataAnalysis:
    """Container for all data and analysis results.
    
    nom_pos         : nominal positions
    blademap        : blade current data and positions
    bpm_pos         : BPM measured positions

    centralsweep_h  : Central sweep blade currents with errors and
                coordinates in horizontal direction
    centralsweep_v  : Central sweep blade currents with errors and
                coordinates in vertical direction

    xbpm_pos_raw_cr : XBPM positions from raw data (cross-blade)
    xbpm_pos_raw_pw : XBPM positions from raw data (pairwise)
    xbpm_pos_scl_cr : XBPM positions from scaled data (cross-blade)
    xbpm_pos_scl_pw : XBPM positions from scaled data (pairwise)

    scale_raw_pair  : Scaling factors for raw pairwise calculation
    scale_raw_cross : Scaling factors for raw cross-blade calculation
    scale_adj_pair  : Scaling factors for adjusted pairwise calculation
    scale_adj_cross : Scaling factors for adjusted cross-blade calculation
    """
    nom_pos         : Optional[Positions] = None
    blademap        : Optional[BladeMap] = None
    bpm_pos         : Optional[Positions] = None
    centralsweep_h  : Optional[CentralSweeps] = None
    centralsweep_v  : Optional[CentralSweeps] = None
    xbpm_pos_raw_cr : Optional[Positions] = None
    xbpm_pos_raw_pw : Optional[Positions] = None
    xbpm_pos_scl_cr : Optional[Positions] = None
    xbpm_pos_scl_pw : Optional[Positions] = None
    scale_raw_pair  : Optional[Scales] = None
    scale_raw_cross : Optional[Scales] = None
    scale_adj_pair  : Optional[Scales] = None
    scale_adj_cross : Optional[Scales] = None
    supmat_pair     : Optional[SupressionMatrix] = None
    supmat_cross    : Optional[SupressionMatrix] = None


class DataStructureBuilder:
    """Builds and enriches Prm parameters from CLI and data sources.

    Consolidates all parameter-related logic:
    - CLI argument parsing
    - Beamline identification from data
    - Grid step inference
    - Data-derived parameter enrichment
    """

    def __init__(self, prm: Optional[Prm] = None):
        """Initialize builder with optional pre-existing parameters."""
        self.prm       : Prm = prm if prm is not None else Prm()
        self.sweepdata : Optional[SweepData] = None
        self.analysis  : Optional[DataAnalysis] = None


    def add_beamline_parameters(self) -> None:
        """Add beamline-specific parameters to prm."""
        self.prm.current = self.sweepdata[0][2]["current"]

        try:
            self.prm.phaseorgap = self.sweepdata[0][1][
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

