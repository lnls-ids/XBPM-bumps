"""Parameter handling and CLI parsing."""

# import sys
from dataclasses import dataclass, field
from typing import Optional, Any, List

import h5py
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


#
# Generic data structures.
#

@dataclass
class Positions:
    """Container for calculated XBPM positions."""
    x : np.ndarray
    y : np.ndarray

    @classmethod
    def from_hdf5_group(cls, data) -> "Positions":
        """Create a Positions instance from x and y arrays."""
        if   isinstance(data, h5py.Group):
            entries = data.keys()
        elif isinstance(data, np.ndarray):
            entries = data.dtype.names
        else:
            raise TypeError(
                "Data must be an h5py.Group or a structured numpy array."
            )

        for ent in entries:
            if ent.startswith('x'):
                x_key = ent
            if ent.startswith('y'):
                y_key = ent

        try:
            return cls(x=data[x_key][:], y=data[y_key][:])
        except (KeyError, ValueError) as err:
            raise KeyError(f"Neither pair of fields found:\n {err}")


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
    def from_hdf5_group(cls, data) -> "Blades":
        """Create a Blades instance from an HDF5 group."""
        datanames = data.dtype.names

        # Check for required datasets in the HDF5 group.
        blades = ['to_mean', 'ti_mean', 'bi_mean', 'bo_mean']
        for blade in blades:
            if blade not in datanames:
                raise ValueError(
                    " WARNING: while reading Average Blade Currents from HDF5"
                    f" file:\n Missing '{blade}' dataset in HDF5 group.")

        # Check for optional standard deviation datasets in the HDF5 group.
        # Some data sets may not have standard deviation information,
        # as blade_map.
        return cls(
            to  = data["to_mean"][:],
            ti  = data["ti_mean"][:],
            bi  = data["bi_mean"][:],
            bo  = data["bo_mean"][:],
            sto = data["to_err"][:] if 'to_err' in datanames else None,
            sti = data["ti_err"][:] if 'ti_err' in datanames else None,
            sbi = data["bi_err"][:] if 'bi_err' in datanames else None,
            sbo = data["bo_err"][:] if 'bo_err' in datanames else None,
        )

#
# Raw data structures.
#

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
        prm    = {key : val for key, val in h5group.attrs.items()}
        nom    = Positions.from_hdf5_group(h5group)
        blades = Blades.from_hdf5_group(h5group)

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
    def from_hdf5_group(cls,
                        h5group: h5py.Group,
                        blade: str) -> "BladeVals":
        """Create a BladeVals instance from an HDF5 group."""
        required_fields = ['val', 'range', 'saturation']
        for fld in required_fields:
            if f"{blade}_{fld}" not in h5group.dtype.names:
                raise ValueError(
                    f" ERROR while reading BladeVals from HDF5 file:\n"
                    f" Missing '{blade}_{fld}' dataset in HDF5 group."
                )

        return cls(
            val        = h5group[f"{blade}_val"][:],
            range      = h5group[f"{blade}_range"][:],
            saturation = h5group[f"{blade}_saturation"][:]
        )


@dataclass
class BPMData:
    descr : str
    pos   : Positions

    @classmethod
    def from_hdf5_group(cls, h5group) -> "BPMData":
        """Create a BPMData instance from an HDF5 group."""
        if "Description" not in h5group.attrs:
            raise ValueError(
                " ERROR while reading BPMData from HDF5 file:\n"
                " Missing 'Description' attribute in HDF5 group."
            )
        if ("x_bpm" not in h5group.dtype.names or
            "y_bpm" not in h5group.dtype.names):
            raise ValueError(
                " ERROR while reading BPMData from HDF5 file:\n"
                " Missing position dataset in HDF5 group."
            )
    
        descr = h5group.attrs["Description"]
        pos  = Positions.from_hdf5_group(h5group)
        return cls(descr=descr, pos=pos)


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
        return cls(
            A = BladeVals.from_hdf5_group(h5group, "A"),
            B = BladeVals.from_hdf5_group(h5group, "B"),
            C = BladeVals.from_hdf5_group(h5group, "C"),
            D = BladeVals.from_hdf5_group(h5group, "D")
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
            prm = dict(h5group.attrs.items())

            # BPM dataset.
            bpm = BPMData.from_hdf5_group(h5group['bpm_data'])

            # Read raw data.
            bld = BladeRawData.from_hdf5_group(h5group["blade_data"])

            # Instantiate SweepData with parameters, BPM data, and raw data.
            return cls(prm=prm, bpm=bpm, blades=bld)
        except Exception as err:
            raise ValueError(
                "### ERROR while reading 'Sweep Data' from HDF5 file:\n"
                f" {err}"
            )


@dataclass
class BeamlineRawData:
    """Container for all sweep data and associated metadata for a beamline.
    
    sweeps: List of SweepData instances
    blade_avg: BladeAvgData instance
    """
    metadata   : dict
    sweeps     : dict[int, SweepData] = field(default_factory=dict)
    blade_avg  : BladeAvgData         = field(default=None)

    @classmethod
    def from_hdf5_group(cls,
                        h5group  : h5py.Group,
                        beamline : str) -> "BeamlineRawData":
        """Extract raw data from the a raw_data HDF5 group."""
        kwargs = dict(metadata=dict(h5group.attrs.items()))

        sweeps = {}
        for key, data in h5group.items():
            # Sweep data.
            if key.startswith('sweep_'):
                # Extract sweep number
                num         = int(key.split('_')[1])
                sweeps[num] = SweepData.from_hdf5_group(h5group=data)

            # Blade averages.
            elif key == "blade_averages":
                # Blade average data is a numpy array structure, not keyed.
                kwargs["blade_avg"] = BladeAvgData.from_hdf5_group(
                    h5group=data
                    )

            else:
                print(f" WARNING: Unknown key '{key}'"
                      f" in beamline '{beamline}'. Skipping.")

        kwargs["sweeps"] = sweeps
        return cls(**kwargs)


#
# Structures for data analysis.
#

@dataclass
class BladeMap:
    """Container for blade current data and associated metadata.

    Attributes:
        blades: Blades (measured currents)
        coords: Horizontal and vertical positions which define the grid of
                measurements.
    """
    prm    : dict
    blades : Blades
    coords : Positions

    @classmethod
    def from_hdf5_group(cls, h5group) -> "BladeMap":
        """Create a BladeMap instance from an HDF5 group."""
        prm    = {key: val for key, val in h5group.attrs.items()}
        blades = Blades.from_hdf5_group(h5group)
        coords = Positions.from_hdf5_group(h5group)

        return cls(prm=prm, blades=blades, coords=coords)


@dataclass
class CentralSweeps:
    """Container for central sweep data.
    
    index: variable coordinate
    fix: fixed coordinate
    blades: Blades
    """
    index    : np.ndarray = field(default_factory=lambda: np.array([]))
    fixed    : np.ndarray = field(default_factory=lambda: np.array([]))
    fixcalc  : np.ndarray = field(default_factory=lambda: np.array([]))
    sfixcalc : np.ndarray = field(default_factory=lambda: np.array([]))
    fixfit   : np.ndarray = field(default_factory=lambda: np.array([]))
    blades   : Blades     = field(default_factory=lambda: Blades(
        to=np.array([]), ti=np.array([]), bi=np.array([]), bo=np.array([]),
        sto=np.array([]), sti=np.array([]), sbi=np.array([]), sbo=np.array([])
    ))

    @classmethod
    def from_hdf5_group(cls,
                        h5group: h5py.Group,
                        direction: str) -> "CentralSweeps":
        """Create a CentralSweeps instance from an HDF5 group."""

        if direction == 'h':
            ind = 'x'
            fix = 'y'
        elif direction == 'v':
            ind = 'y'
            fix = 'x'

        return cls(
            index    = h5group[f"{ind}_index"][:],
            fixed    = h5group[f"{fix}_fix"][:],
            fixcalc  = h5group[f"{fix}_calc"][:],
            sfixcalc = h5group[f"s_{fix}_calc"][:],
            fixfit   = h5group[f"{fix}_fit"][:],
            blades   = Blades.from_hdf5_group(h5group)
        )


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
    standard   : np.ndarray
    stddev     : np.ndarray
    calculated : Optional[np.ndarray] = None
    optimized  : Optional[np.ndarray] = None

    @classmethod
    def from_hdf5_group(cls, h5group) -> "SupressionMatrix":
        """Create a SupressionMatrix instance from an HDF5 group."""
        if ("matrices"   not in h5group or
            "optimized"  not in h5group["matrices"]):
            raise ValueError(
                " ERROR while reading Suppression Matrix from HDF5 file:\n"
                " Missing 'calculated' or 'optimized' dataset in 'matrices' group."
            )

        kwargs = {
            "standard"   : h5group["matrices"]["standard"][:],
            "calculated" : h5group["matrices"]["calculated"][:],
        }

        # Analysis might be incomplete.
        if "optimized" in h5group["matrices"]:
            kwargs["optimized"] = h5group["matrices"]["optimized"][:]

        if "stddev" in h5group["matrices"]:
            kwargs["stddev"] = h5group["matrices"]["stddev"][:]
    
        return cls(**kwargs)


@dataclass
class AnalyzedRawPositions:
    """Container for analyzed positions and associated metadata.
    
    x: horizontal positions
    y: vertical positions
    """
    nom  : Optional[Positions] = None
    calc : Optional[Positions] = None

    @classmethod
    def from_hdf5_group(cls, h5group) -> "AnalyzedRawPositions":
        """Create an AnalyzedRawPositions instance from an HDF5 group."""
        xn  = (Positions.from_hdf5_group(h5group["x_nom"])
               if "x_nom" in h5group else None)
        yn  = (Positions.from_hdf5_group(h5group["y_nom"])
               if "y_nom" in h5group else None)
        nom = Positions(x=xn, y=yn) if xn and yn else None

        xr  = (Positions.from_hdf5_group(h5group["x_raw"])
               if "x_raw" in h5group else None)
        yr  = (Positions.from_hdf5_group(h5group["y_raw"])
               if "y_raw" in h5group else None)
        calc = Positions(x=xr, y=yr) if xr and yr else None

        return cls(nom=nom, calc=calc)


@dataclass
class AnalyzedScaledPositions:
    """Container for analyzed positions and associated metadata.
    
    x: horizontal positions
    y: vertical positions
    """
    nom  : Optional[Positions] = None
    calc : Optional[Positions] = None

    @classmethod
    def from_hdf5_group(cls, h5group) -> "AnalyzedScaledPositions":
        """Create an AnalyzedScaledPositions instance from an HDF5 group."""
        xn = (Positions.from_hdf5_group(h5group["x_nom"])
               if "x_nom" in h5group else None)
        yn = (Positions.from_hdf5_group(h5group["y_nom"])
               if "y_nom" in h5group else None)
        nom = Positions(x=xn, y=yn) if xn and yn else None

        xs = (Positions.from_hdf5_group(h5group["x_scaled"])
               if "x_scaled" in h5group else None)
        ys = (Positions.from_hdf5_group(h5group["y_scaled"])
               if "y_scaled" in h5group else None)
        calc = Positions(x=xs, y=ys) if xs and ys else None

        return cls(nom=nom, calc=calc)


@dataclass
class AnalyzedPositions:
    """Container for analyzed positions and associated metadata.
    
    raw   : AnalyzedRawPositions
    scaled: AnalyzedScaledPositions
    """
    raw    : Optional[AnalyzedRawPositions]    = None
    scaled : Optional[AnalyzedScaledPositions] = None


@dataclass
class DataAnalysis:
    """Container for all data and analysis results.
    
    Instantiation depends on HDF5 group defining the analysis data from a specific beamline.

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
    # Beamline, description and XBPM-source distance.
    prm             : Optional[Prm]               = None

    blademap        : Optional[BladeMap]          = None

    positions       : Optional[AnalyzedPositions] = None

    nom_pos         : Optional[Positions]        = None
    bpm_pos         : Optional[Positions]        = None
    centralsweep_h  : Optional[CentralSweeps]    = None
    centralsweep_v  : Optional[CentralSweeps]    = None
    xbpm_pos_raw_cr : Optional[Positions]        = None
    xbpm_pos_raw_pw : Optional[Positions]        = None
    xbpm_pos_scl_cr : Optional[Positions]        = None
    xbpm_pos_scl_pw : Optional[Positions]        = None
    scale_raw_pair  : Optional[Scales]           = None
    scale_raw_cross : Optional[Scales]           = None
    scale_adj_pair  : Optional[Scales]           = None
    scale_adj_cross : Optional[Scales]           = None
    supmat_pair     : Optional[SupressionMatrix] = None
    supmat_cross    : Optional[SupressionMatrix] = None

    @classmethod
    def from_hdf5_group(cls,
                        h5group: h5py.Group,
                        beamline: str) -> "DataAnalysis":
        """Create a DataAnalysis instance from an HDF5 group."""
        # Extract parameters.
        prm = Prm(**{key: val for key, val in h5group.attrs.items()})

        # Extract blade map.
        blademap = BladeMap(
            blades = Blades.from_hdf5_group(h5group["blademap"]["blades"]),
            coords = Positions.from_hdf5_group(h5group["blademap"]["coords"])
        )

        # Extract other analysis data.
        nom_pos         = Positions.from_hdf5_group(h5group["nom_pos"])
        bpm_pos         = Positions.from_hdf5_group(h5group["bpm_pos"])
        centralsweep_h  = CentralSweeps(
            index=h5group["centralsweep_h"]["index"][:],
            fixed=h5group["centralsweep_h"]["fixed"][:],
            blades=Blades.from_hdf5_group(
                h5group["centralsweep_h"]["blades"]
                ))
        centralsweep_v  = CentralSweeps(
            index=h5group["centralsweep_v"]["index"][:],
            fixed=h5group["centralsweep_v"]["fixed"][:],
            blades=Blades.from_hdf5_group(
                h5group["centralsweep_v"]["blades"]
                ))

        xbpm_pos_raw_cr = Positions.from_hdf5_group(h5group["xbpm_pos_raw_cr"])
        xbpm_pos_raw_pw = Positions.from_hdf5_group(h5group["xbpm_pos_raw_pw"])
        xbpm_pos_scl_cr = Positions.from_hdf5_group(h5group["xbpm_pos_scl_cr"])
        xbpm_pos_scl_pw = Positions.from_hdf5_group(h5group["xbpm_pos_scl_pw"])

        # Extract scaling factors and suppression matrices.
        scale_raw_pair  = Scales(**{
            key: val
            for key, val in h5group["scale_raw_pair"].attrs.items()
            })
        scale_raw_cross = Scales(**{
            key: val for key, val in h5group["scale_raw_cross"].attrs.items()
            })
        scale_adj_pair  = Scales(**{
            key: val for key, val in h5group["scale_adj_pair"].attrs.items()
            })
        scale_adj_cross = Scales(**{
            key: val for key, val in h5group["scale_adj_cross"].attrs.items()
            })
        
        supmat_pair     = SupressionMatrix.from_hdf5_group(
            h5group["supmat_pair"]
            )
        supmat_cross    = SupressionMatrix.from_hdf5_group(
            h5group["supmat_cross"]
            )

        return cls(
            prm             = prm,
            blademap        = blademap,
            nom_pos         = nom_pos,
            bpm_pos         = bpm_pos,
            centralsweep_h  = centralsweep_h,
            centralsweep_v  = centralsweep_v,
            xbpm_pos_raw_cr = xbpm_pos_raw_cr,
            xbpm_pos_raw_pw = xbpm_pos_raw_pw,
            xbpm_pos_scl_cr = xbpm_pos_scl_cr,
            xbpm_pos_scl_pw = xbpm_pos_scl_pw,
            scale_raw_pair  = scale_raw_pair,
            scale_raw_cross = scale_raw_cross,
            scale_adj_pair  = scale_adj_pair,
            scale_adj_cross = scale_adj_cross,
            supmat_pair     = supmat_pair,
            supmat_cross    = supmat_cross
        )


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

