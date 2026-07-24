"""Parameter handling and CLI parsing."""

# import sys
from dataclasses import dataclass, field
from typing import Any, List

import h5py
import numpy as np

# Import DataReader for canonical _extract_beamlines
# from xbpm_bumps.core.readers import DataReader
# from .config import Config
from .constants import ROI_SIZE_H, ROI_SIZE_V


@dataclass
class Prm:
    """Typed container for command-line and runtime parameters.

    This class implements __getitem__/__setitem__ so existing code that
    uses prm["key"] remains compatible while also providing attribute
    access (prm.key).
    """
    beamline         : str
    sr_current       : float
    bpmdist          : float

    showblademap     : bool = False
    centralsweep     : bool = False
    showbladescenter : bool = False
    xbpmpositions    : bool = False
    showbpmpositions : bool = False
    xbpmpositionsraw : bool = False
    usebpmref        : bool = False
    outputfile       : str   | None = None
    xbpmdist         : float | None = None
    workdir          : str   | None = None
    phaseorgap       : dict  | None = None
    section          : str   | None = None
    blademap         : Any   | None = None
    maxradangle      : float = 20.0
    skip             : int   = 0
    scalepolydeg     : int   = 1
    roisize          : List[int] = field(
        default_factory=lambda: [ROI_SIZE_H, ROI_SIZE_V]
        )

    def __getitem__(self, key: str):
        """Dictionary-style access (prm['key']) for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Allow setting attributes via prm['key'] = value."""
        setattr(self, key, value)

    @classmethod
    def from_hdf5_group(cls, bln_grp: h5py.Group) -> "Prm":
        """Create a Prm instance from an HDF5 group."""
        # Extract attributes from the HDF5 group.
        try:
            attrs = {key: val for key, val in bln_grp.attrs.items()}
        except Exception as err:
            raise ValueError(
                "### ERROR while reading 'Prm' from HDF5 group:\n"
                f" {err}"
            )

        # Create a Prm instance with the extracted attributes.
        return cls(**attrs)


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

        x_key = None
        y_key = None
        for ent in entries:
            if ent.startswith('x'):
                x_key = ent
            if ent.startswith('y'):
                y_key = ent

        if x_key is None or y_key is None:
            raise KeyError(
                "Neither 'x' nor 'y' fields found in the provided data."
            )

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
    to  : np.ndarray
    ti  : np.ndarray
    bi  : np.ndarray
    bo  : np.ndarray
    sto : np.ndarray
    sti : np.ndarray
    sbi : np.ndarray
    sbo : np.ndarray

    @classmethod
    def from_hdf5_group(cls, data) -> "Blades":
        """Create a Blades instance from an HDF5 group."""
        datanames = data.dtype.names

        # Check for required datasets in the HDF5 group.
        blades = ['to_mean', 'ti_mean', 'bi_mean', 'bo_mean',
                  'to_err',  'ti_err',  'bi_err',  'bo_err']
        for blade in blades:
            if blade not in datanames:
                raise ValueError(
                    " WARNING: while reading Average Blade Currents from HDF5"
                    f" file:\n Missing '{blade}' dataset in HDF5 group.")

        return cls(
            to  = data["to_mean"][:],
            ti  = data["ti_mean"][:],
            bi  = data["bi_mean"][:],
            bo  = data["bo_mean"][:],
            sto = data["to_err"][:],
            sti = data["ti_err"][:],
            sbi = data["bi_err"][:],
            sbo = data["bo_err"][:],
        )

#
# Raw data structures.
#

@dataclass
class BladeAvgData:
    """Container for averaged blade current data and associated metadata."""
    prm    : dict
    nom    : Positions
    blades : Blades

    @classmethod
    def from_hdf5_group(cls, avg_grp) -> "BladeAvgData":
        """Create a BladeAvgData instance from an HDF5 group."""
        # Extract metadata attributes.
        prm    = {key : val for key, val in avg_grp.attrs.items()}
        nom    = Positions.from_hdf5_group(avg_grp)
        blades = Blades.from_hdf5_group(avg_grp)

        return cls(prm=prm, nom=nom, blades=blades)


@dataclass
class BladeVals:
    """Container for one blade raw data and associated metadata.
    
    val        : measured currents for the blade
    range      : measurement range for the blade
    saturation : saturation levels for the blade
    """
    val        : np.ndarray
    range      : np.ndarray
    saturation : np.ndarray

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
    A : BladeVals
    B : BladeVals
    C : BladeVals
    D : BladeVals

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
    prm    : dict
    bpm    : BPMData
    blades : BladeRawData

    @classmethod
    def from_hdf5_group(cls, swp_grp) -> "SweepData":
        """Create a SweepData instance from an HDF5 group."""
        try:
            # Sweep metadata.
            prm = dict(swp_grp.attrs.items())

            # BPM dataset.
            bpm = BPMData.from_hdf5_group(swp_grp['bpm_data'])

            # Read raw data.
            bld = BladeRawData.from_hdf5_group(swp_grp["blade_data"])

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
    blade_avg  : BladeAvgData | None  = None

    @classmethod
    def from_hdf5_group(cls,
                        raw_grp  : h5py.Group,
                        beamline : str) -> "BeamlineRawData":
        """Extract raw data from the a raw_data HDF5 group."""
        kwargs = dict(metadata=dict(raw_grp.attrs.items()))

        sweeps = {}
        for key, data in raw_grp.items():
            # Sweep data.
            if key.startswith('sweep_'):
                # Extract sweep number
                num         = int(key.split('_')[1])
                sweeps[num] = SweepData.from_hdf5_group(swp_grp=data)

            # Blade averages.
            elif key == "blade_averages":
                # Blade average data is a numpy array structure, not keyed.
                kwargs["blade_avg"] = BladeAvgData.from_hdf5_group(
                    avg_grp=data
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
class SweepLine:
    """Container for central sweep data.
    
    index    : variable coordinate
    fixed    : fixed coordinate
    blades   : values of the blades along the central sweep
    fixcalc  : calculated values for the fixed coordinate
    sfixcalc : std dev of calculated fixed coordinate
    fixfit   : values of fitted affine line to fixed coordinate
    """
    index    : np.ndarray
    fixed    : np.ndarray
    blades   : Blades
    fixcalc  : np.ndarray
    sfixcalc : np.ndarray
    fixfit   : np.ndarray

    @classmethod
    def from_hdf5_group(cls,
                        h5group: h5py.Group,
                        dir: str) -> "SweepLine":
        """Create a SweepLine instance from an HDF5 group.
        
        h5group: HDF5 group containing the central sweep data.
        dir: Direction of the central sweep
            ('h' for horizontal, 'v' for vertical)
        """
        # Check the direction order.
        if dir == 'h':
            ind = 'x'
            fix = 'y'
        elif dir == 'v':
            ind = 'y'
            fix = 'x'
        else:
            raise ValueError(
                " Invalid direction. Use 'h' for horizontal"
                " or 'v' for vertical."
            )

        # Assemble the SweepLine instance.
        return cls(
            index    = h5group[f"{ind}_index"][:],
            fixed    = h5group[f"{fix}_fix"][:],
            fixcalc  = h5group[f"{fix}_calc"][:],
            sfixcalc = h5group[f"s_{fix}_calc"][:],
            fixfit   = h5group[f"{fix}_fit"][:],
            blades   = Blades.from_hdf5_group(h5group)
        )


@dataclass
class CentralSweep:
    """Container for central sweep data and associated metadata.

    Attributes:
        h: SweepLine for horizontal direction
        v: SweepLine for vertical direction
    """
    h : SweepLine
    v : SweepLine

    @classmethod
    def from_hdf5_group(cls, h5group) -> "CentralSweep":
        """Create a SweepCentral instance from an HDF5 group."""
        return cls(
            h = SweepLine.from_hdf5_group(h5group["blades_h"], dir='h'),
            v = SweepLine.from_hdf5_group(h5group["blades_v"], dir='v')
        )


@dataclass
class Scales:
    """Container for scaling factors.
    
    q: quadratic coefficient
    k: linear coefficient
    d: constant offset
    s: standard deviation of the respective coefficient
    """
    kx  : float
    skx : float
    dx  : float
    sdx : float
    ky  : float
    sky : float
    dy  : float
    sdy : float

    qx  : float = 0.0
    sqx : float = 0.0
    qy  : float = 0.0
    sqy : float = 0.0

    @classmethod
    def from_hdf5_group(cls, scl_grp) -> "Scales":
        """Create a Scales instance from an HDF5 group."""
        required_fields = [
            'qx', 'sqx', 'kx', 'skx', 'dx', 'sdx',
            'qy', 'sqy', 'ky', 'sky', 'dy', 'sdy'
        ]
        for fld in required_fields:
            if fld not in scl_grp.attrs:
                raise ValueError(
                    f" ERROR while reading Scales from HDF5 file:\n"
                    f" Missing '{fld}' attribute in HDF5 group."
                )

        return cls(
            qx  = scl_grp.attrs['qx'],
            sqx = scl_grp.attrs['sqx'],
            kx  = scl_grp.attrs['kx'],
            skx = scl_grp.attrs['skx'],
            dx  = scl_grp.attrs['dx'],
            sdx = scl_grp.attrs['sdx'],
            qy  = scl_grp.attrs['qy'],
            sqy = scl_grp.attrs['sqy'],
            ky  = scl_grp.attrs['ky'],
            sky = scl_grp.attrs['sky'],
            dy  = scl_grp.attrs['dy'],
            sdy = scl_grp.attrs['sdy']
        )


@dataclass
class AllScales:
    """Container for all scaling factors.
    
    raw_pw : Scales for raw pairwise calculation
    raw_cr : Scales for raw cross-blade calculation
    trn_pw : Scales for transformed pairwise calculation
    trn_cr : Scales for transformed cross-blade calculation
    """
    raw_pw  : Scales
    raw_cr  : Scales
    trn_pw  : Scales
    trn_cr  : Scales

    @classmethod
    def from_hdf5_group(cls, asc_grp) -> "AllScales":
        """Create an AllScales instance from an HDF5 group."""
        return cls(
            raw_pw = Scales.from_hdf5_group(asc_grp["raw"]["pair"]),
            raw_cr = Scales.from_hdf5_group(asc_grp["raw"]["cross"]),
            trn_pw = Scales.from_hdf5_group(asc_grp["transformed"]["pair"]),
            trn_cr = Scales.from_hdf5_group(asc_grp["transformed"]["cross"])
        )


@dataclass
class SupressionMatrix:
    """Container for suppression matrix data.
    
    matrix: 4x4 numpy array representing the suppression matrix
    """
    standard   : np.ndarray
    calculated : np.ndarray
    stddev     : np.ndarray | None = None
    optimized  : np.ndarray | None = None

    @classmethod
    def from_hdf5_group(cls, mat_grp) -> "SupressionMatrix":
        """Create a SupressionMatrix instance from an HDF5 group."""
        if ("standard" not in mat_grp or
            "calculated" not in mat_grp):
            raise ValueError(
                " ERROR while reading Supression Matrix from HDF5 file:\n"
                " Missing 'standard' or 'calculated' dataset in HDF5 group."
            )

        kwargs = {
            "standard"   : mat_grp["standard"][:],
            "calculated" : mat_grp["calculated"][:],
        }

        # Analysis might be incomplete.
        if "optimized" in mat_grp:
            kwargs["optimized"] = mat_grp["optimized"][:]

        if "stddev" in mat_grp:
            kwargs["stddev"] = mat_grp["stddev"][:]
    
        return cls(**kwargs)


@dataclass
class AnalyzedRawPositions:
    """Container for analyzed positions and associated metadata.

    These data were calculated, but not correct by the suppression matrix.

    x: horizontal positions
    y: vertical positions
    """
    nom : Positions
    bpm : Positions
    pws : Positions
    crs : Positions

    @classmethod
    def from_hdf5_group(cls, h5group) -> "AnalyzedRawPositions":
        """Create an AnalyzedRawPositions instance from an HDF5 group."""
        # Nominal positions.
        gr  = h5group["bpm"] 
        xn  = gr["x_nom"][:]
        yn  = gr["y_nom"][:]
        nom = Positions(x=xn, y=yn)

        # Measured BPM positions.
        xb  = gr["x_bpm"][:]
        yb  = gr["y_bpm"][:]
        bpm = Positions(x=xb, y=yb)

        # Pairwise calculated positions.
        gr  = h5group["xbpm_raw_pairwise"]
        xp  = gr["x_raw"][:]
        yp  = gr["y_raw"][:]
        pws = Positions(x=xp, y=yp)

        # Cross-blade calculated positions.
        gr  = h5group["xbpm_raw_cross"]
        xc  = gr["x_raw"][:]
        yc  = gr["y_raw"][:]
        crs = Positions(x=xc, y=yc)

        return cls(
            nom=nom,
            bpm=bpm,
            pws=pws,
            crs=crs,
            )


@dataclass
class TransformedPositions:
    """Container for analyzed positions and associated metadata.

    These data were calculated and corrected by the suppression matrix.

    x: horizontal positions
    y: vertical positions
    """
    pws : Positions
    crs : Positions

    @classmethod
    def from_hdf5_group(cls, h5group) -> "TransformedPositions":
        """Create a TransformedPositions instance from an HDF5 group."""
        # Cross-blade calculated positions.
        gr  = h5group["xbpm_transformed_pw"]
        xp  = gr["x_trn"][:]
        yp  = gr["y_trn"][:]
        pws = Positions(x=xp, y=yp)

        # Cross-blade calculated positions.
        gr  = h5group["xbpm_transformed_cr"]
        xc  = gr["x_trn"][:]
        yc  = gr["y_trn"][:]
        crs = Positions(x=xc, y=yc)

        return cls(pws=pws, crs=crs)


@dataclass
class AnalyzedPositions:
    """Container for analyzed positions and associated metadata.
    
    raw : AnalyzedRawPositions
    trn : TransformedPositions
    """
    raw : AnalyzedRawPositions
    trn : TransformedPositions | None = None

    @classmethod
    def from_hdf5_group(cls, pos_grp) -> "AnalyzedPositions":
        """Create an AnalyzedPositions instance from an HDF5 group."""
        raw  = AnalyzedRawPositions.from_hdf5_group(pos_grp)

        if ("xbpm_transformed_pw" in pos_grp and
            "xbpm_transformed_cr" in pos_grp):
            trn = TransformedPositions.from_hdf5_group(pos_grp)
        else:
            trn = None

        return cls(raw=raw, trn=trn)


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

    xbpm_pos_raw_cr : AnalyzedRawPositions
    xbpm_pos_raw_pw : AnalyzedRawPositions
    xbpm_pos_scl_cr : TransformedPositions
    xbpm_pos_scl_pw : TransformedPositions

    scale_raw_pw    : Scaling factors for raw pairwise calculation
    scale_raw_cross : Scaling factors for raw cross-blade calculation
    scale_adj_pair  : Scaling factors for adjusted pairwise calculation
    scale_adj_cross : Scaling factors for adjusted cross-blade calculation
    """
    # Beamline, description and XBPM-source distance.
    prm          : Prm
    blademap     : BladeMap          | None = None
    positions    : AnalyzedPositions | None = None
    centralsweep : CentralSweep      | None = None
    scales       : AllScales         | None = None
    supmat       : SupressionMatrix  | None = None

    @classmethod
    def from_hdf5_group(cls, anl_grp: h5py.Group) -> "DataAnalysis":
        """Create a DataAnalysis instance from an HDF5 group."""
        # Extract parameters.
        prm = Prm.from_hdf5_group(anl_grp)

        # Extract blade map.
        blademap = BladeMap.from_hdf5_group(anl_grp["blade_map"])

        # Extract other analysis data.
        positions = AnalyzedPositions.from_hdf5_group(anl_grp["positions"])

        # Central sweeps.
        centralsweep = CentralSweep.from_hdf5_group(anl_grp["central_sweeps"])

        # Extract scaling factors and suppression matrices.
        scales = AllScales.from_hdf5_group(anl_grp["scales"])

        # Extract suppression matrix.        
        supmat = SupressionMatrix.from_hdf5_group(anl_grp["matrices"])

        return cls(
            prm          = prm,
            blademap     = blademap,
            positions    = positions,
            centralsweep = centralsweep,
            scales       = scales,
            supmat       = supmat,
        )
