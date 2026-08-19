"""Parameter handling and CLI parsing."""

# import sys
from dataclasses import dataclass, field
import logging
from typing import List

import h5py
import numpy as np

from xbpm_bumps.core.config import Config
from xbpm_bumps.core.processors import XBPMProcessor, calculate_grid_stats

# Import DataReader for canonical _extract_beamlines
# from xbpm_bumps.core.readers import DataReader
# from .config import Config
from .constants import ROI_SIZE_H, ROI_SIZE_V, MAX_RAD_ANGLE


@dataclass
class Prm:
    """Typed container for command-line and runtime parameters.

    This class implements __getitem__/__setitem__ so existing code that
    uses prm["key"] remains compatible while also providing attribute
    access (prm.key).
    """
    sr_current       : float | None = None      # Synchrotron current

    # File names and analysis parameters.
    inputfile        : str   | None = None      # HDF5 input file name.
    outputfile       : str   | None = None      # HDF5 output file name. 
    phaseorgap       : dict  | None = None      # Phase/gap for the ID.
    maxradangle      : float = MAX_RAD_ANGLE    # Maximum angle of bumps in mrad.

    # What to calculate and show.
    show_bpmpositions     : bool = False
    show_blademap         : bool = False
    show_centralsweep     : bool = False
    show_bladecenter      : bool = False
    show_xbpmpositionsraw : bool = False
    show_xbpmpositions    : bool = False

    def __getitem__(self, key: str):
        """Dictionary-style access (prm['key']) for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Allow setting attributes via prm['key'] = value."""
        setattr(self, key, value)

    @classmethod
    def from_hdf5(cls, dset_grp: h5py.Group) -> "Prm":
        """Create a Prm instance from an HDF5 group."""
        # Extract attributes from the HDF5 group.
        try:
            attrs = {key: val for key, val in dset_grp.attrs.items()}
        except Exception as err:
            raise ValueError(
                "### ERROR while reading 'Prm' from HDF5 group:\n"
                f" {err}"
            )

        # Create a Prm instance with the extracted attributes.
        return cls(**attrs)


@dataclass
class ROISlice:
    sl_v: slice
    sl_h: slice
    sz_v : int = ROI_SIZE_V
    sz_h : int = ROI_SIZE_H

    # def __post_init__(self) -> None:
    #     """Update the ROI slice to the current defaults."""
    #     self.sl_v = slice(0, self.sz_v)
    #     self.sl_h = slice(0, self.sz_h)

    @classmethod
    def update(cls,
               arrayshape: tuple,
               roisize: List[int]
               ) -> "ROISlice":
        nv, nh = arrayshape

        roi_v  = min(roisize[0], nv)   # lines for arrays
        fromv  = max(0, int((nv - roi_v) / 2))
        uptov  = min(nv, fromv + roi_v)

        roi_h  = min(roisize[1], nh)   # columns for arrays
        fromh  = max(0, int((nh - roi_h) / 2))
        uptoh  = min(nh, fromh + roi_h)

        return cls(
            sz_v=roi_v,
            sz_h=roi_h,
            sl_v=slice(fromv, uptov),
            sl_h=slice(fromh, uptoh)
            )


@dataclass
class BeamlinePrm:
    """Typed container for beamline-specific parameters.

    beamline     : Beamline name
    bpmdist      : Distance between adjacent BPMs.
    xbpmdist     : Source-XBPM distance
    skip         : Number of points to skip
    scalepolydeg : Degree of polynomial for scaling fit
    roisize      : ROI size (horizontal, vertical)
    usebpmref    : Whether to use BPM or nominal positions as reference
    """
    beamline     : str   | None = None
    bpmdist      : float | None = None
    xbpmdist     : float | None = None
    skip         : int   = 0
    scalepolydeg : int   = 1
    sector       : list  | None = None
    usebpmref    : bool = False
    roi          : ROISlice = field(default_factory=ROISlice)

    @classmethod
    def from_hdf5(cls, bln_grp: h5py.Group) -> "BeamlinePrm":
        """Create a BeamlinePrm instance from an HDF5 group."""
        try:
            attrs = {key: val for key, val in bln_grp.attrs.items()}

            beamline = attrs.get("beamline", None)
            if "bpmdist" not in attrs:
                attrs["bpmdist"] = Config.XBPMDISTS.get(
                    attrs.get(beamline[:3], ""), None
                )
            if "sector" not in attrs:
                attrs["sector"] = Config.SECTOR.get(
                    attrs.get(beamline[:3], ""), None
                    )
            if "xbpmdist" not in attrs:
                attrs["xbpmdist"] = Config.XBPMDISTS.get(
                    attrs.get(beamline, ""), None
                )
            attrs["roi"] = ROISlice()
        except Exception as err:
            raise ValueError(
                "### ERROR while reading 'BeamlinePrm' from HDF5 group:\n"
                f" {err}"
            )
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
    def from_hdf5(cls, data) -> "Positions":
        """Create a Positions instance from x and y arrays."""
        if   isinstance(data, h5py.Group):
            entries = data.keys()
        elif isinstance(data, np.ndarray):
            entries = data.dtype.names
        elif isinstance(data, h5py.Dataset):
            data = data[()]   # Load the dataset into memory
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
    def from_hdf5(cls, data) -> "Blades":
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
    prm       : dict
    nom       : Positions
    nom_shape : tuple
    blades    : Blades

    @classmethod
    def from_hdf5(cls, avg_grp) -> "BladeAvgData":
        """Create a BladeAvgData instance from an HDF5 group."""
        # Extract metadata attributes.
        prm    = {key : val for key, val in avg_grp.attrs.items()}
        nom    = Positions.from_hdf5(avg_grp)
        blades = Blades.from_hdf5(avg_grp)
        nom_shape = (len(np.unique(nom.y)), len(np.unique(nom.x)))
        return cls(prm=prm, nom=nom, nom_sh=nom_shape, blades=blades)


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
    def from_hdf5(cls, bld_grp: h5py.Group, blade: str) -> "BladeVals":
        """Create a BladeVals instance from an HDF5 group."""
        required_fields = ['val', 'range', 'saturation']
        for fld in required_fields:
            if f"{blade}_{fld}" not in bld_grp.dtype.names:
                raise ValueError(
                    f" ERROR while reading BladeVals from HDF5 file:\n"
                    f" Missing '{blade}_{fld}' dataset in HDF5 group."
                )

        return cls(
            val        = bld_grp[f"{blade}_val"][:],
            range      = bld_grp[f"{blade}_range"][:],
            saturation = bld_grp[f"{blade}_saturation"][:]
        )


@dataclass
class BPMRawData:
    """Container for BPM positions (one sweep) data and its metadata."""
    descr : str
    pos   : Positions

    @classmethod
    def from_hdf5(cls, bpm_grp) -> "BPMRawData":
        """Create a BPMData instance from an HDF5 group."""
        if "Description" not in bpm_grp.attrs:
            raise ValueError(
                " ERROR while reading BPMData from HDF5 file:\n"
                " Missing 'Description' attribute in HDF5 group."
            )
        if ("x_bpm" not in bpm_grp.dtype.names or
            "y_bpm" not in bpm_grp.dtype.names):
            raise ValueError(
                " ERROR while reading BPMData from HDF5 file:\n"
                " Missing position dataset in HDF5 group."
            )
    
        descr = bpm_grp.attrs["Description"]
        pos   = Positions.from_hdf5(bpm_grp)
        return cls(descr=descr, pos=pos)


@dataclass
class BladeRawData:
    """Container for all blades' raw data and associated metadata.
    
    TO, TI, BI, BO: BladeVals for each blade
    """
    TO : BladeVals
    TI : BladeVals
    BI : BladeVals
    BO : BladeVals

    @classmethod
    def from_hdf5(cls, raw_grp: h5py.Group, beamline: str) -> "BladeRawData":
        """Create a BladeRawData instance from an HDF5 group."""
        # Use the checked beamline map to extract data.
        bmap = Config.BLADEMAP.get(beamline, None)
        return cls(
            TO = BladeVals.from_hdf5(raw_grp, bmap["TO"]),
            TI = BladeVals.from_hdf5(raw_grp, bmap["TI"]),
            BI = BladeVals.from_hdf5(raw_grp, bmap["BI"]),
            BO = BladeVals.from_hdf5(raw_grp, bmap["BO"])
        )


@dataclass
class SweepData:
    """Container for sweep data and associated metadata.
    
    prm   : parameters of the sweep
    bpm   : BPM registered orbit at the time of the sweep
    blades: BladeRawData
    """
    prm    : dict
    bpm    : BPMRawData
    blades : BladeRawData

    @classmethod
    def from_hdf5(cls, swp_grp: h5py.Group, beamline: str) -> "SweepData":
        """Create a SweepData instance from an HDF5 group."""
        try:
            # Sweep metadata.
            prm = dict(swp_grp.attrs.items())

            # BPM dataset.
            bpm = BPMRawData.from_hdf5(swp_grp['bpm_data'])

            # Read raw data.
            bld = BladeRawData.from_hdf5(swp_grp["blade_data"], beamline)

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
    
    metadata  : Metadata info for raw_data group. 
    sweeps    : List of SweepData instances
    blade_avg : BladeAvgData instance
    """
    metadata   : dict
    sweeps_bld : dict[int, BladeRawData] = field(default_factory=dict)
    sweeps_bpm : dict[int, BPMRawData]   = field(default_factory=dict)
    blade_avg  : BladeAvgData | None     = None

    @classmethod
    def from_hdf5(cls,
                  raw_grp  : h5py.Group,
                  beamline : str) -> "BeamlineRawData":
        """Extract raw data from the a raw_data HDF5 group."""
        # Group metadata.
        kwargs = dict(metadata=dict(raw_grp.attrs.items()))

        # Run through all stored data.
        sweeps = {}
        for key, data in raw_grp.items():
            # Blade averages.
            if key == "blade_averages":
                # Blade average data is a numpy array structure, not keyed.
                kwargs["blade_avg"] = BladeAvgData.from_hdf5(
                    avg_grp=data
                    )

            # Sweep data.
            elif key.startswith('sweep_'):
                # Extract sweep number
                num         = int(key.split('_')[1])
                sweeps[num] = SweepData.from_hdf5(
                    swp_grp=data, beamline=beamline
                    )

            else:
                print(f" WARNING: Unknown key '{key}'"
                      f" in beamline '{beamline}'. Skipping.")

        # Build structures for blade and BPM data separately.
        blds, bpms = {}, {}
        for key, val in sweeps.items():
            blds[key] = val.blades
            bpms[key] = val.bpm
        kwargs["sweeps_bld"] = blds
        kwargs["sweeps_bpm"] = bpms

        return cls(**kwargs)


#
# Structures for data analysis.
#

@dataclass
class RMSStatistics:
    """Computed RMS statistics between nominal and measured data."""
    # Horizontal, vertical and total differences at each site.
    h      : np.ndarray
    v      : np.ndarray
    t      : np.ndarray

    # Minimum and maximum values of the differences.
    min_h  : float
    max_h  : float
    min_v  : float
    max_v  : float

    # Mean values of the differences.
    mean_h : float
    mean_v : float
    mean_t : float

    @classmethod
    def compute(cls,
                nom_x : np.ndarray,
                nom_y : np.ndarray,
                meas_x : np.ndarray,
                meas_y : np.ndarray
                ) -> "RMSStatistics":
        rms = calculate_grid_stats(
            nom_x, nom_y, meas_x, meas_y
        )
        rms = {
            "h"      : rms['h'],
            "v"      : rms['v'],
            "t"      : rms['t'],
            "min_h"  : rms['min_h'],
            "max_h"  : rms['max_h'],
            "min_v"  : rms['min_v'],
            "max_v"  : rms['max_v'],
            "mean_h" : rms['mean_h'],
            "mean_v" : rms['mean_v'],
            "mean_t" : rms['mean_t'],
        }
        return cls(**rms)

@dataclass
class RMSGridStatistics:
    """Statistics calculated at a given ROI."""
    all      : RMSStatistics             # Full grid statistics.
    roi      : RMSStatistics             # ROI statistics.
    roislice : ROISlice | None = None    # ROI bounds.

    @classmethod
    def compute(cls,
                nom_x    : np.ndarray,
                nom_y    : np.ndarray,
                meas_x   : np.ndarray,
                meas_y   : np.ndarray,
                roislice : ROISlice
                ) -> "RMSGridStatistics":
        """Calculate RMS statistics from position differences in ROI."""
        rms_all = RMSStatistics.compute(
            nom_x,  nom_y,
            meas_x, meas_y
            )

        # Statistics at ROI.
        sl_v, sl_h = roislice.sl_v, roislice.sl_h
        nom_roi_x  = nom_x[sl_v, sl_h]
        nom_roi_y  = nom_y[sl_v, sl_h]
        meas_roi_x = meas_x[sl_v, sl_h]
        meas_roi_y = meas_y[sl_v, sl_h]
        rms_roi = RMSStatistics.compute(
            nom_roi_x,  nom_roi_y,
            meas_roi_x, meas_roi_y
            )

        return cls(
            all=rms_all,
            roi=rms_roi,
            roislice=roislice
        )


@dataclass
class BPMAnalysis:
    """Container for BPM positions and associated metadata.

    Attributes:
        prm : BPM parameters.
        bpm : BPM positions (x, y) calculated from measurements.
        nom : Nominal positions at XBPM site extrapolated from bump-angles.
        rms : RMS values of the differences between bpm and nom positions.
        roi_diffs : estimated standard deviations of the differences between bpm
            and nom positions.
    """
    prm      : BeamlinePrm
    pos_meas : Positions
    pos_nom  : Positions
    rms_diff : RMSGridStatistics

    @classmethod
    def compute(cls, bl_data: "BeamlineData") -> "BPMAnalysis":
        """Create a BPMAnalysis instance from calculated BPM analysis data.
        
        Args:
            bl_data: BeamlineData instance containing the measured BPM
                positions and metadata.

        Returns:
            BPMAnalysis instance with calculated BPM positions and metadata.
        """
        # Link to beamline parameters.
        prm = bl_data.prm

        # Instantiate BPMProcessor to calculate BPM positions.
        from .processors import BPMProcessor as BPMP
        bpm_proc = BPMP(
            rawdata=bl_data.raw_data,
            prm_bml=bl_data.prm,
        )

        rms_diff = RMSGridStatistics.compute(
            bpm_proc.nom_x, bpm_proc.nom_y,
            bpm_proc.meas_x, bpm_proc.meas_y,
            prm.roi
            )

        return cls(
            prm=prm,
            pos_nom=bpm_proc.nominal,
            pos_meas=bpm_proc.measured,
            rms_diff=rms_diff,
            )


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
    def from_hdf5(cls, h5group) -> "BladeMap":
        """Create a BladeMap instance from an HDF5 group."""
        prm    = {key: val for key, val in h5group.attrs.items()}
        blades = Blades.from_hdf5(h5group)
        coords = Positions.from_hdf5(h5group)

        return cls(prm=prm, blades=blades, coords=coords)


@dataclass
class CentralSweepLine:
    """Container for central sweep data.
    
    A sweep is performed along a line (horizontal or vertical) through the center of the blade map. It is supposed that there is no variation in the other direction, but distortions of measurements create an undesired slope, to be evaluated. The sweep data is then used to analyze the behavior of the blades along the sweep line, so the variation in the fixed coordinate is captured.

    blades       : values of the blades along the central sweep
    index        : variable coordinate values along the central sweep
                (x for horizontal sweep, y for vertical sweep)
    fixed        : fixed coordinate values along the central sweep
                (h: x ~ 0 for vertical sweep, v: y ~ 0 for horizontal sweep)
    calc_fix     : calculated values for the fixed coordinate
    calc_fix_err : std dev of calculated fixed coordinate
    fit_fix : values of fitted affine line to fixed coordinate
    """
    blades      : Blades
    index       : np.ndarray
    fixed       : np.ndarray
    calc_pos    : np.ndarray
    fit_pos     : np.ndarray
    fit_pos_err : np.ndarray

    @classmethod
    def from_hdf5(cls,
                  sln_grp: h5py.Group,
                  dir: str) -> "CentralSweepLine":
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
                " Invalid direction."
                " Use 'h' for horizontal or 'v' for vertical."
            )

        # Assemble the SweepLine instance.
        return cls(
            blades=Blades.from_hdf5(sln_grp),
            index=sln_grp[f"{ind}_index"][:],
            fixed=sln_grp[f"{fix}_fix"][:],
            calc_pos=sln_grp[f"{fix}_calc"][:],
            fit_pos=sln_grp[f"{fix}_fit"][:],
            fit_pos_err=sln_grp[f"s_{fix}_fit"][:],
        )


@dataclass
class CentralSweeps:
    """Container for central sweep data and associated metadata.

    Attributes:
        h: SweepLine for horizontal direction
        v: SweepLine for vertical direction
    """
    h : CentralSweepLine | None = None
    v : CentralSweepLine | None = None

    @classmethod
    def from_hdf5(cls, swp_grp) -> "CentralSweeps":
        """Create a CentralSweep instance from an HDF5 group."""
        return cls(
            h = CentralSweepLine.from_hdf5(swp_grp["blades_h"], dir='h'),
            v = CentralSweepLine.from_hdf5(swp_grp["blades_v"], dir='v')
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
    def from_hdf5(cls, scl_grp) -> "Scales":
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
    def from_hdf5(cls, asc_grp) -> "AllScales":
        """Create an AllScales instance from an HDF5 group."""
        return cls(
            raw_pw = Scales.from_hdf5(asc_grp["raw"]["pair"]),
            raw_cr = Scales.from_hdf5(asc_grp["raw"]["cross"]),
            trn_pw = Scales.from_hdf5(asc_grp["transformed"]["pair"]),
            trn_cr = Scales.from_hdf5(asc_grp["transformed"]["cross"])
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
    def from_hdf5(cls, mat_grp) -> "SupressionMatrix":
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
    def from_hdf5(cls, h5group) -> "AnalyzedRawPositions":
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
    def from_hdf5(cls, h5group) -> "TransformedPositions":
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
    def from_hdf5(cls, pos_grp) -> "AnalyzedPositions":
        """Create an AnalyzedPositions instance from an HDF5 group."""
        raw  = AnalyzedRawPositions.from_hdf5(pos_grp)

        if ("xbpm_transformed_pw" in pos_grp and
            "xbpm_transformed_cr" in pos_grp):
            trn = TransformedPositions.from_hdf5(pos_grp)
        else:
            trn = None

        return cls(raw=raw, trn=trn)


@dataclass
class DataAnalysis:
    """Container for all data and analysis results.
 
    Classmethods may read analysis from HDF5 and compute analysis.

    prm          : beamline parameters
    bpm          : BPM calculated positions at XBPM site
    blademap     : blade map of positions
    positions    : Analyzed positions (raw and transformed)   
    centralsweep : Central sweep blade currents with errors
    scales       : Scaling factors
    supmat       : Suppression matrices
    """
    # Beamline, description and XBPM-source distance.
    prm           : BeamlinePrm       | None = None
    bpm           : BPMAnalysis       | None = None
    blademap      : BladeMap          | None = None
    positions     : AnalyzedPositions | None = None
    centralsweeps : CentralSweeps     | None = None
    scales        : AllScales         | None = None
    supmat        : SupressionMatrix  | None = None

    @classmethod
    def from_hdf5(cls, anl_grp: h5py.Group) -> "DataAnalysis":
        """Create a DataAnalysis instance from an HDF5 group."""
        # Extract parameters.
        prm = BeamlinePrm.from_hdf5(anl_grp)

        # Extract BPM analysis data.
        bpm = BPMAnalysis.from_hdf5(anl_grp["bpm_analysis"])

        # Extract blade map.
        blademap = BladeMap.from_hdf5(anl_grp["blade_map"])

        # Extract other analysis data.
        positions = AnalyzedPositions.from_hdf5(anl_grp["positions"])

        # Central sweeps.
        centralsweeps = CentralSweeps.from_hdf5(anl_grp["central_sweeps"])

        # Extract scaling factors and suppression matrices.
        scales = AllScales.from_hdf5(anl_grp["scales"])

        # Extract suppression matrix.        
        supmat = SupressionMatrix.from_hdf5(anl_grp["matrices"])

        return cls(
            prm           = prm,
            bpm           = bpm,
            blademap      = blademap,
            positions     = positions,
            centralsweeps = centralsweeps,
            scales        = scales,
            supmat        = supmat,
        )


@dataclass
class BeamlineData:
    """Encapsulates all data for a single beamline extracted from HDF5.
    
    A structure is created to contain the raw data measured (BeamlineRawData) and, if present, previous analysis results (DataAnalysis) stored in the HDF5 file. Metadata is stored in a parameter dictionary.
    """
    prm      : BeamlinePrm
    raw_data : BeamlineRawData
    analysis : DataAnalysis  = field(default=None)

    @classmethod
    def from_hdf5(cls, bd_grp: h5py.Group,) -> "BeamlineData":
        """Extract the beamline data from an HDF5 group."""
        # Data storage.
        kwargs = {}

        # Beamline parameters.
        kwargs["prm"] =  BeamlinePrm.from_hdf5(bd_grp)
        beamline      = kwargs["prm"].beamline

        # Raw data.
        kwargs["raw_data"] = BeamlineRawData.from_hdf5(
            bd_grp["raw_data"], beamline
            )

        # Analysis data may be not present when data are imported.
        try:
            kwargs["analysis"] = DataAnalysis.from_hdf5(bd_grp["analysis"])
        except Exception as warn:
            logging.warning(
                "### WARNING, while reading 'Data Analysis' from HDF5 file:"
                f"\n {warn}"
            )
        return cls(**kwargs)
