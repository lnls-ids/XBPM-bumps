"""Parameter handling and CLI parsing."""

from io import BytesIO
from typing import ClassVar, Optional, Iterator, List
# from curses import raw
from dataclasses import dataclass, field, fields
from datetime import datetime
import logging
from matplotlib.figure import Figure
# from typing import List, Optional

import h5py
import numpy as np
import xarray as xr

# from xbpm_bumps.core.config import Config
from .config import Config
from .constants import ROI_SIZE_H, ROI_SIZE_V, MAX_RAD_ANGLE, FIGDPI


@dataclass
class GenPrm:
    """Typed container for command-line and runtime parameters.

    This class implements __getitem__/__setitem__ so existing code that
    uses prm["key"] remains compatible while also providing attribute
    access (prm.key).
    """
    beamline              : str   | None = None     # Blade line identifier
    sr_current            : float | None = None     # Synchrotron current

    # File names and analysis parameters.
    inputfile             : str   | None = None     # HDF5 input file name.
    outputfile            : str   | None = None     # HDF5 output file prefix. 
    phaseorgap            : dict  | None = None     # Phase/gap for the IDs.
    maxradangle           : float = MAX_RAD_ANGLE   # Max. bump angle (mrad).
    created               : str   | None = None     # Creation timestamp.
    description           : str   | None = None     # Study description.
    version               : str   | None = None     # Version of the study.
    pv_meter              : str   | None = None     # PV of the XBPM.

    # What to calculate and show.
    show_bpmpositions     : bool = False
    show_blademap         : bool = False
    show_centralsweep     : bool = False
    show_xbpmpositions    : bool = False
    slice_by_roi          : bool = False
    bmap_global_scale     : bool = False
    # show_bladecenter      : bool = False
    # show_xbpmpositionsraw : bool = False

    def __getitem__(self, key: str):
        """Dictionary-style access (prm['key']) for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Allow setting attributes via prm['key'] = value."""
        setattr(self, key, value)

    @classmethod
    def from_hdf5(cls,
                  dset_grp: "h5py.Group",
                  inputfile: str
                  ) -> "GenPrm":
        """Create a Prm instance from an HDF5 group."""
        # Extract attributes from the HDF5 group.
        try:
            prms = {}
            phaseorgap = {}
            for key, val in dset_grp.attrs.items():
                if key.startswith("Gap/Ph"):
                    phaseorgap[key] = float(val)
                else:
                    prms[key] = val
            prms["phaseorgap"] = phaseorgap
            prms["inputfile"] = inputfile
        except Exception as err:
            raise ValueError(
                "### ERROR while reading 'Prm' from HDF5 group:\n"
                f" {err}"
            )

        # Define the output file name based on the input file.
        prms["outputfile"] = cls._set_outputfile_name(inputfile)

        # Create a Prm instance with the extracted attributes.
        return cls(**prms)

    @staticmethod
    def _set_outputfile_name(inputfile: str) -> str:
        """Set the output file prefix for the analysis."""
        return inputfile.rsplit('.', 1)[0]

    def to_hdf5(self,
                prm_grp : "h5py.Group",
                ) -> None:
        """Write a Prm instance to an HDF5 group.
        
        Args:
            self  : The GenPrm instance to write.
            prm_grp : The HDF5 group where the attributes will be stored.
        """
        try:
            for fld in fields(self):
                val = getattr(self, fld.name)
                if val is None:
                    continue

                # Parse phaseorgap values to storing in HDF5.
                if fld.name == "phaseorgap":
                    prm_grp.attrs.update(val)
                    continue

                # Store the attribute in the HDF5 group.
                prm_grp.attrs[fld.name] = val
        except Exception as err:
            print(
                f"### ERROR while writing 'Prm' to {prm_grp.name}"
                f" (HDF5 group):\n {err}"
            )


@dataclass
class ROISlice:
    slice_v : slice
    slice_h : slice
    size_v  : int = ROI_SIZE_V
    size_h  : int = ROI_SIZE_H

    @classmethod
    def update(cls,
               arrayshape: tuple,
               roisize: List[int]
               ) -> "ROISlice":
        """Update the ROI slice based on the array shape and ROI size.

        Args:
            arrayshape : Shape of the array (vertical, horizontal).
            roisize    : Desired ROI size [vertical, horizontal].

        Returns:
            ROISlice   : Updated ROI slice.
        """
        nv, nh = arrayshape

        roi_v  = min(roisize[0], nv)   # lines for arrays
        fromv  = max(0, int((nv - roi_v) / 2))
        uptov  = min(nv, fromv + roi_v)

        roi_h  = min(roisize[1], nh)   # columns for arrays
        fromh  = max(0, int((nh - roi_h) / 2))
        uptoh  = min(nh, fromh + roi_h)

        return cls(
            size_v=roi_v,
            size_h=roi_h,
            slice_v=slice(fromv, uptov),
            slice_h=slice(fromh, uptoh)
            )


@dataclass
class BeamlinePrm:
    """Typed container for beamline-specific parameters.

    beamline     : Beamline name
    description  : Description of the beamline.
    nbumps       : Number of bumps.
    bpmdist      : Distance between adjacent BPMs.
    xbpmdist     : Source-XBPM distance
    skip         : Number of points to skip
    scalepolydeg : Degree of polynomial for scaling fit
    roisize      : ROI size (horizontal, vertical)
    usebpmref    : Whether to use BPM or nominal positions as reference
    """
    beamline     : str             | None = None
    description  : str             | None = None
    nbumps       : int             = 1
    bpmdist      : float           | None = None
    xbpmdist     : float           | None = None
    skip         : int             = 0
    scalepolydeg : int             = 1
    sector       : list            | None = None
    usebpmref    : bool            = False
    grid_shape   : tuple[int, int] | None = None
    roislice     : ROISlice        = None
    updated      : str             | None = None

    # Extra parameters not explicitly defined above.
    extras       : dict | None = None

    @classmethod
    def from_hdf5(cls,
                  bln_grp : h5py.Group,
                  anl_grp : h5py.Group = None,
                  ) -> "BeamlinePrm":
        """Create a BeamlinePrm by merging beamline and analysis attributes.

        Beamline-only fields come from `bln_grp`; analysis-only fields
        (`skip`, `scalepolydeg`, `usebpmref`, `roislice`) come from
        `anl_grp`, falling back to defaults when absent.
        """
        # --- dict 1: beamline attributes ---
        attrs = {key.lower(): val for key, val in bln_grp.attrs.items()}
        attrs["nbumps"] = attrs.pop("# bumps", attrs.get("# sweeps", 1))

        beamline = attrs.get("beamline", None)
        bl_short = beamline[:3]

        if "bpmdist" not in attrs:
            attrs["bpmdist"] = Config.BPMDISTS.get(bl_short, None)

        if "sector" not in attrs:
            attrs["sector"] = Config.SECTOR.get(bl_short, None)
        else:
            sec = attrs["sector"].split()
            attrs["sector"] = [
                sec[0],
                int(sec[1].strip("()"))
                ]

        if "xbpmdist" not in attrs:
            attrs["xbpmdist"] = Config.XBPMDISTS.get(beamline, None)

        if "grid_shape" not in attrs:
            attrs["grid_shape"] = None

        attrs["updated"] = attrs.pop("HDF5 updated on", None)

        # --- dict 2: analysis attributes (merged below) ---
        an_attrs = (
            {key.lower(): val for key, val in anl_grp.attrs.items()}
            if anl_grp is not None else {}
            )

        if "skip" in an_attrs:
            attrs["skip"] = int(an_attrs["skip"])
        if "scalepolydeg" in an_attrs:
            attrs["scalepolydeg"] = int(an_attrs["scalepolydeg"])
        if "usebpmref" in an_attrs:
            attrs["usebpmref"] = bool(an_attrs["usebpmref"])

        # ROI slice: from the Analysis group when present, else default.
        ashape = int(np.sqrt(attrs["nbumps"]))
        roi = an_attrs.get("roi size")
        if roi is not None:
            roisize = [int(x) for x in roi.split(",")]
            attrs["roislice"] = ROISlice.update(
                arrayshape = (ashape, ashape),
                roisize    = roisize,
                )
        else:
            attrs["roislice"] = ROISlice.update(
                arrayshape = (ashape, ashape),
                roisize    = [5, 5],
                )

        # Keep only declared fields; leftovers go to `extras`.
        valid_keys = {fld.name for fld in fields(cls)}
        extras = {
            key: val
            for key, val in attrs.items()
            if key not in valid_keys
            }
        attrs = {
            key: val
            for key, val in attrs.items()
            if key in valid_keys
            }
        attrs["extras"] = extras

        return cls(**attrs)

    def __getattr__(self, name: str):
        """Expose unknown HDF5 attributes from the `extras` dict.

        Replicates the usual behavior for retrieving attributes
        of the class.
        """
        extras = self.__dict__.get("extras")
        if extras is not None and name in extras:
            return extras[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    _ANALYSIS_ATTRS = ("roislice", "scalepolydeg", "skip", "usebpmref")

    def to_hdf5(self,
                bl_grp   : "h5py.Group",
                ) -> None:
        """Write a Prm instance to an HDF5 group.
        
        Args:
            self   : The BeamlinePrm instance to write.
            bl_grp : The HDF5 group where the attributes will be stored.
        """
        # Parameters belonging to the Analysis group are written elsewhere.

        try:
            # Write BeamlinePrm attributes to the HDF5 group.
            attrs = bl_grp.attrs
            for fld in fields(self):
                val = getattr(self, fld.name)
                if val is None:
                    continue
                if fld.name == "extras":
                    continue
                if fld.name in self._ANALYSIS_ATTRS:
                    continue
                if fld.name == "sector":
                    attrs["Sector"] = f"{self.sector[0]} ({self.sector[1]})"
                elif fld.name == "nbumps":
                    attrs["# bumps"] = self.nbumps
                elif fld.name == "updated":
                    attrs["HDF5 updated on"] = datetime.now().isoformat()
                else:
                    attrs[fld.name] = getattr(self, fld.name)
        except Exception as err:
            print(
                f"### ERROR while writing 'Prm' to {bl_grp.name}"
                f" (HDF5 group): \n {err}"
            )

    def to_hdf5_analysis(self,
                         anl_grp : "h5py.Group",
                         ) -> None:
        """Write analysis-specific parameters to the Analysis group attrs."""
        attrs = anl_grp.attrs
        if self.roislice is not None:
            attrs["ROI size"] = (
                f"{self.roislice.size_v}, {self.roislice.size_h}"
                )
        attrs["scalepolydeg"] = self.scalepolydeg
        attrs["skip"]         = self.skip
        attrs["usebpmref"]    = self.usebpmref


#
# Generic data structures.
#

@dataclass
class Positions:
    """Container for calculated XBPM positions."""
    x : np.ndarray
    y : np.ndarray

    @classmethod
    def from_hdf5(cls,
                  h5data: h5py.Group | np.ndarray | h5py.Dataset
                  ) -> "Positions":
        """Create a Positions instance from x and y arrays."""
        ds = None
        if   isinstance(h5data, h5py.Group):
            entries = h5data.keys()
            data    = h5data
        elif isinstance(h5data, np.ndarray):
            entries = h5data.dtype.names
            data    = h5data
        elif isinstance(h5data, h5py.Dataset):
            ds      = h5data
            data    = h5data[()]   # Load the dataset into memory
            entries = h5data.dtype.names
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
            x = data[x_key][:]
            y = data[y_key][:]
        except (KeyError, ValueError) as err:
            raise KeyError(f"Neither pair of fields found:\n {err}")

       # Rebuild the original grid shape when it was flattened.
        if ds is not None and 'positions_shape' in ds.attrs:
            shape = tuple(int(s) for s in ds.attrs['positions_shape'])
            x = x.reshape(shape)
            y = y.reshape(shape)

        return cls(x=x, y=y)

    def to_hdf5(self,
                pos_grp      : h5py.Group,
                append_name  : str = "",
                dataset_name = None
                ) -> "h5py.Dataset":
        """Write a Positions instance to an HDF5 group.
        
        Args:
            self    : The Positions instance to write.
            pos_grp : The HDF5 group where the attributes will be stored.
        """
        xf = f"x_{append_name}" if append_name else "x"
        yf = f"y_{append_name}" if append_name else "y"

        dt = np.dtype([
            (xf, self.x.dtype),
            (yf, self.y.dtype)
            ])
        table = np.empty(self.x.size, dtype=dt)
        table[xf] = self.x.ravel()
        table[yf] = self.y.ravel()

        if dataset_name in pos_grp:
            del pos_grp[dataset_name]
        ds = pos_grp.create_dataset(dataset_name, data=table)

        # Rebuild the grid on read for >1-D positions.
        if self.x.ndim > 1:
            ds.attrs['positions_shape'] = self.x.shape

        return ds


@dataclass
class Blades:
    """Container for blade current data and associated metadata.
    
    to, ti, bi, bo: measured currents for top in, top out, bottom in
                    and bottom out blades
    sto, sti, sbi, sbo: standard deviations of the respective currents
    """
    to      : np.ndarray
    ti      : np.ndarray
    bi      : np.ndarray
    bo      : np.ndarray
    sto     : Optional[np.ndarray]
    sti     : Optional[np.ndarray]
    sbi     : Optional[np.ndarray]
    sbo     : Optional[np.ndarray]
    pos_nom : Positions

    @classmethod
    def from_hdf5(cls,
                  avg_grp: h5py.Group,
                  ) -> "Blades":
        """Create a Blades instance from an HDF5 group."""
        datanames = avg_grp.dtype.names

        # Check for required datasets in the HDF5 group.
        blade_titles = [
            'to_mean', 'ti_mean', 'bi_mean', 'bo_mean',
            'to_err',  'ti_err',  'bi_err',  'bo_err'
            ]
        for blade in blade_titles:
            if blade not in datanames:
                raise ValueError(
                    " WARNING: while reading Average Blade Currents from HDF5"
                    f" file:\n Missing '{blade}' dataset in HDF5 group."
                    )

        # Extract nominal positions.
        pos_nom = Positions.from_hdf5(avg_grp)

        # Extract blades and sort them according to the nominal grid.
        blades = cls._sort_by_grid(avg_grp, pos_nom)
        return cls(**blades)

    @classmethod
    def _sort_by_grid(cls,
                      bldata  : h5py.Group,
                      pos_nom : Positions
                      ) -> dict:
        """Sort the blade data by grid coordinates."""
        bladeset = {}
        for bld in ['to', 'ti', 'bi', 'bo']:
            bladeset[f"{bld}"] = ('points', bldata[f"{bld}_mean"][:])
            if bldata[f"{bld}_err"] is not None:
                bladeset[f"s{bld}"] = ('points', bldata[f"{bld}_err"][:])
        ds = xr.Dataset(
            bladeset,
            coords={
                'x': ('points', pos_nom.x),
                'y': ('points', pos_nom.y),
            }
        )
        # Convert to a 2D grid by setting a MultiIndex and unstacking
        # This requires that every (y,x) combination appears exactly once,
        # otherwise you'll get NaN for missing combinations.
        ds = ds.set_index(points=['y', 'x']).unstack('points')

        # Reshape the coordinate arrays to match the 2D grid.
        xx = ds['x'].broadcast_like(ds['to']).values
        yy = ds['y'].broadcast_like(ds['to']).values

        dsdict = {}
        keys   = ['to', 'ti', 'bi', 'bo', 'sto', 'sti', 'sbi', 'sbo']
        dsdict = {
            key: ds[key].values
            if key in ds else np.full(ds['to'].shape, np.nan)
            for key in keys
            }
        dsdict["pos_nom"] = Positions(x=xx, y=yy)
        return dsdict

    def to_hdf5(self,
                avg_grp      : "h5py.Group",
                dataset_name : str = "blade_averages",
                ) -> None:
        """Write a Blades instance to an HDF5 group.
        
        Args:
            self : The Blades instance to write.
            avg_grp : The HDF5 group where the attributes will be stored.
        """
        # Create a table for storing blade values as a structured NumPy array.
        bld_list = [
            ('x_nom',   'f8', self.pos_nom.x),
            ('y_nom',   'f8', self.pos_nom.y),
            ('to_mean', 'f8', self.to),
            ('ti_mean', 'f8', self.ti),
            ('bi_mean', 'f8', self.bi),
            ('bo_mean', 'f8', self.bo),
            ('to_err',  'f8', self.sto),
            ('ti_err',  'f8', self.sti),
            ('bi_err',  'f8', self.sbi),
            ('bo_err',  'f8', self.sbo),
            ]

        n = self.to.size
        dt = np.dtype([(key, typ) for key, typ, _ in bld_list])
        table = np.full(n, np.nan, dtype=dt)
        for key, _, bld in bld_list:
            # Ignore empty error blades.
            if bld is not None:
                table[key] = bld.ravel()

        if dataset_name in avg_grp:
            del avg_grp[dataset_name]
        return avg_grp.create_dataset(dataset_name, data=table)


#
# Raw data structures.
#


@dataclass
class BladeAvgData:
    """Container for averaged blade current data and associated metadata.
    
    prm       : metadata parameters of the averaged data
    nom       : nominal positions at the XBPM site
    blades    : averaged blade currents and their standard deviations
    """
    prm     : dict
    pos_nom : Positions
    blades  : Blades

    @classmethod
    def from_hdf5(cls,
                  bld_grp : "h5py.Group",
                  ) -> "BladeAvgData":
        """Create a BladeAvgData instance from an HDF5 group."""
        # Extract metadata attributes.
        blades  = Blades.from_hdf5(bld_grp)
        return cls(
            prm     = dict(bld_grp.attrs),
            pos_nom = blades.pos_nom,
            blades  = blades
            )

    def to_hdf5(self,
                bld_grp       : "h5py.Group",
                bladeset_name : str,
                ) -> int:
        """Write a BladeAvgData instance to an HDF5 group."""
        # Get the HDF5 group names for the specified bladeset.
        grp_names = Config._h5_group_name(bladeset_name)

        # Write blade dataset.
        blname = grp_names["blades"]
        ds     = self.blades.to_hdf5(
            bld_grp,
            dataset_name = blname
            )

        # Write the metadata attributes to the blade group.
        for key, val in self.prm.items():
            ds.attrs[key] = val
        # Specific attributes.
        beamline = self.prm.get("beamline", "")
        ds.attrs["Description"] = Config.get_description(
            group    = blname,
            beamline = beamline
            )
        ds.attrs["Beamline"] = beamline


@dataclass
class BladeRawVals:
    """Container for one blade raw data and associated metadata.
    
    val        : measured currents for the blade
    range      : measurement range for the blade
    saturation : saturation levels for the blade
    """
    val        : np.ndarray
    range      : np.ndarray
    saturation : np.ndarray

    @classmethod
    def from_hdf5(cls,
                  bld_grp: h5py.Group,
                  blade: str
                  ) -> "BladeRawVals":
        """Create a BladeRawVals instance from an HDF5 group."""
        required_fields = [
            'val',
            'range',
            'saturation'
            ]
        for fld in required_fields:
            if f"{blade}_{fld}" not in bld_grp.dtype.names:
                raise ValueError(
                    f" ERROR while reading BladeRawVals from HDF5 file:\n"
                    f" Missing '{blade}_{fld}' dataset in HDF5 group."
                )

        return cls(
            val        = bld_grp[f"{blade}_val"][:],
            range      = bld_grp[f"{blade}_range"][:],
            saturation = bld_grp[f"{blade}_saturation"][:]
        )

    def to_hdf5(self,
                table  : np.ndarray,
                blname : str = "",
                ) -> None:
        """Write a BladeRawVals instance to an HDF5 group."""
        for f in fields(self):
            table[f"{blname}_{f.name}"] = getattr(self, f.name)


@dataclass
class BPMRawData:
    """Container for BPM positions (one sweep) data and its metadata."""
    descr : str
    pos   : Positions

    @classmethod
    def from_hdf5(cls,
                  bpm_grp : "h5py.Group"
                  ) -> "BPMRawData":
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

        # Extract the description and positions from the HDF5 group.
        descr = bpm_grp.attrs["Description"]

        # Extract the positions from the HDF5 group.
        pos = Positions.from_hdf5(bpm_grp)
        return cls(
            descr = descr,
            pos   = pos
            )

    def to_hdf5(self,
                bpm_grp : "h5py.Group",
                beamline : str = "",
                bumpnum  : int = None,
                ) -> "h5py.Dataset":
        """Write a BPMRawData instance to an HDF5 dataset."""
        try:
            ds = self.pos.to_hdf5(
                pos_grp      = bpm_grp,
                append_name  = "bpm",
                dataset_name = "bpm_data"
                )
            ds.attrs["Description"] = Config.get_description(
                "bpm_data",
                beamline = beamline,
                bumpnum  = bumpnum
                )
        except Exception as err:
            print(
                "### ERROR while writing 'BPMRawData' instance to"
                f" {bpm_grp.name} (HDF5 group):"
                f"\n {err}"
            )
        return ds


@dataclass
class BladeRawData:
    """Container for all blades' raw data and associated metadata.
    
    TO, TI, BI, BO: BladeVals for each blade
    """
    TO : BladeRawVals
    TI : BladeRawVals
    BI : BladeRawVals
    BO : BladeRawVals

    @classmethod
    def from_hdf5(cls,
                  raw_grp: h5py.Group,
                  beamline: str
                  ) -> "BladeRawData":
        """Create a BladeRawData instance from an HDF5 group."""
        # Use the checked beamline map to extract data.
        bmap = Config.BLADEMAP.get(beamline, None)
        return cls(
            TO = BladeRawVals.from_hdf5(raw_grp, bmap["TO"]),
            TI = BladeRawVals.from_hdf5(raw_grp, bmap["TI"]),
            BI = BladeRawVals.from_hdf5(raw_grp, bmap["BI"]),
            BO = BladeRawVals.from_hdf5(raw_grp, bmap["BO"])
        )

    def to_hdf5(self,
                brd_grp  : h5py.Group,
                beamline : str = ""
                ) -> "h5py.Dataset":
        """Write a BladeRawData instance to an HDF5 group."""
        bmap  = Config.BLADEMAP.get(beamline, None)
        names = ('TO', 'TI', 'BI', 'BO')

        # Prepare the structured array table for storing blade data.
        # ('val', 'range', 'saturation')
        cols = [f.name for f in fields(BladeRawVals)]
        dt   = np.dtype([
            (f"{bmap[n]}_{c}", getattr(getattr(self, n), c).dtype)
            for n in names
            for c in cols
            ])
        table = np.empty(len(self.TO.val), dtype = dt)

        # Delegate the fill to each BladeRawVals instance.
        for name in names:
            getattr(self, name).to_hdf5(table, bmap[name])

        # Create or replace the blade_data dataset in the HDF5 group.
        if "blade_data" in brd_grp:
            del brd_grp["blade_data"]
        ds = brd_grp.require_dataset(
            "blade_data",
            data = table,
            dtype = dt,
            shape = table.shape
            )

        # Add a description attribute to the blade_data dataset.
        ds.attrs["Description"] = Config.get_description(
            "blade_data",
            beamline = beamline
            )
        return ds


@dataclass
class BumpRawData:
    """Container for bump data and associated metadata.
    
    prm   : parameters of the bump
    bpm   : BPM registered orbit at the time of the bump
    blades: BladeRawData
    """
    meta   : dict
    bpm    : BPMRawData
    blades : BladeRawData

    @classmethod
    def from_hdf5(cls,
                  bump_grp : h5py.Group,
                  beamline : str,
                  bumpnum  : int,
                  ) -> "BumpRawData":
        """Create a BumpRawData instance from an HDF5 group."""
        try:
            # Sweep metadata.
            meta = dict(bump_grp.attrs.items())
            meta["Bump #"] = bumpnum

            # Ensure angle keys are present. Workaound for double standars.
            ag = {'x': None, 'y': None}
            ag = {
                d : val
                for d in ['x', 'y']
                for key, val in bump_grp.attrs.items()
                if key.startswith(f'Angle {d}') or
                key.startswith(f'ag{d}')
            }
            meta.update({
                'Angle x' : ag['x'],
                'Angle y' : ag['y'],
            })

            # BPM dataset.
            bpm = BPMRawData.from_hdf5(bump_grp['bpm_data'])

            # Read raw data.
            bld = BladeRawData.from_hdf5(
                bump_grp["blade_data"],
                beamline
                )

            # Instantiate BumpRawData with parameters, BPM data, and raw data.
            return cls(
                meta   = meta,
                bpm    = bpm,
                blades = bld
                )
        except Exception as err:
            raise ValueError(
                "### ERROR while reading 'BumpRawData' from HDF5 file:\n"
                f"  >>> {err}"
            )

    def to_hdf5(self,
                bump_grp  : h5py.Group,
                beamline  : str = "",
                bumpnum   : int = None,
                ) -> int:
        """Write a BumpRawData instance to an HDF5 group."""
        try:
            swp_attrs = [
                'Angle x', 'Angle y',
                'Bump pos x', 'Bump pos y',
                'PV meter', 'SR current',
                'Timestamp',
                'gap/ph.'
                ]
            for attr in swp_attrs:
                if attr in self.meta:
                    bump_grp.attrs[attr] = self.meta[attr]
            if bumpnum is None:
                bumpnum = self.meta.get("bump #")
            bump_grp.attrs["Bump #"] = bumpnum

            # BPM data.
            self.bpm.to_hdf5(
                bpm_grp  = bump_grp,
                beamline = beamline,
                bumpnum  = bumpnum,
                )

            # Blade raw data.
            ds = self.blades.to_hdf5(
                brd_grp  = bump_grp,
                beamline = beamline
                )
            ds.attrs["Description"] = Config.get_description(
                "Bumps",
                beamline = beamline,
                bumpnum  = bumpnum
                )
        except Exception as err:
            print(
                "### ERROR while writing 'BumpRawData' instance to"
                f" {bump_grp.name} (HDF5 group):"
                f"\n {err}"
            )



@dataclass
class BeamlineRawData:
    """Container for all sweep data and associated metadata for a beamline.
    
    metadata  : Metadata info for raw_data group. 
    bumps     : dict of BumppRawData instances for each bump (int)
    blade_avg : BladeAvgData instance
    """
    meta      : dict
    bumps     : dict[int, BumpRawData] = field(default_factory=dict)
    blade_avg : BladeAvgData | None    = None

    @classmethod
    def from_hdf5(cls,
                  raw_grp  : h5py.Group,
                  beamline : str
                  ) -> "BeamlineRawData":
        """Extract raw data from the a raw_data HDF5 group."""
        # Group metadata.
        meta = dict(raw_grp.attrs.items())

        # Blade average data is a numpy array structure.
        blade_avg = BladeAvgData.from_hdf5(
            bld_grp = raw_grp["blade_averages"]
            )

        # Extract sweep data.
        bumps = {}
        for key, data in raw_grp.items():
            if key.startswith('bump_') or key.startswith('sweep_'):
                # Extract sweep number
                num          = int(key.split('_')[1])
                bumps[num]   = BumpRawData.from_hdf5(
                    bump_grp = data,
                    beamline = beamline,
                    bumpnum  = num
                    )

        return cls(
            meta      = meta,
            bumps     = bumps,
            blade_avg = blade_avg
        )

    def to_hdf5(self,
                bl_raw_grp : h5py.Group,
                beamline   : str,
                ) -> None:
        """Write beamline data to an HDF5 group."""
        try:
            # Group attributes.
            meta = dict(self.meta)
            meta.setdefault(
                "Description",
                Config.get_description("raw_data", beamline)
                )
            bl_raw_grp.attrs.update(meta)

            # Write blade averages if available.
            if self.blade_avg is not None:
                self.blade_avg.to_hdf5(
                    bld_grp       = bl_raw_grp,
                    bladeset_name = "AverageData"
                    )

            # Write sweeps data.
            for num, bump in self.bumps.items():
                # Delete existing sweep_xxxx group if it exists,
                # and rewrite it as bump_xxxx.
                name = f"sweep_{num:04d}"
                if name in bl_raw_grp:
                    del bl_raw_grp[name]
                bump_grp = bl_raw_grp.require_group(f"bump_{num:04d}")
                bump.to_hdf5(
                    bump_grp,
                    beamline = beamline,
                    bumpnum  = num,
                    )

        except Exception as e:
            print(" WARNING (BeamlineRawData):"
                  f" Failed to write beamline raw data for '{beamline}':"
                  f"\n \t {e}")


#
# Structures for data analysis.
#


@dataclass
class RMSStatistics:
    """Computed RMS statistics between nominal and measured data."""
    # Horizontal, vertical and total differences at each site.
    diff_h   : np.ndarray
    diff_v   : np.ndarray
    diff_tot : np.ndarray

    # Minimum and maximum values of the differences.
    min_h    : float
    max_h    : float
    min_v    : float
    max_v    : float
    min_tot  : float
    max_tot  : float

    # Mean values of the differences.
    mean_h   : float
    mean_v   : float
    mean_tot : float

    @classmethod
    def from_hdf5(cls,
                  rms_grp: h5py.Group,
                  ) -> "RMSStatistics":
        """Create an RMSStatistics instance from an HDF5 group."""
        ds    = rms_grp['summary']
        row   = ds[()]
        names = ds.dtype.names

        kwargs = {}
        for name in names:
            val = row[name]
            if isinstance(val, np.ndarray):   # defensive scalar extraction
                val = val.item()
            kwargs[name] = val

        return cls(
            diff_h   = rms_grp['diff_h'][()],
            diff_v   = rms_grp['diff_v'][()],
            diff_tot = rms_grp['diff_tot'][()],
            **kwargs,
        )

    def to_hdf5(self,
                rms_grp    : h5py.Group,
                roi_or_all : str = "",
                ) -> None:
        """Write RMSStatistics to an HDF5 group."""
        rms_grp.attrs["Description"] = Config.get_description(
            "RMS statistics",
            roi_or_all = roi_or_all
        )

        # 2-D difference arrays -> datasets (snake_case names).
        for name, arr in (
                ('diff_h',   self.diff_h),
                ('diff_v',   self.diff_v),
                ('diff_tot', self.diff_tot),
                ):
            if name in rms_grp:
                del rms_grp[name]
            rms_grp.create_dataset(name, data=arr)

        # Scalar summary -> single-row dataset with named fields.
        flds = [
            'min_h', 'mean_h', 'max_h',
            'min_v', 'mean_v', 'max_v',
            'min_tot', 'mean_tot', 'max_tot',
            ]
        dt = np.dtype([
            (name, 'f8')
            for name in flds
            ])
        row = np.zeros(1, dtype=dt)

        for name in flds:
            row[name] = getattr(self, name)

        if 'summary' in rms_grp:
            del rms_grp['summary']
        rms_grp.create_dataset('summary', data=row)

@dataclass
class RMSGridStatistics:
    """Statistics calculated at a given ROI."""
    all      : RMSStatistics   # Full grid statistics.
    roi      : RMSStatistics   # ROI statistics.
    roislice : ROISlice        # ROI slice object.

    @classmethod
    def from_hdf5(cls,
                  rmsgrid_grp : h5py.Group,
                  ) -> "RMSGridStatistics":
        """Create an RMSGridStatistics instance from an HDF5 group."""
        # Statistics for the full grid and the ROI.
        all_stats = RMSStatistics.from_hdf5(rmsgrid_grp['all'])
        roi_stats = RMSStatistics.from_hdf5(rmsgrid_grp['roi'])

        # Recalculate ROI slices.
        roislice = ROISlice.update(
            arrayshape = all_stats.diff_h.shape,
            roisize    = roi_stats.diff_h.shape
        )

        return cls(
            all      = all_stats,
            roi      = roi_stats,
            roislice = roislice
        )

    def to_hdf5(self,
                rmsgrid_grp : h5py.Group,
                ) -> None:
        self.all.to_hdf5(
            rmsgrid_grp.require_group('all'),
            roi_or_all='all'
            )
        self.roi.to_hdf5(
            rmsgrid_grp.require_group('roi'),
            roi_or_all='roi'
            )


@dataclass
class BPMAnalysis:
    """Container for BPM positions and associated metadata.

    Attributes:
        prm : BPM parameters.
        bpm : BPM positions (x, y) calculated from measurements.
        nom : Nominal positions at XBPM site extrapolated from bump-angles.
        rms : RMS values of the differences between bpm and nom positions.
        roi_diffs : estimated standard deviations of the differences between
            bpm and nom positions.
    """
    prm      : BeamlinePrm
    pos_nom  : Positions | None = None
    pos_meas : Positions
    rms_diff : RMSGridStatistics

    @classmethod
    def from_hdf5(cls,
                  bpm_grp : "h5py.Group",
                  prm_bml : BeamlinePrm,
                  ) -> "BPMAnalysis":
        """Create a BPMAnalysis instance from an HDF5 group."""
        ds   = bpm_grp['positions']
        data = ds[()]

        shape = None
        if 'positions_shape' in ds.attrs:
            shape = tuple(int(s) for s in ds.attrs['positions_shape'])

        # Reshape the arrays if a shape attribute is present.
        x_nom  = data['x_nom'][:]
        y_nom  = data['y_nom'][:]
        x_meas = data['x_meas'][:]
        y_meas = data['y_meas'][:]
        if shape is not None:
            x_nom  = x_nom.reshape(shape)
            y_nom  = y_nom.reshape(shape)
            x_meas = x_meas.reshape(shape)
            y_meas = y_meas.reshape(shape)

        pos_nom  = Positions(x = x_nom,  y = y_nom)
        pos_meas = Positions(x = x_meas, y = y_meas)
        rms_diff = RMSGridStatistics.from_hdf5(bpm_grp['rms_diff'])

        return cls(
            prm      = prm_bml,
            pos_nom  = pos_nom,
            pos_meas = pos_meas,
            rms_diff = rms_diff,
            )

    def to_hdf5(self,
                bpm_grp : h5py.Group,
                ) -> None:
        """Write BPM analysis data to an HDF5 group."""
        bpm_grp.attrs["Description"] = Config.get_description(
            "bpm_analysis",
            beamline = self.prm.beamline
            )
        bpm_grp.attrs["Sector"] = (
            f"{self.prm.sector[0]} ({self.prm.sector[1]})"
            )
        bpm_grp.attrs["Beamline"] = self.prm.beamline

        # Combined nominal + measured positions -> flat table.
        if self.pos_nom is not None and self.pos_meas is not None:
            dt = np.dtype([
                ('x_nom',  self.pos_nom.x.dtype),
                ('y_nom',  self.pos_nom.y.dtype),
                ('x_meas', self.pos_meas.x.dtype),
                ('y_meas', self.pos_meas.y.dtype),
                ])
            table = np.empty(self.pos_meas.x.size, dtype=dt)
            table['x_nom']  = self.pos_nom.x.ravel()
            table['y_nom']  = self.pos_nom.y.ravel()
            table['x_meas'] = self.pos_meas.x.ravel()
            table['y_meas'] = self.pos_meas.y.ravel()

            # Drop legacy per-object datasets.
            for legacy in ('pos_meas', 'positions'):
                if legacy in bpm_grp:
                    del bpm_grp[legacy]
            ds = bpm_grp.create_dataset('positions', data=table)
            if self.pos_meas.x.ndim > 1:
                ds.attrs['positions_shape'] = self.pos_meas.x.shape

        # RMS grid statistics.
        self.rms_diff.to_hdf5(bpm_grp.require_group('rms_diff'))


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

    qx  : Optional[float] = 0.0
    sqx : Optional[float] = 0.0
    qy  : Optional[float] = 0.0
    sqy : Optional[float] = 0.0

    @classmethod
    def from_hdf5(cls,
                  scl_ds : "h5py.Dataset"
                  ) -> "Scales":
        """Create a Scales instance from a single-row HDF5 dataset."""
        row    = scl_ds[()]
        names  = scl_ds.dtype.names

        kwargs = {}
        for name in names:
            val = row[name]
            if isinstance(val, np.ndarray):   # defensive scalar extraction
                val = val.item()
            kwargs[name] = val

        return cls(**kwargs)



    def to_hdf5(self,
                scl_grp : "h5py.Group",
                ds_name : str = 'coefficients',
                ) -> None:
        """Write a Scales instance as a single-row dataset with named fields.

        Each coefficient becomes a column (field) of the dataset, so the
        stored values are self-describing. No values are duplicated in
        group attributes.
        """
        flds  = [
            'qx', 'sqx',
            'kx', 'skx',
            'dx', 'sdx',
            'qy', 'sqy',
            'ky', 'sky',
            'dy', 'sdy',
            ]
        dt = np.dtype([
            (name, 'f8')
            for name in flds
            ])
        row = np.zeros(1, dtype=dt)

        for name in flds:
            row[name] = getattr(self, name)

        # Replace any previous dataset (e.g. legacy 'value'/'error' format).
        if ds_name in scl_grp:
            del scl_grp[ds_name]
        scl_grp.create_dataset(ds_name, data=row)


@dataclass
class BladeLineFit:
    """Container for the results of a linear fit to blade data.

    Attributes:
        k       : Slope of the linear fit.
        sk      : Standard deviation of the slope.
        d       : Intercept of the linear fit.
        sd      : Standard deviation of the intercept.
        pos     : Positions of the data points.
        bld_raw : Raw data points.
        bld_err : Standard deviations of the blade measurements.
        bld_fit : Fitted data points.
    """
    k       : float
    sk      : float
    d       : float
    sd      : float
    pos     : np.ndarray
    bld_raw : np.ndarray
    bld_err : np.ndarray
    bld_fit : np.ndarray

    @classmethod
    def from_hdf5(cls,
                  blf_grp : h5py.Group,
                  ) -> "BladeLineFit":
        """Create a BladeLineFit instance from an HDF5 group."""
        ds    = blf_grp['coefficients']
        row   = ds[()]
        names = ds.dtype.names

        kwargs = {}
        for name in names:
            val = row[name]
            if isinstance(val, np.ndarray):   # defensive scalar extraction
                val = val.item()
            kwargs[name] = val

        kwargs.update({
            'pos'     : blf_grp['pos'][:],
            'bld_raw' : blf_grp['blade_raw'][:],
            'bld_err' : blf_grp['blade_err'][:],
            'bld_fit' : blf_grp['blade_fit'][:],
        })
        return cls(**kwargs)

    def to_hdf5(self,
                blf_grp : h5py.Group,
                ) -> None:
        """Write the BladeLineFit instance to an HDF5 group."""
        flds = ['k', 'sk', 'd', 'sd']
        dt = np.dtype([
            (name, 'f8')
            for name in flds
            ])
        row = np.zeros(1, dtype=dt)

        for name in flds:
            row[name] = getattr(self, name)

        # Replace any previous dataset (e.g. legacy 'value'/'error' format).
        if 'coefficients' in blf_grp:
            del blf_grp['coefficients']
        blf_grp.create_dataset('coefficients', data=row)

        for name, arr in (
            ('pos',       self.pos),
            ('blade_raw', self.bld_raw),
            ('blade_err', self.bld_err),
            ('blade_fit', self.bld_fit),
            ):
            if name in blf_grp:
                del blf_grp[name]
            blf_grp.create_dataset(name, data=arr)

@dataclass
class BladeCenterAnalysis:
    """Container for the analysis of blade central positions.

    Attributes:
        to   : Fitting results for the top outer blade.
        ti   : Fitting results for the top inner blade.
        bi   : Fitting results for the bottom inner blade.
        bo   : Fitting results for the bottom outer blade.
    """
    to : BladeLineFit
    ti : BladeLineFit
    bi : BladeLineFit
    bo : BladeLineFit

    @classmethod
    def from_hdf5(cls,
                  bca_grp : h5py.Group
                  ) -> "BladeCenterAnalysis":
        """Create a BladeCenterAnalysis instance from an HDF5 group."""
        return cls(
            to = BladeLineFit.from_hdf5(bca_grp['to']),
            ti = BladeLineFit.from_hdf5(bca_grp['ti']),
            bi = BladeLineFit.from_hdf5(bca_grp['bi']),
            bo = BladeLineFit.from_hdf5(bca_grp['bo']),
        )

    def to_hdf5(self,
                bca_grp : h5py.Group,
                ) -> None:
        """Write the BladeCenterAnalysis instance to an HDF5 group."""
        for name, blfit in (
            ('to', self.to),
            ('ti', self.ti),
            ('bi', self.bi),
            ('bo', self.bo),
        ):
            if name in bca_grp:
                del bca_grp[name]
            blf_grp = bca_grp.create_group(name)
            blfit.to_hdf5(blf_grp)


@dataclass
class BCA_HV:
    """Container for horizontal and vertical blade center analyses."""
    h : BladeCenterAnalysis
    v : BladeCenterAnalysis

    def __iter__(self) -> Iterator[tuple[str, BladeCenterAnalysis]]:
        yield 'h', self.h
        yield 'v', self.v

    @classmethod
    def from_hdf5(cls, bcahv_grp) -> "BCA_HV":
        """Create a BCA_HV instance from an HDF5 group."""
        return cls(
            h = BladeCenterAnalysis.from_hdf5(bcahv_grp['h']),
            v = BladeCenterAnalysis.from_hdf5(bcahv_grp['v'])
        )

    def to_hdf5(self,
                bcahv_grp : h5py.Group,
                ) -> None:
        """Write the BCA_HV instance to an HDF5 group."""
        for name, bca in (
            ('h', self.h),
            ('v', self.v),
        ):
            if name in bcahv_grp:
                del bcahv_grp[name]
            bca_grp = bcahv_grp.create_group(name)
            bca.to_hdf5(bca_grp)


@dataclass
class CentralSweepLine:
    """Container for central sweep data.
    
    A sweep is performed along a line (horizontal or vertical) through the
    center of the blade map. It is supposed that there is no variation in
    the other direction, but distortions of measurements create an undesired
    slope, to be evaluated. The sweep data is then used to analyze the
    behavior of the blades along the sweep line, so the variation in the
    fixed coordinate is captured.

    blades       : values of the blades along the central sweep
    index        : variable coordinate values along the central sweep
                (x for horizontal sweep, y for vertical sweep)
    pos_fixed    : fixed coordinate values along the central sweep
                (h: x ~ 0 for vertical sweep, v: y ~ 0 for horizontal sweep)
    pos_calc     : calculated values for the fixed coordinate
    pos_fit      : values of fitted affine line to fixed coordinate
    pos_fit_err  : std dev of calculated fixed coordinate
    """
    blades      : Blades
    pos_index   : np.ndarray
    pos_fixed   : np.ndarray
    pos_calc    : np.ndarray
    pos_fit     : np.ndarray
    pos_fit_err : np.ndarray
    coeffs      : np.ndarray | None = None
    sigmas      : np.ndarray | None = None

    @classmethod
    def from_hdf5(cls,
                  sln_grp: h5py.Group,
                  dir: str
                  ) -> "CentralSweepLine":
        if dir == 'h':
            ind, fix = 'x', 'y'
        elif dir == 'v':
            ind, fix = 'y', 'x'
        else:
            raise ValueError(" Invalid direction. Use 'h' or 'v'.")

        sweep = sln_grp['sweep_data'][:]

        blades = Blades(
            to      = sweep['to_mean'],
            ti      = sweep['ti_mean'],
            bi      = sweep['bi_mean'],
            bo      = sweep['bo_mean'],
            sto     = sweep['to_err'],
            sti     = sweep['ti_err'],
            sbi     = sweep['bi_err'],
            sbo     = sweep['bo_err'],
            pos_nom = Positions(x = sweep['x_nom'], y = sweep['y_nom']),
            )

        def _fit_row(ds_name: str) -> dict | None:
            if ds_name not in sln_grp:
                return None
            row   = sln_grp[ds_name][()]
            names = row.dtype.names
            vals  = {}
            for nm in names:
                v = row[nm]
                if isinstance(v, np.ndarray):
                    v = v.item()
                vals[nm] = v
            return vals

        coeffs = None
        cname  = f"{fix}_fit_coeffs"
        if cname in sln_grp:
            c = _fit_row(cname)
            coeffs = np.array([c['k'], c['d']])

        sigmas = None
        sname  = f"{fix}_fit_sigmas"
        if sname in sln_grp:
            s = _fit_row(sname)
            sigmas = np.array([s['sk'], s['sd']])

        return cls(
            blades      = blades,
            pos_index   = sweep[f"{ind}_index"],
            pos_fixed   = sweep[f"{fix}_fix"],
            pos_calc    = sweep[f"{fix}_calc"],
            pos_fit     = sweep[f"{fix}_fit"],
            pos_fit_err = sweep[f"s_{fix}_fit"],
            coeffs      = coeffs,
            sigmas      = sigmas,
        )

    def to_hdf5(self,
                sln_grp   : "h5py.Group",
                direction : str = "",
                ) -> None:
        if direction == 'h':
            ind, fix = 'x', 'y'
        elif direction == 'v':
            ind, fix = 'y', 'x'
        else:
            raise ValueError("Invalid direction. Use 'h' or 'v'.")

        b = self.blades
        cols = [
            ('x_nom',        b.pos_nom.x),
            ('y_nom',        b.pos_nom.y),
            ('to_mean',      b.to),
            ('ti_mean',      b.ti),
            ('bi_mean',      b.bi),
            ('bo_mean',      b.bo),
            ('to_err',       b.sto),
            ('ti_err',       b.sti),
            ('bi_err',       b.sbi),
            ('bo_err',       b.sbo),
            (f"{ind}_index", self.pos_index),
            (f"{fix}_fix",   self.pos_fixed),
            (f"{fix}_calc",  self.pos_calc),
            (f"{fix}_fit",   self.pos_fit),
            (f"s_{fix}_fit", self.pos_fit_err),
            ]

        dt = np.dtype([(name, 'f8') for name, _ in cols])
        table = np.full(self.pos_index.size, np.nan, dtype=dt)
        for name, arr in cols:
            if arr is not None:
                table[name] = arr

        # Drop legacy per-array datasets.
        for legacy in (
                'blade_averages',
                f"{ind}_index", f"{fix}_fix", f"{fix}_calc",
                f"{fix}_fit", f"s_{fix}_fit",
                ):
            if legacy in sln_grp:
                del sln_grp[legacy]
        sln_grp.create_dataset('sweep_data', data=table)

        # Fit coefficients -> single-row named fields.
        if self.coeffs is not None:
            dtc = np.dtype([('k', 'f8'), ('d', 'f8')])
            row = np.zeros(1, dtype=dtc)
            row['k'] = self.coeffs[0]
            row['d'] = self.coeffs[1]
            if f"{fix}_fit_coeffs" in sln_grp:
                del sln_grp[f"{fix}_fit_coeffs"]
            sln_grp.create_dataset(f"{fix}_fit_coeffs", data=row)

        if self.sigmas is not None:
            dts = np.dtype([('sk', 'f8'), ('sd', 'f8')])
            row = np.zeros(1, dtype=dts)
            row['sk'] = self.sigmas[0]
            row['sd'] = self.sigmas[1]
            if f"{fix}_fit_sigmas" in sln_grp:
                del sln_grp[f"{fix}_fit_sigmas"]
            sln_grp.create_dataset(f"{fix}_fit_sigmas", data=row)


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
    def from_hdf5(cls,
                  swp_grp : "h5py.Group"
                  ) -> "CentralSweeps":
        """Create a CentralSweep instance from an HDF5 group."""
        return cls(
            h = CentralSweepLine.from_hdf5(
                swp_grp["central_sweep_h"],
                dir = 'h'
                ),
            v = CentralSweepLine.from_hdf5(
                swp_grp["central_sweep_v"],
                dir = 'v'
                )
        )

    def to_hdf5(self,
                swp_grp: "h5py.Group",
                ) -> None:
        """Write the CentralSweeps instance to an HDF5 group."""
        if self.h is not None:
            self.h.to_hdf5(
                swp_grp.require_group("central_sweep_h"),
                direction='h'
            )

        if self.v is not None:
            self.v.to_hdf5(
                swp_grp.require_group("central_sweep_v"),
                direction='v'
                )


@dataclass
class AllScales:
    """Container for all scaling factors.
    
    raw_pw : Scales for raw pairwise calculation
    trn_pw : Scales for transformed pairwise calculation
    raw_cr : Scales for raw cross-blade calculation
    trn_cr : Scales for transformed cross-blade calculation
    """
    pairw_std  : Scales
    pairw_trn  : Scales
    cross_std  : Scales
    cross_trn  : Scales

    @classmethod
    def from_hdf5(cls, asc_grp) -> "AllScales":
        """Create an AllScales instance from an HDF5 group."""
        return cls(
            pairw_std = Scales.from_hdf5(asc_grp["pairwise"]["std"]),
            cross_std = Scales.from_hdf5(asc_grp["cross"]["std"]),
            pairw_trn = Scales.from_hdf5(asc_grp["pairwise"]["transf"]),
            cross_trn = Scales.from_hdf5(asc_grp["cross"]["transf"])
        )

    def to_hdf5(self,
                asc_grp: "h5py.Group",
                ) -> None:
        """Write the AllScales instance to an HDF5 group."""
        pair_grp = asc_grp.require_group("pairwise")
        self.pairw_std.to_hdf5(pair_grp, ds_name = 'std')
        self.pairw_trn.to_hdf5(pair_grp, ds_name = 'transf')

        cross_grp = asc_grp.require_group("cross")
        self.cross_std.to_hdf5(cross_grp, ds_name = 'std')
        self.cross_trn.to_hdf5(cross_grp, ds_name = 'transf')


@dataclass
class SuppressionMatrix:
    """Container for suppression matrix data.
    
    matrix: 4x4 numpy array representing the suppression matrix
    """
    standard   : np.ndarray | None = None 
    calculated : np.ndarray | None = None
    stddev     : np.ndarray | None = None
    optimized  : np.ndarray | None = None

    def __post_init__(self) -> None:
        # If both are None, use the standard values
        if self.standard is None and self.stddev is None:
            self.standard, self.stddev = Config.standard_suppression_matrix()

    @classmethod
    def from_hdf5(cls,
                  mat_grp : "h5py.Group"
                  ) -> "SuppressionMatrix":
        """Create a SuppressionMatrix instance from an HDF5 group."""
        if ("standard" not in mat_grp or
            "calculated" not in mat_grp):
            raise ValueError(
                " ERROR while reading Suppression Matrix from HDF5 file:\n"
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

    def to_hdf5(self,
                mat_grp : h5py.Group,
                ) -> None:
        """Write the SuppressionMatrix instance to an HDF5 group."""
        for name in (
            'standard',
            'calculated',
            'stddev',
            'optimized'
            ):
            arr = getattr(self, name)
            if arr is None:
                continue
            if name in mat_grp:
                del mat_grp[name]
            mat_grp.create_dataset(name, data=arr)


@dataclass
class GL2RMatrix:
    """Container for partial delta/sigma transformation matrices.
    
    std   : identity matrix.
    rt_st : rotation plus stretching matrix.
    """
    std  : np.ndarray = field(default_factory=lambda: np.identity(2))
    calc : np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_hdf5(cls,
                  gl2r_grp : "h5py.Group"
                  ) -> "GL2RMatrix":
        """Create a GL2RMatrix instance from an HDF5 group."""
        kwargs = {
            "std"  : gl2r_grp["std"][:],
            "calc" : gl2r_grp["calc"][:],
        }
        return cls(**kwargs)

    def to_hdf5(self,
                gl2r_grp: h5py.Group,
                ) -> None:
        """Write the GL2RMatrix instance to an HDF5 group."""
        gl2r_grp.require_dataset(
            "std",
            data  = self.std,
            dtype = self.std.dtype,
            shape = self.std.shape,
            )
        gl2r_grp.require_dataset(
            "calc",
            data  = self.calc,
            dtype = self.calc.dtype,
            shape = self.calc.shape,
            )


@dataclass
class CalculatedPositions:
    """Container for analyzed positions and associated metadata.

        roi        : ROI slice used in the analysis

        pos_std    : positions calculated from standard formulae
        scale_std  : scaling factors for standard positions
        stat_std   : RMS statistics of the differences between standard
                 positions and nominal or BPM-calculated positions.

        pos_trn    : positions calculated from transformed formulae, through
                 the suppression matrix or the GL2R transformation matrix.
        scale_trn  : scaling factors for transformed positions
        stat_trn   : RMS statistics of the differences between transformed
                 positions and nominal or BPM-calculated positions.
    """
    roi       : np.ndarray

    pos_std   : Positions
    scale_std : Scales
    stat_std  : RMSGridStatistics

    pos_trn   : Positions
    scale_trn : Scales
    stat_trn  : RMSGridStatistics

    @classmethod
    def from_hdf5(cls,
                  pos_grp  : "h5py.Group",
                  roislice : ROISlice,
                  ) -> "CalculatedPositions":
        """Create a CalculatedPositions instance from an HDF5 group."""
        roi = np.array([roislice.size_h, roislice.size_v])

        # Standard positions.
        grs     = pos_grp["xbpm_std"]
        xs      = grs["Standard positions"]["std_x"][:]
        ys      = grs["Standard positions"]["std_y"][:]
        pos_std = Positions(x=xs, y=ys)

        # Scales.
        scale_std = Scales.from_hdf5(grs["scales"])
        stat_std  = RMSGridStatistics.from_hdf5(grs["statistics"])

        # Transformed positions.
        grt     = pos_grp["xbpm_transformed"]
        xt      = grt["Transformed positions"]["trn_x"][:]
        yt      = grt["Transformed positions"]["trn_y"][:]
        pos_trn = Positions(x=xt, y=yt)

        # Scales.
        scale_trn = Scales.from_hdf5(grt["scales"])
        stat_trn  = RMSGridStatistics.from_hdf5(grt["statistics"])

        return cls(
            roi       = roi,
            pos_std   = pos_std,
            pos_trn   = pos_trn,
            scale_std = scale_std,
            scale_trn = scale_trn,
            stat_std  = stat_std,
            stat_trn  = stat_trn
        )

    def to_hdf5(self,
                pos_grp  : "h5py.Group",
                ) -> None:
        """Write CalculatedPositions instance to an HDF5 group."""
        # Standard positions.
        std_grp = pos_grp.require_group("xbpm_std")
        dt = np.dtype([
            ("std_x", self.pos_std.x.dtype),
            ("std_y", self.pos_std.y.dtype)
        ])
        table_s = np.empty(self.pos_std.x.shape, dtype=dt)
        table_s["std_x"] = self.pos_std.x
        table_s["std_y"] = self.pos_std.y
        std_grp.require_dataset(
            "Standard positions",
            data  = table_s,
            dtype = table_s.dtype,
            shape = table_s.shape,
            )

        # Scaling and statistics for the standard positions.
        self.scale_std.to_hdf5(std_grp, ds_name="scales")
        self.stat_std.to_hdf5(std_grp.require_group("statistics"))

        # Transformed positions.
        trn_grp = pos_grp.require_group("xbpm_transformed")
        dt = np.dtype([
            ("trn_x", self.pos_trn.x.dtype),
            ("trn_y", self.pos_trn.y.dtype)
        ])
        table_t = np.empty(self.pos_trn.x.shape, dtype=dt)
        table_t["trn_x"] = self.pos_trn.x
        table_t["trn_y"] = self.pos_trn.y
        trn_grp.require_dataset(
            "Transformed positions",
            data  = table_t,
            dtype = table_t.dtype,
            shape = table_t.shape,
            )

        # Scaling and statistics for the transformed positions.
        self.scale_trn.to_hdf5(trn_grp, ds_name="scales")
        self.stat_trn.to_hdf5(trn_grp.require_group("statistics"))


@dataclass
class XBPMPositions:
    """Container for analyzed positions and associated metadata.
    
    nom   : Nominal positions.
    bpm   : positions calculated from BPM analysis (at XBPM site).
    pairw : positions calculated from pairwise analysis.
    cross : positions calculated from cross-blade analysis.
    """
    nom   : Positions
    pairw : CalculatedPositions
    cross : CalculatedPositions

    @classmethod
    def from_hdf5(cls,
                  pos_grp  : "h5py.Group",
                  roislice : "ROISlice",
                  ) -> "XBPMPositions":
        """Create an XBPMPositions instance from an HDF5 group."""
        nom   = Positions.from_hdf5(pos_grp["nominal"])
        pairw = CalculatedPositions.from_hdf5(
            pos_grp["pairwise"],
            roislice
            )
        cross = CalculatedPositions.from_hdf5(
            pos_grp["cross"],
            roislice
            )

        return cls(
            nom   = nom,
            pairw = pairw,
            cross = cross
        )

    def to_hdf5(self,
                pos_grp    : "h5py.Group",
                ) -> None:
        """Write the current XBPMPositions instance to an HDF5 group."""
        # Nominal positions.
        self.nom.to_hdf5(
            pos_grp      = pos_grp,
            dataset_name = "nominal"
            )

        # Pairwise analyzed positions.
        self.pairw.to_hdf5(pos_grp.require_group("pairwise"))

        # Cross-blade analyzed positions.
        self.cross.to_hdf5(pos_grp.require_group("cross"))


@dataclass
class DataAnalysis:
    """Container for all data and analysis results.
 
    Classmethods may read analysis from HDF5 and compute analysis.

    bpm          : BPM calculated positions at XBPM site
    blademap     : blade map of positions
    bladecenter  : h and v blade center analyses
    xpositions   : Analyzed positions (raw and transformed)   
    centralsweep : Central sweep positions with errors
    scales       : Scaling factors
    supmat       : Suppression matrices
    figures      : dict              # Rendered figures for analysis tabs
    """
    # Beamline, description and XBPM-source distance.
    bpm           : BPMAnalysis       | None = None
    blademap      : BladeAvgData      | None = None
    bladecenter   : BCA_HV            | None = None
    xpositions    : XBPMPositions     | None = None
    centralsweeps : CentralSweeps     | None = None
    scales        : AllScales         | None = None
    supmat        : SuppressionMatrix | None = None
    gl2r          : GL2RMatrix        | None = None
    figures       : dict              | None = None

    @classmethod
    def from_hdf5(cls,
                  anl_grp: "h5py.Group",
                  roislice : ROISlice,
                  prm_bml  : BeamlinePrm,
                  ) -> "DataAnalysis":
        """Create a DataAnalysis instance from an HDF5 group."""
        # Get reference name dictionary.
        dan_names = Config._h5_group_name("DataAnalysis")

        # Extract BPM analysis data.
        bpm_name = dan_names["bpm"]
        bpm = BPMAnalysis.from_hdf5(anl_grp[bpm_name], prm_bml)

        # Extract blade map.
        blademap_name = dan_names["blademap"]
        dataset_name  = Config._h5_group_name("BladeMap")["blades"]
        blademap = BladeAvgData.from_hdf5(
            anl_grp[blademap_name][dataset_name]
            )

        # Extract blade center analysis for h/v directions.
        bc_name = dan_names["bladecenter"]
        bladecenter = BCA_HV.from_hdf5(anl_grp[bc_name])

        # Extract other analysis data.
        xpos_name = dan_names["xpositions"]
        xpositions = XBPMPositions.from_hdf5(
            anl_grp[xpos_name],
            roislice = roislice
            )

        # Central sweeps.
        cs_name = dan_names["centralsweeps"]
        centralsweeps = CentralSweeps.from_hdf5(anl_grp[cs_name])

        # Extract scaling factors and suppression matrices.
        scales_name = dan_names["scales"]
        scales = AllScales.from_hdf5(anl_grp[scales_name])

        # Extract suppression matrix.
        supmat_name = dan_names["supmat"]
        supmat = SuppressionMatrix.from_hdf5(anl_grp[supmat_name])

        # Extract GL2R matrix.
        gl2r_name = dan_names["gl2r"]
        gl2r = GL2RMatrix.from_hdf5(anl_grp[gl2r_name])

        return cls(
            bpm           = bpm,
            blademap      = blademap,
            bladecenter   = bladecenter,
            xpositions    = xpositions,
            centralsweeps = centralsweeps,
            scales        = scales,
            supmat        = supmat,
            gl2r          = gl2r,
        )

    def to_hdf5(self,
                dan_grp : "h5py.Group",
                prm_bml : BeamlinePrm = None,
                ) -> None:
        """Serialize a DataAnalysis instance to an HDF5 group."""
        # Analysis-specific parameters go on the Analysis group itself.
        if prm_bml is not None:
            prm_bml.to_hdf5_analysis(dan_grp)

        # Get reference from dictionary names.
        dan_names = Config._h5_group_name("DataAnalysis")

        flds = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "blademap" and f.name != "figures"
            }
        for f, v in flds.items():
            grp_name = dan_names[f]
            if v is not None:
                v.to_hdf5(dan_grp.require_group(grp_name))

        # Blade map is a specific case for BladeAvgData.
        if self.blademap is not None:
            grp_name = dan_names["blademap"]
            self.blademap.to_hdf5(
                dan_grp.require_group(grp_name),
                bladeset_name = "BladeMap"
                )

        # /figures/<tab key>
        if self.figures:
            fig_group = dan_grp.require_group("figures")
            for key, fig in self.figures.items():
                fig_name  = f"{key}.png"
                self._figure_export(
                    figure      = fig,
                    fig_grp     = fig_group,
                    figure_name = fig_name
                    )


    def _figure_export(
            self,
            figure      : Figure,
            fig_grp     : h5py.Group,
            figure_name : str
            ) -> None:
        """Save the figure to memory as PNG and store it in the HDF5 group."""
        buf = BytesIO()
        figure.savefig(
            buf,
            format      = 'png',
            dpi         = FIGDPI,
            bbox_inches = 'tight'
            )

        # Store the PNG bytes as a uint8 dataset
        buf.seek(0)
        png_bytes = np.frombuffer(
            buf.getvalue(),
            dtype = np.uint8
            )

        fig_grp.create_dataset(figure_name, data=png_bytes)
        buf.close()    


@dataclass
class BeamlineDatum:
    """Encapsulates all data for a single beamline extracted from HDF5.
    
    A structure is created to contain the raw data measured (BeamlineRawData)
    and, if present, previous analysis results (DataAnalysis) stored in the
    HDF5 file. Metadata is stored in a parameter dictionary.
    """
    prm      : BeamlinePrm
    raw_data : BeamlineRawData
    analysis : DataAnalysis     = field(default=None)

    # Class-level, shared by all instances (not a dataclass field).
    _names : ClassVar[dict] = Config._h5_group_name("BeamlineData")
    
    @classmethod
    def from_hdf5(cls,
                  bd_grp: "h5py.Group"
                  ) -> "BeamlineDatum":
        """Extract the beamline data from an HDF5 group."""
        # Data storage.
        kwargs = {}

        # Beamline parameters (analysis attributes merged when present).
        name_an = cls._names["analysis"]
        anl_grp = bd_grp[name_an] if name_an in bd_grp else None
        kwargs["prm"] = BeamlinePrm.from_hdf5(bd_grp, anl_grp)

        # Raw data.
        kwargs["raw_data"] = BeamlineRawData.from_hdf5(
            bd_grp[cls._names["raw_data"]],
            beamline = kwargs["prm"].beamline,
            )

        # Analysis data may be not present when data are imported.
        if anl_grp is not None:
            try:
                kwargs["analysis"] = DataAnalysis.from_hdf5(
                    anl_grp  = anl_grp,
                    roislice = kwargs["prm"].roislice,
                    prm_bml  = kwargs["prm"],
                    )
            except Exception as warn:
                logging.warning(
                    "\n### while reading 'Data Analysis' from HDF5 file:"
                    f"\n {warn}"
                    "\n Instantiating DataAnalysis with default values.\n"
                    )
                kwargs["analysis"] = DataAnalysis()
        else:
            kwargs["analysis"] = DataAnalysis()
        return cls(**kwargs)

    def to_hdf5(self,
                bl_grp : "h5py.Group",
                ) -> None:
        """Serialize the BeamlineData instance to an HDF5 file.

        Args:
            prm (BeamlinePrm): Beamline parameters.
            raw_data (BeamlineRawData): Raw data for the beamline.
            analysis (DataAnalysis): Analysis data for the beamline.
.
        """
        # Store beamline parameters.
        self.prm.to_hdf5(bl_grp)

        # Rewrite raw data only if explicitly requested.
        if self.raw_data is not None:
            bl_raw_grp = bl_grp.require_group(self._names["raw_data"])
            self.raw_data.to_hdf5(
                bl_raw_grp = bl_raw_grp,
                beamline = self.prm.beamline,
                ) 

        # Store analysis data if present.
        if self.analysis is not None:
            dan_grp = bl_grp.require_group(self._names["analysis"])
            self.analysis.to_hdf5(dan_grp = dan_grp, prm_bml = self.prm)

        return bl_raw_grp, dan_grp