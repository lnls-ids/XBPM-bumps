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


class H5Writer:
    """Create HDF5 groups and datasets described by the `Config` registry."""

    @staticmethod
    def group(
        parent : "h5py.Group",
        key    : str,
        name   : str | None = None,
        **fmt,
        ) -> "h5py.Group":
        """Get or create the child group of registry `key`, described.

        `name` overrides the stored name (indexed groups such as bumps);
        `fmt` fills the placeholders of the description.
        """
        grp = parent.require_group(name or Config.group_name(key))
        grp.attrs["Description"] = Config.get_description(key, **fmt)
        return grp

    @staticmethod
    def dataset(
        parent : "h5py.Group",
        key    : str,
        data,
        name   : str | None = None,
        **fmt,
        ) -> "h5py.Dataset":
        """Create the dataset of registry `key`, replacing any previous one."""
        name = name or Config.group_name(key)
        if name in parent:
            del parent[name]
        ds = parent.create_dataset(name, data=data)
        ds.attrs["Description"] = Config.get_description(key, **fmt)
        return ds


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

        attrs["updated"] = attrs.pop("hdf5 updated on", None)

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
                  h5data: h5py.Dataset,
                  xcol : str = "X",
                  ycol : str = "Y",
                  ) -> "Positions":
        """Create a Positions instance from x and y arrays."""
        if isinstance(h5data, h5py.Dataset):
            ds, data = h5data, h5data[()]
        elif isinstance(h5data, np.ndarray):
            ds, data = None, h5data
        else:
            raise TypeError(
                "Data must be an h5py.Dataset or a structured numpy array."
            )

        missing = [c for c in (xcol, ycol) if c not in data.dtype.names]
        if missing:
            where = f" '{ds.name}'" if ds is not None else ""
            raise ValueError(
                f" Missing column(s) {missing} in table{where}.\n"
                " Re-export the file from the source data."
                )

        x, y = data[xcol], data[ycol]

       # Rebuild the original grid shape when it was flattened.
        if ds is not None and 'positions_shape' in ds.attrs:
            shape = tuple(int(s) for s in ds.attrs['positions_shape'])
            x, y = x.reshape(shape), y.reshape(shape)

        return cls(x=x, y=y)

    def to_hdf5(
            self,
            pos_grp      : h5py.Group,
            key  : str,
            xcol : str = "X",
            ycol : str = "Y",
            **fmt,
            ) -> "h5py.Dataset":
        """Write the positions as a flat table named after registry 'key'.
        
        Args:
            self        : The Positions instance to write.
            pos_grp     : The HDF5 group where the attributes will be stored.
            key         : The registry key for naming the dataset.
            xcol        : Column name for x positions.
            ycol        : Column name for y positions.
            fmt         : Additional formatting options for the HDF5 dataset.
        """
        xytab = [
            (xcol, self.x),
            (ycol, self.y)
            ]
        return self.write_table(
            pos_grp, key, xytab, **fmt
            )

    @staticmethod
    def write_table(grp  : h5py.Group,
                    key  : str,
                    cols : list[tuple[str, np.ndarray]],
                    **fmt,
                    ) -> h5py.Dataset:
        """Write arrays of equal shape as the columns of one flat table.

        Also used for tables with extra position columns (nominal +
        measured). The grid shape is stored in `positions_shape` and
        restored by `from_hdf5`.
        """
        shape = cols[0][1].shape
        for name, arr in cols:
            if arr.shape != shape:
                raise ValueError(
                    f" Column '{name}' has shape {arr.shape};"
                    f" expected {shape}."
                    )
        dt    = np.dtype([(name, arr.dtype) for name, arr in cols])
        table = np.empty(cols[0][1].size, dtype=dt)
        for name, arr in cols:
            table[name] = arr.ravel()

        ds = H5Writer.dataset(grp, key, table, **fmt)
        if len(shape) > 1:
            ds.attrs["positions_shape"] = shape
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
    def from_hdf5(
        cls,
        avg_ds: "h5py.Dataset",
        ) -> "Blades":
        """Create a Blades instance from an HDF5 group."""
        col = Config.column

        table    = avg_ds[()]
        required = [
            col(b, k)
            for b in Config.BLADES
            for k in ("mean", "err")
            ]

        missing  = [
            c for c in required
            if c not in table.dtype.names
            ]
        if missing:
            raise ValueError(
                " WARNING: while reading Average Blade Currents from HDF5"
                f" file:\n Missing column(s) {missing} in '{avg_ds.name}'."
                )

        pos_nom = Positions.from_hdf5(
            table,
            xcol = col("x", "nom"),
            ycol = col("y", "nom"),
            )

        blds = cls._sort_by_grid(table, pos_nom)
        return cls(**blds)

    @classmethod
    def _sort_by_grid(
        cls,
        table   : np.ndarray,
        pos_nom : Positions
        ) -> dict:
        """Sort the blade data by grid coordinates."""
        bladeset = {}
        for bld in Config.BLADES:
            bladeset[bld] = (
                'points',
                table[Config.column(bld, "mean")]
                )

            bladeset[f"s{bld}"] = (
                'points',
                table[Config.column(bld, "err")]
                )
            
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
        blds0 = Config.BLADES[0]
        xx    = ds['x'].broadcast_like(ds[blds0]).values
        yy    = ds['y'].broadcast_like(ds[blds0]).values

        keys   = [*Config.BLADES, *(f"s{b}" for b in Config.BLADES)]
        dsdict = {
            key: ds[key].values
            if key in ds else np.full(ds[blds0].shape, np.nan)
            for key in keys
            }
        dsdict["pos_nom"] = Positions(x=xx, y=yy)
        return dsdict

    def to_hdf5(
            self,
            avg_grp  : "h5py.Group",
            key      : str,
            **fmt,
            ) -> "h5py.Dataset":
        """Write a Blades instance to an HDF5 group.
        
        Args:
            self : The Blades instance to write.
            avg_grp : The HDF5 group where the attributes will be stored.
        """
        # Create a table for storing blade values as a structured NumPy array.
        col  = Config.column
        cols = [
            (col("x", "nom"), self.pos_nom.x),
            (col("y", "nom"), self.pos_nom.y),
            *[(col(b, "mean"), getattr(self, b))       for b in Config.BLADES],
            *[(col(b, "err"),  getattr(self, f"s{b}")) for b in Config.BLADES],
            ]

        dt = np.dtype([(name, 'f8') for name, _ in cols])
        table = np.full(self.to.size, np.nan, dtype=dt)
        for name, arr in cols:
            # Empty error blades stay on as NaN.
            if arr is not None:
                table[name] = arr.ravel()

        return H5Writer.dataset(avg_grp, key, table, **fmt)


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
    def from_hdf5(
        cls,
        bld_ds : "h5py.Dataset",
        ) -> "BladeAvgData":
        """Create a BladeAvgData instance from an HDF5 group."""
        # Extract metadata attributes.
        blades  = Blades.from_hdf5(bld_ds)
        return cls(
            prm     = dict(bld_ds.attrs),
            pos_nom = blades.pos_nom,
            blades  = blades
            )

    def to_hdf5(
        self,
        bld_grp  : "h5py.Group",
        key      : str,
        ) -> None:
        """Write a BladeAvgData instance to an HDF5 group."""
        # Write blade dataset.
        ds = self.blades.to_hdf5(bld_grp, key)

        # Write the metadata attributes to the blade group.
        for name, val in self.prm.items():
            if name != "Description":
                ds.attrs[name] = val
        ds.attrs["Beamline"] = self.prm.get("beamline", "")


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
    def from_hdf5(
        cls,
        table : np.ndarray,
        pv    : str
        ) -> "BladeRawVals":
        """Read the columns of one PV ('A'..'D') from the blade table."""
        cols    = {
            f.name: Config.column(pv, f.name)
            for f in fields(cls)
            }

        missing = [
            c for c in cols.values()
            if c not in table.dtype.names
            ]
        if missing:
            raise ValueError(
                f" ERROR while reading BladeRawVals from HDF5 file:\n"
                f" Missing column(s) {missing}."
                )

        bladevals = {
            name: table[col]
            for name, col in cols.items()
            }
        return cls(**bladevals)

    def to_hdf5(
            self,
            table: np.ndarray,
            pv: str
            ) -> None:
        for f in fields(self):
            table[Config.column(pv, f.name)] = getattr(self, f.name)


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

        # Extract the description and positions from the HDF5 group.
        descr = bpm_grp.attrs["Description"]

        # Extract the positions from the HDF5 group.
        col = Config.column
        pos = Positions.from_hdf5(
            bpm_grp,
            xcol = col("x", "bpm"),
            ycol = col("y", "bpm"),
            )

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
        col = Config.column
        ds = self.pos.to_hdf5(
            pos_grp  = bpm_grp,
            key      = "bpm_data",
            xcol     = col("x", "bpm"),
            ycol     = col("y", "bpm"),
            beamline = beamline,
            bumpnum  = bumpnum,
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
    def from_hdf5(
        cls,
        blade_ds: "h5py.Dataset",
        beamline: str
        ) -> "BladeRawData":
        """Create a BladeRawData instance from an HDF5 group."""
        # Use the checked beamline map to extract data.
        bmap  = Config.BLADEMAP[beamline]
        table = blade_ds[()]
        bl_rd = {
            name.upper() : BladeRawVals.from_hdf5(table, bmap[name.upper()])
            for name in Config.BLADES
        }
        return cls(**bl_rd)

    def to_hdf5(
        self,
        brd_grp  : "h5py.Group",
        beamline : str = "",
        bumpnum  : int | None = None,
        ) -> "h5py.Dataset":
        """Write a BladeRawData instance to an HDF5 group."""
        bmap  = Config.BLADEMAP[beamline]
        names = tuple(bld.upper() for bld in Config.BLADES)

        # Prepare the structured array table for storing blade data.
        # ('val', 'range', 'saturation')
        kinds  = [f.name for f in fields(BladeRawVals)]
        dt     = np.dtype([
            (Config.column(bmap[n], k), getattr(getattr(self, n), k).dtype)
            for n in names
            for k in kinds
            ])
        table = np.empty(len(self.TO.val), dtype = dt)

        # Delegate the fill to each BladeRawVals instance.
        for name in names:
            getattr(self, name).to_hdf5(table, bmap[name])

        return H5Writer.dataset(
            brd_grp,
            "blade_data",
            table,
            beamline = beamline,
            bumpnum  = bumpnum
            )


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
    def from_hdf5(
        cls,
        bump_grp : h5py.Group,
        beamline : str,
        bumpnum  : int,
        ) -> "BumpRawData":
        """Create a BumpRawData instance from an HDF5 group."""
        try:
            # Sweep metadata.
            meta = dict(bump_grp.attrs.items())
            meta["Bump #"] = bumpnum

            # Ensure angle keys are present. Workaround for double standars.
            # ag = {'x': None, 'y': None}
            # ag = {
            #     d : val
            #     for d in ['x', 'y']
            #     for key, val in bump_grp.attrs.items()
            #     if key.startswith(f'Angle {d}') or
            #     key.startswith(f'ag{d}')
            # }
            # meta.update({
            #     'Angle x' : ag['x'],
            #     'Angle y' : ag['y'],
            # })

            # BPM dataset.
            bpm = BPMRawData.from_hdf5(
                bump_grp[Config.group_name('bpm_data')]
                )

            # Read blade raw data.
            bld = BladeRawData.from_hdf5(
                bump_grp[Config.group_name("blade_data")],
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

    def to_hdf5(
        self,
        raw_grp  : "h5py.Group",
        beamline : str = "",
        bumpnum  : int = None,
        ) -> int:
        """Write a BumpRawData instance to an HDF5 group."""
        if bumpnum is None:
            bumpnum = self.meta.get("Bump #")
        try:
            bump_grp = H5Writer.group(
                parent   = raw_grp,
                key      = "bump",
                name     = self.bump_name(bumpnum),
                beamline = beamline,
                bumpnum  = bumpnum,
                )
            # Every attribute is kept (Angle x/y, Disp pos x/y, Gap/Ph. <ID>);
            # the description is regenerated from the registry.
            for name, val in self.meta.items():
                if name != "Description" and val is not None:
                    bump_grp.attrs[name] = val
            bump_grp.attrs["Bump #"] = bumpnum

            # Blade raw data.
            self.blades.to_hdf5(
                bld_grp  = bump_grp,
                beamline = beamline,
                bumpnum  = bumpnum
                )

            # BPM data.
            self.bpm.to_hdf5(
                bpm_grp  = bump_grp,
                beamline = beamline,
                bumpnum  = bumpnum,
                )

        except Exception as err:
            print(
                "### ERROR while writing 'BumpRawData' instance to"
                f" {raw_grp.name}, bump # {bumpnum} (HDF5 group):"
                f"\n {err}"
            )

    @staticmethod
    def bump_name(num: int) -> str:
        """Stored group name of a bump, e.g. 'Bump 0007'."""
        return f"{Config.group_name('bump')} {num:04d}"

    @staticmethod
    def bump_number(name: str) -> int | None:
        """Inverse of `bump_name`; None when `name` is not a bump group."""
        prefix, _, tail = name.rpartition(" ")
        if prefix == Config.group_name("bump") and tail.isdigit():
            return int(tail)
        return None



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
    def from_hdf5(
        cls,
        raw_grp  : "h5py.Group",
        beamline : str
        ) -> "BeamlineRawData":
        """Extract raw data from the a raw_data HDF5 group."""
        # Group metadata.
        meta = dict(raw_grp.attrs.items())

        # Extract sweep data.
        bumps = {}
        for name, grp in raw_grp.items():
            num = BumpRawData.bump_number(name)
            if num is not None:
                # Extract sweep number
                bumps[num]   = BumpRawData.from_hdf5(
                    bump_grp = grp,
                    beamline = beamline,
                    bumpnum  = num
                    )

        # Blade average data is a numpy array structure.
        blade_avg = BladeAvgData.from_hdf5(
            bld_ds = raw_grp[Config.group_name("blade_averages")]
            )

        return cls(
            meta      = meta,
            bumps     = bumps,
            blade_avg = blade_avg
        )

    def to_hdf5(
        self,
        bl_grp : h5py.Group,
        beamline   : str,
        ) -> None:
        """Create and fill raw data group inside beamline group."""
        raw_grp = H5Writer.group(
            bl_grp,
            "raw_data",
            beamline = beamline
            )
        raw_grp.attrs.update({
            k: v
            for k, v in self.meta.items()
            if k != "Description"
            })
        if self.blade_avg is not None:
            self.blade_avg.to_hdf5(
                bld_grp = raw_grp,
                key     = "blade_averages"
                )
        for num, bump in self.bumps.items():
            bump.to_hdf5(
                raw_grp,
                beamline = beamline,
                bumpnum  = num
                )


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

    _SUMMARY : ClassVar[tuple[str, ...]] = (
        'min_h',   'mean_h',   'max_h',
        'min_v',   'mean_v',   'max_v',
        'min_tot', 'mean_tot', 'max_tot',
        )

    # RMSStatistics
    @staticmethod
    def _column(field: str) -> str:
        """'min_h' -> 'Horizontal minimum'."""
        kind, direction = field.split("_")
        return Config.column(direction, kind)

    @classmethod
    def from_hdf5(
        cls,
        rms_grp: "h5py.Group"
        ) -> "RMSStatistics":
        """Create an RMSStatistics instance from an HDF5 group."""
        row      = rms_grp[Config.group_name("summary")][0]
        diff_h   = rms_grp[Config.group_name("diff_h")][()]
        diff_v   = rms_grp[Config.group_name("diff_v")][()]
        diff_tot = rms_grp[Config.group_name("diff_tot")][()]
        stats    = {
            f : float(row[cls._column(f)])
            for f in cls._SUMMARY
            }

        return cls(
            diff_h   = diff_h,
            diff_v   = diff_v,
            diff_tot = diff_tot,
            **stats
            )

    def to_hdf5(
        self,
        parent : "h5py.Group",
        key    : str
        ) -> None:
        """Create the 'All grid'/'ROI' group (`key`: 'rms_all', 'rms_roi')."""
        grp = H5Writer.group(parent, key)
        for dkey in ("diff_h", "diff_v", "diff_tot"):
            H5Writer.dataset(grp, dkey, getattr(self, dkey))

        row = np.zeros(1, dtype=np.dtype([
            (self._column(f), 'f8')
            for f in self._SUMMARY
            ]))
        for f in self._SUMMARY:
            row[self._column(f)] = getattr(self, f)

        H5Writer.dataset(grp, "summary", row)


@dataclass
class RMSGridStatistics:
    """Statistics calculated at a given ROI."""
    all      : RMSStatistics   # Full grid statistics.
    roi      : RMSStatistics   # ROI statistics.
    roislice : ROISlice        # ROI slice object.

    @classmethod
    def from_hdf5(
        cls,
        rmsgrid_grp : h5py.Group,
        ) -> "RMSGridStatistics":
        """Create an RMSGridStatistics instance from an HDF5 group."""
        # Statistics for the full grid and the ROI.
        all_stats = RMSStatistics.from_hdf5(
            rmsgrid_grp[Config.group_name('rms_all')]
            )

        roi_stats = RMSStatistics.from_hdf5(
            rmsgrid_grp[Config.group_name('rms_roi')]
            )

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

    def to_hdf5(
        self,
        parent : "h5py.Group",
        key    : str,
        ) -> None:
        grp = H5Writer.group(parent, key)
        self.all.to_hdf5(grp, 'rms_all')
        self.roi.to_hdf5(grp, 'rms_roi')


@dataclass
class BPMAnalysis:
    """Container for BPM positions and associated metadata.

    Attributes:
        prm      : beamline parameters.
        pos_nom  : nominal positions at XBPM site, from bump-angles.
        pos_meas : measured positions at the XBPM site.
        rms_diffs: estimated std devs of differences between nominal and
                   measured positions.
    """
    prm      : BeamlinePrm
    pos_nom  : Positions
    pos_meas : Positions
    rms_diff : RMSGridStatistics

    @classmethod
    def from_hdf5(
        cls,
        bpm_grp : "h5py.Group",
        prm_bml : BeamlinePrm,
        ) -> "BPMAnalysis":
        """Create a BPMAnalysis instance from an HDF5 group."""
        name = Config.group_name('bpm_positions')
        col  = Config.column
        ds   = bpm_grp[name]

        pos_nom  = Positions.from_hdf5(ds, col('x', 'nom'),  col('y', 'nom'))
        pos_meas = Positions.from_hdf5(ds, col('x', 'meas'), col('y', 'meas'))
        rms_diff = RMSGridStatistics.from_hdf5(
            bpm_grp[Config.group_name('rms_diff')]
            )

        return cls(
            prm      = prm_bml,
            pos_nom  = pos_nom,
            pos_meas = pos_meas,
            rms_diff = rms_diff,
            )

    def to_hdf5(
        self,
        parent : "h5py.Group",
        ) -> None:
        """Write BPM analysis data to an HDF5 group."""
        bpm_grp = H5Writer.group(parent, "bpm_analysis")
        bpm_grp.attrs["Sector"]   = (
            f"{self.prm.sector[0]} ({self.prm.sector[1]})")
        bpm_grp.attrs["Beamline"] = self.prm.beamline

        col = Config.column
        if self.pos_nom is not None and self.pos_meas is not None:
            Positions.write_table(bpm_grp, "bpm_positions", [
                (col("x", "nom"),  self.pos_nom.x),
                (col("y", "nom"),  self.pos_nom.y),
                (col("x", "meas"), self.pos_meas.x),
                (col("y", "meas"), self.pos_meas.y),
                ])
        self.rms_diff.to_hdf5(bpm_grp, "rms_diff")


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

    @staticmethod
    def _column(field: str) -> str:
        """'kx' -> 'X slope', 'sqy' -> 'Y quadratic error'."""
        return Config.column(field[-1], field[:-1])

    @classmethod
    def from_hdf5(
        cls,
        scl_ds : "h5py.Dataset"
        ) -> "Scales":
        """Create a Scales instance from a single-row HDF5 dataset."""
        row    = scl_ds[0]
        names  = scl_ds.dtype.names
        kwargs = {
            f.name : float(row[cls._column(f.name)])
            for f in fields(cls)
            if cls._column(f.name) in names
            }
        return cls(**kwargs)

    def to_hdf5(
        self,
        parent : "h5py.Group",
        key    : str,
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
            (self._column(name), 'f8')
            for name in flds
            ])
        row = np.zeros(1, dtype = dt)
        for name in flds:
            row[self._column(name)] = getattr(self, name)
        H5Writer.dataset(parent, key, row)


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

    _COEFS  : ClassVar[tuple[str, ...]] = ('k', 'sk', 'd', 'sd')
    _CURVES : ClassVar[tuple[str, ...]] = ('raw', 'err', 'fit')

    @classmethod
    def from_hdf5(
        cls,
        data   : np.ndarray,      # Fit data table
        coeffs : np.void,         # raw coefficients table
        blade  : str,             # 'to', 'ti', 'bi', 'bo'
        ) -> "BladeLineFit":
        """Create a BladeLineFit instance from an HDF5 group."""
        col = Config.column
        cfs = {
            c : float(coeffs[col(blade, c)])
            for c in cls._COEFS
            }
        pos     = data[Config.H5_COLUMNS["pos"]]
        bld_raw = data[col(blade, "raw")]
        bld_err = data[col(blade, "err")]
        bld_fit = data[col(blade, "fit")]
        return cls(
            **cfs,
            pos     = pos,
            bld_raw = bld_raw,
            bld_err = bld_err,
            bld_fit = bld_fit
        )

    def to_hdf5(
        self,
        data   : np.ndarray,        # preallocated 'Fit data' table
        coeffs : np.void,           # row 0 of the 'Coefficients' table
        blade  : str,
        ) -> None:
        """Fill this blade's columns in the two shared tables."""
        col = Config.column
        data[Config.H5_COLUMNS["pos"]] = self.pos   # same axis for all blades
        for c in self._CURVES:
            data[col(blade, c)] = getattr(self, f"bld_{c}")
        for c in self._COEFS:
            coeffs[col(blade, c)] = getattr(self, c)


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
    def from_hdf5(
        cls,
        bca_grp : h5py.Group
        ) -> "BladeCenterAnalysis":
        """Create a BladeCenterAnalysis instance from an HDF5 group."""
        data   = bca_grp[Config.group_name('fit_data')][()]
        coeffs = bca_grp[Config.group_name('coefficients')][0]
        blades = {
            bld: BladeLineFit.from_hdf5(data, coeffs, bld)
            for bld in Config.BLADES
            }
        return cls(**blades)

    def to_hdf5(
        self,
        parent    : "h5py.Group",
        direction : str,
        ) -> None:
        """Write the BladeLineFit instance to an HDF5 group."""
        bca_grp = H5Writer.group(parent, f"direction_{direction}")

        col     = Config.column
        fit_dt  = [
            (Config.H5_COLUMNS["pos"], 'f8')
            ] + [
            (col(b, c), 'f8')
            for b in Config.BLADES
            for c in BladeLineFit._CURVES
            ]

        coef_dt = [
            (col(b, c), 'f8')
            for b in Config.BLADES
            for c in BladeLineFit._COEFS
            ]

        data    = np.full(
            self.to.pos.size,
            np.nan,
            dtype=np.dtype(fit_dt)
            )

        coeffs  = np.zeros(1, dtype=np.dtype(coef_dt))

        for bld in Config.BLADES:
            getattr(self, bld).to_hdf5(data, coeffs[0], bld)

        H5Writer.dataset(bca_grp, "fit_data", data)
        H5Writer.dataset(bca_grp, "coefficients", coeffs)


@dataclass
class BCA_HV:
    """Container for horizontal and vertical blade center analyses."""
    h : BladeCenterAnalysis
    v : BladeCenterAnalysis

    def __iter__(self) -> Iterator[tuple[str, BladeCenterAnalysis]]:
        yield 'h', self.h
        yield 'v', self.v

    @classmethod
    def from_hdf5(cls, bcahv_grp : "h5py.Group") -> "BCA_HV":
        """Create a BCA_HV instance from an HDF5 group."""
        return cls(
            h = BladeCenterAnalysis.from_hdf5(
                bcahv_grp[Config.group_name("direction_h")]
                ),
            v = BladeCenterAnalysis.from_hdf5(
                bcahv_grp[Config.group_name("direction_v")]
                )
        )

    def to_hdf5(self, parent: "h5py.Group") -> None:
        bc_grp = H5Writer.group(parent, "blade_center")
        self.h.to_hdf5(bc_grp, direction='h')
        self.v.to_hdf5(bc_grp, direction='v')


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
    def from_hdf5(
        cls,
        sln_grp   : h5py.Group,
        direction : str
        ) -> "CentralSweepLine":
        ind, fix = {'h': ('x', 'y'), 'v': ('y', 'x')}[direction]
        col      = Config.column
        sweep    = sln_grp[Config.group_name("sweep_data")][()]

        blades = Blades(
            to      = sweep[col("to", "mean")],
            ti      = sweep[col("ti", "mean")],
            bi      = sweep[col("bi", "mean")],
            bo      = sweep[col("bo", "mean")],
            sto     = sweep[col("to", "err")],
            sti     = sweep[col("ti", "err")],
            sbi     = sweep[col("bi", "err")],
            sbo     = sweep[col("bo", "err")],
            pos_nom = Positions(
                x = sweep[col("x", "nom")],
                y = sweep[col("y", "nom")]
                ),
            )

        coeffs = sigmas = None
        name   = Config.group_name("sweep_coeffs")
        if name in sln_grp:
            row    = sln_grp[name][0]
            coeffs = np.array([row[col(fix, "k")],  row[col(fix, "d")]])
            sigmas = np.array([row[col(fix, "sk")], row[col(fix, "sd")]])

        return cls(
            blades      = blades,
            pos_index   = sweep[col(ind, "index")],
            pos_fixed   = sweep[col(fix, "fixed")],
            pos_calc    = sweep[col(fix, "calc")],
            pos_fit     = sweep[col(fix, "fit")],
            pos_fit_err = sweep[col(fix, "fit_err")],
            coeffs      = coeffs,
            sigmas      = sigmas,
        )

    def to_hdf5(
        self,
        parent    : "h5py.Group",
        direction : str = "",
        ) -> None:
        ind, fix = {'h': ('x', 'y'), 'v': ('y', 'x')}[direction]
        sln_grp  = H5Writer.group(parent, f"sweep_{direction}")
        col, blds = Config.column, self.blades

        cols = [
            (col("x",  "nom"),     blds.pos_nom.x),
            (col("y",  "nom"),     blds.pos_nom.y),
            (col("to", "mean"),    blds.to),
            (col("ti", "mean"),    blds.ti),
            (col("bi", "mean"),    blds.bi),
            (col("bo", "mean"),    blds.bo),
            (col("to", "err"),     blds.sto),
            (col("ti", "err"),     blds.sti),
            (col("bi", "err"),     blds.sbi),
            (col("bo", "err"),     blds.sbo),
            (col(ind,  "index"),   self.pos_index),
            (col(fix,  "fixed"),   self.pos_fixed),
            (col(fix,  "calc"),    self.pos_calc),
            (col(fix,  "fit"),     self.pos_fit),
            (col(fix,  "fit_err"), self.pos_fit_err),
            ]

        dt = np.dtype([(name, 'f8') for name, _ in cols])
        table = np.full(self.pos_index.size, np.nan, dtype=dt)
        for name, arr in cols:
            if arr is not None:
                table[name] = arr
        H5Writer.dataset(
            sln_grp,
            "sweep_data",
            table
        )

        # Fit coefficients -> single-row named fields.
        if self.coeffs is not None:
            sigmas = (self.sigmas
                      if self.sigmas is not None
                      else (np.nan, np.nan)
                      )
            dtc = np.dtype([
                (col(fix, c), 'f8')
                for c in ("k", "sk", "d", "sd")
                ])
            row = np.zeros(1, dtype=dtc)
            row[col(fix, 'k')],  row[col(fix, 'd')]  = self.coeffs
            row[col(fix, 'sk')], row[col(fix, 'sd')] = sigmas
            H5Writer.dataset(
                sln_grp,
                "sweep_coeffs",
                row
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
    def from_hdf5(cls,
                  swp_grp : "h5py.Group"
                  ) -> "CentralSweeps":
        """Create a CentralSweep instance from an HDF5 group."""
        name_h = Config.group_name("sweep_h")
        h = (
            CentralSweepLine.from_hdf5(swp_grp[name_h], direction = 'h')
            if name_h in swp_grp
            else None
        )

        name_v = Config.group_name("sweep_v")
        v = (
            CentralSweepLine.from_hdf5(swp_grp[name_v], direction = 'v')
            if name_v in swp_grp
            else None
        )
        return cls(h=h, v=v)

    def to_hdf5(
            self,
            parent : "h5py.Group",
            ) -> None:
        """Write the CentralSweeps instance to an HDF5 group."""
        swp_grp = H5Writer.group(parent, "central_sweeps")
        if self.h is not None:
            self.h.to_hdf5(swp_grp, direction='h')
        if self.v is not None:
            self.v.to_hdf5(swp_grp, direction='v')


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
    def from_hdf5(
        cls,
        asc_grp : "h5py.Group",
        ) -> "AllScales":
        """Create an AllScales instance from an HDF5 group."""
        pairw = asc_grp[Config.group_name("scales_pairwise")]
        cross = asc_grp[Config.group_name("scales_cross")]
        std   = Config.group_name("scales_std")
        trn   = Config.group_name("scales_trn")
        return cls(
            pairw_std = Scales.from_hdf5(pairw[std]),
            pairw_trn = Scales.from_hdf5(pairw[trn]),
            cross_std = Scales.from_hdf5(cross[std]),
            cross_trn = Scales.from_hdf5(cross[trn])
        )

    def to_hdf5(
        self,
        parent: "h5py.Group",
        ) -> None:
        """Write the AllScales instance to an HDF5 group."""
        asc_grp = H5Writer.group(parent, "scales")
        pairw   = H5Writer.group(asc_grp, "scales_pairwise")
        cross   = H5Writer.group(asc_grp, "scales_cross")
        self.pairw_std.to_hdf5(pairw, "scales_std")
        self.pairw_trn.to_hdf5(pairw, "scales_trn")
        self.cross_std.to_hdf5(cross, "scales_std")
        self.cross_trn.to_hdf5(cross, "scales_trn")


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
    def from_hdf5(
        cls,
        mat_grp : "h5py.Group"
        ) -> "SuppressionMatrix":
        """Create a SuppressionMatrix instance from an HDF5 group."""
        std  = Config.group_name("sup_standard")
        calc = Config.group_name("sup_calculated")
        sdev = Config.group_name("sup_stddev")
        opt  = Config.group_name("sup_optimized")

        if std not in mat_grp or calc not in mat_grp:
            raise ValueError(
                " ERROR while reading Suppression Matrix from HDF5 file:\n"
                f" Missing {std} or {calc} dataset in HDF5 group."
            )

        kwargs = {
            "standard"   : mat_grp[std][:],
            "calculated" : mat_grp[calc][:],
        }

        # Analysis might be incomplete.
        if opt in mat_grp:
            kwargs["optimized"] = mat_grp[opt][:]

        if sdev in mat_grp:
            kwargs["stddev"] = mat_grp[sdev][:]
    
        return cls(**kwargs)

    def to_hdf5(self,
                parent : h5py.Group,
                ) -> None:
        """Write the SuppressionMatrix instance to an HDF5 group."""
        mat_grp = H5Writer.group(parent, "suppression")
        for key, arr in (
            ('sup_standard',   self.standard),
            ('sup_calculated', self.calculated),
            ('sup_stddev',     self.stddev),
            ('sup_optimized',  self.optimized)
            ):
            if arr is not None:
                H5Writer.dataset(mat_grp, key, arr)


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
            "std"  : gl2r_grp[Config.group_name("gl2r_std")][:],
            "calc" : gl2r_grp[Config.group_name("gl2r_calc")][:],
        }
        return cls(**kwargs)

    def to_hdf5(self,
                parent: "h5py.Group",
                ) -> None:
        """Write the GL2RMatrix instance to an HDF5 group."""
        gl2r_grp = H5Writer.group(parent, "gl2r")
        H5Writer.dataset(gl2r_grp, "gl2r_std",  self.std)
        H5Writer.dataset(gl2r_grp, "gl2r_calc", self.calc)


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
        data_s  = grs["Standard positions"][()]
        shape_s = None
        if "positions_shape" in grs["Standard positions"].attrs:
            shape_s = tuple(
                int(s)
                for s in grs["Standard positions"].attrs["positions_shape"]
                )
        xs = data_s["std_x"][:]
        ys = data_s["std_y"][:]
        if shape_s is not None:
            xs = xs.reshape(shape_s)
            ys = ys.reshape(shape_s)
        pos_std = Positions(x=xs, y=ys)

        scale_std = Scales.from_hdf5(grs["scales"])
        stat_std  = RMSGridStatistics.from_hdf5(grs["statistics"])

        # Transformed positions.
        grt     = pos_grp["xbpm_transformed"]
        data_t  = grt["Transformed positions"][()]
        shape_t = None
        if "positions_shape" in grt["Transformed positions"].attrs:
            shape_t = tuple(
                int(s)
                for s in grt["Transformed positions"].attrs["positions_shape"]
                )
        xt = data_t["trn_x"][:]
        yt = data_t["trn_y"][:]
        if shape_t is not None:
            xt = xt.reshape(shape_t)
            yt = yt.reshape(shape_t)
        pos_trn = Positions(x=xt, y=yt)

        # Scales.
        scale_trn = Scales.from_hdf5(grt["scales"])
        
        # Statistics.
        stat_trn  = RMSGridStatistics.from_hdf5(grt["statistics"])

        return cls(
            roi       = roi,
            pos_std   = pos_std,
            pos_trn   = pos_trn,
            scale_std = scale_std,
            scale_trn = scale_trn,
            stat_std  = stat_std,
            stat_trn  = stat_trn,
        )

    def to_hdf5(self,
                pos_grp : "h5py.Group",
                pos_nom : "Positions" = None,
                ) -> None:
        """Write CalculatedPositions instance to an HDF5 group."""
        # Standard positions.
        std_grp = pos_grp.require_group("xbpm_std")
        self._write_position_table(
            std_grp,
            "Standard positions",
            pos_meas = self.pos_std,
            pos_nom  = pos_nom,
            xf = "std_x",
            yf = "std_y",
            )
        self.scale_std.to_hdf5(std_grp, ds_name="scales")
        self.stat_std.to_hdf5(std_grp.require_group("statistics"))

        # Transformed positions.
        trn_grp = pos_grp.require_group("xbpm_transformed")
        self._write_position_table(
            trn_grp,
            "Transformed positions",
            pos_meas = self.pos_trn,
            pos_nom  = pos_nom,
            xf = "trn_x",
            yf = "trn_y",
            )
        self.scale_trn.to_hdf5(trn_grp, ds_name="scales")
        self.stat_trn.to_hdf5(trn_grp.require_group("statistics"))

    @staticmethod
    def _write_position_table(
            grp      : "h5py.Group",
            ds_name  : str,
            pos_meas : "Positions",
            pos_nom  : "Positions" = None,
            xf       : str = "x",
            yf       : str = "y",
            ) -> h5py.Dataset:
        """Write measured (and optional nominal) positions as a flat table."""
        if pos_nom is not None and pos_nom.x.shape != pos_meas.x.shape:
            raise ValueError(
                f"Nominal shape {pos_nom.x.shape} does not match measured"
                f" shape {pos_meas.x.shape}."
                )

        cols = []
        if pos_nom is not None:
            cols += [
                ('x_nom', pos_nom.x.dtype, pos_nom.x),
                ('y_nom', pos_nom.y.dtype, pos_nom.y),
                ]
        cols += [
            (xf, pos_meas.x.dtype, pos_meas.x),
            (yf, pos_meas.y.dtype, pos_meas.y),
            ]

        dt = np.dtype([(name, typ) for name, typ, _ in cols])
        table = np.empty(pos_meas.x.size, dtype=dt)
        for name, _, arr in cols:
            table[name] = arr.ravel()

        if ds_name in grp:
            del grp[ds_name]
        ds = grp.create_dataset(ds_name, data=table)
        if pos_meas.x.ndim > 1:
            ds.attrs['positions_shape'] = pos_meas.x.shape
        return ds


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
    def from_hdf5(
        cls,
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
                pos_grp : "h5py.Group",
                ) -> None:
        """Write the current XBPMPositions instance to an HDF5 group."""
        # Nominal positions (flat x, y).
        self.nom.to_hdf5(
            pos_grp      = pos_grp,
            dataset_name = "nominal"
            )

        # Pairwise analyzed positions (with nominal reference).
        self.pairw.to_hdf5(
            pos_grp.require_group("pairwise"),
            pos_nom = self.nom
            )

        # Cross-blade analyzed positions (with nominal reference).
        self.cross.to_hdf5(
            pos_grp.require_group("cross"),
            pos_nom = self.nom
            )


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
