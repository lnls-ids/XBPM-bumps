"""Parameter handling and CLI parsing."""

from typing import Optional, Iterator, List
# from curses import raw
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime
import logging
# from typing import List, Optional

import h5py
import numpy as np
import xarray as xr

# from xbpm_bumps.core.config import Config
from .config import Config
from .constants import ROI_SIZE_H, ROI_SIZE_V, MAX_RAD_ANGLE


@dataclass
class GenPrm:
    """Typed container for command-line and runtime parameters.

    This class implements __getitem__/__setitem__ so existing code that
    uses prm["key"] remains compatible while also providing attribute
    access (prm.key).
    """
    beamline              : str   | None = None     # Blade line identifier
    sr_current            : float | None = None      # Synchrotron current

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
            attrs = {key: val for key, val in dset_grp.attrs.items()}
            attrs["inputfile"] = inputfile
        except Exception as err:
            raise ValueError(
                "### ERROR while reading 'Prm' from HDF5 group:\n"
                f" {err}"
            )

        # Define the output file name based on the input file.
        attrs["outputfile"] = cls._set_outputfile_name(inputfile)

        # Create a Prm instance with the extracted attributes.
        return cls(**attrs)

    @staticmethod
    def _set_outputfile_name(inputfile: str) -> str:
        """Set the output file prefix for the analysis."""
        return inputfile.rsplit('.', 1)[0]

    def to_hdf5(self,
                prm_grp : "h5py.Group",
                ) -> int:
        """Write a Prm instance to an HDF5 group.
        
        Args:
            self  : The GenPrm instance to write.
            prm_grp : The HDF5 group where the attributes will be stored.
        
        Returns:
            int: 0 if successful, 1 if an error occurred.

        """
        try:
            attrs = prm_grp.require_group("attributes").attrs
            for fld in fields(self):
                attrs[fld.name] = getattr(self, fld.name)
        except Exception as err:
            print(
                f"### ERROR while writing 'Prm' to {prm_grp.name}"
                f" (HDF5 group):\n {err}"
            )
            return 1

        return 0


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
    roislice     : ROISlice        = field(default_factory=ROISlice)
    updated      : str             | None = None

    # Extra parameters not explicitly defined above.
    extras       : dict | None = None

    @classmethod
    def from_hdf5(cls,
                  bln_grp: h5py.Group
                  ) -> "BeamlinePrm":
        """Create a BeamlinePrm instance from an HDF5 group."""
        try:
            attrs = {key.lower(): val for key, val in bln_grp.attrs.items()}

            # Rewrite '# bumps' to 'nbumps'.
            attrs["nbumps"] = attrs.pop("# bumps", attrs.get("# sweeps", 1))

            beamline = attrs.get("beamline", None)
            bl_short = beamline[:3]

            if "bpmdist" not in attrs:
                attrs["bpmdist"] = Config.BPMDISTS.get(bl_short, None)

            if "sector" not in attrs:
                attrs["sector"] = Config.SECTOR.get(bl_short, None)

            if "xbpmdist" not in attrs:
                attrs["xbpmdist"] = Config.XBPMDISTS.get(beamline, None)

            if "grid_shape" not in attrs:
                attrs["grid_shape"] = None

            # Initialize ROI data.
            ashape  = int(np.sqrt(attrs["nbumps"]))
            roisize = [int(x) for x in attrs["ROI size"]]
            attrs["roislice"] = ROISlice.update(
                arrayshape=(ashape, ashape),
                roisize = roisize
            )

            # Rewrite 'HDF5 updated on' to 'updated'.
            attrs["updated"] = attrs.pop("hdf5 updated on", None)

        except Exception as err:
            raise ValueError(
                "### ERROR while reading 'BeamlinePrm' from HDF5 group:\n"
                f" {err}"
            )

        # Ignore attributes that are not declared fields of this dataclass.
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


    def to_hdf5(self,
                bl_grp  : "h5py.Group",
                ) -> None:
        """Write a Prm instance to an HDF5 group.
        
        Args:
            self   : The BeamlinePrm instance to write.
            bl_grp : The HDF5 group where the attributes will be stored.
        """
        try:
            # Write BeamlinePrm attributes to the HDF5 group.
            attrs = bl_grp.require_group("attributes").attrs
            for fld in fields(self):
                if fld.name == "extras":
                    continue
                elif fld.name == "roislice":
                    rs = self.roislice
                    attrs["ROI size"] = rs.size_x, rs.size_y
                else:
                    attrs[fld.name] = getattr(self, fld.name)
        except Exception as err:
            print(
                f"### ERROR while writing 'Prm' to {bl_grp.name}"
                f" (HDF5 group): \n {err}"
            )


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
        if   isinstance(h5data, h5py.Group):
            entries = h5data.keys()
        elif isinstance(h5data, np.ndarray):
            entries = h5data.dtype.names
        elif isinstance(h5data, h5py.Dataset):
            h5data = h5data[()]   # Load the dataset into memory
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
            return cls(x=h5data[x_key][:], y=h5data[y_key][:])
        except (KeyError, ValueError) as err:
            raise KeyError(f"Neither pair of fields found:\n {err}")

    def to_hdf5(self,
                pos_grp     : h5py.Group,
                append_name : str = "",
                dataset_name = None
                ) -> None:
        """Write a Positions instance to an HDF5 group.
        
        Args:
            self    : The Positions instance to write.
            pos_grp : The HDF5 group where the attributes will be stored.
        """
        xf = f"x_{append_name}" if append_name else "x"
        yf = f"y_{append_name}" if append_name else "y"
        if dataset_name is None:
            for name, arr in ((xf, self.x), (yf, self.y)):
                if name in pos_grp:
                    del pos_grp[name]
                pos_grp.create_dataset(name, data=arr)
        else:
            dt = np.dtype([
                (xf, self.x.dtype),
                (yf, self.y.dtype)
                ])
            table = np.empty(self.x.size, dtype=dt)
            table[xf] = self.x
            table[yf] = self.y
            if dataset_name in pos_grp:
                del pos_grp[dataset_name]
            pos_grp.create_dataset(dataset_name, data=table)


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
    sto     : np.ndarray
    sti     : np.ndarray
    sbi     : np.ndarray
    sbo     : np.ndarray
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
                      bldata: h5py.Group,
                      pos_nom: Positions
                      ) -> dict:
        """Sort the blade data by grid coordinates."""
        ds = xr.Dataset(
            {
                'to'  : ('points', bldata["to_mean"][:]),
                'ti'  : ('points', bldata["ti_mean"][:]),
                'bi'  : ('points', bldata["bi_mean"][:]),
                'bo'  : ('points', bldata["bo_mean"][:]),
                'sto' : ('points', bldata["to_err"][:]),
                'sti' : ('points', bldata["ti_err"][:]),
                'sbi' : ('points', bldata["bi_err"][:]),
                'sbo' : ('points', bldata["bo_err"][:]),
            },
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

        keys   = ['to', 'ti', 'bi', 'bo', 'sto', 'sti', 'sbi', 'sbo']
        dsdict = {key: ds[key].values for key in keys}
        dsdict["pos_nom"] = Positions(x=xx, y=yy)
        return dsdict

    def to_hdf5(self,
                avg_grp : h5py.Group,
                ) -> int:
        """Write a Blades instance to an HDF5 group.
        
        Args:
            self : The Blades instance to write.
            avg_grp : The HDF5 group where the attributes will be stored.
        
        Returns:
            int: 0 if successful, 1 if an error occurred.

        """
        dt = np.dtype([
            ('x_nom',   'f8'), ('y_nom',   'f8'),
            ('to_mean', 'f8'), ('to_err',  'f8'),
            ('ti_mean', 'f8'), ('ti_err',  'f8'),
            ('bi_mean', 'f8'), ('bi_err',  'f8'),
            ('bo_mean', 'f8'), ('bo_err',  'f8'),
        ])
        n = self.to.size
        table = np.empty(n, dtype=dt)
        table['x_nom']   = self.pos_nom.x.ravel()
        table['y_nom']   = self.pos_nom.y.ravel()
        table['to_mean'] = self.to.ravel()
        table['to_err']  = self.sto.ravel()
        table['ti_mean'] = self.ti.ravel()
        table['ti_err']  = self.sti.ravel()
        table['bi_mean'] = self.bi.ravel()
        table['bi_err']  = self.sbi.ravel()
        table['bo_mean'] = self.bo.ravel()
        table['bo_err']  = self.sbo.ravel()

        if 'blade_averages' in avg_grp:
            del avg_grp['blade_averages']
        return avg_grp.create_dataset('blade_averages', data=table)


#
# Raw data structures.
#

@dataclass
class BladeAvgData:
    """Container for averaged blade current data and associated metadata.
    
    prm       : metadata parameters of the averaged data
    nom       : nominal positions at the XBPM site
    nom_shape : shape of the nominal positions grid
    blades    : averaged blade currents and their standard deviations
    """
    prm     : dict
    pos_nom : Positions
    blades  : Blades

    @classmethod
    def from_hdf5(cls,
                  avg_grp : "h5py.Group",
                  ) -> "BladeAvgData":
        """Create a BladeAvgData instance from an HDF5 group."""
        # Extract metadata attributes.
        prm     = {key : val for key, val in avg_grp.attrs.items()}
        blades  = Blades.from_hdf5(avg_grp)
        pos_nom = blades.pos_nom
        return cls(
            prm     = prm,
            pos_nom = pos_nom,
            blades  = blades
            )

    def to_hdf5(self,
                raw_grp        : h5py.Group,
                beamline       : str = ""
                ) -> int:
        """Write a BladeAvgData instance to an HDF5 group."""
        # Write blade average data.
        ds = self.blades.to_hdf5(raw_grp)

        # Write the metadata attributes to the "Blade averages" group.
        for key, val in self.prm.items():
            ds.attrs[key] = val
        ds.attrs["Description"] = Config.get_description(
            "Blade averages", 
            beamline
            )


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
        bld = blade.lower()

        required_fields = [
            'val',
            'range',
            'saturation'
            ]
        for fld in required_fields:
            if f"{bld}_{fld}" not in bld_grp.dtype.names:
                raise ValueError(
                    f" ERROR while reading BladeRawVals from HDF5 file:\n"
                    f" Missing '{bld}_{fld}' dataset in HDF5 group."
                )

        return cls(
            val        = bld_grp[f"{bld}_val"][:],
            range      = bld_grp[f"{bld}_range"][:],
            saturation = bld_grp[f"{bld}_saturation"][:]
        )

    def to_hdf5(self,
                raw_grp : h5py.Group,
                blname  : str = "",
                ) -> None:
        """Write a BladeRawVals instance to an HDF5 group."""
        try:
            bld = blname.lower()
            raw_grp.require_dataset(
                f"{bld}_val",
                data=self.val
                )
            raw_grp.require_dataset(
                f"{bld}_range",
                data=self.range
                )
            raw_grp.require_dataset(
                f"{bld}_saturation",
                data=self.saturation
                )
        except Exception as err:
            print(
                "### ERROR while writing 'BladeRawVals' instance to"
                f" {raw_grp.name} (HDF5 group):"
                f"\n {err}"
            )


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
                bpm_grp      : h5py.Group,
                beamline     : str = "",
                bumpnum      : int = None,
                ) -> int:
        """Write a BPMRawData instance to an HDF5 group."""
        try:
            bpm_grp.attrs["Description"] = Config.get_description(
                "BPM data",
                beamline = beamline,
                bumpnum  = bumpnum
                )   
            self.pos.to_hdf5(
                bpm_grp,
                append_name = "bpm",
                dataset_name = "bpm_data"
                )
        except Exception as err:
            print(
                "### ERROR while writing 'BPMRawData' instance to"
                f" {bpm_grp.name} (HDF5 group):"
                f"\n {err}"
            )


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
    def from_hdf5(cls, raw_grp: h5py.Group, beamline: str) -> "BladeRawData":
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
                raw_grp  : h5py.Group,
                beamline : str = ""
                ) -> int:
        """Write a BladeRawData instance to an HDF5 group."""
        raw_grp.attrs["Description"] = Config.get_description(
            "Blade data",
            beamline = beamline
            )
        bmap  = Config.BLADEMAP.get(beamline, None)
        names = ('TO', 'TI', 'BI', 'BO')

        # ('val', 'range', 'saturation')
        cols = [f.name for f in fields(BladeRawVals)]
        dt   = np.dtype([
            (f"{bmap[n]}_{c}", getattr(getattr(self, n), c).dtype)
            for n in names
            for c in cols
            ])
        table = np.empty(len(self.TO.val), dtype = dt)

        # Delegate the fill.
        for name in names:
            getattr(self, name).to_hdf5(table, bmap[name])

        if "blade_data" in raw_grp:
            del raw_grp["blade_data"]
        raw_grp.create_dataset("blade_data", data=table)


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
            bld = BladeRawData.from_hdf5(bump_grp["blade_data"], beamline)

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
            # Write sweep metadata to HDF5 group attributes.
            bump_grp.attrs["Description"] = Config.get_description(
                "Bumps",
                beamline = beamline,
                bumpnum = self.meta.get("Bump #")
                )
            swp_attrs = ['Angle x', 'Angle y',
                         'Bump pos x', 'Bump pos y',
                         'PV meter', 'SR current',
                         'Timestamp',
                         'gap/ph.'
                         ]
            for attr in swp_attrs:
                if attr in self.meta:
                    bump_grp.attrs[attr] = self.meta[attr]
            if bumpnum is not None:
                bump_grp.attrs["Bump #"] = bumpnum

            # BPM data.
            BPMRawData.to_hdf5(
                bump_grp["bpm_data"],
                self.bpm,
                beamline = beamline
                )

            # Blade raw data.
            BladeRawData.to_hdf5(
                bump_grp["blade_data"],
                self.blades,
                beamline = beamline
                )
        except Exception as err:
            print(
                "### ERROR while writing 'BumpRawData' instance to"
                f" {bump_grp.name} (HDF5 group):"
                f"\n {err}"
            )
            return 1

        return 0


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
            avg_grp = raw_grp["blade_averages"],
            )

        # Extract sweep data.
        bumps = {}
        for key, data in raw_grp.items():
            if key.startswith('bump_'):
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
                raw_grp  : h5py.Group,
                beamline : str,
                nbumps   : int,
                ) -> int:
        """Write beamline data to an HDF5 group."""
        try:
            # Group attributes.
            descr = Config.get_description("Raw data", beamline)
            meta = {
                "# bumps"          : nbumps,
                "Beamline"         : beamline,
                "Description"      : descr,
                "HDF5 updated on"  : (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ),
                "Source dist. (m)" : Config.XBPMDISTS[beamline]
                }
            raw_grp.attrs.update(meta)

            # Write blade averages if available.
            if self.blade_avg is not None:
                BladeAvgData.to_hdf5(raw_grp, self.blade_avg)

            # Write sweeps data.
            for num, bump in self.bumps.items():
                bump_grp = raw_grp.require_group(f"bump_{num}")
                BumpRawData.to_hdf5(
                    bump_grp,
                    bump.bpm,
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
        names = ds.attrs['names']
        vals  = {
            name: val
            for name, val in zip(names, ds['value'])
            }

        return cls(
            diff_h   = rms_grp['Diff. h'][()],
            diff_v   = rms_grp['Diff. v'][()],
            diff_tot = rms_grp['Diff. tot'][()],
            min_h    = vals['min_h'],
            mean_h   = vals['mean_h'],
            max_h    = vals['max_h'],
            min_v    = vals['min_v'],
            mean_v   = vals['mean_v'],
            max_v    = vals['max_v'],
            min_tot  = vals['min_tot'],
            max_tot  = vals['max_tot'],
            mean_tot = vals['mean_tot'],
        )

    def to_hdf5(self,
                rms_grp    : h5py.Group,
                roi_or_all : str = "",
                ) -> None:
        """Write RMSStatistics to an HDF5 group."""
        rms_grp.attrs["Description"] = Config.get_description(
            "RMS statistics", roi_or_all=roi_or_all
        )

        # 2-D difference arrays -> datasets
        for name, arr in (
                ('Diff. h',   self.diff_h),
                ('Diff. v',   self.diff_v),
                ('Diff. tot', self.diff_tot),
                ):
            # Safe guard for shape variability.
            if name in rms_grp:
                del rms_grp[name]
            rms_grp.create_dataset(name, data=arr)

        # Scalar resume values -> tidy table (one 'value' column).
        names  = [
            'min_h',   'mean_h',   'max_h',
            'min_v',   'mean_v',   'max_v',
            'min_tot', 'mean_tot', 'max_tot',
            ]
        values = [
            self.min_h,   self.mean_h,   self.max_h,
            self.min_v,   self.mean_v,   self.max_v,
            self.min_tot, self.mean_tot, self.max_tot
            ]

        dt = np.dtype([('value', 'f8')])
        ds = rms_grp.require_dataset(
            'summary',
            shape=(len(names),),
            dtype=dt
            )
        ds[...] = values
        ds.attrs['names'] = names


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
    pos_meas : Positions
    rms_diff : RMSGridStatistics

    @classmethod
    def from_hdf5(cls,
                  bpm_grp : "h5py.Group"
                  ) -> "BPMAnalysis":
        """Create a BPMAnalysis instance from an HDF5 group."""
        prm      = {
            key: val
            for key, val in bpm_grp.attrs.items()
            }

        pos_meas = Positions.from_hdf5(bpm_grp['pos_meas'])

        rms_diff = RMSGridStatistics.from_hdf5(bpm_grp['rms_diff'])

        return cls(
            prm      = prm,
            pos_meas = pos_meas,
            rms_diff = rms_diff
            )

    def to_hdf5(self,
                bpm_grp   : h5py.Group,
                ) -> None :
        # Write relevant attributes to group.
        bpm_grp["Description"] = Config.get_description(
            "BPM Analysis",
            beamline = self.prm.beamline
            )
        bpm_grp.attrs["Sector"]   = self.prm.sector
        bpm_grp.attrs["Beamline"] = self.prm.beamline

        # Write BPM positions.
        self.pos_meas.to_hdf5(
            bpm_grp.require_group('pos_meas')
            )

        # Write RMS grid statistics.
        self.rms_diff.to_hdf5(
            bpm_grp.require_group('rms_diff')
            )


@dataclass
class BladeMap:
    """Container for blade current data and associated metadata.

    Attributes:
        prm    : Metadata parameters of the blade map.
        pos    : Horizontal and vertical positions which define the grid of
            measurements.
        blades : Blades (measured currents)
    """
    prm    : dict
    pos    : Positions
    blades : Blades

    @classmethod
    def from_hdf5(cls,
                  bmp_grp : "h5py.Group"
                  ) -> "BladeMap":
        """Create a BladeMap instance from an HDF5 group."""
        prm    = {
            key: val
            for key, val in bmp_grp.attrs.items()
            }

        blades = Blades.from_hdf5(bmp_grp['blades'])

        pos    = Positions.from_hdf5(bmp_grp['positions'])
        return cls(
            prm    = prm,
            blades = blades,
            pos    = pos
            )

    def to_hdf5(self,
                bmp_grp : h5py.Group,
                ) -> None:
        """Write BladeMap to an HDF5 group."""
        # Write metadata attributes to the HDF5 group.
        for key, val in self.prm.items():
            bmp_grp.attrs[key] = val

        # Write position data.
        self.pos.to_hdf5(bmp_grp.require_group('positions'))

        # Write blade current data.
        self.blades.to_hdf5(bmp_grp.require_group('blades'))


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
                  scl_grp : "h5py.Group"
                  ) -> "Scales":
        """Create a Scales instance from an HDF5 group."""
        ds    = scl_grp['coefficients']
        names = ds.attrs['names']

        kwargs = {}
        for name, val, err in zip(names, ds['value'], ds['error']):
            if isinstance(name, bytes):   # defensive: in case stored as bytes
                name = name.decode()
            kwargs[name]       = val
            kwargs['s' + name] = err

        return cls(**kwargs)

    def to_hdf5(self,
                scl_grp : h5py.Group,
                ) -> None:
        """Write a Scales instance to an HDF5 group."""
        # Write all attributes of the Scales instance to the HDF5 group.
        for key, val in self.__dict__.items():
            scl_grp.attrs[key] = val

        names  = [
            'qx', 'kx', 'dx',
            'qy', 'ky', 'dy',
            ]
        values = [
            self.qx, self.kx, self.dx,
            self.qy, self.ky, self.dy,
            ]
        errors = [
            self.sqx, self.skx, self.sdx,
            self.sqy, self.sky, self.sdy,
            ]

        dt = np.dtype([
            ('value', 'f8'),
            ('error', 'f8')
            ])
        data = np.array(
            list(zip(values, errors)),
            dtype=dt
        )

        ds = scl_grp.require_dataset(
            'coefficients',
            shape=(len(values),),
            dtype=dt
        )
        ds[...] = data
        ds.attrs['names'] = names


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
        names = ds.attrs['names']
        vals  = {
            name : (val, error)
            for name, val, error in zip(names, ds['value'], ds['error'])
            }

        k, sk = vals['k']
        d, sd = vals['d']
        return cls(
            k=k, sk=sk, d=d, sd=sd,
            pos     = blf_grp['pos'][:],
            bld_raw = blf_grp['blade_raw'][:],
            bld_err = blf_grp['blade_err'][:],
            bld_fit = blf_grp['blade_fit'][:],
        )

    def to_hdf5(self,
                blf_grp : h5py.Group,
                ) -> None:
        """Write the BladeLineFit instance to an HDF5 group."""
        names = ['k', 'd']
        values = [self.k,  self.d]
        errors = [self.sk, self.sd]
        dt = np.dtype([
            ('value', 'f8'),
            ('error', 'f8')
        ])
        ds = blf_grp.require_dataset(
            'coefficients',
            shape=(len(values),),
            dtype=dt
        )

        ds[...] = np.array(
            list(zip(values, errors)),
            dtype=dt
        )
        ds.attrs['names'] = names

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
        """Create a CentralSweepLine instance from an HDF5 group.
        
        sln_grp: HDF5 group containing the central sweep data.
        dir: Direction of the central sweep
            ('h' for horizontal, 'v' for vertical)
        """
        # Check the direction order.
        if dir == 'h':
            ind, fix = 'x', 'y'
        elif dir == 'v':
            ind, fix = 'y', 'x'
        else:
            raise ValueError(" Invalid direction. Use 'h' or 'v'.")

        coeffs  = (
            sln_grp[f"{fix}_fit_coeffs"][:]
            if f"{fix}_fit_coeffs" in sln_grp
            else None
            )
        sigmas  = (
            sln_grp[f"{fix}_fit_sigmas"][:]
            if f"{fix}_fit_sigmas"
            in sln_grp else None
            )

        # Assemble the SweepLine instance.
        return cls(
            blades      = Blades.from_hdf5(sln_grp["blade_averages"]),
            pos_index   = sln_grp[f"{ind}_index"][:],
            pos_fixed   = sln_grp[f"{fix}_fix"][:],
            pos_calc    = sln_grp[f"{fix}_calc"][:],
            pos_fit     = sln_grp[f"{fix}_fit"][:],
            pos_fit_err = sln_grp[f"s_{fix}_fit"][:],
            coeffs      = coeffs,
            sigmas      = sigmas,
        )

    def to_hdf5(self,
                sln_grp   : "h5py.Group",
                direction : str = ""
                ) -> None:
        if direction == 'h':
            ind, fix = 'x', 'y'
        elif direction == 'v':
            ind, fix = 'y', 'x'
        else:
            raise ValueError("Invalid direction. Use 'h' or 'v'.")

        # Write the blades data to the HDF5 group.
        self.blades.to_hdf5(sln_grp)

        for name, arr in (
                (f"{ind}_index", self.pos_index),
                (f"{fix}_fix",   self.pos_fixed),
                (f"{fix}_calc",  self.pos_calc),
                (f"{fix}_fit",   self.pos_fit),
                (f"s_{fix}_fit", self.pos_fit_err),
                ):
            if name in sln_grp:
                del sln_grp[name]
            sln_grp.require_dataset(
                name,
                data = arr,
                dtype = arr.dtype,
                shape = arr.shape,
                )

        if self.coeffs is not None:
            sln_grp.require_dataset(
                f"{fix}_fit_coeffs",
                data = self.coeffs,
                dtype = self.coeffs.dtype,
                shape = self.coeffs.shape,
                )

        if self.sigmas is not None:
            sln_grp.require_dataset(
                f"{fix}_fit_sigmas",
                data  = self.sigmas,
                dtype = self.sigmas.dtype,
                shape = self.sigmas.shape,
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

    @classmethod
    def to_hdf5(cls,
                asc_grp: "h5py.Group",
                asc: "AllScales"
                ) -> None:
        """Write the AllScales instance to an HDF5 group."""
        pair_grp = asc_grp.create_group("pairwise")
        Scales.to_hdf5(
            pair_grp.create_group("std"),
            asc.pairw_std
            )
        Scales.to_hdf5(
            pair_grp.create_group("transf"),
            asc.pairw_trn
            )

        cross_grp = asc_grp.create_group("cross")
        Scales.to_hdf5(
            cross_grp.create_group("std"),
            asc.cross_std
            )
        Scales.to_hdf5(
            cross_grp.create_group("transf"),
            asc.cross_trn
            )


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
                mat     : "SuppressionMatrix",
                ) -> None:
        """Write the SuppressionMatrix instance to an HDF5 group."""
        for name in (
            'standard',
            'calculated',
            'stddev',
            'optimized'
            ):
            arr = getattr(mat, name)
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
                gl2r: "GL2RMatrix",
                ) -> None:
        """Write the GL2RMatrix instance to an HDF5 group."""
        for name in ("std", "calc"):
            if name in gl2r_grp:
                del gl2r_grp[name]
            gl2r_grp.create_dataset(
                name,
                data = getattr(self, name)
                )


@dataclass
class CalculatedPositions:
    """Container for analyzed positions and associated metadata.

    These data were calculated, but not correct by the suppression matrix.

        roi        : ROI slice used in the analysis

        pos_std    : positions calculated from standard formulae
        scale_std  : scaling factors for standard positions
        stat_std   : RMS statistics of the differences between standard
                 positions and nominal or BPM-calculated positions.

        pos_trn    : positions calculated from transformed formulae, through
                 the suppression matrix.
        scale_trn  : scaling factors for transformed positions
        stat_trn   : RMS statistics of the differences between transformed
                 positions and nominal or BPM-calculated positions.
    """
    roi       : ROISlice

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
        # Standard positions.
        gr      = pos_grp["xbpm_std"]
        xs      = gr["nom_x"][:]
        ys      = gr["nom_y"][:]
        pos_std = Positions(x=xs, y=ys)

        # Scales.
        scale_std = Scales.from_hdf5(pos_grp["scales_std"])
        stat_std  = RMSGridStatistics.from_hdf5(pos_grp["stat_std"])

        # Transformed positions.
        gr      = pos_grp["xbpm_transformed"]
        xt      = gr["trn_x"][:]
        yt      = gr["trn_y"][:]
        pos_trn = Positions(x=xt, y=yt)

        # Scales.
        scale_trn = Scales.from_hdf5(pos_grp["scales_trn"])
        stat_trn  = RMSGridStatistics.from_hdf5(pos_grp["stat_trn"])

        return cls(
            roi       = roislice,
            pos_std   = pos_std,
            pos_trn   = pos_trn,
            scale_std = scale_std,
            scale_trn = scale_trn,
            stat_std  = stat_std,
            stat_trn  = stat_trn
        )

    @classmethod
    def to_hdf5(cls,
                pos_grp  : "h5py.Group",
                calc_pos : "CalculatedPositions"
                ) -> None:
        """Write a CalculatedPositions instance to an HDF5 group."""
        # Standard positions.
        std_grp = pos_grp.create_group("xbpm_std")
        for name, arr in (
            ("nom_x", calc_pos.pos_std.x),
            ("nom_y", calc_pos.pos_std.y)
            ):
            if name in std_grp:
                del std_grp[name]
            std_grp.create_dataset(name, data=arr)

        # Scaling for the standard positions.
        Scales.to_hdf5(
            pos_grp.require_group("scales_std"),
            calc_pos.scale_std
            )

        # Statistics for the standard positions.
        RMSGridStatistics.to_hdf5(
            pos_grp.require_group("stat_std"),
            calc_pos.stat_std
            )

        # Transformed positions.
        trn_grp = pos_grp.create_group("xbpm_transformed")
        for name, arr in (
            ("trn_x", calc_pos.pos_trn.x),
            ("trn_y", calc_pos.pos_trn.y)
            ):
            if name in trn_grp:
                del trn_grp[name]
            trn_grp.create_dataset(name, data=arr)

        # Scaling for the transformed positions.
        Scales.to_hdf5(
            pos_grp.require_group("scales_trn"),
            calc_pos.scale_trn
            )

        # Statistics for the transformed positions.
        RMSGridStatistics.to_hdf5(
            pos_grp.require_group("stat_trn"),
            calc_pos.stat_trn
            )


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
                  pos_grp : "h5py.Group",
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

    @classmethod
    def to_hdf5(self,
                pos_grp    : "h5py.Group",
                xpositions : "XBPMPositions"
                ) -> None:
        """Write an XBPMPositions instance to an HDF5 group."""
        # Nominal positions.
        Positions.to_hdf5(
            pos_grp.create_group("nominal"),
            xpositions.nom
            )

        # Pairwise analyzed positions.
        CalculatedPositions.to_hdf5(
            pos_grp.require_group("pairwise"),
            xpositions.pairw,
            dtype = xpositions.pairw.coeffs.dtype,
            shape = xpositions.pairw.coeffs.shape,
            )

        # Cross-blade analyzed positions.
        CalculatedPositions.to_hdf5(
            pos_grp.require_group("cross"),
            xpositions.cross,
            dtype = xpositions.cross.coeffs.dtype,
            shape = xpositions.cross.coeffs.shape,
            )


@dataclass
class DataAnalysis:
    """Container for all data and analysis results.
 
    Classmethods may read analysis from HDF5 and compute analysis.

    prm          : beamline parameters
    bpm          : BPM calculated positions at XBPM site
    blademap     : blade map of positions
    bladecenter  : h and v blade center analyses
    xpositions   : Analyzed positions (raw and transformed)   
    centralsweep : Central sweep positions with errors
    scales       : Scaling factors
    supmat       : Suppression matrices
    """
    # Beamline, description and XBPM-source distance.
    beamline_prm  : BeamlinePrm       | None = None
    bpm           : BPMAnalysis       | None = None
    blademap      : BladeMap          | None = None
    bladecenter   : BCA_HV            | None = None
    xpositions    : XBPMPositions     | None = None
    centralsweeps : CentralSweeps     | None = None
    scales        : AllScales         | None = None
    supmat        : SuppressionMatrix | None = None
    gl2r          : GL2RMatrix        | None = None

    @classmethod
    def from_hdf5(cls, anl_grp: h5py.Group) -> "DataAnalysis":
        """Create a DataAnalysis instance from an HDF5 group."""
        # Extract parameters.
        bl_prm = BeamlinePrm.from_hdf5(anl_grp)

        # Extract BPM analysis data.
        bpm = BPMAnalysis.from_hdf5(anl_grp["bpm_analysis"])

        # Extract blade map.
        blademap = BladeMap.from_hdf5(anl_grp["blade_map"])

        # Extract blade center analysis for h/v directions.
        bladecenter = BCA_HV(
            h = BladeCenterAnalysis.from_hdf5(anl_grp["blade_center/h"]),
            v = BladeCenterAnalysis.from_hdf5(anl_grp["blade_center/v"])
        )

        # Extract other analysis data.
        xpositions = XBPMPositions.from_hdf5(
            anl_grp["xpositions"],
            roislice = bl_prm.roislice
            )

        # Central sweeps.
        centralsweeps = CentralSweeps.from_hdf5(anl_grp["central_sweeps"])

        # Extract scaling factors and suppression matrices.
        scales = AllScales.from_hdf5(anl_grp["scales"])

        # Extract suppression matrix.        
        supmat = SuppressionMatrix.from_hdf5(anl_grp["suppression_matrices"])

        # Extract GL2R matrix.
        gl2r = GL2RMatrix.from_hdf5(anl_grp["gl2r"])

        return cls(
            beamline_prm  = bl_prm,
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
                dan_grp : h5py.Group,
                datanl  : "DataAnalysis",
                ) -> None:
        """Serialize a DataAnalysis instance to an HDF5 group."""
        kwargs = {}
        kwargs['beamline_prm']  = datanl.beamline_prm
        kwargs['bpm']           = datanl.bpm
        kwargs['blademap']      = datanl.blademap
        kwargs['bladecenter']   = datanl.bladecenter
        kwargs['centralsweeps'] = datanl.centralsweeps
        kwargs['scales']        = datanl.scales
        kwargs['supmat']        = datanl.supmat
        kwargs['gl2r']          = datanl.gl2r
        kwargs['xpositions']    = datanl.xpositions

        try:
            dan_grp.require_group("data_analysis")
            datanl.to_hdf5(dan_grp["data_analysis"])
        except Exception as e:
            logging.error("Failed to create HDF5 group"
                          f" '{datanl.name}': {e}")

        # Store the keyword arguments as attributes in the HDF5 group.
        for key, value in kwargs.items():
            if value is not None:
                dan_grp.attrs[key] = value

        # Serialize each component to the HDF5 group.
        if datanl.bpm is not None:
            BPMAnalysis.to_hdf5(
                dan_grp.require_group("bpm_analysis"),
                datanl.bpm
                )

        if datanl.blademap is not None:
            BladeMap.to_hdf5(
                dan_grp.require_group("blademap"),
                datanl.blademap
                )

        if datanl.bladecenter is not None:
            BCA_HV.to_hdf5(
                dan_grp.require_group("bladecenter"),
                datanl.bladecenter
                )

        if datanl.centralsweeps is not None:
            CentralSweeps.to_hdf5(
                dan_grp.require_group("central_sweeps"),
                datanl.centralsweeps
                )

        if datanl.scales is not None:
            AllScales.to_hdf5(
                dan_grp.require_group("scales"),
                datanl.scales
                )

        if datanl.supmat is not None:
            SuppressionMatrix.to_hdf5(
                dan_grp.require_group("suppression_matrices"),
                datanl.supmat
                )

        if datanl.gl2r is not None:
            GL2RMatrix.to_hdf5(
                dan_grp.require_group("gl2r"),
                datanl.gl2r
                )

        if datanl.xpositions is not None:
            XBPMPositions.to_hdf5(
                dan_grp.require_group("xpositions"),
                datanl.xpositions
                )


@dataclass
class BeamlineData:
    """Encapsulates all data for a single beamline extracted from HDF5.
    
    A structure is created to contain the raw data measured (BeamlineRawData)
    and, if present, previous analysis results (DataAnalysis) stored in the
    HDF5 file. Metadata is stored in a parameter dictionary.
    """
    prm      : BeamlinePrm
    raw_data : BeamlineRawData
    analysis : DataAnalysis     = field(default=None)

    @classmethod
    def from_hdf5(cls,
                  bd_grp: "h5py.Group"
                  ) -> "BeamlineData":
        """Extract the beamline data from an HDF5 group."""
        # Data storage.
        kwargs = {}

        # Beamline parameters.
        kwargs["prm"] = BeamlinePrm.from_hdf5(bd_grp)
        beamline      = kwargs["prm"].beamline

        # Raw data.
        kwargs["raw_data"] = BeamlineRawData.from_hdf5(
            bd_grp["raw_data"],
            beamline
            )

        # Analysis data may be not present when data are imported.
        try:
            kwargs["analysis"] = DataAnalysis.from_hdf5(bd_grp["analysis"])
        except Exception as warn:
            logging.warning(
                "### while reading 'Data Analysis' from HDF5 file:"
                f"\n {warn}"
                "\n Instantiating DataAnalysis with default values.\n"
                )
            kwargs["analysis"] = DataAnalysis()
        return cls(**kwargs)

    def to_hdf5(self,
                bldata  : "BeamlineData",
                bld_grp : h5py.Group,
                ) -> None:
        """Serialize the BeamlineData instance to an HDF5 file.

        Args:
            prm (BeamlinePrm): Beamline parameters.
            raw_data (BeamlineRawData): Raw data for the beamline.
            analysis (DataAnalysis): Analysis data for the beamline.
.
        """
        # Create a temporary HDF5 file in memory.
        try:
            # Create a group for the beamline data.
            bl_group = bld_grp.require_group(bldata.prm.beamline)

            # Rewrite raw data only if explicitly requested.
            if bldata.raw_data is not None:
                raw_group = bl_group.require_group("raw_data")
                bldata.raw_data.to_hdf5(
                    raw_grp = raw_group,
                    bldata  = bldata.raw_data
                    )

            # Store analysis data if present.
            if bldata.analysis is not None:
                anl_group = bl_group.require_group("analysis")
                bldata.analysis.to_hdf5(
                    dan_grp = anl_group,
                    datanl  = bldata.analysis
                    )
        except Exception as err:
            logging.error(f"Failed to write BeamlineData to HDF5: {err}")
