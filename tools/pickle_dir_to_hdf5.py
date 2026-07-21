#!/usr/bin/env python3

import argparse
from datetime import datetime
import h5py
import numpy as np
import os
import pickle  # noqa: S403
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

from xbpm_bumps.core.config import Config      #noqa: E402

HELP_DESCRIPTION = (
"""This program extract data from pickle files created by
XBPM-bumps experiments. The pickle files must be provided
in a directory.

The data is parsed and an HDF5 is created with the
same name as the directory, containing the data in a
structured format.
"""
)


def cmd_line() -> argparse.Namespace:
    """Parse command-line arguments using argparse.

    Args:
        argv: List of command-line arguments. If None, uses sys.argv[1:].
    """
    parser = argparse.ArgumentParser(
        prog="pickle_dir_to_hdf5",
        description="Convert XBPM data from pickle files to HDF5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"{HELP_DESCRIPTION}\n")

    # Boolean flags
    parser.add_argument(
        '-d', '--dir', type=str, default='./', required=True,
        help='Directory containing pickle files.'
    )
    return parser.parse_args(sys.argv[1:])


def pick_beamline(beamlines: list) -> str:
    """Prompt user to select a beamline from the list.

    Default is to select all beamlines if no input is provided.

    Args:
        beamlines: List of unique beamline names.

    Returns:
        beamline: str, Selected beamline name.
    """
    print(f"### Unique beamlines found: {', '.join(beamlines)}")
    for i, beamline in enumerate(beamlines, start=1):
        print(f"{i}. {beamline}")
    print(f"{i+1}. All beamlines (default)")
    while True:
        try:
            choice = input("Select a beamline by number: ")

            # If default.
            if choice == "":
                return beamlines

            # Otherwise.
            choice = int(choice)
            if 1 <= choice <= len(beamlines):
                break
            elif choice == len(beamlines) + 1:
                return beamlines
            else:
                print("\nInvalid choice. Please select a number"
                      f" between 1 and {len(beamlines) + 1}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return [beamlines[choice - 1]]


def _get_timestamp_from_filename(filename: str) -> str:
    ts = filename.strip('.pickle').split('_')[-1]
    ts = datetime.fromtimestamp(float(ts)).strftime('%Y-%m-%d %H:%M:%S')
    return ts


def read_pickle_dir(path: str) -> list:
    """Read pickle directory and return rawdata as a list of tuples.

    Args:
        path: Directory path.

    Returns:
        rawdata: List of (meta, grid, bpm_dict) tuples.
    """
    # List pickle files
    allfiles = os.listdir(path)
    picklefiles = [pf for pf in allfiles if pf.endswith("pickle")]
    if len(picklefiles) == 0:
        print(f"No pickle files found in directory '{path}'. Aborting.")
        sys.exit(0)
    sfiles = sorted(picklefiles, key=lambda name: name.split('_')[-1])

    rawdata = []

    # Run through each pickle file and load the data.
    for file in sfiles:
        with open(os.path.join(path, file), 'rb') as df:
            filedata = pickle.load(df)  # noqa: S301

            # If entry is already a tuple (meta, grid, bpm_dict), use as is
            if isinstance(filedata, (list, tuple)) and len(filedata) == 3:
                filedata[2]['timestamp'] = _get_timestamp_from_filename(file)
                rawdata.append(filedata)

            elif isinstance(filedata, dict):
                # Otherwise, try to wrap in expected structure
                # Add timestamp to meta data in the appropriate dict.
                if 'current' in filedata.keys():
                    filedata['timestamp'] = _get_timestamp_from_filename(file)

                rawdata.append((filedata, None, {}))

            else:
                rawdata.append(({}, None, {}))

    return rawdata


def extract_beamlines(rawdata: list) -> list:
    """Extract unique beamlines from pickle raw data headers."""
    try:
        beamlines = set()
        for record in rawdata or []:
            if not (isinstance(record, (list, tuple)) and len(record) > 0):
                continue
            header = record[0]
            if isinstance(header, dict):
                for key in header.keys():
                    if isinstance(key, str):
                        beamlines.add(key)
            elif isinstance(header, (list, tuple)):
                for item in header:
                    if isinstance(item, str):
                        beamlines.add(item)
        return sorted(list(beamlines))
    except Exception:
        return []


def _reverse_blade_map(beamline: str) -> dict:
    """Return the blade mapping for the given beamline.
    
    Reverse blade mapping from TO → A to A → TO etc.

    Args:
        beamline: str, Selected beamline name.

    Returns:
        revmap: dict, Mapping of blade names to their corresponding PVs.
    """
    bmap = Config.BLADEMAP.get(beamline, {})
    return {v: k for k, v in bmap.items()}


def _blade_average(blade: list) -> tuple:
    """Calculate the average of blades' values for current beamline.
    
    Values are converted to Amperes using Config.AMPSUB (unit map).

    Args:
        blade: list of (value, unit) tuples for each blade.

    Returns:
        avg  : float, Average value of the blades.
        std  : float, Standard deviation of the blades.
        vals : np.ndarray, Array of processed blade values.
    """
    vals = np.array([
        vv * Config.AMPSUB[un] for vv, un in blade
    ])
    return np.average(vals), np.std(vals), vals


def extract_and_average_blade_data(rawdata0: dict, beamline: str) -> dict:
    """Extract and average blade data for the given beamline.
    
    'rawdata0' contains raw blade data for the selected beamline for a single grid point. This script extracts the data for the given beamline and computes the average and standard deviation for each blade for that point.

    Args:
        rawdata0: dict, Raw data for a single grid point.
        beamline: str, Selected beamline name.

    Returns:
        rawblades: dict, Raw blade data for the selected beamline.
        avgblades: dict, Average and std dev for each blade.
    """
    rawblades = {}
    avgblades = {}
    # Reversed blade map (get ).
    revmap = _reverse_blade_map(beamline)

    # Run through each PV.
    pvs    = ['A', 'B', 'C', 'D']
    for pv in pvs:
        # Get average, std dev, and raw values for the blade.
        avg, std, vals = _blade_average(rawdata0[pv + "_val"])

        # Store raw values: _val, _range, _saturation.
        rawblades[pv + "_val"]   = np.array(vals)
        rawblades[pv + "_range"] = np.array(
            rawdata0.get(pv + "_range", None)
            )
        rawblades[pv + "_saturation"] = np.array(
            rawdata0.get(pv + "_saturation", None)
            )

        # Store each avg / std. dev. value.
        bl     = revmap.get(pv)
        blname = f"{bl.lower()}_"
        avgblades[blname + "mean"] = avg
        avgblades[blname + "err"]  = std

    return rawblades, avgblades


def parse_rawdata(rawdata: list, beamline: str) -> list:
    """Parse rawdata to extract beamline-specific data.

    Args:
        rawdata: List of (meta, grid, bpm_dict) tuples for the selected beamline.

    Returns:
        blade_data: List of (meta, grid, bpm_dict) tuples for the selected beamline.
    """
    # Electronics PV.
    pv_meter = rawdata[0][0][beamline].get('prefix', None)
    xbpmdist = Config.XBPMDISTS.get(beamline, 1.0)

    # Data dictionaries: raw parsed data and averaged blade data.
    bpm_data   = {}
    blade_data = {}
    avgblades  = {}
    rawmeta    = {}

    # Run through each sweep record.
    for ii, record in enumerate(rawdata):
        # Define index for experience numbering.
        jj = ii + 1

        # Storage ring current.
        current = record[2]['current']

        # Time.
        timestamp = record[2].get('timestamp', "N/A")

        # Nominal positions of the bumps by changing the beam angles.
        angle_x = record[2]['agx']
        angle_y = record[2]['agy']

        # Alternative bumping method, by displacing the beam with defined
        # positions. If bumps are made by angle, these are tipically zero.
        pos_x   = record[2]['posx']
        pos_y   = record[2]['posy']

        # BPM registered positions (orbx, orby) for the current bump.
        orbx    = record[2]['orbx']
        orby    = record[2]['orby']
        bpm_data[jj] = {
            'x_bpm' : orbx,
            'y_bpm' : orby
            }

        # Extract and average blade data for the current beamline.
        rawblades, avgbld = extract_and_average_blade_data(
            record[0][beamline], beamline
            )

        # Store raw blade data.
        blade_data[jj] = rawblades

        # Store averaged blade data.
        avgblades.setdefault(jj, {})
        avgblades[jj]['x_nom'] = angle_x * xbpmdist
        avgblades[jj]['y_nom'] = angle_y * xbpmdist
        avgblades[jj].update(avgbld)

        # Create metadata dictionary.
        # Description.
        description = (
            f"Raw data for sweep {jj:04d} from XBPM-bumps experiments."
            )
        rawmeta[jj] = {
            'Description'     : description,
            'Timestamp'       : timestamp,
            'PV meter'        : pv_meter,
            'Angle x [mrad]'  : angle_x,
            'Angle y [mrad]'  : angle_y,
            'Bump pos x'      : pos_x,
            'Bump pos y'      : pos_y,
            'SR current [mA]' : current,
            }

        # Gaps / phases of the IDs. It is originally a dict in the form
        # {'cnb' : 80.0, 'cat' : 80.0, 'mnc' : 0.1481, etc}
        id_gaps = record[1]
        gap_phase = {
            f"gap/ph. {key}" : value for key, value in id_gaps.items()
            }
        rawmeta[jj].update(gap_phase)

    return (rawmeta, bpm_data, blade_data, avgblades)


def _write_group_data(groupname, group: h5py.Group, data: dict) -> None:
    """Write datasets to the given HDF5 group."""
    # Convert data dictionary to structured array for HDF5 export.
    names  = list(data.keys())
    arrays = list(data.values())
    # automatic dtype deduction
    rec_data = np.rec.fromarrays(arrays, names=names)
    dset = group.create_dataset(groupname, data=rec_data)
    return dset


def export_rawdata_to_hdf5(dataset: tuple, outname: str, append=False) -> None:
    """Export parsed rawdata to HDF5 file.

    Args:
        dataset    : tuple, Containing rawmeta, bpm_data, blade_data, and
                     avgblades.
        outname    : str,  Output HDF5 file name.
        append     : bool, Whether to append to the existing HDF5 file.
    """
    # Create attribute lists for HDF5 export.
    beamlines = list(dataset.keys())

    w_or_a = "a" if append else "w"
    with h5py.File(outname, w_or_a) as h5file:
        for beamline in beamlines:
            metadata  = dataset[beamline][0]
            bpmdata   = dataset[beamline][1]
            bladedata = dataset[beamline][2]
            avgblades = dataset[beamline][3]

            # One raw data set per beamline.
            mastergrp = h5file.create_group(f"{beamline}")

            # Set metadata attributes for the beamline group.
            descr = (f"Data for {beamline} beamline from XBPM experiments.")
            meta = {
                "# sweeps"         : len(metadata),
                "Beamline"         : beamline,
                "Description"      : descr,
                "Experience start" : metadata[1].get('Timestamp', "N/A"),
                "HDF5 created on"  : datetime.now().isoformat(),
                }
            mastergrp.attrs.update(meta)

            # Write raw data.
            for ii in metadata.keys():
                # Create a subgroup for each sweep and write its attributes.
                grp_sweep = mastergrp.create_group(f"sweep_{ii:04d}")
                grp_sweep.attrs.update(metadata[ii])

                # Subgroup for BPM data.
                datasetname = "bpm_data"
                dset = _write_group_data(datasetname, grp_sweep, bpmdata[ii])
                dset.attrs['Description'] =  (
                    f"All BPM registered positions for the sweep {ii:04d}."
                    )

                # Subgroup for raw blade data.
                datasetname = "raw_data"
                dset = _write_group_data(datasetname, grp_sweep, bladedata[ii])
                dset.attrs['Description'] = (
                    f"Raw blade measurements for the sweep {ii:04d}."
                    )

            # Write averaged data.
            datasetname = "blade_averages"
            
            # Data must be put into a table structure.
            # Extract column names from the first entry of avgdata,
            # and sort rows by sweep index.
            columns     = list(next(iter(avgblades.values())).keys()) 
            rows_sorted = [avgblades[k] for k in sorted(avgblades)]

            # Extract column arrays.
            col_arrays  = {
                col: np.array([
                    row[col]
                    for row in rows_sorted])
                    for col in columns
                    }

            # Create a structured array for HDF5 export.
            struct_array = np.rec.fromarrays(
                [col_arrays[col] for col in columns],
                names=columns
                )
            dset = mastergrp.create_dataset(datasetname, data=struct_array)

            # Select some metadata attributes to define the averaged data.
            avg_meta = {
                'PV meter'   : metadata[1]['PV meter'],
                'SR current' : metadata[1]['SR current'],
                }
            avg_meta.update({k: v for k, v in metadata[1].items()
                             if k.startswith('gap')})
            dset.attrs.update(avg_meta)
            dset.attrs['Description'] = (
                "Averaged blade measurements for each sweep sampling "
                f"from {beamline}, with standard deviations."
                )


def hdf5_handler(filepath: str) -> tuple:
    """Determine HDF5 output file name and handle overwrite/append.
    
    Args:
        filepath: str,  Input directory path.
    
    Returns:
        outfile : str,  Output HDF5 file name.
        append  : bool, Whether to append to existing file or overwrite.
    """
    # Set file name.
    outfile = filepath + "_test" + ".h5"
    print(f"\n### Exporting data to HDF5 file:\n '{outfile}'")

    # Check whether HDF5 file already exists and ask for confirmation
    # to overwrite.
    append = False
    if os.path.exists(outfile):
        overwrite = input(f"\n### WARNING: File '{outfile}' already exists."
                          " Overwrite? [Y/n]: ")
        if overwrite.lower() == 'n':
            print("\n>>> Appending data to existing file.")
            append = True
        else:
            print("\n>>> Overwriting existing file.")

    return outfile, append


def main() -> None:
    """Main function to convert pickle directory to HDF5."""
    # Get directory from command line.
    args = cmd_line()

    # Read pickle files in the directory.
    rawdata = read_pickle_dir(args.dir)

    # Get available beamlines from the rawdata and choose each one to process.
    # Default is to process all.
    beamlines = extract_beamlines(rawdata)
    beamlines = pick_beamline(beamlines)
    print(f"\n>>> Selected beamline(s): {', '.join(beamlines)}")
    print(f"\n### Found {len(rawdata)} pickle files in directory \n### "
          f"'{args.dir}'")
    
    # Extract data for each selected beamline and store in a dictionary.
    # Data is extracted as-is and averaged data is also computed.
    dataset = {}
    for bline in beamlines:
        dataset[bline] = parse_rawdata(rawdata, bline)

    # Set file name and handle overwrite/append.
    outfile, append = hdf5_handler(args.dir)

    export_rawdata_to_hdf5(dataset, outfile, append)
    print("###  done.")


if __name__ == "__main__":
    main()