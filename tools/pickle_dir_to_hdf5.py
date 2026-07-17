#!/usr/bin/env python3

import argparse
from datetime import datetime
import h5py
import numpy as np
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

from xbpm_bumps.core.reader_pickle import (    #noqa: E402
    read_pickle_dir, extract_beamlines
    )
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
        pos_x = record[2]['posx']
        pos_y = record[2]['posy']

        # BPM registered positions (orbx, orby) for the current bump.
        orbx = record[2]['orbx']
        orby = record[2]['orby']
        bpm_data[jj] = {
            'bpm_x' : orbx,
            'bpm_y' : orby
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
            'Description' : description,
            'Timestamp'   : timestamp,
            'pv_meter'    : pv_meter,
            'angle_x'     : angle_x,
            'angle_y'     : angle_y,
            'bump_pos_x'  : pos_x,
            'bump_pos_y'  : pos_y,
            'SR current'  : current,
            }

        # Gaps / phases of the IDs. It is originally a dict in the form
        # {'cnb' : 80.0, 'cat' : 80.0, 'mnc' : 0.1481, etc}
        id_gaps = record[1]
        gap_phase = {
            f"gap/ph. {key}" : value for key, value in id_gaps.items()
            }
        rawmeta[jj].update(gap_phase)

    return rawmeta, bpm_data, blade_data, avgblades


def _write_group_data(groupname, group: h5py.Group, data: dict) -> None:
    """Write datasets to the given HDF5 group."""
    # Convert data dictionary to structured array for HDF5 export.
    names  = list(data.keys())
    arrays = list(data.values())
    # automatic dtype deduction
    rec_data = np.rec.fromarrays(arrays, names=names)
    dset = group.create_dataset(groupname, data=rec_data)
    return dset


def export_rawdata_to_hdf5(rawmeta: dict,
                           bpm_data: dict,
                           blade_data: dict,
                           avgblades: dict,
                           outname: str,
                           append=False) -> None:
    """Export parsed rawdata to HDF5 file.

    Args:
        bpm_data: dict, BPM data for the selected beamline.
        blade_data: dict, Parsed blade data for the selected beamline.
        avgblades: dict, Averaged blade data for the selected beamline.
        outname: str, Output HDF5 file name.
    """
    # Create attribute lists for HDF5 export.
    beamlines = list(rawmeta.keys())

    w_or_a = "a" if append else "w"
    with h5py.File(outname, w_or_a) as h5file:
        for beamline in beamlines:
            bladedata = blade_data[beamline]
            bpmdata   = bpm_data[beamline]
            avgdata   = avgblades[beamline]
            metadata  = rawmeta[beamline]

            # One raw data set per beamline.
            mastergrp = h5file.create_group(f"{beamline}")
            mastergrp.attrs['Description'] = (
                f"Raw data for beamline {beamline}"
                " from XBPM-bumps experiments."
                )
            mastergrp.attrs['Beamline'] = beamline
            mastergrp.attrs['HDF5 created on'] = datetime.now().isoformat()
            mastergrp.attrs['# sweeps'] = len(metadata)
            mastergrp.attrs['Experience start'] = metadata[1].get('Timestamp',
                                                                  "N/A")

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
            columns = list(next(iter(avgdata.values())).keys()) 
            rows_sorted = [avgdata[k] for k in sorted(avgdata)]
            # Extract column arrays.
            col_arrays = {
                col: np.array([
                    row[col]
                    for row in rows_sorted])
                    for col in columns
                    }
            # Create a structured array for HDF5 export.
            structured_array = np.rec.fromarrays(
                [col_arrays[col] for col in columns],
                names=columns
                )
            dset = mastergrp.create_dataset(datasetname,
                                            data=structured_array)
            # Select some metadata attributes to define the averaged data.
            avg_meta = {
                'pv_meter'   : metadata[1]['pv_meter'],
                'SR current' : metadata[1]['SR current'],
                }
            avg_meta.update({k: v for k, v in metadata[1].items()
                             if k.startswith('gap')})
            dset.attrs.update(avg_meta)
            dset.attrs['Description'] = (
                "Averaged blade measurements for each sweep sampling "
                f"from {beamline}, with standard deviations."
                )


def main() -> None:
    """Main function to convert pickle directory to HDF5."""
    # Get directory from command line.
    args = cmd_line()

    # Read pickle files in the directory.
    rawdata = read_pickle_dir(args.dir)

    # Get available beamlines from the rawdata.
    beamlines = extract_beamlines(rawdata)

    # Choose beamline to process. Default is to process all.
    beamlines = pick_beamline(beamlines)

    print(f"\n>>> Selected beamline(s): {', '.join(beamlines)}")

    # DEBUG
    # print(f"\n##### RAWDATA Meta: type = {type(rawdata)},"
    #       f" length = {len(rawdata)}")
    # END DEBUG
    print(f"\n### Found {len(rawdata)} pickle files in directory \n### "
          f"'{args.dir}'")
    
    # Extract data for each selected beamline and store in a dictionary.
    # Data is extracted as-is and averaged data is also computed.
    raw_meta, bpm_data, blade_data, avgblade = {}, {}, {}, {}
    for bline in beamlines:
        (raw_meta[bline],   bpm_data[bline],
         blade_data[bline], avgblade[bline]) = parse_rawdata(rawdata, bline)

    # Export to HDF5.
    outfile = args.dir + "_test" + ".h5"
    print(f"\n### Exporting data to HDF5 file:\n '{outfile}'")
    # Check whether HDF5 file already exists and ask for confirmation
    # to overwrite.
    if os.path.exists(outfile):
        append = False
        overwrite = input(f"\n### WARNING: File '{outfile}' already exists."
                          " Overwrite? [Y/n]: ")
        if overwrite.lower() == 'n':
            print("\n>>> Appending data to existing file.")
            append = True

    export_rawdata_to_hdf5(raw_meta, bpm_data, blade_data, avgblade,
                           outfile, append=append)
    print("###  done.")


if __name__ == "__main__":
    main()