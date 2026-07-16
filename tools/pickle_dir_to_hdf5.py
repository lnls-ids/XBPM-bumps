#!/usr/bin/env python3

import argparse
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

from xbpm_bumps.core.reader_pickle import read_pickle_dir, extract_beamlines
from xbpm_bumps.core.config import Config

HELP_DESCRIPTION = (
"""This program extract data from pickle files created by
XBPM-bumps experiments. The pickle files must be provided
in a directory.

The data is parsed and an HDF5 is created with the
same name as the directory, containing the data in a
structured format.
"""
)

def parse_args() -> argparse.Namespace:
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
    print(f"Unique beamlines found: {', '.join(beamlines)}")
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


def parameters_read(rawdata: list, beamline: str) -> dict:
    """Extract parameters for the selected beamline from rawdata.

    Args:
        rawdata: List of (meta, grid, bpm_dict) tuples.
        beamline: str, Selected beamline name.

    Returns:
        params: dict, Extracted parameters for the selected beamline.
    """
    prm = {}
    prm['xbpmdist'] = Config.XBPMDISTS.get(beamline, 1.0)

    return prm


def blade_map(beamline: str) -> dict:
    """Return the blade mapping for the given beamline."""
    bmap = Config.BLADEMAP.get(beamline, {})
    return {v: k for k, v in bmap.items()}


def _blade_average(blade: list, beamline: str) -> tuple:
    """Calculate the average of blades' values for current beamline."""
    if beamline in ["MGN", "MNC"]:
        return np.average(blade), np.std(blade), blade

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
    pvs    = ['A', 'B', 'C', 'D']
    # Reversed blade map.
    revmap = blade_map(beamline)

    # Run through each PV.
    for pv in pvs:
        # Get average, std dev, and raw values for the blade.
        avg, std, vals = _blade_average(rawdata0[pv + "_val"], beamline[:3])

        # Store each avg / std. dev. value.
        bl     = revmap.get(pv)
        blname = f"{bl.lower()}_"
        avgblades[blname + "mean"] = avg
        avgblades[blname + "err"]  = std

        # Store raw values: _val, _range, _saturation.
        rawblades[pv + "_val"]        = np.array(vals)
        rawblades[pv + "_range"]      = np.array(rawdata0[pv + "_range"])
        rawblades[pv + "_saturation"] = np.array(rawdata0[pv + "_saturation"])

    return rawblades, avgblades


def parse_rawdata(rawdata: list, beamline: str) -> list:
    """Parse rawdata to extract beamline-specific data.

    Args:
        rawdata: List of (meta, grid, bpm_dict) tuples.

    Returns:
        parsed_data: List of (meta, grid, bpm_dict) tuples for the selected beamline.
    """
    parsed_data = {}
    avgblades = {}
    for ii, record in enumerate(rawdata):
        rawblades, avgbld = extract_and_average_blade_data(
            record[0][beamline], beamline
            )

        # Gaps / phases of the IDs
        id_gaps = record[1]
        # Storage ring current.
        current = record[2]['current']
        agx     = record[2]['agx']
        agy     = record[2]['agy']
        orbx    = record[2]['orbx']
        orby    = record[2]['orby']

        avgblades[ii] = {}
        avgblades[ii].update(avgbld)
        avgblades[ii]['x_nom'] = agx
        avgblades[ii]['y_nom'] = agy

        parsed_data[(agx, agy)] = {
            'id_gaps' : id_gaps,
            'current' : current,
            'bpm_data' : {
                'orbx' : orbx,
                'orby' : orby
                },
            'rawblades' : rawblades,
            }
    return parsed_data, avgblades


def export_rawdata_to_hdf5(parsed_data: dict,
                           avgblades: dict,
                           outname: str) -> None:
    """Export parsed rawdata to HDF5 file.

    Args:
        parsed_data: dict, Parsed data for the selected beamline.
        outname: str, Output HDF5 file name.
    """
    import h5py

    with h5py.File(outname, 'w') as h5file:
        mstrgrp = h5file.create_group('raw_data')
        for ii, ((agx, agy), data) in enumerate(parsed_data.items()):
            group_name = f"sweep_{ii}"
            grp = mstrgrp.create_group(group_name)
            grp.attrs['id_gaps'] = data['id_gaps']
            grp.attrs['current'] = data['current']
            grp.create_dataset('orbx', data=data['bpm_data']['orbx'])
            grp.create_dataset('orby', data=data['bpm_data']['orby'])
            for blade, values in data['rawblades'].items():
                grp.create_dataset(f'raw_{blade}', data=values)
        for beamline, vals in avgblades.items():
            group_name = f"avg_{beamline}"
            grpavg = h5file.create_group(group_name)
            grpavg.attrs['description'] = (
                "Averaged blade measurements for all sweeps "
                "(full table/grid, matches original "
                f"{beamline} measurements)."
                )
            grpavg.attrs['HDF5 type'] = "COMPOUND"
            
            for key in enumerate(vals.keys()):
                grpavg.create_dataset(f"{key}", data=vals[key])


def main() -> None:
    """Main function to convert pickle directory to HDF5."""
    args      = parse_args()
    rawdata   = read_pickle_dir(args.dir)
    beamlines = extract_beamlines(rawdata)
    beamlines = pick_beamline(beamlines)

    print(f"\nSelected beamline: {', '.join(beamlines)}\n")
    print(f"\n##### RAWDATA Meta: type = {type(rawdata)},"
          f" length = {len(rawdata)}")
    
    parsed_data = {}
    avgblades = {}
    for bl in beamlines:
        parsed_data[bl], avgblades[bl] = parse_rawdata(rawdata, bl)


    print(f"\n##### Meta: type = {type(parsed_data)},"
          f" length = {len(parsed_data)}")
    for k, pd in parsed_data.items():
        print(f"\n>>>>> key = {k},\t type = {type(pd)},\t length = {len(pd)}")
    print(f"Found {len(rawdata)} pickle files in directory '{args.dir}'.")

    outname = args.dir + ".h5"
    export_rawdata_to_hdf5(parsed_data, avgblades, outname)

if __name__ == "__main__":
    main()