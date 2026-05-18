#!/bin/env python3

"""Get distance between BPMs.

This script uses a Sirius storage ring model to calculate the distance
between neighbour BPMs in a given sector, provided as an integer number 
as a command line argument.

>>> Run this script under "sirius' environment, to use
>>> pymodels and pyaccel libraries.
"""

import numpy as np
import pymodels
import pyaccel
import sys


def main():
    """Extract BPM positions from Sirius model and calculate distance."""
    try:
        sector = int(sys.argv[1])
    except Exception as err:
        print(f"\n ERROR: {err}")
        print("\n Please, type the sector number.\n")
        help("bpm_distance")
        sys.exit()

    print(f"\n Calculating distance for sector {sector}...")

    # Sirius model.
    modelsi = pymodels.si.create_accelerator()

    # Find BPM section.
    bpmidxar = np.array(pymodels.si.get_family_data(modelsi)['BPM']['index'])
    bpmidx = bpmidxar.ravel()
    fam_names_mia = pyaccel.lattice.find_indices(modelsi, 'fam_name', 'mia')
    fam_names_mib = pyaccel.lattice.find_indices(modelsi, 'fam_name', 'mib')
    fam_names_mip = pyaccel.lattice.find_indices(modelsi, 'fam_name', 'mip')

    fam_names = sorted(fam_names_mia + fam_names_mib + fam_names_mip)
    idx = fam_names[sector - 1]

    diffs = bpmidx - idx
    minidx = np.argmin(np.abs(diffs))
    bminidx = bpmidx[minidx]
    bminidxnext = bpmidx[minidx + 1]

    # Find the distance between BPMs for the position calculation.
    twiss, *_ = pyaccel.optics.calc_twiss(modelsi)
    bpmsdist = twiss.spos[bminidxnext] - twiss.spos[bminidx]
    print(f" Distance = {bpmsdist}")

    # print(f" bpm idx = {bpmidx}, \t idx = {idx}, minidx = {minidx}")
    # print(f" bpm idx = {diffs}, \t idx = {idx}")
    # print(f" fam_names = {fam_names}")
    # print(f" bpm_min_idx  = {bminidx};   next =  {bminidxnext} ")
    # print(f" shape bpmidx = {bpmidx.shape}")


if __name__ == "__main__":
    main()
