"""Pickle directory backend for XBPM DataReader."""
import pickle  # noqa: S403
import os
import sys


def read_pickle_dir(path):
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
    for file in sfiles:
        with open(os.path.join(path, file), 'rb') as df:
            entry = pickle.load(df)  # noqa: S301
            # If entry is already a tuple (meta, grid, bpm_dict), use as is
            if isinstance(entry, (list, tuple)) and len(entry) == 3:
                rawdata.append(entry)
            # Otherwise, try to wrap in expected structure
            elif isinstance(entry, dict):
                rawdata.append((entry, None, {}))
            else:
                rawdata.append(({}, None, {}))
    return rawdata
