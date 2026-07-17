"""Pickle directory backend for XBPM DataReader."""

from datetime import datetime
import pickle  # noqa: S403
import os
import sys


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

            # DEBUG
            # print(f" Reading file: {file}"
            #     f"\t Timestamp: {ts}"
            #     f"\t Rawdata ype: {type(rawdata)}")
            # for ii, rd in enumerate(rawdata):
            #     print(f"  rawdata[{ii}]: {type(rd[2])}")
            # DEBUG

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
