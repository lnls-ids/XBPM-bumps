"""Text file reader for XBPM-bumps."""

import re
import logging


def is_empty(line):
    """Check if a line is empty or contains only whitespace."""
    return bool(re.match(r'^\s*$', line))


def is_header(line):
    """Check if a line is a header line starting with '#'."""
    return bool(re.match(r'^\s*#', line))


def parse_header_line(line, header_meta, beamline_ref):
    """Parse a header line and update metadata dictionary."""
    cline = line.strip('#')
    kprm, vprm = cline.split(':', 1)
    key = kprm.strip().lower()
    val = vprm.strip()
    if re.search(r'beamline', key, re.IGNORECASE):
        beamline_ref[0] = val
    header_meta[key] = val


def parse_data_line(line, data):
    """Parse a data line and update data dictionary."""
    parts = line.strip().split()
    if len(parts) >= 2:
        try:
            x, y = float(parts[0]), float(parts[1])
            data[(x, y)] = [float(v) for v in parts[2:]]
        except Exception as e:
            logging.warning("Failed to parse data line '%s': %s",
                            line.strip(), e)


def read_text_file(path):
    """Read text file and return rawdata as a list of tuples.

    Tuples are meta, grid, bpm_dict

    Args:
        path: Path to text file.

    Returns:
        rawdata: List of (meta, grid, bpm_dict) tuples.
    """
    data = {}
    header_meta = {}
    beamline_ref = [None]

    with open(path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if is_empty(line):
                continue
            if is_header(line):
                parse_header_line(line, header_meta, beamline_ref)
                continue
            parse_data_line(line, data)
    if beamline_ref[0]:
        header_meta['beamline'] = beamline_ref[0]

    rawdata = [(header_meta, data, {})] if header_meta else []
    return rawdata
