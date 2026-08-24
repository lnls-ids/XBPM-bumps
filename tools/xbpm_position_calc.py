#!/usr/bin/env python3
"""Fetch data from any XBPMs position calculation PVs and plot them"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from siriuspy.clientarch import Time, PVDataSet
from datetime import datetime
from epics import caget

os.environ["EPICS_CA_ADDR_LIST"] = "10.0.38.59:62000 10.30.13.22 10.30.14.19"

BLADES = ["AmplA-Mon", "AmplB-Mon", "AmplC-Mon", "AmplD-Mon"]
ADJUST = ["PosKx-RB","PosXOffset-RB","PosKy-RB", "PosYOffset-RB"]
MATRIX = "SupMat-RB"

BEAMLINES = {
    "SI-06SBFE": "CNB",
    "SI-07SPFE": "CAT",
    "SI-09SAFE": "MNC",
    "SI-10BCFE": "MGN",
}

def _get_beamline_name(prefix):
    curr_bl = str()
    for bl in BEAMLINES:
        if prefix == bl:
            curr_bl = BEAMLINES[bl]
            return curr_bl
            

def set_pv_names(prefix, xbpm_number):
    """Set PV names for the specified XBPM number with the given prefix"""
    if xbpm_number not in [1, 2]:
        raise ValueError("XBPM number must be either 1 or 2.")
    
    xbpm_prefix = f"{prefix}:DI-PBPM-{xbpm_number}"

    pvnames = {
        "BLADES": [],
        "ADJUST": [],
        "MATRIX": [],
    }

    pvnames["BLADES"] = [f"{xbpm_prefix}:{blade}" for blade in BLADES]
    pvnames["ADJUST"] = [f"{xbpm_prefix}:{adj}" for adj in ADJUST]
    pvnames["MATRIX"] = xbpm_prefix+":"+MATRIX

    return pvnames


def _parse_dates(dt):
    """Parse dates from date string."""
    year, month, day = dt.year, dt.month, dt.day
    hour, minute = dt.hour, dt.minute
    return [year, month, day, hour, minute]


def get_pvdata(pvnames, initdate, enddate, timeout):
    """Fetch data from the EPICS archiver for the given PV names and time range."""
    if isinstance(pvnames, str):
        pvnames = [pvnames]

    idt = _parse_dates(initdate)
    edt = _parse_dates(enddate)

    pvs_data = PVDataSet(pvnames)
    pvs_data.timeout = timeout
    pvs_data.time_start = Time(*idt, 0)
    pvs_data.time_stop  = Time(*edt, 0)
    pvs_data.update(mean_sec=10)

    t0 = pvs_data[pvnames[0]].timestamp[0]
    return pvs_data, t0, pvs_data[pvnames[0]].timestamp


def calculate_positions(blades, adj, matrix):
    ampA = blades[0].value
    ampB = blades[1].value
    ampC = blades[2].value
    ampD = blades[3].value

    print(f"Gain X: {adj[0]} nm")
    print(f"Delta X: {adj[1]} nm")

    print(f"Gain Y: {adj[2]} nm")
    print(f"Delta Y: {adj[3]} nm")

    supmat_vec = np.array(matrix)
    supmat = supmat_vec.reshape(4, 4)
    print(f"Suppression Matrix:\n {supmat}")

    positions = list()
    for val in range(len(ampA)):
        blades_vec = np.array([ampA[val], ampB[val], ampC[val], ampD[val]])
        res = supmat @ blades_vec

        raw_x = res[0]/res[1]
        raw_y = res[2]/res[3]

        X = raw_x * adj[0] - adj[1]
        Y = raw_y * adj[2] - adj[3]

        positions.append([X, Y])

    return positions


def plot_data(result, idt, edt, prefix, xnum):
    x_data = list()
    y_data = list()
    time   = list()

    ndots  = len(result)
    ninterval = ((edt - idt) / ndots)

    count  = idt
    for i in range(ndots):
        x_data.append(result[i][0] / 1000)  # Change units from nm to µm
        y_data.append(result[i][1] / 1000)
        time.append(count)
        count += ninterval

    curr_bl = _get_beamline_name(prefix)
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(8, 5))
    fig.suptitle(f"{curr_bl} XBPM{xnum} X and Y calculated positions - {idt.date()}")

    ax_x.plot(time, x_data, color='red',label='X position')
    ax_x.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_x.set_ylabel("X Position (µm)")
    ax_x.set_xlabel ("Time (hh:mm)")
    ax_x.legend()
    ax_x.grid()

    ax_y.plot(time, y_data, color='blue', label='Y position')
    ax_y.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_y.set_ylabel("Y Position (µm)")
    ax_y.set_xlabel ("Time (hh:mm)")
    ax_y.legend()
    ax_y.grid()

    plt.tight_layout()
    plt.show()
    

def cmd_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Read XBPMs positions PVs from EPICS archiver")

    parser.add_argument(
        '-p', '--prefix', type=str, required=True,
        help='Prefix for XBPMs PVs'
        )
    parser.add_argument(
        '-x', '--xbpm_number', type=int, required=True, choices=[1, 2],
        help='XBPM number (1 or 2)'
        )
    parser.add_argument(
        '-i', '--init_date', type=str, required=True,
        help="Initial date in ISO format (YYYY-MM-DD [HH:MM), without seconds."
    )
    parser.add_argument(
        '-e', '--end_date', type=str, required=True,
        help=("End date in ISO format (YYYY-MM-DD [HH:MM]), without seconds.")
    )

    parser.add_argument(
        '-g', '--graph', default=False, action='store_true',
        help="Plot graph of the data and fittings. (default: False)"
    )

    args = parser.parse_args()

    # Rearrange date order if needed.
    date1, date2 = args.init_date, args.end_date
    d1 = datetime.fromisoformat(date1)
    d2 = datetime.fromisoformat(date2)
    args.init_date, args.end_date = min(d1, d2), max(d1, d2)

    if not args.prefix:
        parser.error(f"Prefix for XBPMs PVs is required.")

    return args


def main():
    args = cmd_args()

    pvnames = set_pv_names(args.prefix, args.xbpm_number)

    adj_data = [caget(pv) for pv in pvnames["ADJUST"]]
    matrix = caget(pvnames["MATRIX"])

    # Fetch data from the EPICS archiver for the specified time range
    data, _, _ = get_pvdata(pvnames["BLADES"], args.init_date, args.end_date, timeout=1)

    positions = calculate_positions(data, adj_data, matrix)

    if args.graph:
        plot_data(positions, args.init_date, args.end_date, args.prefix, args.xbpm_number)


if __name__ == "__main__":
    main()
