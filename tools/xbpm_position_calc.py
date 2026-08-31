#!/usr/bin/env python3
"""Fetch data from any XBPMs position calculation PVs and plot them"""

import argparse
import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from epics import caget
from siriuspy.clientarch import PVDataSet, Time

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

XBPM_DIST_DIFF = {
    "CAT": 3.850,
    "CNB": 3.850,
    "MGN": 5.930,
    "MNC": 3.850,
}

def _get_beamline_name(prefix):
    curr_bl = ""
    for bl_id, bl in BEAMLINES.items():
        if prefix == bl_id:
            curr_bl = bl
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
    pvs_data.update(mean_sec=5)

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

    positions = []
    for val in range(len(ampA)):
        blades_vec = np.array([ampA[val], ampB[val], ampC[val], ampD[val]])
        res = supmat @ blades_vec

        raw_x = res[0]/res[1]
        raw_y = res[2]/res[3]

        X = raw_x * adj[0] - adj[1]
        Y = raw_y * adj[2] - adj[3]

        positions.append([X, Y])

    return positions


def _calculate_angle(xbpm1_pos, xbpm2_pos, beamline):
    xbpms_dist = 0
    diff = abs(xbpm2_pos - xbpm1_pos) / 1000

    for bl, dist in XBPM_DIST_DIFF.items():
        if beamline == bl:
            xbpms_dist = dist

    if xbpms_dist == 0:
        raise ValueError("Invalid distance between XBPMs")
    
    angle = diff / xbpms_dist
    return angle


def plot_simple_data(result, idt, edt, prefix, xnum):
    x_data = []
    y_data = []
    time   = []

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
    

def plot_comparison_data(xbpm1_pos, xbpm2_pos, idt, edt, prefix):
    x1_data   = []
    y1_data   = []
    x2_data   = []
    y2_data   = []
    avg_pos_x = []
    avg_pos_y = []
    angle_x   = []
    angle_y   = []
    time      = []

    curr_bl = _get_beamline_name(prefix)

    ndots  = len(xbpm1_pos)
    ninterval = ((edt - idt) / ndots)

    count  = idt
    for i in range(ndots):
        x1_data.append(xbpm1_pos[i][0] / 1000)  # Change units from nm to µm
        y1_data.append(xbpm1_pos[i][1] / 1000)
        x2_data.append(xbpm2_pos[i][0] / 1000)
        y2_data.append(xbpm2_pos[i][1] / 1000)

        avg_pos_x.append((xbpm2_pos[i][0]+xbpm1_pos[i][0])/2000)
        avg_pos_y.append((xbpm2_pos[i][1]+xbpm1_pos[i][1])/2000)

        angle_x.append(_calculate_angle(xbpm1_pos[i][0], xbpm2_pos[i][0], curr_bl))
        angle_y.append(_calculate_angle(xbpm1_pos[i][1], xbpm2_pos[i][1], curr_bl))
        
        time.append(count)
        count += ninterval

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"{curr_bl} XBPMs X and Y calculated positions comparison - {idt.date()}")

    # XBPM1 X position plot
    ax_x.plot(time, x1_data, color='red',label='XBPM1')
    ax_x.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_x.set_ylabel("X1 Position (µm)")
    ax_x.set_xlabel ("Time (hh:mm)")

    # XBPM2 X position in the same graph
    ax_x2 = ax_x.twinx()
    ax_x2.plot(time, x2_data, color='brown', label='XBPM2')
    ax_x2.set_ylabel("X2 Position (µm)")

    # Append both labels and legends
    x1_lines, x1_labels = ax_x.get_legend_handles_labels()
    x2_lines, x2_labels = ax_x2.get_legend_handles_labels()

    ax_x.legend(x1_lines + x2_lines, x1_labels + x2_labels)
    ax_x.grid()

    # XBPM1 Y position plot
    ax_y.plot(time, y1_data, color='lightblue', label='XBPM1')
    ax_y.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_y.set_ylabel("Y1 Position (µm)")
    ax_y.set_xlabel ("Time (hh:mm)")

    # XBPM2 Y position in the same graph
    ax_y2 = ax_y.twinx()
    ax_y2.plot(time, y2_data, color='navy', label='XBPM2')
    ax_y2.set_ylabel("Y2 Position (µm)")

    # Append both labels and legends
    y1_lines, y1_labels = ax_y.get_legend_handles_labels()
    y2_lines, y2_labels = ax_y2.get_legend_handles_labels()

    ax_y.legend(y1_lines + y2_lines, y1_labels + y2_labels)
    ax_y.grid()

    # Plot positions analysis
    fig, (ax_x_avg, ax_y_avg) = plt.subplots(2, 1, figsize=(8, 5))
    fig.suptitle(f"{curr_bl} XBPMs average positions comparison - {idt.date()}")

    ax_x_avg.plot(time, avg_pos_x, color='red', label='Avg. X position')
    ax_x_avg.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_x_avg.set_ylabel("Position (µm)")
    ax_x_avg.set_xlabel("Time (hh:mm)")
    ax_x_avg.legend()
    ax_x_avg.grid()

    ax_y_avg.plot(time, avg_pos_y, color='blue', label='Avg. Y position')
    ax_y_avg.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_y_avg.set_ylabel("Position (µm)")
    ax_y_avg.set_xlabel("Time (hh:mm)")
    ax_y_avg.legend()
    ax_y_avg.grid()

    # Plot angle analysis
    fig, (ax_angx, ax_angy) = plt.subplots(2, 1, figsize=(8, 5))
    fig.suptitle(f"{curr_bl} XBPMs angle comparison - {idt.date()}")

    ax_angx.plot(time, angle_x, color='red', label='X angle')
    ax_angx.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_angx.set_ylabel("Angle (µrad)")
    ax_angx.set_xlabel("Time (hh:mm)")
    ax_angx.legend()
    ax_angx.grid()

    ax_angy.plot(time, angle_y, color='blue', label='Y angle')
    ax_angy.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_angy.set_ylabel("Angle (µrad)")
    ax_angy.set_xlabel("Time (hh:mm)")
    ax_angy.legend()
    ax_angy.grid()

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
        '-x', '--xbpm_number', type=int, required=True, choices=[0, 1, 2],
        help='XBPM number (0 for both, 1 for XBPM1, 2 for XBPM2)'
        )
    parser.add_argument(
        '-i', '--init_date', type=str, required=True,
        help="Initial date in ISO format (YYYY-MM-DD [HH:MM), without seconds."
    )
    parser.add_argument(
        '-e', '--end_date', type=str, required=True,
        help=("End date in ISO format (YYYY-MM-DD [HH:MM]), without seconds.")
    )

    args = parser.parse_args()

    # Rearrange date order if needed.
    date1, date2 = args.init_date, args.end_date
    d1 = datetime.fromisoformat(date1)
    d2 = datetime.fromisoformat(date2)
    args.init_date, args.end_date = min(d1, d2), max(d1, d2)

    if not args.prefix:
        parser.error("Prefix for XBPMs PVs is required.")

    return args


def main():
    args = cmd_args()

    if args.xbpm_number == 0:
        xbpm1_pvnames = set_pv_names(args.prefix, 1)
        xbpm2_pvnames = set_pv_names(args.prefix, 2)

        # Fetch data for both XBPMs
        xbpm1_data, _, _ = get_pvdata(xbpm1_pvnames["BLADES"], args.init_date, args.end_date, timeout=5)
        xbpm2_data, _, _ = get_pvdata(xbpm2_pvnames["BLADES"], args.init_date, args.end_date, timeout=5)

        # Calculate positions for both XBPMs
        xbpm1_pos = calculate_positions(
            xbpm1_data, 
            [caget(pv) for pv in xbpm1_pvnames["ADJUST"]], 
            caget(xbpm1_pvnames["MATRIX"])
        )  

        xbpm2_pos = calculate_positions(
            xbpm2_data, 
            [caget(pv) for pv in xbpm2_pvnames["ADJUST"]], 
            caget(xbpm2_pvnames["MATRIX"])
        )

        plot_comparison_data(xbpm1_pos, xbpm2_pos, args.init_date, args.end_date, args.prefix)

    else:
        pvnames = set_pv_names(args.prefix, args.xbpm_number)

        adj_data = [caget(pv) for pv in pvnames["ADJUST"]]
        matrix = caget(pvnames["MATRIX"])
        # Fetch data from the EPICS archiver for the specified time range
        data, _, _ = get_pvdata(pvnames["BLADES"], args.init_date, args.end_date, timeout=5)

        positions = calculate_positions(data, adj_data, matrix)
        plot_simple_data(positions, args.init_date, args.end_date, args.prefix, args.xbpm_number)

if __name__ == "__main__":
    main()
