#!/usr/bin/env python3
"""Fetch data from any XBPMs position calculation PVs and plot them"""

import argparse
import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from epics import caget
from matplotlib.patches import Ellipse
from siriuspy.clientarch import PVDataSet, Time

EPICS_CA_ADDR_LIST = ["10.0.38.59:62000", "10.30.13.22:62000", "10.30.14.19:62000"]
os.environ["EPICS_CA_ADDR_LIST"] = " ".join(EPICS_CA_ADDR_LIST)

BLADES = ["AmplA-Mon", "AmplB-Mon", "AmplC-Mon", "AmplD-Mon"]
ADJUST = ["PosKx-RB","PosXOffset-RB","PosKy-RB", "PosYOffset-RB"]
MATRIX = "SupMat-RB"

BEAMLINES = {
    "CNB": ["SI-06SBFE", 3.850],
    "CAT": ["SI-07SPFE", 3.850],
    "MNC": ["SI-09SAFE", 3.850],
    "MGN": ["SI-10BCFE", 5.930],
}

def get_beamline_data(name):
    for bl_name, bl in BEAMLINES.items():
        if name == bl_name:
            return bl


def set_pv_names(prefix, xbpm_number):
    """Set PV names for the specified XBPM number with the given prefix"""
    if xbpm_number not in [1, 2]:
        raise ValueError("XBPM number must be either 1 or 2.")
    
    xbpm_prefix = prefix + ":DI-PBPM-" + str(xbpm_number)

    pvnames = {
        "BLADES": [],
        "ADJUST": [],
        "MATRIX": [],
    }

    pvnames["BLADES"] = [f"{xbpm_prefix}:{blade}" for blade in BLADES]
    pvnames["ADJUST"] = [f"{xbpm_prefix}:{adj}" for adj in ADJUST]
    pvnames["MATRIX"] = xbpm_prefix + ":" + MATRIX

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


def calculate_positions(blades, adj, matrix, xbpm_number):
    ampA = blades[0].value
    ampB = blades[1].value
    ampC = blades[2].value
    ampD = blades[3].value

    supmat_vec = np.array(matrix)
    supmat = supmat_vec.reshape(4, 4)

    positions = []
    for val in range(len(ampA)):
        blades_vec = np.array([ampA[val], ampB[val], ampC[val], ampD[val]])
        res = supmat @ blades_vec

        raw_x = res[0]/res[1]
        raw_y = res[2]/res[3]

        X = (raw_x * adj[0] - adj[1]) / 1000 # Convert from nm to µm
        Y = (raw_y * adj[2] - adj[3]) / 1000

        positions.append([X, Y])

    print("\n"+"#"*50)
    print(f"##### XBPM {xbpm_number} Position Calculation Parameters #####")
    print(f"\nGain X : {adj[0]:>12.2f} nm")
    print(f"Delta X: {adj[1]:>12.2f} nm")

    print(f"\nGain Y : {adj[2]:>12.2f} nm")
    print(f"Delta Y: {adj[3]:>12.2f} nm")

    print("\nSuppression Matrix:")
    with np.printoptions(formatter={'float': '{:>8.4f}'.format}):
        print(supmat)

    return positions


def _calculate_angle(xbpm1_pos, xbpm2_pos, xbpms_dist):
    diff = (xbpm2_pos - xbpm1_pos)

    if xbpms_dist == 0:
        raise ValueError("Invalid distance between XBPMs")
    
    return diff / xbpms_dist

def plot_simple_data(result, idt, edt, beamline, xnum):
    x_data, y_data, time = [], [], []

    ndots  = len(result)
    ninterval = ((edt - idt) / ndots)

    count  = idt
    for i in range(ndots):
        x_data.append(result[i][0])
        y_data.append(result[i][1])
        time.append(count)
        count += ninterval

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(8, 5))
    fig.suptitle(f"{beamline} XBPM{xnum} X and Y calculated positions - {idt.date()}")

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
    

def plot_comparison_data(xbpm1_pos, xbpm2_pos, idt, edt, beamline, xbpms_dist):
    x1_data, y1_data = [], []
    x2_data, y2_data = [], []
    angle_x, angle_y = [], []
    time = []

    ndots  = len(xbpm1_pos)
    ninterval = ((edt - idt) / ndots)

    count  = idt
    for i in range(ndots):
        x1_data.append(xbpm1_pos[i][0])
        y1_data.append(xbpm1_pos[i][1])
        x2_data.append(xbpm2_pos[i][0])
        y2_data.append(xbpm2_pos[i][1])

        angle_x.append(_calculate_angle(xbpm1_pos[i][0], xbpm2_pos[i][0], xbpms_dist))
        angle_y.append(_calculate_angle(xbpm1_pos[i][1], xbpm2_pos[i][1], xbpms_dist))
        
        time.append(count)
        count += ninterval

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"{beamline} XBPMs X and Y calculated positions comparison - {idt.date()}")

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

    # Plot angle analysis
    x1_std = np.std(x1_data, ddof=1)
    y1_std = np.std(y1_data, ddof=1)
    x2_std = np.std(x2_data, ddof=1)
    y2_std = np.std(y2_data, ddof=1)

    tot_std_x1 = np.sqrt(x1_std**2 + y1_std**2)
    tot_std_x2 = np.sqrt(x2_std**2 + y2_std**2)

    ang_x_std = (1/xbpms_dist) * np.sqrt(x1_std**2 + x2_std**2)
    ang_y_std = (1/xbpms_dist) * np.sqrt(y1_std**2 + y2_std**2)

    print("\n"+"#"*50)
    print("############ XBPM Statistics Analysis ############")
    print(f"\nXBPM1 X deviation    : {x1_std:>6.2f} µm")
    print(f"XBPM1 Y deviation    : {y1_std:>6.2f} µm")
    print(f"XBPM1 total deviation: {tot_std_x1:>6.2f} µm")

    print(f"\nXBPM2 X deviation    : {x2_std:>6.2f} µm")
    print(f"XBPM2 Y deviation    : {y2_std:>6.2f} µm")
    print(f"XBPM2 total deviation: {tot_std_x2:>6.2f} µm")

    print(f"\nX angle deviation: {ang_x_std:>6.2f} µrad")
    print(f"Y angle deviation: {ang_y_std:>6.2f} µrad")

    print(f"\nXBPM 1 X average position: {np.mean(x1_data):>7.2f} µm")
    print(f"XBPM 1 Y average position: {np.mean(y1_data):>7.2f} µm")
    print(f"\nXBPM 2 X average position: {np.mean(x2_data):>7.2f} µm")
    print(f"XBPM 2 Y average position: {np.mean(y2_data):>7.2f} µm")

    fig, (ax_ang_x, ax_ang_y) = plt.subplots(2, 1, figsize=(8, 5))
    fig.suptitle(f"{beamline} XBPMs angle comparison - {idt.date()}")

    ax_ang_x.plot(time, angle_x, color='red', label='X angle')
    ax_ang_x.fill_between(
        time, angle_x - ang_x_std, angle_x + ang_x_std, color="red", alpha=0.15, label="±1 Std Dev."
    )
    ax_ang_x.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_ang_x.set_ylabel("Angle (µrad)")
    ax_ang_x.legend()
    ax_ang_x.grid()

    ax_ang_y.plot(time, angle_y, color='blue', label='Y angle')
    ax_ang_y.fill_between(
        time, angle_y - ang_y_std, angle_y + ang_y_std, color="blue", alpha=0.15, label="±1 Std Dev."
    )
    ax_ang_y.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_ang_y.set_ylabel("Angle (µrad)")
    ax_ang_y.set_xlabel("Time (hh:mm)")
    ax_ang_y.legend()
    ax_ang_y.grid()

    # Plot position map
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(np.mean(x1_data), np.mean(y1_data), color='red', label='XBPM1')
    ax.scatter(np.mean(x2_data), np.mean(y2_data), color='blue', label='XBPM2')

    shadow1 = Ellipse(
    xy=(np.mean(x1_data), np.mean(y1_data)),
    width=tot_std_x1 * 2,
    height=tot_std_x1 * 2,
    angle=0,
    color="red",
    alpha=0.2,
    label="±1 SD Region",
    zorder=2,
    )

    shadow2 = Ellipse(
    xy=(np.mean(x2_data), np.mean(y2_data)),
    width=tot_std_x2 * 2,
    height=tot_std_x2 * 2,
    angle=0,
    color="blue",
    alpha=0.2,
    label="±1 SD Region",
    zorder=2,
    )

    ax.add_patch(shadow1)
    ax.add_patch(shadow2)
    ax.set_title(f"{beamline} XBPMs Position Map - {idt.date()}")
    ax.set_xlabel("X Position (µm)")
    ax.set_ylabel("Y Position (µm)")

    if beamline == 'MGN':
        ax.set_xlim(10, -10) # Invert X-axis to match SR coordinates
        ax.set_ylim(-1000, 1000)
    else:
        ax.set_xlim(150, -150)
        ax.set_ylim(-150, 150)
    ax.legend()
    ax.grid()

    plt.tight_layout()  
    plt.show()

def cmd_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Read XBPMs positions PVs from EPICS archiver")

    parser.add_argument(
        '-b', '--beamline', type=str, required=True,
        help='Beamline name (e.g. MNC, MGN, CNB, CAT)'
        )
    parser.add_argument(
        '-x', '--xbpm_number', type=int, required=True, choices=[0, 1, 2],
        help='XBPM number (0 for both, 1 for XBPM1, 2 for XBPM2)'
        )
    parser.add_argument(
        '-i', '--init_date', type=str, required=True,
        help="Initial date in ISO format (YYYY-MM-DD [HH:MM]), without seconds."
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

    return args


def main():
    args = cmd_args()
    prefix, xbpms_dist = get_beamline_data(args.beamline)

    if args.xbpm_number == 0:
        xbpm1_pvnames = set_pv_names(prefix, 1)
        xbpm2_pvnames = set_pv_names(prefix, 2)

        # Fetch data for both XBPMs
        xbpm1_data, _, _ = get_pvdata(xbpm1_pvnames["BLADES"], args.init_date, args.end_date, timeout=10)
        xbpm2_data, _, _ = get_pvdata(xbpm2_pvnames["BLADES"], args.init_date, args.end_date, timeout=10)

        # Calculate positions for both XBPMs
        xbpm1_pos = calculate_positions(
            xbpm1_data, 
            [caget(pv) for pv in xbpm1_pvnames["ADJUST"]], 
            caget(xbpm1_pvnames["MATRIX"]),
            1
        )  

        xbpm2_pos = calculate_positions(
            xbpm2_data, 
            [caget(pv) for pv in xbpm2_pvnames["ADJUST"]], 
            caget(xbpm2_pvnames["MATRIX"]),
            2
        )

        plot_comparison_data(xbpm1_pos, xbpm2_pos, args.init_date, args.end_date, args.beamline, xbpms_dist)

    else:
        pvnames = set_pv_names(prefix, args.xbpm_number)

        adj_data = [caget(pv) for pv in pvnames["ADJUST"]]
        matrix = caget(pvnames["MATRIX"])
        # Fetch data from the EPICS archiver for the specified time range
        data, _, _ = get_pvdata(pvnames["BLADES"], args.init_date, args.end_date, timeout=10)

        positions = calculate_positions(data, adj_data, matrix, args.xbpm_number)
        plot_simple_data(positions, args.init_date, args.end_date, args.beamline, args.xbpm_number)

if __name__ == "__main__":
    main()
