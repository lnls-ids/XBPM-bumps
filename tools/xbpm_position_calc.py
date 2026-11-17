#!/usr/bin/env python3
"""
##### XBPM Position Calculation Tool #####

This script calculates the positions of XBPMs based on the amplitudes of the blades, 
adjustment parameters, and suppression matrices. It fetches data from the EPICS archiver
for the specified time range and plots the calculated positions for analysis.

The script supports calculating positions for a single XBPM or comparing positions between two XBPMs.
"""
import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

# Configuration
EPICS_CA_ADDR_LIST = [
     "10.0.38.59:62000",
     "10.30.13.22:62000",
     "10.30.14.19:62000"
     ]

BLADES = ["AmplA-Mon", "AmplB-Mon", "AmplC-Mon", "AmplD-Mon"]
ADJUST = ["PosKx-RB","PosXOffset-RB","PosKy-RB", "PosYOffset-RB"]
MATRIX = "SupMat-RB"

BEAMLINES = {
    "CNB": ["SI-06SBFE", 3.850],
    "CAT": ["SI-07SPFE", 3.850],
    "MNC": ["SI-09SAFE", 3.850],
    "MGN": ["SI-10BCFE", 5.930],
}


@dataclass(frozen=True)
class PVArchiveData:
    """Archiver samples and their shared timestamps."""

    data: Any
    pvnames: tuple[str, ...]
    timestamps: np.ndarray


@dataclass(frozen=True)
class PositionSeries:
    """Calculated horizontal and vertical positions associated with timestamps."""

    timestamps: np.ndarray
    x: np.ndarray
    y: np.ndarray


# PV naming and archiver access
def configure_epics() -> None:
    """Configure EPICS Channel Access for executable use."""
    os.environ["EPICS_CA_ADDR_LIST"] = " ".join(EPICS_CA_ADDR_LIST)


def get_pv_value(pvname: str) -> Any:
    """Read a live EPICS PV after applying this script's connection settings."""
    configure_epics()
    from epics import caget

    return caget(pvname)


def get_beamline_data(name: str) -> list:
    try:
        return BEAMLINES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown beamline {name!r}. Choose from: {', '.join(BEAMLINES)}."
        ) from exc


def set_pv_names(prefix: str,
                 xbpm_number: int
                 ) -> dict:
    """Set PV names for the specified XBPM number with the given prefix.

    Arguments:
        prefix (str): Prefix for the PV names.
        xbpm_number (int): XBPM number (1 or 2).
    
    Returns:
        dict: Dictionary containing PV names for blades, adjustment parameters, 
        and suppression matrix.
    """
    if xbpm_number not in [1, 2]:
        raise ValueError("XBPM number must be either 1 or 2.")
    
    xbpm_prefix = prefix + ":DI-PBPM-" + str(xbpm_number)

    pvnames = {}
    pvnames["BLADES"] = [
        f"{xbpm_prefix}:{blade}"
        for blade in BLADES
        ]
    pvnames["ADJUST"] = [
        f"{xbpm_prefix}:{adj}"
        for adj in ADJUST
        ]
    pvnames["MATRIX"] = xbpm_prefix + ":" + MATRIX

    return pvnames


def _parse_dates(dt: datetime) -> list:
    """Parse dates from date string.

    Given the date in datetime format, this function extracts the 
    year, month, day, hour, and minute components.

    Arguments:
        dt (datetime): Date in datetime format.

    Returns:
        list: List of date components [year, month, day, hour, minute].
    """
    year, month, day = dt.year, dt.month, dt.day
    hour, minute = dt.hour, dt.minute

    return [year, month, day, hour, minute]


def get_pvdata(pvnames: Union[list, str],
               initdate: datetime,
               enddate: datetime,
               timeout: int) -> PVArchiveData:
    """Fetch data from the EPICS archiver for the given PV names and time range.
    
    Arguments:
        pvnames (list): List of PV names to fetch data for.
        initdate (datetime): Initial date and time for data fetching.
        enddate (datetime): End date and time for data fetching.
        timeout (int): Timeout in seconds for the data fetching operation.

    Returns:
        PVArchiveData: The PVDataSet object and aligned timestamps.
    """
    if isinstance(pvnames, str):
        pvnames = [pvnames]
    if not pvnames:
        raise ValueError("At least one PV name is required.")
    if enddate < initdate:
        raise ValueError("End date must be on or after the initial date.")

    idt = _parse_dates(initdate)
    edt = _parse_dates(enddate)

    configure_epics()
    from siriuspy.clientarch import PVDataSet, Time

    pvs_data = PVDataSet(pvnames)
    pvs_data.timeout = timeout
    pvs_data.time_start = Time(*idt, 0)
    pvs_data.time_stop  = Time(*edt, 0)
    pvs_data.update(mean_sec=5)

    timestamps = np.asarray(pvs_data[pvnames[0]].timestamp)
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError(
            f"PV {pvnames[0]!r} returned no valid one-dimensional timestamps."
        )

    for pvname in pvnames:
        pv_timestamps = np.asarray(pvs_data[pvname].timestamp)
        values = np.asarray(pvs_data[pvname].value)
        if pv_timestamps.size == 0 or values.size == 0:
            raise ValueError(f"No archived samples returned for PV {pvname!r}.")
        if values.ndim != 1:
            raise ValueError(
                f"Expected scalar samples for PV {pvname!r}; "
                f"got shape {values.shape}."
            )
        if len(pv_timestamps) != len(values):
            raise ValueError(
                f"PV {pvname!r} has {len(pv_timestamps)} timestamps but "
                f"{len(values)} values."
            )
        if not np.array_equal(pv_timestamps, timestamps):
            raise ValueError(
                f"Timestamp samples for PV {pvname!r} do not align with "
                f"PV {pvnames[0]!r}; position calculation requires matching "
                "blade timestamps."
            )

    return PVArchiveData(
        data=pvs_data,
        pvnames=tuple(pvnames),
        timestamps=timestamps,
    )


# Position and angle calculations
def calculate_positions(archive: PVArchiveData,
                        adj: list,
                        matrix: list,
                        xbpm_number: int
                        ) -> PositionSeries:
    """Calculate XBPM positions based on blades amplitudes, adjustment parameters, 
    and suppression matrix.
    
    Given the amplitudes of the blades (A, B, C, D), the adjustment parameters 
    (GainX, OffsetX, GainY, OffsetY), and the suppression matrix, this function calculates 
    the X and Y positions of the XBPM.

    Arguments:
        archive (PVArchiveData): Aligned blade samples and timestamps.
        adj (list): List of 4 adjustment parameters (GainX, OffsetX, GainY, OffsetY).
        matrix (list): List of 16 elements representing the suppression matrix.
        xbpm_number (int): XBPM number (1 or 2).
    
    Returns:
        PositionSeries: Calculated X/Y positions associated with timestamps.
    """
    blades = [archive.data[pvname] for pvname in archive.pvnames]
    if len(blades) != 4:
        raise ValueError(f"Expected 4 blade PVs; got {len(blades)}.")
    blade_values = [np.asarray(blade.value) for blade in blades]
    if any(values.ndim != 1 for values in blade_values):
        raise ValueError("Expected one-dimensional blade sample arrays.")
    sample_counts = [len(values) for values in blade_values]
    if not sample_counts or sample_counts[0] == 0:
        raise ValueError(f"No blade samples available for XBPM {xbpm_number}.")
    if len(set(sample_counts)) != 1:
        raise ValueError(
            "Blade sample counts do not match: "
            + ", ".join(str(count) for count in sample_counts)
            + "."
        )

    try:
        adj = np.asarray(adj, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Adjustment parameters must be 4 numeric values.") from exc
    if adj.shape != (4,):
        raise ValueError(
            "Adjustment parameters must be a one-dimensional list of 4 values."
        )
    if not np.all(np.isfinite(adj)):
        raise ValueError("Adjustment parameters must all be finite numbers.")

    try:
        supmat = np.asarray(matrix, dtype=float).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError("Suppression matrix must contain 16 numeric values.") from exc
    if not np.all(np.isfinite(supmat)):
        raise ValueError("Suppression matrix values must all be finite numbers.")

    positions = []
    for val in range(sample_counts[0]):
        blades_vec = np.array([values[val] for values in blade_values])
        res = supmat @ blades_vec

        if res[1] == 0 or res[3] == 0:
            raise ValueError(
                f"Suppression matrix denominator is zero at sample {val} "
                f"for XBPM {xbpm_number}."
            )
        raw_x = res[0]/res[1]
        raw_y = res[2]/res[3]

        X = (raw_x * adj[0] - adj[1]) / 1000 # Convert from nm to µm
        Y = (raw_y * adj[2] - adj[3]) / 1000

        if not np.isfinite(X) or not np.isfinite(Y):
            raise ValueError(
                f"Non-finite position calculated at sample {val} "
                f"for XBPM {xbpm_number}."
            )
        positions.append([X, Y])

    positions = np.asarray(positions)
    return PositionSeries(
        timestamps=archive.timestamps,
        x=positions[:, 0],
        y=positions[:, 1],
    )


def _calculate_angle(xbpm1_pos: float,
                     xbpm2_pos: float,
                     xbpms_dist: float
                     ) -> float:
    """
    Calculate the angle between two XBPM positions based on their positions 
    and the distance between them.
    
    Arguments:
        xbpm1_pos (float): Position of the first XBPM.
        xbpm2_pos (float): Position of the second XBPM.
        xbpms_dist (float): Distance between the two XBPMs.
    
    Returns:
        float: Calculated angle in µrad.
    """
    diff = (xbpm2_pos - xbpm1_pos)

    if xbpms_dist == 0:
        raise ValueError("Invalid distance between XBPMs")
    
    return diff / xbpms_dist


# Console reporting
def print_position_parameters(adj: list,
                              matrix: list,
                              xbpm_number: int) -> None:
    """Print the adjustment parameters and suppression matrix for an XBPM."""
    supmat = np.asarray(matrix, dtype=float).reshape(4, 4)
    print("\n"+"#"*50)
    print(f"##### XBPM {xbpm_number} Position Calculation Parameters #####")
    print(f"\nGain X : {adj[0]:>12.2f} nm")
    print(f"Delta X: {adj[1]:>12.2f} nm")

    print(f"\nGain Y : {adj[2]:>12.2f} nm")
    print(f"Delta Y: {adj[3]:>12.2f} nm")

    print("\nSuppression Matrix:")
    with np.printoptions(formatter={'float': '{:>8.4f}'.format}):
        print(supmat)


# Plot data validation and conversion
def _validate_plot_data(series: PositionSeries,
                        label: str
                        ) -> PositionSeries:
    timestamps = np.asarray(series.timestamps)
    x_positions = np.asarray(series.x, dtype=float)
    y_positions = np.asarray(series.y, dtype=float)

    if (timestamps.ndim != 1 or x_positions.ndim != 1 or
            y_positions.ndim != 1 or
            len(timestamps) != len(x_positions) or
            len(timestamps) != len(y_positions)):
        raise ValueError(
            f"{label} positions and timestamps must be one-dimensional "
            "arrays of equal nonzero length."
        )
    if len(timestamps) == 0:
        raise ValueError(f"No position samples available for {label}.")
    if np.issubdtype(timestamps.dtype, np.number) and not np.all(
            np.isfinite(timestamps)):
        raise ValueError(f"{label} timestamps contain non-finite values.")
    if (not np.all(np.isfinite(x_positions)) or
            not np.all(np.isfinite(y_positions))):
        raise ValueError(f"{label} positions contain non-finite values.")

    if len(np.unique(timestamps)) != len(timestamps):
        raise ValueError(f"{label} timestamps must be unique.")

    order = np.argsort(timestamps)
    return PositionSeries(
        timestamps=timestamps[order],
        x=x_positions[order],
        y=y_positions[order],
    )


def _to_plot_times(timestamps: np.ndarray) -> list:
    """Convert Unix timestamps to datetimes for Matplotlib date axes."""
    numeric_timestamps = np.issubdtype(timestamps.dtype, np.number)
    if timestamps.dtype == object:
        numeric_timestamps = all(
            isinstance(timestamp, (int, float, np.number))
            for timestamp in timestamps
        )
    if numeric_timestamps:
        return [datetime.fromtimestamp(float(timestamp)) for timestamp in timestamps]
    return timestamps


def _format_time_axis(ax: plt.Axes,
                      initdate: datetime,
                      enddate: datetime) -> None:
    """Apply consistent date formatting, label, and tick rotation."""
    is_multiday = (enddate - initdate).total_seconds() > 24 * 60 * 60
    if is_multiday:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel("Date and time")
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_xlabel("Time (hh:mm)")


def _covariance_ellipse(x_positions: np.ndarray,
                        y_positions: np.ndarray,
                        color: str,
                        label: str) -> tuple[Ellipse, float, float]:
    covariance = np.cov(np.vstack((x_positions, y_positions)), ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    ellipse = Ellipse(
        xy=(np.mean(x_positions), np.mean(y_positions)),
        width=2 * np.sqrt(eigenvalues[0]),
        height=2 * np.sqrt(eigenvalues[1]),
        angle=angle,
        color=color,
        alpha=0.2,
        label=label,
        zorder=2,
    )
    return ellipse, np.sqrt(covariance[0, 0]), np.sqrt(covariance[1, 1])


def _set_position_map_limits(ax: plt.Axes) -> None:
    ax.set_xlim(200, -200)
    ax.set_ylim(-200, 200)
    ax.set_aspect("equal", adjustable="box")


# Plotting
def plot_simple_data(series: PositionSeries,
                     idt: datetime,
                     edt: datetime,
                     beamline: str,
                     xnum: int) -> Figure:
    """Plot the calculated XBPM positions over time.
    
    Arguments:
        series (PositionSeries): Calculated positions and their timestamps.
        idt (datetime): Initial date and time for the plot.
        edt (datetime): End date and time for the plot.
        beamline (str): Name of the beamline.
        xnum (int): XBPM number (1 or 2).
    """
    series = _validate_plot_data(series, f"XBPM{xnum}")
    time = _to_plot_times(series.timestamps)

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(8, 5))
    fig.suptitle(f"{beamline} XBPM{xnum} X and Y"
                 f" calculated positions - {idt.date()}")

    ax_x.plot(time, series.x, color='red',label='X position')
    _format_time_axis(ax_x, idt, edt)
    ax_x.set_ylabel("X Position (µm)")
    ax_x.legend()
    ax_x.grid()

    ax_y.plot(time, series.y, color='blue', label='Y position')
    _format_time_axis(ax_y, idt, edt)
    ax_y.set_ylabel("Y Position (µm)")
    ax_y.legend()
    ax_y.grid()

    fig.tight_layout()
    return fig
    

def plot_comparison_data(xbpm1_pos: PositionSeries,
                         xbpm2_pos: PositionSeries,
                         idt: datetime,
                         edt: datetime,
                         beamline: str,
                         xbpms_dist: float) -> tuple[Figure, Figure, Figure]:
    """Plot the calculated positions of two XBPMs and their angle comparison over time.
    Also it plots the average position of both XBPMs with their standard deviation
    as a shadow area.

    Arguments:
        xbpm1_pos (PositionSeries): XBPM1 positions and their timestamps.
        xbpm2_pos (PositionSeries): XBPM2 positions and their timestamps.
        idt (datetime): Initial date and time for the plot.
        edt (datetime): End date and time for the plot.
        beamline (str): Name of the beamline.
        xbpms_dist (float): Distance between the two XBPMs.
    """
    xbpm1_pos = _validate_plot_data(xbpm1_pos, "XBPM1")
    xbpm2_pos = _validate_plot_data(xbpm2_pos, "XBPM2")
    if not np.isfinite(xbpms_dist) or xbpms_dist <= 0:
        raise ValueError("Distance between XBPMs must be a positive finite number.")
    time, indices1, indices2 = np.intersect1d(
        xbpm1_pos.timestamps,
        xbpm2_pos.timestamps,
        return_indices=True,
    )
    if len(time) < 2:
        raise ValueError(
            "XBPM1 and XBPM2 need at least two matching timestamps for "
            "comparison and sample standard deviations; "
            f"found {len(time)}."
        )

    xbpm1_pos = PositionSeries(
        timestamps=time,
        x=xbpm1_pos.x[indices1],
        y=xbpm1_pos.y[indices1],
    )
    xbpm2_pos = PositionSeries(
        timestamps=time,
        x=xbpm2_pos.x[indices2],
        y=xbpm2_pos.y[indices2],
    )
    time = _to_plot_times(time)
    x1_data, y1_data = xbpm1_pos.x, xbpm1_pos.y
    x2_data, y2_data = xbpm2_pos.x, xbpm2_pos.y
    angle_x, angle_y = [], []

    for x1, x2, y1, y2 in zip(x1_data, x2_data, y1_data, y2_data):
        angle_x.append(_calculate_angle(x1, x2, xbpms_dist))
        angle_y.append(_calculate_angle(y1, y2, xbpms_dist))

    fig_positions, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 8))
    fig_positions.suptitle(
        f"{beamline} XBPMs X and Y calculated positions comparison - {idt.date()}"
    )

    # Plot both XBPMs against a shared position scale.
    ax_x.plot(time, x1_data, color='red',label='XBPM1')
    ax_x.plot(time, x2_data, color='brown', label='XBPM2')
    _format_time_axis(ax_x, idt, edt)
    ax_x.set_ylabel("X Position (µm)")
    ax_x.legend()
    ax_x.grid()

    # Plot both XBPMs against a shared position scale.
    ax_y.plot(time, y1_data, color='lightblue', label='XBPM1')
    ax_y.plot(time, y2_data, color='navy', label='XBPM2')
    _format_time_axis(ax_y, idt, edt)
    ax_y.set_ylabel("Y Position (µm)")
    ax_y.legend()
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

    fig_angles, (ax_ang_x, ax_ang_y) = plt.subplots(2, 1, figsize=(8, 6))
    fig_angles.suptitle(f"{beamline} XBPMs angle comparison - {idt.date()}")

    ax_ang_x.plot(time, angle_x, color='red', label='X angle')
    ax_ang_x.fill_between(
        time, angle_x - ang_x_std, angle_x + ang_x_std, color="red", alpha=0.15, label="±1 Std Dev."
    )
    _format_time_axis(ax_ang_x, idt, edt)
    ax_ang_x.set_ylabel("Angle (µrad)")
    ax_ang_x.legend()
    ax_ang_x.grid()

    ax_ang_y.plot(time, angle_y, color='blue', label='Y angle')
    ax_ang_y.fill_between(
        time, angle_y - ang_y_std, angle_y + ang_y_std, color="blue", alpha=0.15, label="±1 Std Dev."
    )
    _format_time_axis(ax_ang_y, idt, edt)
    ax_ang_y.set_ylabel("Angle (µrad)")
    ax_ang_y.legend()
    ax_ang_y.grid()

    # Plot position map
    fig_map, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(np.mean(x1_data), np.mean(y1_data), color='red', label='XBPM1')
    ax.scatter(np.mean(x2_data), np.mean(y2_data), color='blue', label='XBPM2')

    center1 = (np.mean(x1_data), np.mean(y1_data))
    center2 = (np.mean(x2_data), np.mean(y2_data))
    shadow1, x1_spread, y1_spread = _covariance_ellipse(
        x1_data, y1_data, "red", "1 SD covariance ellipse"
    )
    shadow2, x2_spread, y2_spread = _covariance_ellipse(
        x2_data, y2_data, "blue", "_nolegend_"
    )

    ax.add_patch(shadow1)
    ax.add_patch(shadow2)
    ax.set_title(f"{beamline} XBPMs Position Map - {idt.date()}")
    ax.set_xlabel("X Position (µm)")
    ax.set_ylabel("Y Position (µm)")

    _set_position_map_limits(ax)
    ax.legend()
    ax.grid()

    fig_positions.tight_layout()
    fig_angles.tight_layout()
    fig_map.tight_layout()
    return fig_positions, fig_angles, fig_map


# Command-line interface
def cmd_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Read XBPMs positions PVs from EPICS archiver"
        )

    parser.add_argument(
        '-b', '--beamline', type=str, required=True,
        choices=BEAMLINES,
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


def main() -> None:
    configure_epics()
    args = cmd_args()
    prefix, xbpms_dist = get_beamline_data(args.beamline)

    if args.xbpm_number == 0:
        xbpm1_pvnames = set_pv_names(prefix, 1)
        xbpm2_pvnames = set_pv_names(prefix, 2)

        # Fetch data for both XBPMs
        xbpm1_data = get_pvdata(
            xbpm1_pvnames["BLADES"],
            args.init_date,
            args.end_date,
            timeout=10
            )

        xbpm2_data = get_pvdata(
            xbpm2_pvnames["BLADES"],
            args.init_date,
            args.end_date,
            timeout=10
            )

        # Calculate positions for both XBPMs
        xbpm1_adj = [get_pv_value(pv) for pv in xbpm1_pvnames["ADJUST"]]
        xbpm1_matrix = get_pv_value(xbpm1_pvnames["MATRIX"])
        xbpm1_pos = calculate_positions(
            xbpm1_data,
            xbpm1_adj,
            xbpm1_matrix,
            1
        )
        print_position_parameters(xbpm1_adj, xbpm1_matrix, 1)

        xbpm2_adj = [get_pv_value(pv) for pv in xbpm2_pvnames["ADJUST"]]
        xbpm2_matrix = get_pv_value(xbpm2_pvnames["MATRIX"])
        xbpm2_pos = calculate_positions(
            xbpm2_data,
            xbpm2_adj,
            xbpm2_matrix,
            2
        )
        print_position_parameters(xbpm2_adj, xbpm2_matrix, 2)

        plot_comparison_data(
            xbpm1_pos,
            xbpm2_pos,
            args.init_date,
            args.end_date,
            args.beamline,
            xbpms_dist
            )

    else:
        pvnames = set_pv_names(prefix, args.xbpm_number)

        adj_data = [get_pv_value(pv) for pv in pvnames["ADJUST"]]
        matrix = get_pv_value(pvnames["MATRIX"])
        # Fetch data from the EPICS archiver for the specified time range
        data = get_pvdata(
            pvnames["BLADES"], args.init_date, args.end_date, timeout=10
        )

        positions = calculate_positions(
            data,
            adj_data,
            matrix,
            args.xbpm_number
            )
        print_position_parameters(adj_data, matrix, args.xbpm_number)
        plot_simple_data(
            positions,
            args.init_date,
            args.end_date,
            args.beamline,
            args.xbpm_number
            )

    plt.show()


if __name__ == "__main__":
    main()
