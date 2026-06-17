#!/bin/env python3

"""
Calculate positions from MNC1 and MNC2 XBPMs using their estimated
suppression matrixes and compare with BPM readings.

>>> Run this script under "sirius' environment, to use
>>> pymodels and pyaccel libraries.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from epics import caget, camonitor

os.environ["EPICS_CA_ADDR_LIST"] = "10.0.38.59:62000 10.30.13.22 10.30.14.19"

BPM_M1 = {
    "X":"SI-09M1:DI-BPM:PosX-Mon",
    "Y":"SI-09M1:DI-BPM:PosY-Mon"
}
BPM_M2 = {
    "X":"SI-09M2:DI-BPM:PosX-Mon",
    "Y":"SI-09M2:DI-BPM:PosY-Mon"
}

MNC1_supmat = np.array([[1, -0.2029, -0.2420,  0.6981],
                        [1,  0.2029,  0.2420,  0.6981],
                        [1,  2.3967, -1.0477, -1.9829],
                        [1,  2.3967,  1.0477,  1.9829]])

MNC2_supmat = np.array([[1, -0.2204, -0.3916,  0.1056],
                        [1,  0.2204,  0.3916,  0.1056],
                        [1,  0.9625, -1.0029, -0.3625],
                        [1,  0.9625,  1.0029,  0.3625]])

def calculate_positions(device_name, supmat, kx=1, deltax=0, ky=1, deltay=0):
    # Get the PVs for the four blades
    blade_pvs = {
        "TO": f"{device_name}:AmplA-Mon",
        "TI": f"{device_name}:AmplB-Mon",
        "BI": f"{device_name}:AmplC-Mon",
        "BO": f"{device_name}:AmplD-Mon"
    }
    
    # Read the positions from the PVs
    positions = {blade: caget(pv) for blade, pv in blade_pvs.items()}
    num_positions = np.array(list(positions.values()))

    print(f"Positions for {device_name}: {positions}")

    corr_values = supmat @ num_positions
    print(corr_values)
    xpos = (corr_values[0]/corr_values[1]*kx) + deltax
    ypos = (corr_values[2]/corr_values[3]*ky) + deltay
    
    return xpos, ypos

if __name__ == "__main__":
    MNC1 = "SI-09SAFE:DI-PBPM-1"
    MNC2 = "SI-09SAFE:DI-PBPM-2"

    xpos1, ypos1 = calculate_positions(MNC1, MNC1_supmat, 5063.57, -3550.25, 6036.06, -100.71)
    print(f"Calculated X Position: {xpos1} um")
    print(f"Calculated Y Position: {ypos1} um") 

    xpos2, ypos2 = calculate_positions(MNC2, MNC2_supmat, 2346.18, -1329.13, 7304.28, -376.33)
    print(f"Calculated X Position: {xpos2} um")
    print(f"Calculated Y Position: {ypos2} um")

    bpm_m1_x = caget(BPM_M1["X"])
    bpm_m1_y = caget(BPM_M1["Y"])
    bpm_m2_x = caget(BPM_M2["X"])
    bpm_m2_y = caget(BPM_M2["Y"])

    x = [xpos1, xpos2, bpm_m1_x/1000, bpm_m2_x/1000]
    y = [ypos1, ypos2, bpm_m1_y/1000, bpm_m2_y/1000]
    labels = ['MNC1', 'MNC2', 'BPM M1', 'BPM M2']

    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.scatter(x, y)
    plt.title("MNC XBPM Positions")
    plt.xlabel("X Position (um)")
    plt.ylabel("Y Position (um)")
    plt.grid()
    plt.show()
