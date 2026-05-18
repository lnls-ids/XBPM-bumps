#!/bin/env python3

"""Get relationships between XBPM suppression matrixes.

This script uses the suppression matrixes obtained by the XBPM analisys
to calculate and check if there is a significant relationship between matrixes
given they were acquired with different gaps for the same beamline undulator.

>>> Run this script under "sirius' environment, to use
>>> pymodels and pyaccel libraries.
"""

import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Analyze relationships between XBPM suppression matrixes.")
parser.add_argument("--xbpm", help="XBPM name (e.g., MNC1, MNC2)")
parser.add_argument("--file_path", help="Path to XBPM data files")
args = parser.parse_args()

xbpm_data = {}
phase = [5.419, 2.818, 0.148, 4.536]

# Read HDF5 files containing the suppression matrixes
for file in glob.glob("/home/caio.santos/Documents/XBPM_analysis/MNC1/MNC1_*.h5"):
    print(f"Found file: {file}")
    with h5py.File(file, "r") as f:
        idx = int(file.split("_")[-1].split(".")[0])
        xbpm_data[idx] = {}
        xbpm_data[idx]['supmat'] = f["/analysis_MNC1/matrices/calculated"][:]
        xbpm_data[idx]['stddev'] = f["/analysis_MNC1/matrices/stddev"][:]
        xbpm_data[idx]['phase'] = phase[int(file.split("_")[-1].split(".")[0]) - 1]

print(f"MNC1 data: {xbpm_data}")

def _get_matrixes_coefs(data, row, col):
    idx = []
    for i in range(len(data)):
        matrix = data[i+1]['supmat']
        idx.append(matrix[row, col])
    return idx


def _get_matrixes_errors(data, row, col):
    errors = []
    for i in range(len(data)):
        stddev = data[i+1]['stddev']
        errors.append(stddev[row, col])
    return errors


def _fit_coef_curve(phase, coefs, degree):
    fit = np.polyfit(sorted(phase), coefs, degree)
    print(f"{degree}-degree fit coefficients: {fit}")
    return np.polyval(fit, sorted(phase))


def _plot_coefs_results(ax, phase, coefs, fitted_1vals, fitted_2vals, errors, title):
    ax.plot(sorted(phase), fitted_1vals, label='Linear Fit', linestyle='--', marker='x')
    ax.plot(sorted(phase), fitted_2vals, label='Quadratic Fit', linestyle='--', marker='x')
    ax.plot(sorted(phase), coefs, label='Matrixes Coefs.', marker='o')
    ax.errorbar(sorted(phase), coefs, yerr=errors, fmt='o', color='green', capsize=2, label='Propagated Error')
    ax.set_xticks(sorted(phase))
    ax.set_xlabel('Phase (mm)')
    ax.set_ylabel('Coefficient value (a.u.)')
    ax.set_title(title)
    ax.grid()
    ax.legend()


if __name__ == "__main__":
    # Plot coeffitiens vs phase with linear fit
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
    fig.suptitle("XBPM Suppression Matrixes Coefficients and Undulator Phase Correlations - MNC1", fontsize=16)
    for i in range(0, 4, 2):
        for j in range(1, 4):
            print(f"\nCoefficient a[{i}{j}]")
            coef = _get_matrixes_coefs(xbpm_data, i, j)
            print(f"Coefficients for each phase: {coef}")
            errors = _get_matrixes_errors(xbpm_data, i, j)
            print(f"Standard deviations for each phase: {errors}")
            fitted_curve_1 = _fit_coef_curve(sorted(phase), coef, degree=1)
            fitted_curve_2 = _fit_coef_curve(sorted(phase), coef, degree=2)
            if (i//2) < 1:
                _plot_coefs_results(ax[i//2, j-1], phase, coef, fitted_curve_1, fitted_curve_2, errors, f'{chr(916)}x Coef a[{i}{j}] vs Phase')
            else:
                _plot_coefs_results(ax[i//2, j-1], phase, coef, fitted_curve_1, fitted_curve_2, errors, f'{chr(916)}y Coef a[{i}{j}] vs Phase')

    plt.tight_layout()
    plt.show()
