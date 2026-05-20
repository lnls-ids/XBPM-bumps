#!/bin/env python3

"""Get relationships between XBPM suppression matrixes.

This script uses the suppression matrixes obtained by the XBPM analisys
to calculate and check if there is a significant relationship between matrixes
given they were acquired with different gaps for the same beamline undulator.
"""

import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

desc = """This script uses the suppression matrixes obtained by the XBPM analisys
to calculate and check if there is a significant relationship between matrixes
given they were acquired with different gaps for the same beamline undulator."""
coefs = ['B', 'C', 'D']

def _check_file_path(path):
    if not glob.glob(f"{path}"):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    if not glob.glob(f"{path}/*.h5") and not glob.glob(f"{path}/*.hdf5"):
        raise FileNotFoundError(f"No HDF5 files found in the specified path: {path}")


def _check_hdf5_structure(file):
    required_groups = [
        f"/analysis_{args.xbpm}/matrices/calculated", 
        f"/analysis_{args.xbpm}/matrices/stddev", 
        "/raw_data/sweep_0000"
        ]
    with h5py.File(file, "r") as f:
        for group in required_groups:
            if group not in f:
                raise KeyError(f"Required group '{group}' not found in file: {file}")
            if group == "/raw_data/sweep_0000":
                if f[group].attrs.get(f'{args.xbpm[:3].lower()} gap') is None:
                    raise KeyError(f"Required attribute '{args.xbpm[:3].lower()} gap' not"
                                   "found in group '/raw_data/sweep_0000' of file: {file}")


def _get_matrixes_coefs(data, row, col):
    idx = list()
    for i in data:
        matrix = abs(data[i]['supmat'])
        idx.append(matrix[row, col])
    return idx


def _get_matrixes_errors(data, row, col):
    errors = list()
    for i in data:
        stddev = data[i]['stddev']
        errors.append(stddev[row, col])
    return errors


def _fit_coef_curve(phase, coefs, degree):
    fit = np.polyfit(phase, coefs, degree)
    print(f"{degree}-degree fit coefficients: {fit}")
    return np.polyval(fit, phase)


def _plot_coefs_results(ax, phase, coefs, fitted_1vals, fitted_2vals, errors, title):
    ax.plot(phase, fitted_1vals, label='Linear Fit', color='red', linestyle='--', marker='x')
    ax.plot(phase, fitted_2vals, label='Quadratic Fit', color='orange', linestyle='--', marker='s')
    ax.errorbar(phase, coefs, yerr=errors, fmt='o-', color='tab:blue', capsize=2, label='Matrix Coeffs.')
    ax.set_xticks(phase)
    ax.set_xlabel('Phase/Gap (mm)')
    ax.set_ylabel('Coefficient value (a.u.)')
    ax.set_title(title)
    ax.grid()
    ax.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="matrix_gaps_analysis", description=desc)
    parser.add_argument("-x", "--xbpm", required=True, help="XBPM name (e.g., MNC1, MNC2)")
    parser.add_argument("-p", "--path", required=True, help="Path to XBPM HDF5 processed data files")
    args = parser.parse_args()

    # Read HDF5 files containing the suppression matrixes
    xbpm_data = dict()
    idx = 0
    _check_file_path(f"{args.path}")
    for file in glob.glob(f"{args.path}/*.h5"):
        print(f"Found file: {file}")
        _check_hdf5_structure(file)
        with h5py.File(file, "r") as f:
            xbpm_data[idx] = {}
            xbpm_data[idx]['supmat'] = f[f"/analysis_{args.xbpm}/matrices/calculated"][:]
            xbpm_data[idx]['stddev'] = f[f"/analysis_{args.xbpm}/matrices/stddev"][:]
            gap = f["/raw_data/sweep_0000"].attrs[f'{args.xbpm[:3].lower()} gap']
            xbpm_data[idx]['gap'] = round(gap, 4)
            idx += 1

    xbpm_data = dict(sorted(xbpm_data.items(), key=lambda item: item[1]['gap']))
    print(f"{args.xbpm} data sorted by phase/gap:\n {xbpm_data}")

    # Plot coeffitiens vs phase with linear fit
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    fig.suptitle(f"{args.xbpm} XBPM Suppression Matrices Coeffs. & Undulator Phase/Gap Relationship", fontsize=16)
    for i in range(0, 4, 2):
        for j in range(1, 4):
            print(f"\nCoefficient a[{i}{j}]")
            coef = _get_matrixes_coefs(xbpm_data, i, j)
            print(f"Coefficients for each phase/gap: {coef}")
            errors = _get_matrixes_errors(xbpm_data, i, j)
            print(f"Standard deviations for each phase/gap: {errors}")
            fitted_curve_1 = _fit_coef_curve([xbpm_data[idx]['gap'] for idx in xbpm_data], coef, degree=1)
            fitted_curve_2 = _fit_coef_curve([xbpm_data[idx]['gap'] for idx in xbpm_data], coef, degree=2)
            if (i//2) < 1:
                _plot_coefs_results(ax[i//2, j-1], [xbpm_data[idx]['gap'] for idx in xbpm_data], coef, 
                                    fitted_curve_1, fitted_curve_2, errors, f'{chr(916)}x {coefs[j-1]} vs Phase/Gap')
            else:
                _plot_coefs_results(ax[i//2, j-1], [xbpm_data[idx]['gap'] for idx in xbpm_data], coef, 
                                    fitted_curve_1, fitted_curve_2, errors, f'{chr(916)}y {coefs[j-1]} vs Phase/Gap')

    plt.tight_layout()
    plt.show()
