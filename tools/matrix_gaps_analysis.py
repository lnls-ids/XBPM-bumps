#!/bin/env python3

"""
Get relationships between XBPM suppression matrices.

This script uses the suppression matrices obtained from the XBPM analisys
to fit curves, check standard deviations, and plot the results to check if 
there is a significant relationship between matrices calculated for different 
gaps/phases from the acquisitions of the same beamline undulator.
"""

import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

desc = """This script uses the suppression matrices obtained from the XBPM analisys
to fit curves, check standard deviations, and plot the results to check if 
there is a significant relationship between matrices calculated for different 
gaps/phases from the acquisitions of the same beamline undulator."""
coefs = ['B', 'C', 'D']

def _check_file_path(path):
    if not glob.glob(f"{path}"):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    if not glob.glob(f"{path}/*.h5") and not glob.glob(f"{path}/*.hdf5"):
        raise FileNotFoundError(f"No HDF5 files found in the specified path: {path}")


def _check_hdf5_structure(file, xbpm):
    required_groups = [
        f"/analysis_{xbpm}/matrices/calculated", 
        f"/analysis_{xbpm}/matrices/stddev", 
        "/raw_data/sweep_0000"
        ]
    with h5py.File(file, "r") as f:
        for group in required_groups:
            if group not in f:
                raise KeyError(f"Required group '{group}' not found in file: {file}")
            if group == "/raw_data/sweep_0000":
                if f[group].attrs.get(f'{xbpm[:3].lower()} gap') is None:
                    raise KeyError(f"Required attribute '{xbpm[:3].lower()} gap' not"
                                   "found in group '/raw_data/sweep_0000' of file: {file}")


def _load_xbpm_data(path, xbpm):
    # Read HDF5 files containing the suppression matrices
    xbpm_data = dict()
    idx = 0
    _check_file_path(f"{path}")
    for file in glob.glob(f"{path}/*.h5"):
        print(f"Found file: {file}")
        _check_hdf5_structure(file, xbpm)
        with h5py.File(file, "r") as f:
            xbpm_data[idx] = {}
            xbpm_data[idx]['supmat'] = f[f"/analysis_{xbpm}/matrices/calculated"][:]
            xbpm_data[idx]['stddev'] = f[f"/analysis_{xbpm}/matrices/stddev"][:]
            gap = f["/raw_data/sweep_0000"].attrs[f'{xbpm[:3].lower()} gap']
            xbpm_data[idx]['gap'] = round(gap, 4)
            idx += 1

    xbpm_data = dict(sorted(xbpm_data.items(), key=lambda item: item[1]['gap']))
    print(f"{xbpm} data sorted by phase/gap:\n {xbpm_data}")

    return xbpm_data


def _get_matrices_coefs(data, row, col):
    idx = list()
    for i in data:
        matrix = abs(data[i]['supmat'])
        idx.append(matrix[row, col])
    return idx


def _get_matrices_errors(data, row, col):
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
    parser.add_argument("-c", "--compare_xbpm", help="Other XBPM name for comparison (e.g., MNC1, MNC2)."\
                        " It can be the same as --xbpm to compare different datasets of the same XBPM.")
    parser.add_argument("-cp", "--compare_path", help="Path to other XBPM HDF5 processed data files for comparison")
    args = parser.parse_args()

    # Load XBPM data
    xbpm_data = _load_xbpm_data(args.path, args.xbpm)

    # Plot coeffitiens vs phase with linear fit
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    fig.suptitle(f"{args.xbpm} XBPM Suppression Matrices Coeffs. & Undulator Phase/Gap Relationship", fontsize=16)
    for i in range(0, 4, 2):
        for j in range(1, 4):
            print(f"\nCoefficient a[{i}{j}]")
            coef = _get_matrices_coefs(xbpm_data, i, j)
            print(f"Coefficients for each phase/gap: {coef}")
            errors = _get_matrices_errors(xbpm_data, i, j)
            print(f"Standard deviations for each phase/gap: {errors}")
            fitted_curve_1 = _fit_coef_curve([xbpm_data[idx]['gap'] for idx in xbpm_data], coef, degree=1)
            fitted_curve_2 = _fit_coef_curve([xbpm_data[idx]['gap'] for idx in xbpm_data], coef, degree=2)
            if (i//2) < 1:
                _plot_coefs_results(ax[i//2, j-1], [xbpm_data[idx]['gap'] for idx in xbpm_data], coef, 
                                    fitted_curve_1, fitted_curve_2, errors, f'{chr(916)}x {coefs[j-1]} vs Phase/Gap')
            else:
                _plot_coefs_results(ax[i//2, j-1], [xbpm_data[idx]['gap'] for idx in xbpm_data], coef, 
                                    fitted_curve_1, fitted_curve_2, errors, f'{chr(916)}y {coefs[j-1]} vs Phase/Gap')

    if args.compare_xbpm != None:
        if args.compare_xbpm == args.xbpm and args.compare_path != args.path:
            print("\nComparing different datasets of the same XBPM...")
        elif args.compare_xbpm == args.xbpm and args.compare_path == args.path:
            raise ValueError("Please provide a different dataset for comparison.")
        else:
            print(f"\nComparing with {args.compare_xbpm} XBPM...")
        
        comparison_xbpm_data = _load_xbpm_data(args.compare_path, args.compare_xbpm)

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        fig.suptitle(f"Comparison of XBPMs Suppression Matrices Coefficients Relationship", fontsize=16)
        for i in range(0, 4, 2):
            for j in range(1, 4):
                coef_1 = _get_matrices_coefs(xbpm_data, i, j)
                errors_1 = _get_matrices_errors(xbpm_data, i, j)
                coef_2 = _get_matrices_coefs(comparison_xbpm_data, i, j)
                errors_2 = _get_matrices_errors(comparison_xbpm_data, i, j)
                phase_1 = [xbpm_data[idx]['gap'] for idx in xbpm_data]
                phase_2 = [comparison_xbpm_data[idx]['gap'] for idx in comparison_xbpm_data]

                if (i//2) < 1:
                    ax[i//2, j-1].errorbar(
                        phase_1, 
                        coef_1, 
                        yerr=errors_1, 
                        fmt='o-',
                        color='tab:blue',
                        capsize=2,
                        label=f'{args.xbpm} Coeffs.'
                    )
                    
                    ax[i//2, j-1].errorbar(
                        phase_2,
                        coef_2,
                        yerr=errors_2,
                        fmt='s-',
                        color='tab:orange',
                        capsize=2,
                        label=f'{args.compare_xbpm} Coeffs.'
                    )
                    
                    ax[i//2, j-1].set_title(f'{chr(916)}x {coefs[j-1]} vs Phase/Gap')
                else:
                    ax[i//2, j-1].errorbar(
                        phase_1, 
                        coef_1, 
                        yerr=errors_1, 
                        fmt='o-', 
                        color='tab:blue', 
                        capsize=2, 
                        label=f'{args.xbpm} Coeffs.'
                    )

                    ax[i//2, j-1].errorbar(
                        phase_2,
                        coef_2,
                        yerr=errors_2,
                        fmt='s-',
                        color='tab:orange',
                        capsize=2,
                        label=f'{args.compare_xbpm} Coeffs.'
                    )

                    ax[i//2, j-1].set_title(f'{chr(916)}y {coefs[j-1]} vs Phase/Gap')
                ax[i//2, j-1].set_xticks(sorted(set(phase_1 + phase_2)))
                ax[i//2, j-1].set_xlabel('Phase/Gap (mm)')
                ax[i//2, j-1].set_ylabel('Coefficient value (a.u.)')
                ax[i//2, j-1].grid()
                ax[i//2, j-1].legend()

    plt.tight_layout()
    plt.show()
