"""Human-readable summary of a DataAnalysis for the GUI info panel."""

import numpy as np

from xbpm_bumps.core.data_structure import (
    BCA_HV,
    BPMAnalysis,
    DataAnalysis,
    Scales,
    CentralSweeps,
    RMSGridStatistics,
    SuppressionMatrix,
    GL2RMatrix
    )


def _f(
        value : float,
        digits: int = 2
        ) -> str:
    """Format a float value with a given number of significant digits.

    Args:
        value: The float value to format.
        digits: The number of significant digits.
    
    Returns:
        The formatted float as a string.
    """
    try:
        return f"{float(value):6.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _err(
        value: float,
        err: float,
        digits: int = 1,
        ) -> str:
    """Format a value with its associated error.

    Args:
        value: The central value.
        err: The associated error.

    Returns:
        A string representation in the form "value (err)".
    """
    return f"{_f(value)} ({float(err):.{digits}g})"


def format_analysis_info(
        analysis: "DataAnalysis",
        ) -> dict[str, str]:
    """Format the analysis information for a given DataAnalysis object.

    Args:
        analysis : The DataAnalysis object containing the results.

    Returns:
        A dictionary mapping tab names to human-readable strings summarizing
        the analysis for each tab.
    """
    if analysis is None:
        return {}

    # BPM and central sweeps information.
    tab_info = {
        "bpm" : (
            _bpm_stats(analysis.bpm)
            if analysis.bpm is not None else ""
            ),
        "blade_central_sweeps": (
            _blades(analysis.bladecenter)
            if analysis.bladecenter is not None else ""
            ),
        "position_central_sweeps" : (
            _sweeps(analysis.centralsweeps)
            if analysis.centralsweeps is not None else ""
            ),
    }

    # Add position statistics to the tab information.
    if analysis.positions is None:
        return tab_info

    # Table for pairwise and cross position statistics.
    anpos = analysis.positions
    stat_table = {
        "xbpm_pairwise_raw" : (
        "Pairwise Standard Deviation, not transformed",
        anpos.pairw.scale_std,
        anpos.pairw.stat_std,
        "std"
        ),
        "xbpm_pairwise_trn" : (
        "Pairwise Standard Deviation, transformed",
        anpos.pairw.scale_trn,
        anpos.pairw.stat_trn,
        "pw_tr"
        ),
        "xbpm_cross_raw" : (
        "Cross Standard Deviation, not transformed",
        anpos.cross.scale_std,
        anpos.cross.stat_std,
        "cr_std"
        ),
        "xbpm_cross_trn" : (
        "Cross Standard Deviation, transformed",
        anpos.cross.scale_trn,
        anpos.cross.stat_trn,
        "cr_rot"
        ),
    }

    supmat_lines = _supmat(analysis.supmat, analysis.gl2r)
    for tab_name, (description, scales, stat, mat) in stat_table.items():
        tab_info[tab_name] = (
            description + '\n' +
            _scales(scales) +
            _xbpm_stats(stat) +
            supmat_lines.get(mat, "")
        )

    return tab_info


def _bpm_stats(
        bpm: "BPMAnalysis",
        ) -> str:
    """Format XBPM statistics for the given positions.
    
    Args:
        bpm : "BPMAnalysis", the BPM analysis results to format

    Returns:
        A string representing the formatted BPM statistics.
    """
    lines = []
    lines.append("\n### BPM Statistics:\n")
    lines.append(_xbpm_stats(bpm.rms_diff))
    return ''.join(lines)


def _scales(
        scl: "Scales"
        ) -> str:
    """Format the scale information for the positions.
    
    Args:
        scl : "Scales", the scale information to format

    Returns:
        A string representing the formatted scale information.
    """
    sclset = {
        "kx" : (scl.kx, scl.skx),
        "dx" : (scl.dx, scl.sdx),
        "ky" : (scl.ky, scl.sky),
        "dy" : (scl.dy, scl.sdy),
        "qx" : (scl.qx, scl.sqx),
        "qy" : (scl.qy, scl.sqy)
    }
    headline = "\n### Scales:\n\n"
    lines = [
        f"  {name}  = {_err(value, error)}"
        for name, (value, error) in sclset.items()
        ]
    return headline + '\n'.join(lines)


def _xbpm_stats(
                stat: "RMSGridStatistics",
                ) -> str:
    """Format a single block of RMS grid statistics and append to lines.
    
    Args:
        stat  : "RMSGridStatistics", the RMS grid statistics to format
    """
    lines = []
    lines.append(
        "\n*  ROI Slice :"
        f" H = {_f(stat.roislice.size_h)},\t"
        f" V = {_f(stat.roislice.size_v)}\n"
    )
    for case, allroi in [("All", stat.all),
                         ("ROI", stat.roi)]:
        lines.append(
            f"  {case} (min / avg / max):\n"
            f"\t H = {_f(allroi.min_h)} / "
            f" {_f(allroi.mean_h)} / "
            f" {_f(allroi.max_h)}\n"
            f"\t V = {_f(allroi.min_v)} / "
            f" {_f(allroi.mean_v)} / "
            f" {_f(allroi.max_v)}\n"
        )
    return ''.join(lines)


def _sweeps(
        sweeps: "CentralSweeps",
        ) -> str:
    """Format central sweeps for the given positions.

    Args:
        sweeps : "CentralSweeps", the central sweeps to format
    """
    lines = []
    lines.append("\n### Central Sweeps:\n")
    if sweeps.h is not None:
        lines.append("  H sweep:\n \t")
        coeffs_h = sweeps.h.coeffs
        sigmas_h = sweeps.h.sigmas
        if coeffs_h is not None:
            lines.append(f"kx : {_err(coeffs_h[0], sigmas_h[0])}\t")
            lines.append(f"dx : {_err(coeffs_h[1], sigmas_h[1])}\n")

    if sweeps.v is not None:
        lines.append("  V sweep:\n \t")
        coeffs_v = sweeps.v.coeffs
        sigmas_v = sweeps.v.sigmas
        if coeffs_v is not None:
            lines.append(f"ky : {_err(coeffs_v[0], sigmas_v[0])}\t")
            lines.append(f"dy : {_err(coeffs_v[1], sigmas_v[1])}\n")
    return ''.join(lines)


def _blades(
        blades: "BCA_HV",
        ) -> str:
    """Format blade analysis for the given positions.
    
    Args:
        blades : dict, the blade central analysis coefficients to format
    """
    lines = []
    lines.append("\n### Blade Analysis ###\n")
    for direction, bl in blades:
        lines.append(f"  {direction.capitalize()} sweep:\n")
        lines.append(f"  TO : k = {_err(bl.to.k, bl.to.sk)},\t"
                     f"   delta = {_err(bl.to.d, bl.to.sd)}\n"
                     f"  TI : k = {_err(bl.ti.k, bl.ti.sk)},\t"
                     f"   delta = {_err(bl.ti.d, bl.ti.sd)}\n"
                     f"  BI : k = {_err(bl.bi.k, bl.bi.sk)},\t"
                     f"   delta = {_err(bl.bi.d, bl.bi.sd)}\n"
                     f"  BO : k = {_err(bl.bo.k, bl.bo.sk)},\t"
                     f"   delta = {_err(bl.bo.d, bl.bo.sd)}\n\n"
                     )
    lines.append("###\n")
    return ''.join(lines)


def _supmat(
        supmat: "SuppressionMatrix",
        rotmat: GL2RMatrix,
        ) -> dict[str, str]:
    """Format suppression matrix and return as a dictionary of strings.
    
    Args:
        supmat : SuppressionMatrix, the suppression matrix to format
    """
    if supmat is None:
        return {k : "" for k in ["std", "pw_tr", "cr_std", "cr_rot"]}

    lines = {
        "std":   _matrix_print("Standard",  supmat.standard,  None),
        "pw_tr": _matrix_print("Calculated", supmat.calculated, supmat.stddev),
    }
    if rotmat is not None:
        lines["cr_std"] = _matrix_print("Cross Standard", rotmat.std,  None)
        lines["cr_rot"] = _matrix_print("Cross Rotation", rotmat.calc, None)
    else:
        lines["cr_std"] = ""
        lines["cr_rot"] = ""
    return lines


def _matrix_print(
        title: str,
        mat: "np.ndarray",
        err: "np.ndarray | None",
        ) -> str:
    """Print the given matrix with optional error matrix and title.
    
    Args:
        title : str
        mat : numpy.ndarray
        err : numpy.ndarray or None
    """
    if mat is None:
        return f"### WARNING: Matrix {title} is not defined.\n"

    lines = [f"\n### Matrix - {title}:\n"]
    mm, nn = mat.shape
    for ii in range(mm):
        row = []
        for jj in range(nn):
            row.append(
                _err(mat[ii, jj], err[ii, jj])
                if err is not None
                else _f(mat[ii, jj])
                )
        lines.append(" " + "  ".join(row) + "\n")
    return ''.join(lines)

