"""Human-readable summary of a DataAnalysis for the GUI info panel."""

import numpy as np

from xbpm_bumps.core.data_structure import (
    BPMAnalysis,
    DataAnalysis,
    AnalyzedPositions,
    AllScales,
    CentralSweeps,
    RMSGridStatistics,
    SuppressionMatrix,
    )


def _f(
        value : float,
        digits: int = 6
        ) -> str:
    """Format a float value with a given number of significant digits.

    Args:
        value: The float value to format.
        digits: The number of significant digits.
    
    Returns:
        The formatted float as a string.
    """
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _err(value: float, err: float) -> str:
    """Format a value with its associated error.

    Args:
        value: The central value.
        err: The associated error.

    Returns:
        A string representation in the form "value (err)".
    """
    return f"{_f(value)} ({_f(err, 2)})"


def format_analysis_info(analysis: "DataAnalysis") -> str:
    """Format the analysis information for a given DataAnalysis object.

    Args:
        analysis: The DataAnalysis object containing the results.

    Returns:
        A human-readable string summarizing the analysis for the specified tab.
    """
    if analysis is None:
        return "No analysis available yet."

    lines: list[str] = []
    if analysis.positions is not None:
        _scales(lines, analysis.scales)
        _xbpm_stats(lines, analysis.positions)

    if analysis.bpm is not None:
        _bpm_stats(lines, analysis.bpm)

    if analysis.centralsweeps is not None:
        _sweeps(lines, analysis.centralsweeps)

    if analysis.bladecenter is not None:
        _blades(lines, analysis.bladecenter)

    if analysis.supmat is not None:
        _supmat(lines, analysis.supmat)

    return "\n".join(lines)


def _scales(
        lines: list[str],
        scales: "AllScales"
        ) -> None:
    """Format the scale information for the positions and append to lines."""
    scl_case = [
        ("Raw Pairwise",         scales.raw_pw),
        ("Transformed Pairwise", scales.trn_pw),
        ("Raw Crossed",          scales.raw_cr),
        ("Transformed Crossed",  scales.trn_cr),
    ]
    for name, scl in scl_case:
        # Example implementation, replace with actual logic
        sclset = [
            ("kx", scl.kx, scl.skx),
            ("dx", scl.dx, scl.sdx),
            ("ky", scl.ky, scl.sky),
            ("dy", scl.dy, scl.sdy),
            ("qx", scl.qx, scl.sqx),
            ("qy", scl.qy, scl.sqy)
        ]
        lines.append(f"\n### Scales, {name}:\n")
        for name, value, error in sclset:
            lines.append(f"  {name}  = {_f(value)}\t ({_f(error)})\n")
        lines.append("\n")


def _bpm_stats(
        lines: list[str],
        bpm: "BPMAnalysis",
        ) -> None:
    """Format XBPM statistics for the given positions and append to lines.
    
    Args:
        lines : list[str], string table to append the formatted statistics to
        bpm   : "BPMAnalysis", the BPM analysis results to format
    """
    lines.append("\n### BPM Statistics:\n")
    _stat_block(lines, bpm.rms_diff)
    return lines


def _xbpm_stats(
        lines: list[str],
        positions: "AnalyzedPositions",
        ) -> None:
    """Format XBPM statistics for the given positions and append to lines."""
    lines.append("\n### XBPM Statistics:\n")

    # Table for pairwise and cross statistics, with and without transformation
    stat_table =[
        ("Pairwise Standard Deviation, not transformed",
         positions.pairw.stat_std),
        ("Pairwise Standard Deviation, transformed",
         positions.pairw.stat_trn),
        ("Cross Standard Deviation, not transformed",
         positions.cross.stat_std),
        ("Cross Standard Deviation, transformed",
         positions.cross.stat_trn),

    ]
    # Run through the statistics table and format each block.
    for title, stat in stat_table:
        lines.append(f"\n* {title}:\n")
        _stat_block(lines, stat)
    return lines


def _stat_block(lines: list[str],
                stat: "RMSGridStatistics",
                ) -> None:
    """Format a single block of RMS grid statistics and append to lines.
    
    Args:
        lines : list[str], the string table
        stat  : "RMSGridStatistics", the RMS grid statistics to format
    """
    lines.append(
        "\n*  ROI Slice :"
        f" H = {_f(stat.roislice.sz_h)},\t"
        f" V = {_f(stat.roislice.sz_v)}\n"
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


def _sweeps(
        lines : list[str],
        sweeps: "CentralSweeps",
        ) -> None:
    """Format central sweeps for the given positions and append to lines.
    
    Args:
        lines  : list[str], string table to append the formatted statistics to
        sweeps : "CentralSweeps", the central sweeps to format
    """
    lines.append("\n### Central Sweeps:\n")
    if sweeps.h is not None:
        lines.append("  H sweep:\n \t")
        coeffs_h = sweeps.h.coeffs
        sigmas_h = sweeps.h.sigmas
        lines.append(f"kx : {_err(coeffs_h[0], sigmas_h[0])}\t")
        lines.append(f"dx : {_err(coeffs_h[1], sigmas_h[1])}\n")

    if sweeps.v is not None:
        lines.append("  V sweep:\n \t")
        coeffs_v = sweeps.v.coeffs
        sigmas_v = sweeps.v.sigmas
        lines.append(f"ky : {_err(coeffs_v[0], sigmas_v[0])}\t")
        lines.append(f"dy : {_err(coeffs_v[1], sigmas_v[1])}\n")


def _blades(
        lines: list[str],
        blades: dict,
        ) -> None:
    """Format blade analysis for the given positions and append to lines.
    
    Args:
        lines  : list[str], string table to append the formatted statistics to
        blades : dict, the blade central analysis coefficients to format
    """
    lines.append("\n### Blade Analysis ###\n")
    for direction, bl in blades.items():
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


def _supmat(
        lines: list[str],
        supmat: "SuppressionMatrix",
        ) -> None:
    """Format suppression matrix.
    
    Args:
        lines  : list[str], strings to append to output 
        supmat : SuppressionMatrix, the suppression matrices to format
    """
    lines.append("\n### Suppression Matrix:\n")
    for mat, err, title in [
        (supmat.standard,   None,          "Standard"),
        (supmat.calculated, supmat.stddev, "Calculated"),
        (supmat.optimized,  None,          "Optimized")
        ]:
        _matrix_print(lines, mat, err, title)


def _matrix_print(lines: list[str],
                  mat: "np.ndarray",
                  err: "np.ndarray | None",
                  title: str):
    """Print the given matrix with optional error matrix and title.
    
    Args:
        mat : numpy.ndarray
        err : numpy.ndarray or None
        title : str
    """
    lines.append(f"\n### {title} :\n")
    if mat is None:
        lines.append(f"### WARNING: Matrix {title} is not defined.\n")
        return

    mm, nn = mat.shape
    for ii in range(mm):
        for jj in range(nn):
            if err is not None:
                lines.append(_err(mat[ii, jj], err[ii, jj]))
            else:
                lines.append(_f(mat[ii, jj]))
        lines.append("\n")

