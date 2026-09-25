"""Human-readable summary of a DataAnalysis for the GUI info panel."""

from xbpm_bumps.core.data_structure import (
    BPMAnalysis,
    DataAnalysis,
    AnalyzedPositions,
    AllScales,
    CentralSweeps,
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


def _err(value, err) -> str:
    """Format a value with its associated error.

    Args:
        value: The central value.
        err: The associated error.

    Returns:
        A string representation in the form "value (err)".
    """
    return f"{_f(value)} ({_f(err)})"


def format_analysis_info(
        analysis: "DataAnalysis",
        tab_key: str
        ) -> str:
    """Format the analysis information for a given tab.

    Args:
        analysis: The DataAnalysis object containing the results.
        tab_key: The key of the tab for which to format the information.

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
        _supmat(lines, analysis.supmat, tab_key)
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
    """Format XBPM statistics for the given positions and append to lines."""
    lines.append("\n### BPM Statistics:\n")


    lines.append(
        "\n*  ROI Slice :"
        f" H = {_f(bpm.rms_diff.roislice.sz_h)},\t"
        f" V = {_f(bpm.rms_diff.roislice.sz_v)}\n"
    )
    for case, allroi in [("All", bpm.rms_diff.all),
                         ("ROI", bpm.rms_diff.roi)]:
        lines.append(
            f"  {case} (min / avg / max):\n"
            f"\t H = {_f(allroi.min_h)} / "
            f" {_f(allroi.mean_h)} / "
            f" {_f(allroi.max_h)}\n"
            f"\t V = {_f(allroi.min_v)} / "
            f" {_f(allroi.mean_v)} / "
            f" {_f(allroi.max_v)}\n"
        )
    return lines


def _xbpm_stats(
        lines: list[str],
        positions: "AnalyzedPositions",
        ) -> None:
    """Format XBPM statistics for the given positions and append to lines."""
    lines.append("\n### XBPM Statistics:\n")

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

    for title, stat in stat_table:
        lines.append(f"\n* {title}:\n")
        lines.append(
            "\n*  ROI Slice :"
            f" H = {_f(stat.roislice.sz_h)},\t"
            f" V = {_f(stat.roislice.sz_v)}\n"
        )
        for case, allroi in [("All", stat.all), ("ROI", stat.roi)]:
            lines.append(
                f"  {case} (min / avg / max):\n"
                f"\t H = {_f(allroi.min_h)} / "
                f" {_f(allroi.mean_h)} / "
                f" {_f(allroi.max_h)}\n"
                f"\t V = {_f(allroi.min_v)} / "
                f" {_f(allroi.mean_v)} / "
                f" {_f(allroi.max_v)}\n"
            )
    return lines


def _sweeps(
        lines : list[str],
        sweeps: "CentralSweeps",
        ) -> None:
    """Format central sweeps for the given positions and append to lines."""
    pass


def _blades(
        lines: list[str],
        blades: dict,
        ) -> None:
    """Format blade analysis for the given positions and append to lines."""
    pass


def _supmat(
        lines: list[str],
        supmat: "SuppressionMatrix",
        ) -> None:
    """Format supplementary material analysis for the given positions and append to lines."""
    pass
