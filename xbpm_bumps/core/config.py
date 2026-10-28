"""Beamline configuration constants."""

# from typing import Optional

from collections import defaultdict

class Config:
    """Beamline configuration and constants."""

    # Power relative to Ampere subunits.
    AMPSUB = {
        0    : 1.0,    # no unit defined.
        "0"  : 1.0,    # no unit defined.
        "mA" : 1e-3,   # mili
        "uA" : 1e-6,   # micro
        "nA" : 1e-9,   # nano
        "pA" : 1e-12,  # pico
        "fA" : 1e-15,  # femto
        "aA" : 1e-18,  # atto
    }

    # Map of blades positions in each XBPM.
    # TO, TI, BO, BI : top/bottom, in/out, relative to the storage ring;
    # A, B, C, D : names of respective P.V.s
    XBPM_BEAMLINES = [
        "CAT", "CAT1",
        "CNB", "CNB1",
        "MGN", "MGN1", "MGN2",
        "MNC", "MNC1", "MNC2",
        "SIMUL"
        ]
    BLADEMAP = {
        key : {"TO": "A", "TI": "B", "BI": "C", "BO": "D"}
        for key in XBPM_BEAMLINES
    }

    LEGACY_BLADEMAP = {
        "MNC1" : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
        "MNC2" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        "CAT1" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        "CNB1" : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
        "CNB2" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        # "MGN"  : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
        "MGN1" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        "MGN2" : {"TO": 'B', "TI": 'C', "BI": 'A', "BO": 'D'},
        "SIMUL": {"TO": 'A', "TI": 'B', "BI": 'C', "BO": 'D'},
        }
    # The name of the beamline equals to first XBPM id.
    for blade in ["CAT", "CNB", "MGN", "MNC"]:
        LEGACY_BLADEMAP[blade] = LEGACY_BLADEMAP[blade + "1"]

    # The XBPM beamlines.
    BEAMLINENAME = {
        "CAT" : "Caterete",
        "CNB" : "Carnauba",
        "MGN" : "Mogno",
        "MNC" : "Manaca",
        "N/A" : "Not defined",
    }

    # Distances between two adjacent BPMs around source of bump at each line.
    BPMDISTS = {
        "CAT" : 6.175495,
        "CNB" : 6.175495,
        "MGN" : 2.2769999999999015,
        "MNC" : 7.035495,
    }

    # Distance from source (its center) to XBPM at each beamline.
    # Obtained from comissioning reports.
    XBPMDISTS = {
        "CAT"  : 15.740,
        "CAT1" : 15.740,
        "CAT2" : 19.590,
        "CNB"  : 15.740,
        "CNB1" : 15.740,
        "CNB2" : 19.590,
        "MGN1" : 10.237,
        "MGN2" : 16.167,
        "MNC1" : 15.740,
        "MNC2" : 19.590,
    }

    # Sections of the ring for each beamline.
    SECTOR = {
        "CAT" : ["subsec:07SP", 7],
        "CNB" : ["subsec:06SB", 6],
        "MGN" : ["subsec:10BC", 10],
        "MNC" : ["subsec:09SA", 9]
    }

    # -----------------------------------------------------------------------
    # Graph titles, grouped by tab / figure.
    # Keys within each tab are the subplot role: 'total', 'roi', 'heatmap',
    # 'h' (horizontal panel), 'v' (vertical panel), 'suptitle'.
    # Values may be plain strings or format-strings with {beamline} / {xbpm}.
    # -----------------------------------------------------------------------
    PLOT_TITLES = {
        # BPM-derived positions tab
        "bpm": {
            "total"   : "BPM @ {beamline}",
            "roi"     : "BPM @ {beamline} (ROI)",
            "heatmap" : r"RMS $\Delta$ @ ROI",
        },

        # "Blade Map" tab
        "blade_map": {
            "suptitle" : "Blade Map – {beamline}",
        },

        # "Blades at sweeps" tab
        "blades_at_sweeps": {
            "h"        : "Horizontal",
            "v"        : "Vertical",
            "suptitle" : "Blade Currents at Center",
        },

        # "Positions along sweeps" tab
        "sweeps": {
            "h"        : "Central Horizontal Sweeps",
            "v"        : "Central Vertical Sweeps",
        },

        # XBPM position tabs (pairwise / cross, raw / transformed)
        "xbpm_positions": {
            "total"    : (
                r"XBPM{xbpmnum}@{beamline}: {ct}, {rort}"
                ),
            "roi"      : (
                r"XBPM{xbpmnum}@{beamline}: {ct}, {rort} (ROI)"
                ),
            "heatmap"  : (
                r"RMS $\Delta$ @ ROI"
                ),
        },
    }

    # Consistent naming for pairwise/cross and raw/transformed modes
    # used by non-plot UI text (analysis info box, exports, logs).
    POSITION_CALC_LABELS = {
        "pair"     : "Δ/Σ",
        "pairwise" : "Δ/Σ",
        "cross"    : "partial Δ/Σ",
    }

    POSITION_SCOPE_LABELS = {
        "raw"     : "raw",
        "scaled"  : "transformed",
    }

    # ------------------------------------------------------------------
    # HDF5 registry: key -> (stored name, description template).
    # Templates may use {beamline}, {bumpnum}, {figure}. Stored names must
    # not contain "/" (HDF5 path separator).
    # ------------------------------------------------------------------
    H5_GROUPS = {
        # Beamline level.
        "raw_data": (
            "Raw data",
            "Raw data for {beamline} beamline from XBPM bumping experiments."
            ),

        "analysis": (
            "Analysis",
            "Results of the analysis of {beamline} XBPM bumping experiments."
            ),

        # Raw data.
        "blade_averages": (
            "Blade averages",
            "Averaged blade currents at each grid position, with standard"
            " deviations."
            ),

        "bump": (
            "Bump", "Measurements for bump #{bumpnum} of {beamline}."
            ),

        "bpm_data": (
            "BPM data",
            "Positions registered by the BPMs during bump #{bumpnum}."
            ),

        "blade_data": (
            "Blade data",
            "Currents, ranges and saturation levels of the four blades during"
            " bump #{bumpnum}."
            ),

        # BPM analysis.
        "bpm_analysis": (
            "BPM analysis",
            "Beam positions at the XBPM site derived from BPM tangents and"
            " compared with the nominal grid."
            ),
        "bpm_positions": (
            "Positions",
            "Nominal and measured positions at the XBPM site."
            ),

        "rms_diff": (
            "RMS differences",
            "Differences between the reference and the calculated positions."
            ),

        "rms_all":  (
            "All grid",
            "Differences over the full grid."
            ),

        "rms_roi":  (
            "ROI",
            "Differences inside the ROI."
            ),

        "diff_h":   (
            "Horizontal differences",
            "Horizontal component of the differences."
            ),

        "diff_v":   (
            "Vertical differences",
            "Vertical component of the differences."
            ),

        "diff_tot": (
            "Total differences",
            "Total (horizontal and vertical) differences."
            ),

        "summary":  (
            "Summary",
            "Minimum, mean and maximum of the differences."
            ),

        # Blade map.
        "blade_map": (
            "Blade map",
            "Averaged blade currents mapped on the grid."
            ),

        "blades": (
            "Blades",
            "Blade currents and their standard deviations at each grid"
            " position."
            ),

        # Blade central sweep.
        "blade_center": (
            "Blade central sweep",
            "Linear fits to the blade currents along the central sweep lines."
            ),

        "direction_h": (
            "Horizontal",
            "Blade behavior along the horizontal central line."
            ),

        "direction_v": (
            "Vertical",
            "Blade behavior along the vertical central line."
            ),

        "fit_data": (
            "Fit data",
            "Sweep positions and the measured, error and fitted currents of"
            " the four blades."
            ),

        "coefficients": (
            "Coefficients",
            "Slope and intercept of the linear fit of each blade, with"
            " standard deviations."
            ),

        # XBPM calculated positions.
        "xbpm_positions": (
            "XBPM calculated positions",
            "Positions calculated with the delta-sigma and partial"
            " delta-sigma formulae, before and after transformation."),

        "pairwise": (
            "Delta-Sigma",
            "Positions from the delta/sigma (pairwise) formula."
            ),

        "cross": (
            "Partial Delta-Sigma",
            "Positions from the partial delta/sigma (cross-blade) formula."
            ),

        "xbpm_std": (
            "Standard",
            "Positions from the standard formulae."
            ),

        "xbpm_transformed": (
            "Transformed",
            "Positions after the suppression-matrix / GL2R transformation."
            ),

        "std_positions": (
            "Standard positions",
            "Scaled standard positions, with the nominal reference."
            ),

        "trn_positions": (
            "Transformed positions",
            "Scaled transformed positions, with the nominal reference."
            ),

        "pos_scales": (
            "Scales",
            "Scaling coefficients applied to the calculated grid positions."
            ),

        "statistics": (
            "Statistics",
            "RMS statistics of the calculated positions against the reference."
            ),

        # Central sweeps.
        "central_sweeps": (
            "Central sweeps",
            "Position behavior along the central sweeps."
            ),

        "sweep_h": (
            "Horizontal sweep",
            "Measurements along the horizontal central sweep."
            ),

        "sweep_v": (
            "Vertical sweep",
            "Measurements along the vertical central sweep."
            ),

        "sweep_data": (
            "Sweep data",
            "Blade currents and positions along the sweep."
            ),

        "sweep_coeffs": (
            "Fit coefficients",
            "Slope and intercept of the fixed-coordinate fit,"
            " with their standard deviations."
            ),

        # Scale factors.
        "scales": (
            "Scale factors",
            "Scaling factors of the four position calculations."
            ),

        "scales_pairwise": (
            "Delta-Sigma",
            "Scaling factors for the delta/sigma positions."
            ),

        "scales_cross": (
            "Partial Delta-Sigma",
            "Scaling factors for the partial delta/sigma positions."
            ),

        "scales_std": (
            "Standard",
            "Scaling factors for the standard positions."
            ),

        "scales_trn": (
            "Transformed",
            "Scaling factors for the transformed positions."
            ),

        # Matrices.
        "suppression": (
            "Suppression matrices",
            "Standard, calculated and optimized suppression"
            " matrices."
            ),

        "sup_standard": (
            "Standard",
            "Standard suppression matrix (fixed 1/-1 pattern)."
            ),

        "sup_calculated": (
            "Calculated",
            "Suppression matrix calculated from the blade"
            " fits."
            ),

        "sup_stddev": (
            "Standard deviations",
            "Standard deviations of the calculated suppression"
            " matrix."
            ),

        "sup_optimized": (
            "Optimized",
            "Suppression matrix optimized against the"
            " reference positions."
            ),

        "gl2r": (
            "GL2R matrices",
            "GL2R transformation matrices for the cross-blade"
            " positions."
            ),

        "gl2r_std":  (
            "Standard",
            "Standard (identity) GL2R matrix."
            ),

        "gl2r_calc": (
            "Calculated",
            "Calculated GL2R matrix."
            ),

        # Figures (dataset names are given explicitly: "<tab>.png").
        "figures": (
            "Figures",
            "Figures of the analysis, stored as PNG byte streams."
            ),

        "figure":  (
            "{figure}",
            "Figure '{figure}' of the analysis."
            ),
    }

    # ------------------------------------------------------------------
    # Column names: <axis | blade | PV> + <kind>.
    # ------------------------------------------------------------------
    BLADES = ("to", "ti", "bi", "bo")

    # Prefix of a column name: an axis, a blade, a PV or a direction.
    H5_SUBJECTS = {
        "x": "X",
        "y": "Y",

        "to": "TO",
        "ti": "TI",
        "bi": "BI",
        "bo": "BO",
    
        "A": "Blade A",
        "B": "Blade B",
        "C": "Blade C",
        "D": "Blade D",

        "h": "Horizontal",
        "v": "Vertical",
        "tot": "Total",
    }

    H5_COLUMNS = {
        # Position kinds (with an axis: "X nominal", "Y BPM", ...).
        "nom"   : "nominal",    
        "meas"  : "measured",
        "std"   : "standard",   
        "trn"   : "transformed",
        "bpm"   : "BPM",

        # Table kinds (with a blade or an axis).
        "mean"  : "mean",
        "err"   : "error",
        "raw"   : "raw",
        "fit"   : "fit",
        "fit_err": "fit error",
        
        "index" : "index",
        "fixed" : "fixed",
        "calc"  : "calculated",
        "val"   : "value",
        
        "range" : "range",
        "saturation" : "saturation",
        "pos"   : "Position",

        # Fit coefficients (blade or axis): slope, intercept, quadratic.
        "k"     : "slope (k)",
        "sk"    : "slope err. (sk)",
        "d"     : "offset (d)",
        "sd"    : "offset err. (sd)",
        "q"     : "quad. (q)",
        "sq"    : "quad. err. (sq)",

        # RMS summary: <direction> <kind>.
        "min"   : "minimum",
        "max"   : "maximum",
        "h"     : "Horizontal",
        "v"     : "Vertical",
        "tot"   : "Total",
    }

    @classmethod
    def group_name(cls, key: str) -> str:
        """Stored (human-readable) name of a registry key."""
        return cls.H5_GROUPS[key][0]

    @classmethod
    def get_description(
        cls,
        key: str,
        **fmt
        ) -> str:
        """Description of a registry key; unknown placeholders become ''."""
        template = cls.H5_GROUPS.get(key, (None, None))[1]
        if template is None:
            return "N/A"
        return template.format_map(defaultdict(str, fmt))

    @classmethod
    def column(cls, subject: str, kind: str) -> str:
        """'X nominal, Top outer mean, Blade A value, Horizontal minimum."""
        return f"{cls.H5_SUBJECTS[subject]} {cls.H5_COLUMNS[kind]}"


    # ------------------------- column helpers -------------------------
    # @classmethod
    # def get_beamline_name(cls, code: str) -> str:
    #     """Get full beamline name from code."""
    #     return cls.BEAMLINENAME.get(code, "N/A")

    @classmethod
    def get_plot_title(cls,
                       beamline   : str,
                       graph_type : str,
                       ax_type    : str,
                       calc_type  : str = "",
                       rort       : str = "",
                       ) -> str:
        """Return a graph title from the central registry.
    
        Args:
            graph_type: Top-level key in PLOT_TITLES (e.g. 'sweeps').
            ax_type:  Subplot role key
                       ('total', 'roi', 'heatmap', 'h', 'v', 'suptitle').
            beamline:  Beamline 3-letter code plus XBPM number (e.g. 'MNC1').
            calc_type: calculation type (e.g. 'pairwise' or 'cross').
            rort:      raw or transformed.

        Returns:
            Formatted title string, or empty string if key not found.
        """
        # Select beamline code and XBPM number from input, if provided.
        if beamline is not None:
            bline, xbpmnum = beamline[:3], beamline[-1]
        else:
            bline, xbpmnum = "", ""

        # Select calc type name from pairwise / cross, if provided.
        if calc_type:
            calc_type = (
                r"$\Delta/\Sigma$"
                if calc_type == "pairwise"
                else r"partial $\Delta/\Sigma$"
                )

        # Set raw / transformed string for title, if provided.
        rort = (
            "Raw" if rort == "raw"
            else "Transf." if rort == "trn"
            else ""
            )

        # Get template from registry and format it with provided values.
        template = cls.PLOT_TITLES.get(graph_type, {}).get(ax_type, "")
        return template.format(
            beamline = bline,
            xbpmnum  = xbpmnum,
            rort     = rort,
            ct       = calc_type,
        )

    @classmethod
    def get_position_subject(cls,
                             scope: str,
                             calc_type: str
                             ) -> str:
        """Return normalized subject label for XBPM position metadata.

        Examples:
            ('raw', 'pair')   -> 'raw Δ/Σ'
            ('raw', 'cross')  -> 'raw partial Δ/Σ'
            ('scaled', 'pair')-> 'transformed Δ/Σ'
        """
        scope_label = cls.POSITION_SCOPE_LABELS.get(scope, str(scope or ""))
        calc_label  = cls.POSITION_CALC_LABELS.get(calc_type,
                                                   str(calc_type or ""))
        return f"{scope_label} {calc_label}".strip()

    @staticmethod
    def standard_suppression_matrix() -> tuple:
        """Return the standard suppression matrix with 1/-1 pattern.

        This is the fixed pattern used for raw position calculations,
        independent of blade behavior or data fitting.

        Returns:
            np.ndarray: 4x4 standard suppression matrix
            np.ndarray: 4x4 zero matrix for standard deviation (no fit)
        """
        import numpy as np
        supmat =  np.array([
            [1, -1, -1,  1],
            [1,  1,  1,  1],
            [1,  1, -1, -1],
            [1,  1,  1,  1],
        ], dtype=float)
        # No standard deviation for fixed matrix
        stddevmat = np.zeros_like(supmat)
        return supmat, stddevmat
