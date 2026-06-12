"""Beamline configuration constants."""


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
    #
    # CAVEAT: the map is based on the current configuration of the beamlines, 
    # based on the the machine studies. The aim is to reassign the cables in
    # the XBPMs so the blades sequence correspond directly to the
    # PVs (A, B, C, D). The following map corrects the configuration and helps 
    # in finding the correct wiring.
    #
    BLADEMAP = {
        #"MNC"  : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
        #"MNC1"  : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
        ##"MNC1" : {"TO": 'B', "TI": 'D', "BI": 'A', "BO": 'C'},
        #"MNC2" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        #"MGN1" : {"TO": 'B', "TI": 'C', "BI": 'A', "BO": 'D'},
        #"MGN2" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},

        # "MNC1"  : {"TO": 'A', "TI": 'B', "BI": 'C', "BO": 'D'},
        "MNC1" : {"TO": 'D', "TI": 'C', "BI": 'B', "BO": 'A'},
        "MNC2" : {"TO": 'A', "TI": 'B', "BI": 'C', "BO": 'D'},

        # "MGN1" : {"TO": 'A', "TI": 'B', "BI": 'C', "BO": 'D'},
        "MGN1" : {"TO": 'A', "TI": 'C', "BI": 'B', "BO": 'D'},
        "MGN2" : {"TO": 'A', "TI": 'B', "BI": 'C', "BO": 'D'},

        "CAT"  : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        "CAT1" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},

        # ## To be checked: ## #
        # "CAT2": {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        "CNB"  : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        "CNB1" : {"TO": 'B', "TI": 'A', "BI": 'D', "BO": 'C'},

        # "CNB"  : {"TO": 'A', "TI": 'B', "BI": 'C', "BO": 'D'},
        # "CNB1" : {"TO": 'A', "TI": 'B', "BI": 'C', "BO": 'D'},
        # "CNB2" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        # "MGN"  : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
        # "MGN1" : {"TO": 'A', "TI": 'D', "BI": 'B', "BO": 'C'},
        # "MGN2" : {"TO": 'A', "TI": 'B', "BI": 'D', "BO": 'C'},

        "SIMUL": {"TO": 'A', "TI": 'B', "BI": 'C', "BO": 'D'},
    }

    # The XBPM beamlines.
    BEAMLINENAME = {
        "CAT": "Caterete",
        "CNB": "Carnauba",
        "MGN": "Mogno",
        "MNC": "Manaca",
        "N/A": "Not defined",
    }

    # Distances between two adjacent BPMs around source of bump at each line.
    BPMDISTS = {
        "CAT": 6.175495,
        "CNB": 6.175495,
        "MGN": 2.2769999999999015,
        "MNC": 7.035495,
    }

    # Distance from source (its center) to XBPM at each beamline.
    # Obtained from comissioning reports.
    XBPMDISTS = {
        "CAT":  15.740,
        "CAT1": 15.740,
        "CAT2": 19.590,
        "CNB": 15.740,
        "CNB1": 15.740,
        "CNB2": 19.590,
        "MGN1": 10.237,
        "MGN2": 16.167,
        "MNC1": 15.740,
        "MNC2": 19.590,
    }

    # Sections of the ring for each beamline.
    SECTIONS = {
        "CAT": "subsec:07SP",
        "CNB": "subsec:06SB",
        "MGN": "subsec:10BC",
        "MNC": "subsec:09SA"
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
            "heatmap" : "RMS Differences at ROI",
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
                "XBPM{xbpmnum}@{beamline}: {ct} $\Delta/\Sigma$, {rort}"
                ),
            "roi"      : (
                "XBPM{xbpmnum}@{beamline}: {ct} $\Delta/\Sigma$, {rort} (ROI)"
                ),
            "heatmap"  : (
                "XBPM{xbpmnum}@{beamline}: RMS in ROI"
                ),
        },
    }

    @classmethod
    def get_beamline_name(cls, code: str) -> str:
        """Get full beamline name from code."""
        return cls.BEAMLINENAME.get(code, "N/A")

    @classmethod
    def get_plot_title(cls, tab: str,
                       graph: str,
                       beamline: str = None,
                       rort: str = "",
                       calc_type: str = "") -> str:
        """Return a graph title from the central registry.

        Args:
            tab:       Top-level key in PLOT_TITLES (e.g. 'sweeps').
            graph:     Subplot role key
                       ('total', 'roi', 'heatmap', 'h', 'v', 'suptitle').
            beamline:  Beamline 3-letter code plus XBPM number (e.g. 'MNC1').
            rort:      raw or transformed.
            calc_type: calculation type (e.g. 'pairwise' or 'cross').

        Returns:
            Formatted title string, or empty string if key not found.
        """
        # Select beamline code and XBPM number from input, if provided.
        if beamline is not None:
            bline, xbpmnum = beamline[:3], beamline[-1]
        else:
            bline, xbpmnum = "", ""

        # Set raw / transformed string for title, if provided.
        rort = "Raw" if rort == "R" else "Transf."

        # Select cacl type name from pairwise / cross, if provided.
        if calc_type:
            calc_type = "" if calc_type == "pairwise" else "Part."

        # Get template from registry and format it with provided values.
        template = cls.PLOT_TITLES.get(tab, {}).get(graph, "")
        return template.format(
            beamline=bline,
            xbpmnum=xbpmnum,
            rort=rort,
            ct=calc_type,
        )
