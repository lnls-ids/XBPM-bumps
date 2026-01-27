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
    BLADEMAP = {
        "MNC"  : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
        "MNC1" : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
        "MNC2" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        "CAT"  : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        "CAT1" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},

        # ## To be checked: ## #
        # "CAT2": {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        "CNB"  : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
        "CNB1" : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
        "CNB2" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        "MGN"  : {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
        "MGN1" : {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
        "MGN2" : {"TO": 'B', "TI": 'C', "BI": 'A', "BO": 'D'},
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

    @classmethod
    def get_beamline_name(cls, code: str) -> str:
        """Get full beamline name from code."""
        return cls.BEAMLINENAME.get(code, "N/A")
