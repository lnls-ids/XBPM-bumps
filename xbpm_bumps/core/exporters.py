"""Data export functionality."""

import numpy as np

from .parameters import Prm


class Exporter:
    """Handles persistence of calc. artifacts (positions, blades, supmat)."""

    def __init__(self, prm: Prm):
        """Keep reference parameters for naming and context during exports."""
        self.prm = prm

    def write_supmat(self, supmat: np.ndarray) -> None:
        """Write suppression matrix to disk."""
        outfile = f"supmat_{self.prm.beamline}.dat"
        with open(outfile, 'w') as fs:
            for lin in supmat:
                for col in lin:
                    fs.write(f" {col:12.6f}")
                fs.write("\n")

    def data_dump(self, data, positions, sup: str = "") -> None:
        """Dump blades data and calculated positions to files."""
        outfile = f"xbpm_blades_values_{self.prm.beamline}.dat"
        print(f"\n Writing out data to file {outfile} ...", end='')
        with open(outfile, 'w') as df:
            for key, val in data.items():
                df.write(f"{key[0]}  {key[1]}")
                for vv in val:
                    df.write(f"  {vv[0]} {vv[1]}")
                df.write("\n")

        pos_pair, pos_cr = positions

        outfilep = f"xbpm_positions_pair_{sup}_{self.prm.beamline}.dat"
        print("\n Writing out pairwise blade calculated positions to file"
              f" {outfilep} ...", end='')
        with open(outfilep, 'w') as fp:
            for key, val in pos_pair.items():
                fp.write(f"{key[0]}  {key[1]}")
                fp.write(f"  {val[0]} {val[1]}\n")

        outfilec = f"xbpm_positions_cross_{sup}_{self.prm.beamline}.dat"
        print("\n Writing out cross-blade calculated positions to file"
              f" {outfilec} ...", end='')
        with open(outfilec, 'w') as fc:
            for key, val in pos_cr.items():
                fc.write(f"{key[0]}  {key[1]}")
                fc.write(f"  {val[0]} {val[1]}\n")

        print("done.\n")

    def data_dump_with_prefix(self, prefix: str, data,
                              positions, sup: str = "") -> None:
        """Dump blades data and calculated positions using a filename prefix.

        Args:
            prefix: Base path (without extension) to prepend to output files.
            data: Raw blades data dictionary.
            positions: Tuple of (pair_positions_dict, cross_positions_dict).
            sup: Suffix indicating 'raw' or 'scaled'.
        """
        blades_file = f"{prefix}_blades_values_{self.prm.beamline}.dat"
        print(f"\n Writing out data to file {blades_file} ...", end='')
        with open(blades_file, 'w') as df:
            for key, val in data.items():
                df.write(f"{key[0]}  {key[1]}")
                for vv in val:
                    df.write(f"  {vv[0]} {vv[1]}")
                df.write("\n")

        pos_pair, pos_cr = positions

        pair_file = f"{prefix}_positions_pair_{sup}_{self.prm.beamline}.dat"
        print("\n Writing out pairwise blade calculated positions to file"
              f" {pair_file} ...", end='')
        with open(pair_file, 'w') as fp:
            for key, val in pos_pair.items():
                fp.write(f"{key[0]}  {key[1]}")
                fp.write(f"  {val[0]} {val[1]}\n")

        cross_file = f"{prefix}_positions_cross_{sup}_{self.prm.beamline}.dat"
        print("\n Writing out cross-blade calculated positions to file"
              f" {cross_file} ...", end='')
        with open(cross_file, 'w') as fc:
            for key, val in pos_cr.items():
                fc.write(f"{key[0]}  {key[1]}")
                fc.write(f"  {val[0]} {val[1]}\n")

        print("done.\n")
