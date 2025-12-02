#!/bin/env python3

"""Plot data to compare calculated outputs from XBPM measurements.

Compare calculated data without suppression matrix, with suppression matrix
and with modified suppression matrix  by random walks (annealing).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# Distance from source (its center) to XBPM at each beamline.
# Obtained from comissioning reports.
XBPMDISTS = {
    "CAT":  15.740,
    "CAT1": 15.740,
    "CAT2": 19.590,
    "CNB1": 15.740,
    "CNB2": 19.590,
    "MGN1": 10.237,
    "MGN2": 16.167,
    "MNC1": 15.740,
    "MNC2": 19.590,
}


def datafiles(beamline="<BEAMLINE>"):
    """Define file names to be read."""
    return [
        [f"xbpm_positions_pair_raw_{beamline}.dat", "Pair, Raw Calc."],
        [f"xbpm_positions_pair_scaled_sort_{beamline}.dat", "Pair, Simple"],
        [f"rand_positions_sort_{beamline}.dat", "Pair, Random Walk"]
    ]


DATAFILENAMES = [f"{df[0]}" for df in datafiles()]

FIGDPI = 300  # DPI


def plot_graph(data, stype, dist, grname):
    """."""
    print(f"##### Type of analysis: {stype} ")

    xn = data[:, 0].reshape(11, 11) * dist
    yn = data[:, 1].reshape(11, 11) * dist
    xc = data[:, 2].reshape(11, 11) * dist
    yc = data[:, 3].reshape(11, 11) * dist

    fr, to = 3, 8
    fig, (axall, axroi, axcolor) = plt.subplots(1, 3, figsize=(15, 4))
    axall.plot(xn.ravel(), yn.ravel(), 'r+', label='Nom')
    axall.plot(xc.ravel(), yc.ravel(), 'bo', label='Calc')

    xnroi = xn[fr:to, fr:to]
    ynroi = yn[fr:to, fr:to]
    xcroi = xc[fr:to, fr:to]
    ycroi = yc[fr:to, fr:to]

    axroi.plot(xnroi.ravel(),
            ynroi.ravel(), 'r+', label='Nom')
    axroi.plot(xcroi.ravel(),
            ycroi.ravel(), 'bo', label='Calc')

    for ax, tt in ([axall, "All sites"], [axroi, "Close up"]):
        ax.set_xlabel(u'$x$ [$\\mu$m]')
        ax.set_ylabel(u'$y$ [$\\mu$m]')
        ax.set_title(f"XBPM 1 @ MNC, {stype}, {tt}")
        ax.axis('equal')
        ax.grid()
        ax.legend()
        ax.minorticks_on()

    # fig, (axhist, axcolor) = plt.subplots(1, 2, figsize=(14, 6))
    # nsites = (to - fr) ** 2
    # diff = np.sqrt((xn[fr:to, fr:to] - xc[fr:to, fr:to]) ** 2 +
    #                (yn[fr:to, fr:to] - yc[fr:to, fr:to]) ** 2) / nsites
    diffx2 = (xn[fr:to, fr:to] - xc[fr:to, fr:to]) ** 2
    diffy2 = (yn[fr:to, fr:to] - yc[fr:to, fr:to]) ** 2
    diffx = np.sqrt(diffx2)
    diffy = np.sqrt(diffy2)

    print("### At ROI:")
    print(f"Min difference  (x): {np.min(diffx):.2f}  um")
    print(f"Mean difference (x): {np.mean(diffx):.2f} um")
    print(f"Max difference  (x): {np.max(diffx):.2f}  um\n")

    print(f"Min difference  (y): {np.min(diffy):.2f}  um")
    print(f"Mean difference (y): {np.mean(diffy):.2f} um")
    print(f"Max difference  (y): {np.max(diffy):.2f}  um\n")

    diff = np.sqrt((diffx2 + diffy2))
    print(f"Min difference  (tot): {np.min(diff):.2f}  um")
    print(f"Mean difference (tot): {np.mean(diff):.2f} um")
    print(f"Max difference  (tot): {np.max(diff):.2f}  um\n")

    # axhist.hist(diff.reshape(-1), bins=20)
    # axhist.set_xlabel('Difference (um)')
    # axhist.set_ylabel('Frequency')
    # axhist.set_title('Histogram of Differences')

    # alldiff = np.sqrt(((xn - xc) ** 2 + (yn - yc) ** 2))
    axcolor.imshow(diff, origin='lower',
                extent=(xnroi.min(), xnroi.max(),
                        ynroi.min(), ynroi.max()))
    axcolor.set_xlabel(u'$x$ [$\\mu$m]')
    axcolor.set_ylabel(u'$y$ [$\\mu$m]')
    axcolor.set_title('Color Map of Differences')
    plt.colorbar(axcolor.images[0], ax=axcolor, label='Difference [$\\mu$m]')
    plt.grid(which="minor")
    plt.tight_layout()
    plt.savefig(grname, transparent=False, facecolor="white", dpi=FIGDPI)


HELP_DESCRIPTION = f"""
The program tries to open three files with calculated positions.

Data files' names must be:

{DATAFILENAMES[0]}
{DATAFILENAMES[1]}
{DATAFILENAMES[2]}

where <BEAMLINE> is defined with key -b, as above.

Each data file must contain a set of four columns. First two are
the 'nominal' values (x, y) of beam position; the next two are the 
calculated horizontal and vertical positions, respectively.
"""


def cmd_options(argv=None):
    """Get beamline choice from command line."""
    parser = argparse.ArgumentParser(
        prog="plot_res",
        description="Plot XBPM's calculated positions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"{HELP_DESCRIPTION}\n")

    parser.add_argument(
        '-b', '--beamline', type=str, required=True, dest="beamline",
        help='Beamline to be analysed.'
        )

    args = parser.parse_args(argv)
    return args.beamline


def main():
    """The main function."""
    # Define beamline.
    beamline = cmd_options()   # "MNC1"

    if beamline not in XBPMDISTS.keys():
        print(f" ERROR: {beamline} beamline not recognized. Aborting.")
        return 1

    # Distances.
    print(f" Working in beamline {beamline}.\n")
    dist = XBPMDISTS[beamline]

    # Data files.
    wdir = "./"
    datafilestr = datafiles(beamline=beamline)
    for num, (wfile, meastype) in enumerate(datafilestr):
        try:
            data = np.genfromtxt(wdir + wfile)
            grname = f"{wdir}/compare_{num+1}_{wfile.strip('.dat')}.png"
            plot_graph(data, meastype, dist, grname)
        except Exception as err:
            print(" ERROR: when trying to open file "
                  f"{wdir + wfile}:\n{err}\n")

        print("#######################\n")
    plt.show()


if __name__ == "__main__":
    main()
