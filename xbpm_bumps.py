#!/usr/bin/python3

"""Extract XBPM's data and calculate the beam position.

Usage:
    xbpm_bumps.py [OPTION] [VALUE]

where options are:
  -h : this help

Input parameters:
  -w <directory> : the directory with measured data
  -d <distance>  : distance from source to XBPM (optional; if not given,
                   the standard ones from Sirius are used)
  -g <step>      : step between neighbour sites in the grid, default = 4.0
  -s <number>    : initial data to be skipped, default = 0

Output parameters (information to be shown):
  -b             : positions calculated from BPM data
  -m             : the blade map (to check blades' positions)
  -s             : anaysis of blades' behaviour by sweeping through the center
                   of canvas. Fit lines to data.
  -x             : positions calculated from XBPM data

This program extract data from pickle files containing the measurements
of XBPM blades' currents of a few beamlines (CAT, CNB, MNC, MGN) and
calculates the respective positions based on pairwise and cross-blade
formulas.

The data is treated with linear transformations first, to correct for
distortions promoted by different gains in each blade. Firstly, gains
(or their reciprocal, 1/G, which is the suppression) are estimated by
analyzing the slope of the line formed by central points in vertical
and horizontal sweeping. The supression is applied to the set of points
to correct the grid, then a linear scaling is calculated to set distances
in micrometers.

"""

import getopt
import matplotlib.pyplot as plt
import numpy as np
import pickle                 # noqa: S403
import os
import sys
from copy import deepcopy

FILE_EXTENSION = ".pickle"    # Data file type.
GRIDSTEP = 2                  # Default grid step.

# Power relative to Ampere subunits.
AMPSUB = {
    0    : 1.0,      # if not unit is defined.
    "mA" : 1e-3,
    "uA" : 1e-6,
    "nA" : 1e-9,
    "pA" : 1e-12,
}


# Map of blades positions in each XBPM.
# TO, TI, BO, BI : top/bottom, in/out, relative to the storage ring;
# A, B, C, D : names of respective P.V.s
BLADEMAP = {
    "MNC": {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
    "MNC1": {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
    "MNC2": {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
    "CAT":  {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
    "CAT1": {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},

    # ## To be checked: ## #
    # "CAT2": {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
    "CNB1": {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
    "CNB2": {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
    "MGN": {"TO": 'C', "TI": 'A', "BI": 'D', "BO": 'B'},
    "MGN1": {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
    "MGN2": {"TO": 'B', "TI": 'A', "BI": 'C', "BO": 'D'},
    "SIM":  {"TO": 'A', "TI": 'B', "BI": 'C', "BO": 'D'},
}

# The XBPM beamlines.
BEAMLINENAME = {
    "CAT": "Cateretê",
    "CNB": "Carnaúba",
    "MGN": "Mogno",
    "MNC": "Manacá",
}

# Distances between two adjacent BPMs around the source of bump at each line.
BPMDISTS = {
    "CAT": 6.175495,
    "CNB": 6.175495,
    "MGN": 2.2769999999999015,
    "MNC": 7.035495,
}

# Distance from source to XBPM at each beamline.
XBPMDISTS = {
    "CAT":  15,
    "CAT1": 15,
    "CNB1": 15,
    "CNB2": 15,
    "MGN1": 15,
    "MGN2": 15,
    "MNC1": 15,
    "MNC2": 15,
}

# Sections of the ring for each beamline.
SECTIONS = {
    "CAT": "subsec:07SP",
    "CNB": "subsec:06SB",
    "MGN": "subsec:10BC",
    "MNC": "subsec:09SA"
}


# ## Get command line options and data from work directory.

def cmd_options():  # noqa: C901
    """Read command line parameters."""
    # Read options, if available.
    try:
        opts = getopt.getopt(sys.argv[1:], "hbcmsxd:g:k:o:w:")
    except getopt.GetoptError as err:
        print("\n\n ERROR: ", str(err), "\b.")
        sys.exit(1)

    # Initialize parameters.
    prm = {
        "showblademap"     : False,
        "sweep"            : False,
        "showbladescenter" : False,
        "xbpmpositions"    : False,
        "xbpmfrombpm"      : False,
        "outputfile"       : None,
        "xbpmdist"         : None,
        "workdir"          : None,
        "skip"             : 0,
        "gridstep"         : GRIDSTEP,
        "maxradangle"      : 20.0,
    }

    for op in opts[0]:
        if op[0] == "-h":
            """Help message."""
            help("xbpm_bumps")
            sys.exit(0)

        if op[0] == "-b":
            # Show positions from BPM data.
            prm["xbpmfrombpm"] = True

        if op[0] == "-c":
            # Show blades' central lines response while sweeping.
            prm["showbladescenter"] = True

        if op[0] == "-m":
            # Map of baldes' response to beam position.
            prm["showblademap"] = True

        if op[0] == "-s":
            # Analysis of sweeping through central lines.
            prm["sweep"] = True

        if op[0] == "-x":
            # Beam position from XBPM data.
            prm["xbpmpositions"] = True

        if op[0] == "-d":
            # Distance to XBPM.
            prm["xbpmdist"] = float(op[1])

        if op[0] == "-k":
            # Skip initial values in data.
            prm["skip"] = int(op[1])

        if op[0] == "-o":
            # Dump data .
            prm["outputfile"] = op[1]

        if op[0] == "-w":
            # Working directory.
            prm["workdir"] = op[1]

    if prm["workdir"] is None:
        print(" The working directory was not defined (option -w). Aborting.")
        sys.exit(0)
    return prm


def get_pickle_data(prm):
    """Open pickle files in directory and extract data.

    Args:
        prm (dict): parameters with directory and section infrmation

    Returns:
        rawdata (list): all data collected from pickle files in working
            directory
    """
    # Get files from directory.
    allfiles = os.listdir(prm['workdir'])
    picklefiles = [pf for pf in allfiles if pf.endswith("pickle")]

    if len(picklefiles) == 0:
        print(f"No pickle files found in directory \'{prm['workdir']}\'."
              "Aborting.")
        sys.exit(0)

    with open(prm["workdir"] + "/" + picklefiles[0], 'rb') as df:
        firstfile = pickle.load(df)           # noqa: S301
    prm["beamline"] = next(iter(firstfile[0]))
    beamline = prm["beamline"][:3]
    print(f"### Working beamline     :\t {BEAMLINENAME[beamline]}"
          f" ({prm['beamline']})")

    prm["current"]  = firstfile[2]["current"]
    print(f"### Storage ring current :\t {prm['current']} mA")

    try:
        prm["phaseorgap"] = firstfile[1][beamline.lower()]
        print(f"### Phase / Gap ({prm['beamline']})   :\t"
              f" {prm['phaseorgap']}")
    except Exception:
        print(f"\n WARNING: no phase/gap defined for {prm['beamline']}.")

    prm["bpmdist"]  = BPMDISTS[beamline]
    prm["section"]  = SECTIONS[beamline]
    prm["blademap"] = BLADEMAP[beamline]
    if prm["xbpmdist"] is None:
        prm["xbpmdist"] = XBPMDISTS[prm["beamline"]]
        print("\n WARNING: distance of XBPM from source not defined, "
              f"setting it to the {prm['beamline']} default.")

    # List of files in working directory.
    files = [
        pf for pf in picklefiles
        if pf.endswith(FILE_EXTENSION)
        # and prm["section"] in pf
        ]

    # Order files by timestamp.
    sfiles = sorted(files, key=lambda name: lastfield(name, '_'))
    # Assemble all data from files.
    rawdata = list()
    for file in sfiles:
        with open(prm["workdir"] + "/" + file, 'rb') as df:
            rawdata.append(pickle.load(df))           # noqa: S301
    return rawdata


def lastfield(name, fld=' '):
    """Get the last part of string separated by given character.

    Args:
        name (str) : string to be splitted;
        fld (str) : separator character.

    Returns:
        last field of splitted string 'name'.
    """
    return name.split(fld)[-1]


def gridstep_get(rawdata):
    """Try and calculate grid step from data.

    Args:
        rawdata (list) : raw data read from the BPMs and the XBPM's blades.
    """
    agx = [rawdata[ii][2]['agx'] for ii in range(len(rawdata))]
    agy = [rawdata[ii][2]['agy'] for ii in range(len(rawdata))]

    # Try it from horizontal sweep.
    xset = list(set(agx))  # Strip redundancies.
    gridstepx = 0 if len(xset) == 1 else np.abs(xset[1] - xset[0])
    # Try it from vertical sweep.
    yset = list(set(agy))  # Strip redundancies.
    gridstepy = 0 if len(yset) == 1 else np.abs(yset[1] - yset[0])

    if gridstepx != gridstepy:
        print(f"\n WARNING: horizontal grid step ({gridstepx}) differs from"
              f" vertical grid step ({gridstepy})."
              "\n          I'll try it with the smaller value, if not zero.")

    if gridstepx < gridstepy and gridstepx != 0:
        return gridstepx
    elif gridstepy != 0:
        return gridstepy
    else:
        print(" ERROR: I could not infer the grid step size. "
              " Please, rerun and provide the value manually,"
              " with option -g. Aborting.")
        sys.exit(0)


# ## Estimate positions from BPM data.

def beam_positions_from_bpm(rawdata, prm):
    """Calculate the beam position at the XBPM from BPM data.

    Args:
        rawdata (list) : raw data with bpms and xbpms measurements.
        prm (dict) : set of parameters of the analysis.
    """
    fig, ax = plt.subplots()

    # Extract the index sector of the beamline and calculate the tangents.
    sector = int(prm["section"].split(':')[1][:2])
    idx = 8 * (sector - 1) - 1
    tangents = tangents_calc(rawdata, idx, prm)

    print(f"# Distance between BPMs            = {prm['bpmdist']:8.4f}  m\n"
          f"# Distance between source and XBPM = {prm['xbpmdist']:8.4f} m\n")
    # xtob = prm["xbpmdist"] / prm["bpmdist"]

    xbpm_pos = positions_calc_from_bpm(tangents, prm["xbpmdist"])
    xpos, ypos = list(), list()
    xnom, ynom = list(), list()
    for key, val in xbpm_pos.items():
        xnom.append(key[0])
        ynom.append(key[1])
        xpos.append(val[0])
        ypos.append(val[1])

    # Standard deviations.
    std_dev_bpm_estimate(xnom, ynom, xpos, ypos)

    # vmax = np.max(ynom) / np.max(ypos)
    # ax.plot(xpos, np.array(ypos) * vmax, 'bo', label="measured")
    ax.plot(xpos, np.array(ypos), 'bo', label="measured")
    ax.plot(xnom, ynom, 'r+', label="nominal")

    ax.set_xlabel("$x$ [$\mu$m]")  # noqa: W605
    ax.set_ylabel("$y$ [$\mu$m]")  # noqa: W605
    ax.set_title(f"Beam positions @ {prm['beamline']} from BPM values")

    # fig.canvas.draw_idle()
    # lim = np.max(xnom + ynom) * 1.2
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    ax.legend()
    ax.grid()

    fig.savefig(f"xbpm_from_bpm_{prm['beamline']}.png")


def tangents_calc(rawdata, idx, prm):
    """Calculate the tangents of the beam angles between neighbour BPMs.

    The tangents in x and y of angles of beam's displacement, caused by
    bumping it, at each point of the scanned grid. The calculations
    correspond to measurements of beam positions at two neighbour BPMs.

    Args:
        rawdata (list) : raw data with bpms and xbpms measurements.
        idx (int) : index for specific sector of the storage ring.
        prm (dict) : general parameters of the analysis.

    Returns:
        tangents (dict) : the tangents in x and y of angles of beam's
            displacement.
    """
    idx, nextidx = idx, idx + 1
    offset_x_sect, offset_y_sect = 0, 0
    offset_x_next, offset_y_next = 0, 0
    offsetfound = False

    # Find the offset.
    for dt in rawdata:
        if dt[2]['agx'] == 0 and dt[2]['agy'] == 0:
            offset_x_sect = dt[2]['orbx'][idx]
            offset_y_sect = dt[2]['orby'][idx]
            offset_x_next = dt[2]['orbx'][nextidx]
            offset_y_next = dt[2]['orby'][nextidx]
            offsetfound = True
            break

    # Try to define an offset from the existent data.
    if not offsetfound:
        (offset_x_sect, offset_x_next,
         offset_y_sect, offset_y_next) = offset_search(rawdata, idx)

    # Calculate the tangents (differences) between nieghbour BPMs.
    tangents = dict()
    for dt in rawdata:
        tx = (((dt[2]['orbx'][nextidx] - offset_x_next)) -
               (dt[2]['orbx'][idx]     - offset_x_sect)) / prm["bpmdist"]
        ty = (((dt[2]['orby'][nextidx] - offset_y_next)) -
               (dt[2]['orby'][idx]     - offset_y_sect)) / prm["bpmdist"]
        agx, agy = dt[2]['agx'], dt[2]['agy']
        tangents[agx, agy] = np.array([tx, ty])  # * K_um
    return tangents


def offset_search(rawdata, idx):
    """Try to find an offset from the data itself.

    Extrapolate the data to zero to find out the offset. This is an attempt
    to find offsets for the orbits registered by the BPMs when the reference
    orbit is not available.

    Args:
        rawdata (list) : raw data with bpms and xbpms measurements.
        idx (int) : index for specific sector of the storage ring.

    Returns:
        offsets in x and y of both BPMs.
    """
    nextidx = idx + 1
    agx    = np.array([dt[2]['agx']           for dt in rawdata])  # noqa: E272
    orbx   = np.array([dt[2]['orbx'][idx]     for dt in rawdata])  # noqa: E272
    n_orbx = np.array([dt[2]['orbx'][nextidx] for dt in rawdata])
    #
    agxmax = np.max(agx)
    agxmin = np.min(agx)
    #
    osx = np.array(sorted(list(set(orbx))))
    oxmin, oxmax = osx[0], osx[-1]
    offset_x_sect = (oxmin * agxmax - oxmax * agxmin) / (agxmax - agxmin)
    #
    onx = np.array(sorted(list(set(n_orbx))))
    onxmin, onxmax = onx[0], onx[-1]
    offset_x_next = (onxmin * agxmax - onxmax * agxmin) / (agxmax - agxmin)
    agy    = np.array([dt[2]['agy']           for dt in rawdata])  # noqa: E272
    orby   = np.array([dt[2]['orby'][idx]     for dt in rawdata])  # noqa: E272
    n_orby = np.array([dt[2]['orby'][nextidx] for dt in rawdata])
    #
    agymax = np.max(agy)
    agymin = np.min(agy)
    #
    osy = np.array(sorted(list(set(orby))))
    oymin, oymax = osy[0], osy[-1]
    offset_y_sect = (oymin * agymax - oymax * agymin) / (agymax - agymin)
    #
    ony = np.array(sorted(list(set(n_orby))))
    onymin, onymax = ony[0], ony[-1]
    offset_y_next = (onymin * agymax - onymax * agymin) / (agymax - agymin)

    return (offset_x_sect, offset_x_next,
            offset_y_sect, offset_y_next)


def positions_calc_from_bpm(tangents, xbpm_dist):
    """Calculate the beam position out of BPM data.

    Given the grid of positions formed by the displacements of the beam due
    to bumps on it, and their corresponding tangents at the BPMs
    surrounnding the bumpings, calulate the positions at the XBPMs.

    Args:
        tangents (dict) : tangent of angles corresponding to each grid position.
        xbpm_dist (float) : distance from source to XBPM at the beamline.
    """
    positions = dict()
    for key, tg in tangents.items():
        newkey = (key[0]* xbpm_dist, key[1] * xbpm_dist)
        positions[newkey] = tg * xbpm_dist
    return positions


def std_dev_bpm_estimate(xnom, ynom, xpos, ypos):
    """Estimate std dev of positions based on BPM measurements.

    Calculate RMS values at each site when compared to formally established
    grid of points. It corresponds to the deviation of each measured value
    from the nominal in at the grid.

    Args:
        xnom (list) : the nominal value of x-coordinate;
        ynom (list) : the nominal value of y-coordinate;
        xpos (list) : the measured value of x-coordinate;
        ypos (list) : the measured value of y-coordinate.
    """
    npxnom = np.array(xnom)
    npynom = np.array(ynom)
    npxpos = np.array(xpos)
    npypos = np.array(ypos)

    # Values for all sites in the grid.
    nfh = npxnom.shape[0]
    nfv = npynom.shape[0]
    diff_h = np.abs(npxnom.ravel() - npxpos.ravel())
    diff_h_max = np.max(diff_h)
    sig2_h = np.sum(diff_h**2) / nfh
    #
    diff_v = np.abs(npynom.ravel() - npypos.ravel())
    diff_v_max = np.max(diff_v)
    sig2_v = np.sum(diff_v**2) / nfh

    print("Sigmas:\n"
        f"   All sites,     H = {np.sqrt(sig2_h):.4f}\n"
        f"   All sites,     V = {np.sqrt(sig2_v):.4f},\n"
        f"   All sites, total = {np.sqrt(sig2_h + sig2_v):.4f}\n"
        "Maximum difference:\n"
        f"   All sites,     H = {diff_h_max:.4f}\n"
        f"   All sites,     V = {diff_v_max:.4f},\n")

    # Values for ROI.
    nshx, nshy = len(set(xnom)), len(set(ynom))
    nmax = nshx * nshy

    if (nmax > nfh or nmax > nfv or nshx == 1 or nshy == 1):
        print("\n WARNING: sweeping looks incomplete, no ROI was defined. "
              " (Maybe just one line swept?)")
    else:
        fr, upto = int(nshx / 2 - 2), int(nshy / 2 + 2)
        npxnom_cut = npxnom.reshape(nshx, nshy)[fr:upto, fr:upto]
        npynom_cut = npynom.reshape(nshx, nshy)[fr:upto, fr:upto]
        npxpos_cut = npxpos.reshape(nshx, nshy)[fr:upto, fr:upto]
        npypos_cut = npypos.reshape(nshx, nshy)[fr:upto, fr:upto]

        sig2_v_roi = np.sum((npynom_cut.ravel() -
                             npypos_cut.ravel())**2) / nfv
        sig2_h_roi = np.sum((npxnom_cut.ravel() -
                             npxpos_cut.ravel())**2) / nfh

        print("\nValues in ROI"
              f" (H: {np.min(npxnom_cut)}, {np.max(npxnom_cut)};"
              f"  V: {np.min(npynom_cut)}, {np.max(npynom_cut)})\n"
              f"   ROI,     H = {np.sqrt(sig2_h_roi):.4f}\n"
              f"   ROI,     V = {np.sqrt(sig2_v_roi):.4f},\n"
              f"   ROI, total = {np.sqrt(sig2_h_roi + sig2_v_roi):.4f}")


# ## Select data.

def blades_fetch(data, beamline):
    """Retrieve each blade's data and average over their values.

    Obs.: A map of the blade positions must be provided for each beamline.

    Args:
        data (list) : acquired data from bpm and xbpm measurements.
        beamline (str) : beamline to be analysed.

    Returns:
        balde_vals (dict) : averaged and std dev values from blades' measured
            data; bpm h and v angular positions (in urad), taken as 'real'
            positions, work as indices.
    """
    blade_vals = dict()
    for dt in data:
        xbpm = dt[0][beamline]
        vals = list()
        for blade in BLADEMAP[beamline].values():
            # Average and std dev over measured values of current blade.
            av, sd = blade_average(xbpm[f'{blade}_val'], beamline)
            vals.append((av, sd))
        bpm_x = dt[2]['agx']
        bpm_y = dt[2]['agy']
        blade_vals[bpm_x, bpm_y] = np.array(vals)
    return blade_vals


def blade_average(blade, beamline):
    """Calculate the average of blades' values.

    Args:
        blade (numpy array) : raw data measured by given blade.
        beamline (str) : beamline identifier to define the type of data
            for averaging.

    Returns: averaged measured value for the blade and its standard deviation.
    """
    # Decide the the type of data to be average over.
    if beamline in ["MGN", "MNC"]:
        return np.average(blade), np.std(blade)
    vals = np.array([
        vv * AMPSUB[un] for vv, un in blade
    ])
    return np.average(vals), np.std(vals)


# ## Map of XBPM blades' sequence.

def blades_map_show(data, prm):
    """Show graphs of intensities measured by each blade at each position.

    Given the grid of positions formed by the displacements of the beam due
    to bumps on it, show the intensity (colour) map of current from each blade.

    Args:
        data (dict) : blades' values indexed by grid positions.
        prm (dict) : general parameters of the analysis.
    """
    blades, stddevs, maxval = data_parse(data, prm["gridstep"])
    to, ti, bi, bo = blades
    # sto, sti, sbi, sbo = stddevs

    fig, rx = plt.subplots(2, 2, figsize=(8, 5))
    extent = (-maxval, maxval, -maxval, maxval)

    quad  = [[ti, to], [bi, bo]]
    names = [["TI", "TO"], ["BI", "BO"]]
    imgs  = []
    for idy in range(2):
        for idx in range(2):
            imgs.append(rx[idy][idx].imshow(quad[idy][idx],
                                            extent=extent))
            rx[idy][idx].set_xlabel(u"$x$ [$\\mu$rad]")
            rx[idy][idx].set_ylabel(u"$y$ [$\\mu$rad]")
            rx[idy][idx].set_title(names[idy][idx])

    fig.tight_layout(pad=0., w_pad=-15., h_pad=2.)
    fig.savefig(f"blade_map_{prm['beamline']}.png")
    # plt.show()q
    # fig.colorbar(imgs[3], ax=ax)


def data_parse(data, gridstep=GRIDSTEP):
    """Extract each blade's data from whole data.

    Args:
        data (dict): keys are (x, y) grid positions, values are blades'
            measurements with errors (order: to, ti, bi, bo).
        ngrid (tuple): grid size, lin x col.
        gridstep (float): interval between two ajacent coordinates.

    Returns:
        to, ti, bi, bo (numpy arrays): the values read by each blade.

    Note: the keys in data are the positions in the coordinate system, but the
        indexing of the numpy arrrays to, ti, bi, bo are the conventional ones,
        so the [0, 0] index corresponds to the left-upmost position etc.
    """
    maxval = np.max(np.abs(list(data.keys())))
    nlin = int(np.sqrt(len(data.keys())))
    ngrid = (nlin, nlin)
    to,  ti  = np.zeros(ngrid), np.zeros(ngrid)
    bo,  bi  = np.zeros(ngrid), np.zeros(ngrid)
    sto, sti = np.zeros(ngrid), np.zeros(ngrid)
    sbo, sbi = np.zeros(ngrid), np.zeros(ngrid)
    for key in data.keys():
        # Map spatial coordinates onto matrix indexes.
        lin = int((maxval - key[1]) / gridstep)
        col = int((maxval + key[0]) / gridstep)

        try:
            to[lin, col]  = data[key][0, 0]
            ti[lin, col]  = data[key][1, 0]
            bi[lin, col]  = data[key][2, 0]
            bo[lin, col]  = data[key][3, 0]

            sto[lin, col] = data[key][0, 1]
            sti[lin, col] = data[key][1, 1]
            sbi[lin, col] = data[key][2, 1]
            sbo[lin, col] = data[key][3, 1]
        except Exception as err:
            print(f"\n WARNING: {err}\n"
                  " when trying to parse blade data.")
    return [to, ti, bi, bo], [sto, sti, sbi, sbo], maxval


# ## Dump XBPM's selected and averaged data to file.

def data_dump(data, prm):
    """Dump data to file.

    Args:
        data (dict) : calculated beam positions at the grid indexed by
            their nominal position values.
        prm (dict) : general parameters of the analysis.
    """
    print(f"\n Writing out data do file {prm['outputfile']} ...", end='')
    with open(prm["outputfile"], 'w') as df:
        for key, val in data.items():
            df.write(f"{key[0]}  {key[1]}")
            for v in val:
                df.write(f"  {v[0]} {v[1]}")
            df.write("\n")
    print("done.\n")


# ## Sweep through horizontal and vertical central lines.

def central_sweeps(data, prm, show=False):
    """Check each blade's behaviour at symmetric positions.

    Args:
        data (dict) : calculated beam positions at the grid indexed by
            their nominal position values.
        prm (dict) : general parameters of the analysis.
        show (bool) : show the graphics of the sweepings or not.    

    Returns:
        hrange_cut, vrange_cut (numpy array) : vertical and horizontal range
            of positions swept.
        blades_h, blades_v (dict) : the values measured by the blades along
            the grid points fo central sweeping.
    """
    # Split data.
    # bpm_pos = np.array(list(data.keys()))
    # bpm_x, bpm_y = bpm_pos[:, 0], bpm_pos[:, 1]

    # Values of blades' counting at center lines (x = 0 and y = 0).
    # Obs.: values not as indices, but real positions.

    keys = np.array(list(data.keys()))
    hrange, vrange = np.unique(keys[:, 0]), np.unique(keys[:, 1])

    # nrange = np.unique(np.abs(list(data.keys()))[:, 0])
    nh, nv = hrange.shape[0], vrange.shape[0]
    halfh = int(nh/2)
    halfv = int(nv/2)
    abh = 3 if hrange[-1] > 3 else halfh
    fromh, uptoh = halfh - abh, halfh + abh + 1
    abv = 3 if vrange[-1] > 3 else halfv
    prm["nroi"] = [abh, abv]
    fromv, uptov = halfv - abv, halfv + abv + 1

    # Central line intervals.
    hrange_cut = hrange[fromh:uptoh]
    vrange_cut = hrange[fromv:uptov]

    # data indices are tuples of the grid coordinates: (x, y).
    # Run through central horizontal (ch) line.
    try:
        to_ch  = np.array([data[jj, 0][0] for jj in hrange_cut])
        ti_ch  = np.array([data[jj, 0][1] for jj in hrange_cut])
        bi_ch  = np.array([data[jj, 0][2] for jj in hrange_cut])
        bo_ch  = np.array([data[jj, 0][3] for jj in hrange_cut])
        blades_h = {"to": to_ch, "ti": ti_ch, "bi": bi_ch, "bo": bo_ch}
    except Exception as err:
        print("\n WARNING: horizontal sweeping interrupted,"
              f" data grid may be incomplete: {err}")
        norm = np.array([[1., 0], [0., 0.]])
        return ([norm, norm, norm, norm], [norm, norm, norm, norm],
                hrange, vrange)

    # Vertical positions along central horizontal line.
    pos_to_ti_v = (to_ch + ti_ch)
    pos_bi_bo_v = (bo_ch + bi_ch)
    pos_ch_v = (pos_to_ti_v - pos_bi_bo_v) / (pos_to_ti_v + pos_bi_bo_v)

    # Fit a line to the central horizontal sweep.
    fit_ch_v  = np.polyfit(hrange_cut, pos_ch_v, deg=1)

    # Run through central vertical (cv) line.
    try:
        to_cv  = np.array([data[0, jj][0] for jj in vrange_cut])
        ti_cv  = np.array([data[0, jj][1] for jj in vrange_cut])
        bi_cv  = np.array([data[0, jj][2] for jj in vrange_cut])
        bo_cv  = np.array([data[0, jj][3] for jj in vrange_cut])
        blades_v = {"to": to_cv, "ti": ti_cv, "bi": bi_cv, "bo": bo_cv}
    except Exception as err:
        print("\n WARNING: vertical sweeping interrupted,"
              f" data grid may be incomplete: {err}")
        norm = np.array([[1., 0], [0., 0.]])
        return ([norm, norm, norm, norm], [norm, norm, norm, norm],
                hrange, vrange)

    # Horizontal positions along central vertical line.
    pos_to_ti_h = (to_cv + ti_cv)
    pos_bi_bo_h = (bo_cv + bi_cv)
    pos_cv_h = (pos_to_ti_h - pos_bi_bo_h) / (pos_to_ti_h + pos_bi_bo_h)

    # Fit a line to the central vertical sweep.
    fit_cv_h  = np.polyfit(vrange_cut, pos_cv_h, deg=1)

    if show:
        fig, ax = plt.subplots(figsize=(12, 5))

        hline = (fit_ch_v[0, 0] * hrange + fit_ch_v[1, 0]) * prm["xbpmdist"]
        ax.plot(hrange * prm["xbpmdist"], hline, '^-', label="H fit")
        ax.plot(hrange_cut * prm["xbpmdist"],
                pos_ch_v[:, 0] * prm["xbpmdist"], 'o-', label="H sweep")

        vline = (fit_cv_h[0, 0] * vrange + fit_cv_h[1, 0]) * prm["xbpmdist"]
        ax.plot(pos_cv_h[:, 0]  * prm["xbpmdist"],
                vrange_cut * prm["xbpmdist"], '^-', label="V sweep")
        ax.plot(vline, vrange * prm["xbpmdist"], '^-', label="V fit")
        # ax.plot(vrange_cut, h_cv_pos[:, 0], 'o-', label="V sweep")
        # ax.plot(vrange, vline, '^-', label="V fit")
        # ax.plot(vrange_cut, h_cv_pos[:, 0], 'o-', label="V sweep")

        ax.set_xlabel(u"$x$ [$\\mu$rad]")
        ax.set_ylabel(u"$y$ [$\\mu$rad]")
        ax.set_title("Central Horizontal / Vertical Sweeps")
        ax.grid(True)
        ax.legend()

        fig.savefig(f"xbpm_sweeps_{prm['beamline']}.png")

    return (hrange_cut, vrange_cut, blades_h, blades_v)


# ## Show XBPM data at central lines.

def blades_show_at_center(range_h, range_v, blades_h, blades_v):
    """Show graphics of blade value along the central sweeping points.

    Given the grid of positions formed by the displacements of the beam due
    to bumps on it, show the intensities measured by each blade along the
    sweeping.

    Args:
        range_h (numpy array) : set of points of the grid swept at the
            central horizontal line.
        range_v (numpy array) : set of points of the grid swept at the
            central vertical line.
        blades_h (dict) : blades' measurements at each grid's central
            horizontal point.
        blades_v (dict) : blades' measurements at each grid's central
            vertical point.
    """
    fig, (axh, axv) = plt.subplots(1, 2, figsize=(10, 5))

    # DEBUG
    print(f"\n (BLADES SHOW) H KEYS = {blades_h.keys()}")
    # DEBUG

    for key, val in blades_h.items():

        # DEBUG
        # print(f"\n\n ##### BLADES SHOW\nkey = {key}, \nval =\n{val[:3]}")
        # DEBUG
        weight = 1. / val[:, 1]
        (acoef, bcoef) = np.polyfit(range_h, val[:, 0], deg=1, w=weight)
        axh.plot(range_h, range_h * acoef + bcoef, "o-", label=f"{key} fit")
        axh.errorbar(range_h, val[:, 0], val[:, 1], fmt='^-', label=key)

        axh.plot()

    for key, val in blades_v.items():
        weight = 1. / val[:, 1]
        (acoef, bcoef) = np.polyfit(range_v, val[:, 0], deg=1, w=weight)
        axv.plot(range_v, range_v * acoef + bcoef, "o-", label=f"{key} fit")
        axv.errorbar(range_v, val[:, 0], val[:, 1], fmt='^-', label=key)

    axh.set_title("Horizontal")
    axv.set_title("Vertical")
    axh.legend()
    axv.legend()
    axh.grid()
    axv.grid()
    xlabel = u"$x$ $\\mu$m"
    ylabel = u"$I$ [A] / # (counts)"
    axh.set_xlabel(xlabel)
    axv.set_xlabel(xlabel)
    axh.set_ylabel(ylabel)
    axv.set_ylabel(ylabel)
    fig.tight_layout()


# ## Beam position from XBPM data.

def xbpm_position_calc(data, prm, range_h, range_v, blades_h, blades_v):
    """Calculate positions from blades' measured data.

    Args:
        data    (dict)  : nominal positions and respective data from blades.
        prm     (dict)  : all parameters of the data and the analysis.
        coefs_h (list)  : coefficients of fittings at horizontal central
            sweeps for each blade.
        coefs_v (list)  : coefficients of fittings at vertical central
            sweeps for each blade.
        range_h (numpy array) : horizontal range swept by the beam
        range_v (numpy array) : vertical range swept by the beam
        blades_h (numpy array) : measured data from blades in the horizontal
                                central line
        blades_v (numpy array) : measured data from blades in the vertical
                                central line
        show (boolean) : show blades' behaviour at center or not.

    Returns:
        pos_pair_h/_v (list) : the pairwise calculated positions
        pos_cr_h/_v (list)   : the cross-blade calculated positions.
    """
    blades, stddevs, maxval = data_parse(data)
    supmat = suppression_matrix(range_h, range_v,
                                blades_h, blades_v,
                                show=True, nosuppress=False)
    pos_pair  = beam_position_pair(data, supmat)
    (pos_nom_h, pos_nom_v,
     pos_pair_h, pos_pair_v) = position_dict_parse(pos_pair)
    pos_cr_h, pos_cr_v = beam_position_cross(blades)

    # Adjust to real distance.
    pos_nom_h *= prm["xbpmdist"]
    pos_nom_v *= prm["xbpmdist"]

    # Indices of central ranges for scaling.
    keys = np.array(list(data.keys()))
    range_h, range_v = np.unique(keys[:, 0]), np.unique(keys[:, 1])
    nh, nv = range_h.shape[0], range_v.shape[0]
    halfh = int(nh / 2)
    halfv = int(nv / 2)
    ab = 3 if range_h[-1] > 3 else halfh
    frh, uph = halfh - ab, halfh + ab + 1
    ab = 3 if range_v[-1] > 3 else halfv
    frv, upv = halfv - ab, halfv + ab + 1

    # Slices relative to central ranges.
    # Offsets.
    pos_pair_h    -= pos_pair_h[halfh, halfv]
    pos_pair_v    -= pos_pair_v[halfh, halfv]
    # ROIs.
    pos_nom_h_roi  = pos_nom_h[frh:uph, frv:upv]
    pos_nom_v_roi  = pos_nom_v[frh:uph, frv:upv]
    pos_pair_h_roi = pos_pair_h[frh:uph, frv:upv]
    pos_pair_v_roi = pos_pair_v[frh:uph, frv:upv]

    # Scaling coefficients, pairwise calculation.
    kxp, deltaxp   = np.polyfit(pos_pair_h_roi.T.ravel(),
                                pos_nom_h_roi.ravel(),
                                deg=1)
    kyp, deltayp   = np.polyfit(pos_pair_v_roi.T.ravel(),
                                pos_nom_v_roi.ravel(), deg=1)
    print("\n#### Pairwise blades:")
    print(f"kx = {kxp:.4f}, \t deltax = {deltaxp:.4f}")
    print(f"ky = {kyp:.4f}, \t deltay = {deltayp:.4f}")
    scaled_pos_pair_h = kxp * pos_pair_h
    scaled_pos_pair_v = kyp * pos_pair_v

    # Central ranges for cross-blades.
    x0c, y0c      = pos_cr_h[halfh, halfv], pos_cr_v[halfh, halfv]
    pos_cr_h     -= x0c
    pos_cr_v     -= y0c
    pos_cr_h_roi  = pos_cr_h[frh:uph, frv:upv]
    pos_cr_v_roi  = pos_cr_v[frh:uph, frv:upv]

    # Scaling coefficients, cross-blades calculation.
    try:
        kxc, deltaxc = np.polyfit(pos_cr_h_roi.T.ravel(),
                                  pos_nom_h_roi.ravel(), deg=1)
        kyc, deltayc = np.polyfit(pos_cr_v_roi.T.ravel(),
                                  pos_nom_v_roi.ravel(), deg=1)
    except Exception as err:
        print(f"\n WARNING: when scaling cross-blades position, \n {err}."
              "\n Scaling not done.")
        kxc, deltaxc = 1., 0.
        kyc, deltayc = 1., 0.
    print("\n#### Cross blades:")
    print(f"kx = {kxc:.4f}, \t deltax = {deltaxc:.4f}")
    print(f"ky = {kyc:.4f}, \t deltay = {deltayc:.4f}")
    scaled_pos_cr_h = kxc * pos_cr_h + deltaxc
    scaled_pos_cr_v = kyc * pos_cr_v + deltayc

    fig, (axcross, axpair) = plt.subplots(1, 2, figsize=(12, 6))
    # fig.tight_layout()

    # axpair.plot(grid[:, 0], grid[:, 1], 'r+')
    axpair.plot(pos_nom_h, pos_nom_v, 'r+')
    axpair.plot(scaled_pos_pair_h, scaled_pos_pair_v, 'bo')
    # axpair.plot(pos_pair_h, pos_pair_v, 'bo-')
    axpair.set_title(f"Pairwise-blades positions @ {prm['beamline']}")
    axpair.grid()

    axpair.set_xlabel(u"$x$ [$\\mu$rad]")
    axpair.set_ylabel(u"$y$ [$\\mu$rad]")
    axcross.set_xlabel(u"$x$ [$\\mu$rad]")
    axcross.set_ylabel(u"$y$ [$\\mu$rad]")

    # axcross.plot(grid[:, 0], grid[:, 1], 'r+')
    # hmin, hmax = np.min(pos_cr_h), np.max(pos_cr_h)
    # vmin, vmax = np.min(pos_cr_v), np.max(pos_cr_v)
    # axcross.set_xlim(hmin, hmax)
    # axcross.set_ylim(vmin, vmax)
    # axcross.plot(pos_cr_h, pos_cr_v, 'bo-')
    axcross.plot(scaled_pos_cr_h, scaled_pos_cr_v, 'bo-')
    axcross.set_title(f"Cross-blades positions @ {prm['beamline']}")
    axcross.grid()

    fig.tight_layout()
    fig.savefig(f"xbpm_scaled_fitted_{prm['beamline']}.svg")

    return [pos_pair_h, pos_pair_v], [pos_cr_h, pos_cr_v]


# Pairwise blades calculation.

def suppression_matrix(range_h, range_v, blades_h, blades_v,
                       show=False, nosuppress=False):
    """Calculate the suppression matrix.

    Args:
        range_h (numpy array) : horizontal coordinates to be analysed.
        range_v (numpy array) : vertical coordinates to be analysed.
        blades_h (numpy array) : data recorded by blade's measurements at
            horizontal central line.
        blades_v (numpy array) : data recorded by blade's measurements at
            vertical central line.

        show (bool) : show blades' curves.
        nosuppress (bool) : return a matrix where gains are set to 1.

    Returns:
        the suppression matrix (numpy array)
    """
    # Linear fittings to each blade's data through horizontal and vertical
    # central lines.
    pch = list()
    for blade in blades_h.values():
        weight = 1. / blade[:, 1]
        pch.append(np.polyfit(range_h, blade[:, 0], deg=1, w=weight))

    pcv = list()
    for blade in blades_v.values():
        weight = 1. / blade[:, 1]
        pcv.append(np.polyfit(range_v, blade[:, 0], deg=1, w=weight))

    if show:
        blades_center_show(range_h, range_v, blades_h, blades_v,
                           np.array(pch), np.array(pcv))

    # Normalize by the first blade and define suppression as 1/m.
    pcv = pcv[0] / np.abs(pcv)
    pch = pch[0] / np.abs(pch)

    # Set all suppressions to 1.
    if nosuppress:
        pch = np.ones(8).reshape(4, 2)
        pcv = np.ones(8).reshape(4, 2)

    supmat = np.array([
        [pch[0, 0],  pch[1, 0], -pch[2, 0], -pch[3, 0]],   # noqa: E241
        [pch[0, 0],  pch[1, 0],  pch[2, 0],  pch[3, 0]],   # noqa: E241
        [pcv[0, 0], -pcv[1, 0], -pcv[2, 0],  pcv[3, 0]],   # noqa: E241
        [pcv[0, 0],  pcv[1, 0],  pcv[2, 0],  pcv[3, 0]],   # noqa: E241
    ])

    # DEBUG
    print(f" (SUP MAT) SUP MATRIX (nosuppress = {nosuppress}) = \n{supmat}\n")
    # DEBUG

    return supmat


def beam_position_pair(data, supmat):
    """Calculate beam position from blades' currents.

    Args:
        data (dict) : fetched data from bpm and xbpm (blades) measurements.
        supmat (numpy array): supression matrix.

    Returns:
        positions (numpy array) : h and v calculated positions for
            bpm and xbpm.
    """
    positions = dict()
    for pos, bld in data.items():
        dsps = supmat @ bld[:, 0]  # .T
        # Position is calculated as delta over sigma.
        positions[pos] = np.array([dsps[0] / dsps[1], dsps[2] / dsps[3]])
    zero_origin(positions)
    return positions


def zero_origin(positions):
    """Subtract the values at the center of the grid to correct offset.

    Args:
        positions (dict) : calculated values of positions, indexed by nominal
        values at the grid.
    """
    zero = deepcopy(positions[0, 0])
    for val in positions.values():
        val -= zero


def position_dict_parse(data):
    """Parse data from XBPM dictionary (scaled from BPM positions).

    Args:
        data (dict) :  calculated beam positions indexed by their respective
            nominal positions.

    Returns:
        xbpm_nom_h, xbpm_nom_v (numpy array) : beam nominal positions.
        xbpm_meas_h, xbpm_meas_v (numpy array) : beam calculated positions
            from measured blades' currents.
    """
    gridlist = np.array(list(data.keys()))
    gridpoints = list(set(gridlist.ravel()))
    gridstep = np.abs(gridpoints[1] - gridpoints[0])
    rsh = int(np.sqrt(gridlist.shape[0]))
    xbpm_nom_h, xbpm_nom_v = np.zeros((rsh, rsh)), np.zeros((rsh, rsh))
    xbpm_meas_h, xbpm_meas_v = np.zeros((rsh, rsh)), np.zeros((rsh, rsh))

    maxval = np.max(gridlist)
    for key, val in data.items():
        lin = int((maxval + key[1]) / gridstep)
        col = int((maxval + key[0]) / gridstep)
        try:
            xbpm_nom_h[lin, col]  = key[0]
            xbpm_nom_v[lin, col]  = key[1]
            xbpm_meas_h[lin, col] = val[0]
            xbpm_meas_v[lin, col] = val[1]
        except Exception as err:
            print(f"\n WARNING: when parsing positions dictionary:\n{err}")
            continue

    return (xbpm_nom_h, xbpm_nom_v, xbpm_meas_h, xbpm_meas_v)


# Cross-blades calculation.

def beam_position_cross(blades):
    """Calculate beam position from blades' currents.

    Args:
        blades (list) : averaged values measured for each blade.
        prm (dict) : general parameters of the analysis.

    Returns:
        pos_h, pos_v (numpy array): calculated positions by crossing
            differences.
    """
    to, ti, bi, bo = blades
    v1 = (to - bi) / (to + bi)
    v2 = (ti - bo) / (ti + bo)
    pos_h = (v1 - v2)
    pos_v = (v1 + v2)
    return [pos_h, pos_v]


def blades_center_show(range_h, range_v, blades_h, blades_v, pch, pcv):
    """Show blades' behaviour and fittings at central sweep lines."""
    fig, (axh, axv) = plt.subplots(1, 2, figsize=(13, 6))

    idx = 0
    for lbl, blade in blades_h.items():
        bld = blade[:, 0]  # / blade[0, 0]
        axh.plot(range_h, bld, 'o-', label=lbl)
        bfit = pch[idx, 0] * range_h + pch[idx, 1]
        axh.plot(range_h, bfit, '^',
                 linestyle="dashed", label=f"{lbl} fit")
        idx += 1

    idx = 0
    for lbl, blade in blades_v.items():
        axv.plot(range_v, blade[:, 0], 'o-', label=lbl)
        bfit = pcv[idx, 0] * range_v + pcv[idx, 1]
        axv.plot(range_v, bfit, '^',
                 linestyle="dashed", label=f"{lbl} fit")
        idx += 1

    for ax in [axh, axv]:
        ax.set_ylabel(u"blade counts [a.u]")
        ax.legend()
        ax.grid()

    axh.set_xlabel(u"$x$ [a.u]")
    axh.set_title("Horizontal sweeps")
    #
    axv.set_xlabel(u"$y$ [a.u]")
    axv.set_title("Vertical sweeps")
    fig.tight_layout()


# ## Main function.

def main():
    """."""
    # Read command line options.
    prm = cmd_options()

    # Get raw data. These are the data from all sectors.
    # alldata = get_pickle_data(prm)
    # rawdata = alldata[prm["skip"]:]
    rawdata = get_pickle_data(prm)[prm["skip"]:]
    prm["gridstep"] = gridstep_get(rawdata)
    print(f"###  Setting gridstep to {prm['gridstep']}.\n")

    # Beam position at XBPM calculated from BPM data solely.
    # The sector is selected from 'section' parameter.
    if prm["xbpmfrombpm"]:
        beam_positions_from_bpm(rawdata, prm)

    # Extract data from raw data.
    data = blades_fetch(rawdata, prm["beamline"])

    # Dictionary with measured data from blades for each nominal position.
    if prm["showblademap"]:
        blades_map_show(data, prm)

    # Dump data to file.
    if prm["outputfile"] is not None:
        data_dump(data, prm)

    # Show central sweeping results.
    if prm["sweep"]:
        csweeps = central_sweeps(data, prm, show=True)

    # Calculate beam position from XBPM data.
    if prm["showbladescenter"]:
        blades_show_at_center(*central_sweeps(data, prm, show=False))

    # Calculate beam position from XBPM data.
    if prm["xbpmpositions"]:
        csweeps = central_sweeps(data, prm, show=False)
        ([pos_pair_h, pos_pair_v],
         [pos_cr_h, pos_cr_v]) = xbpm_position_calc(data, prm, *csweeps)

    plt.show()


if __name__ == "__main__":
    main()
    print("\n\n Done.")
