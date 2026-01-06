"""Constants for XBPM analysis."""

FILE_EXTENSION = ".pickle"    # Data file type.
GRIDSTEP = 2                  # Default grid step.
STD_ROI_SIZE = 2              # Default range for ROI.
FIGDPI = 300                  # Figure dpi saving parameter.

HELP_DESCRIPTION = (
"""
This program extract data from a file or a directory with the
measurements of XBPM blades' currents of a few beamlines
(CAT, CNB, MNC, MGN) and calculates the respective positions
based on pairwise and cross-blade formulas.

If data is read from a text file, it  may have a header with parameters,
starting with '#', like:
# SR current: 300.0
# Beamline: CAT
# Phase/Gap: 6.0
# XBPM distance: 15.74
# Inter BPM distance: 6.175495

The data lines must have the format:

nom_x nom_y to err_to ti err_ti bi err_bi bo err_bo

where nom_x and nom_y are the horizontal and vertical nominal
positions of the beam,
to, ti, bi, bo are the blade currents and
err_to, err_ti, err_bi, err_bo are the respective errors.

If data is read from a directory, it must contain pickle files with
data saved.

The data is treated with linear transformations first, to correct for
distortions promoted by different gains in each blade. Firstly, gains
(or their reciprocal, 1/G, which is the suppression) are estimated by
analyzing the slope of the line formed by central points in vertical
and horizontal sweeping. The supression is applied to the set of points
to correct the grid, then a linear scaling is calculated to set distances
in micrometers.
""")
