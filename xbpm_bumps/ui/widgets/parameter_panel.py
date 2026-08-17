"""Parameter input panel for XBPM analysis."""

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QCheckBox, QGroupBox, QDoubleSpinBox, QSpinBox, QPushButton, QLabel,
    QLineEdit
)
from PyQt5.QtCore import pyqtSignal

from xbpm_bumps.core.data_structure import Positions

from ...core.constants import ROI_SIZE_H, ROI_SIZE_V


class ParameterPanel(QWidget):
    """Widget for inputting XBPM analysis parameters.

    Emits parametersChanged signal when any parameter is modified.
    """

    def __init__(self, parent: QWidget = None):
        """Initialize the parameter panel.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.all_check : bool = False  # Track toggle state
        self.inputfile : str  = ""
        self.beamline  : str  | None = None
        self.setup_ui()

    parametersChanged = pyqtSignal()  # noqa: N815

    def set_beamline(self, beamline: str):
        """Set the beamline value for persistence in the panel."""
        self.beamline = beamline
        self.parametersChanged.emit()

    def get_beamline(self) -> str | None:
        """Get the currently set beamline, if any."""
        return self.beamline

    def setup_ui(self) -> None:
        """Initialize the UI layout and widgets."""
        layout = QVBoxLayout(self)

        # Current working directory display (read-only)
        inputfile_row = QHBoxLayout()
        inputfile_row.addWidget(QLabel("Input file:"))
        self.inputfile_field = QLineEdit()
        self.inputfile_field.setReadOnly(True)
        self.inputfile_field.setPlaceholderText("(not set)")
        self.inputfile_field.setMinimumWidth(260)
        inputfile_row.addWidget(self.inputfile_field, 1)
        layout.addLayout(inputfile_row)

        # Parameters group
        layout.addWidget(self._create_parameters_group())

        # Reference selection group (BPM vs nominal)
        layout.addWidget(self._create_reference_group())

        # Analysis options group
        layout.addWidget(self._create_options_group())

        layout.addStretch()

    def _create_parameters_group(self) -> QGroupBox:
        """Create the numerical parameters group."""
        group  = QGroupBox("Parameters")
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        # XBPM distance
        self.xbpmdist_spin = QDoubleSpinBox()
        self.xbpmdist_spin.setRange(0.0, 100.0)
        self.xbpmdist_spin.setValue(15.74)
        self.xbpmdist_spin.setDecimals(3)
        self.xbpmdist_spin.setSuffix(" m")
        self.xbpmdist_spin.setSpecialValueText("Auto (from beamline)")
        self.xbpmdist_spin.setMaximumWidth(115)
        self.xbpmdist_spin.valueChanged.connect(self.parametersChanged.emit)
        layout.addRow("XBPM Distance:", self.xbpmdist_spin)

        # ROI sizes (horizontal/vertical)
        self.roi_h_spin = QSpinBox()
        self.roi_h_spin.setRange(1, 999)
        self.roi_h_spin.setValue(int(ROI_SIZE_H))
        self.roi_h_spin.setSuffix(" pts")
        self.roi_h_spin.setMaximumWidth(115)
        self.roi_h_spin.valueChanged.connect(self.parametersChanged.emit)
        layout.addRow("ROI Size H:", self.roi_h_spin)

        self.roi_v_spin = QSpinBox()
        self.roi_v_spin.setRange(1, 999)
        self.roi_v_spin.setValue(int(ROI_SIZE_V))
        self.roi_v_spin.setSuffix(" pts")
        self.roi_v_spin.setMaximumWidth(115)
        self.roi_v_spin.valueChanged.connect(self.parametersChanged.emit)
        layout.addRow("ROI Size V:", self.roi_v_spin)

        # Skip initial data
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 1000)
        self.skip_spin.setValue(0)
        self.skip_spin.setSuffix(" points")
        self.skip_spin.setMaximumWidth(115)
        self.skip_spin.valueChanged.connect(self.parametersChanged.emit)
        layout.addRow("Skip Initial:", self.skip_spin)

        # Polynomial degree
        self.scalepolydeg = QSpinBox()
        self.scalepolydeg.setRange(1, 2)
        self.scalepolydeg.setValue(1)
        # self.scalepolydeg.setSuffix("")
        self.scalepolydeg.setMaximumWidth(115)
        self.scalepolydeg.valueChanged.connect(self.parametersChanged.emit)

        layout.addRow("Polynomial degree:", self.scalepolydeg)
        group.setLayout(layout)
        return group

    def _create_reference_group(self) -> QGroupBox:
        """Create the reference selection group (BPM vs nominal)."""
        group = QGroupBox()
        group.setTitle("")  # No title for minimal appearance
        layout = QVBoxLayout()

        # Use BPM positions as nominal reference
        self.bpm_ref_check = QCheckBox(
            "Use BPM positions as nominal reference"
        )
        self.bpm_ref_check.setChecked(True)  # Default to BPM reference
        self.bpm_ref_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.bpm_ref_check)

        group.setLayout(layout)
        return group

    def _create_options_group(self) -> QGroupBox:
        """Create the analysis options checkboxes group."""
        group = QGroupBox("Analysis Options")
        layout = QVBoxLayout()

        # 1. Calculate BPM positions
        self.bpm_check = QCheckBox("Calculate BPM positions (-b)")
        self.bpm_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.bpm_check)

        # 2. Show blade map
        self.blademap_check = QCheckBox("Show blade map (-m)")
        self.blademap_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.blademap_check)

        # 3. Show central line sweeps
        self.central_check = QCheckBox("Show central line sweeps (-c)")
        self.central_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.central_check)

        # 4. Show positions at center
        self.center_check = QCheckBox("Show positions at center (-s)")
        self.center_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.center_check)

        # 5. XBPM positions without suppression
        self.xbpm_raw_check = QCheckBox(
            "XBPM positions without suppression (-r)"
        )
        self.xbpm_raw_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.xbpm_raw_check)

        # 6. Calculate XBPM positions (scaled)
        self.xbpm_check = QCheckBox("Calculate XBPM positions (-x)")
        self.xbpm_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.xbpm_check)

        # Button row for "All" option
        layout.addSpacing(10)
        button_layout = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.clicked.connect(self._check_all_options)
        button_layout.addWidget(all_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        group.setLayout(layout)
        return group

    def _check_all_options(self) -> None:
        """Toggle all analysis option checkboxes."""
        self._all_checked = not self._all_checked
        self.bpm_check.setChecked(self._all_checked)
        self.blademap_check.setChecked(self._all_checked)
        self.central_check.setChecked(self._all_checked)
        self.center_check.setChecked(self._all_checked)
        self.xbpm_raw_check.setChecked(self._all_checked)
        self.xbpm_check.setChecked(self._all_checked)
        self.parametersChanged.emit()

    def show_inputfile(self, inputfile: str) -> None:
        """Set the input file path programmatically.

        This replaces the old editable working directory field with an input file field; triggers parametersChanged.
        """
        inputfile = inputfile or ""
        if inputfile != self.inputfile:
            self.inputfile = inputfile
            display = inputfile if inputfile else ""
            self.inputfile_field.setText(display)
            self.inputfile_field.setToolTip(display)
            self.parametersChanged.emit()

    def set_roi_defaults_from_grid(self, coords: Positions) -> None:
        """Set ROI defaults from available grid points in each axis."""
        nh = len(np.unique(coords.x))
        nv = len(np.unique(coords.y))
        self.roi_h_spin.setValue(max(1, nh))
        self.roi_v_spin.setValue(max(1, nv))
        return (nh, nv)  # Return the grid shape for reference

    def get_parameters(self) -> dict:
        """Extract current parameter values as a dictionary.

        Returns:
            Dictionary with parameter names and values compatible with
            ParameterBuilder.from_cli() format.
        """
        params = {
            'inputfile'             : self.inputfile,
            'outputfile'            : self.outputfile,
            'beamline'              : self.beamline,
            'xbpmdist'              : self.xbpmdist_spin.value(),
            'roisize'               : [
                int(self.roi_v_spin.value()),
                int(self.roi_h_spin.value()),
                ],
            'show_blademap'         : self.blademap_check.isChecked(),
            'show_centralsweep'     : self.central_check.isChecked(),
            'show_bladecenter'      : self.center_check.isChecked(),
            'show_xbpmpositions'    : self.xbpm_check.isChecked(),
            'show_bpmpositions'     : self.bpm_check.isChecked(),
            'show_xbpmpositionsraw' : self.xbpm_raw_check.isChecked(),

            'skip'                  : self.skip_spin.value(),
            'scalepolydeg'          : self.scalepolydeg.value(),
            'usebpmref'             : self.bpm_ref_check.isChecked(),
        }

        # Define variables in parameters dataclass.
        return params

    # def set_parameters(self, params: dict):
    #     """Set parameter values from a dictionary.

    #     Args:
    #         params: Dictionary with parameter names and values.
    #     """
    #     # Text and numeric parameters
    #     if 'workdir' in params:
    #         self.set_workdir(params['workdir'])
    #     if 'xbpmdist' in params and params['xbpmdist'] is not None:
    #         self.xbpmdist_spin.setValue(params['xbpmdist'])
    #     if 'scalepolydeg' in params and params['scalepolydeg'] is not None:
    #         self.scalepolydeg.setValue(params['scalepolydeg'])
    #     if 'roisize' in params and params['roisize']:
    #         try:
    #             self.roi_h_spin.setValue(int(params['roisize'][0]))
    #             self.roi_v_spin.setValue(int(params['roisize'][1]))
    #         except Exception:  # noqa: S110
    #             pass
    #     if 'skip' in params:
    #         self.skip_spin.setValue(params['skip'])

    #     # Boolean checkboxes - map parameter name to widget
    #     checkboxes = {
    #         'xbpmpositionsraw' : self.xbpm_raw_check,
    #         'xbpmpositions'    : self.xbpm_check,
    #         'xbpmfrombpm'      : self.bpm_check,
    #         'usebpmref'         : self.bpm_ref_check,
    #         'showblademap'     : self.blademap_check,
    #         'centralsweep'     : self.central_check,
    #         'showbladescenter' : self.center_check,
    #     }
    #     for param, checkbox in checkboxes.items():
    #         if param in params:
    #             checkbox.setChecked(params[param])
