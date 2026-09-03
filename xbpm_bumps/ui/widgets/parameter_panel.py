"""Parameter input panel for XBPM analysis."""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QCheckBox, QGroupBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QLabel, QLineEdit
)
from PyQt5.QtCore import QSignalBlocker, pyqtSignal
from ...core.data_structure import Positions, Prm, BeamlinePrm
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
        self.inputfile  : str  = ""
        self.outputfile : str  = ""
        self.beamline   : str  | None = None
        self.setup_ui()

    parametersChanged = pyqtSignal()  # noqa: N815

    def set_beamline(self, beamline: str):
        """Set the beamline value for persistence in the panel."""
        self.beamline = beamline

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
        self.bpm_check = QCheckBox(
            "Show positions from BPM measurements"
            )
        self.bpm_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.bpm_check)

        # 2. Show blade map
        self.blademap_check = QCheckBox("Show blade map")
        self.blademap_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.blademap_check)

        # 3. Show central line sweeps
        self.blade_central_check = QCheckBox(
            "Show blades at central line sweeps"
            )
        self.blade_central_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.blade_central_check)

        # 4. Show positions at center
        self.position_center_check = QCheckBox(
            "Show positions at center line sweeps"
            )
        self.position_center_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.position_center_check)

        # 5. XBPM positions without suppression
        self.xbpm_raw_check = QCheckBox(
            "XBPM positions without suppression"
        )
        self.xbpm_raw_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.xbpm_raw_check)

        # 6. Calculate XBPM positions (scaled)
        self.xbpm_calc_check = QCheckBox(
            "Calculate XBPM positions"
            )
        self.xbpm_calc_check.toggled.connect(self.parametersChanged.emit)
        layout.addWidget(self.xbpm_calc_check)

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
        boxes = (
                self.bpm_check,
                self.blademap_check,
                self.blade_central_check,
                self.position_center_check,
                self.xbpm_raw_check,
                self.xbpm_calc_check,
            )
        checked = not all(box.isChecked() for box in boxes)

        for box in boxes:
            box.setChecked(checked)

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

    def set_roi_defaults_from_grid(self,
                                   coords: Positions,
                                   ) -> None:
        """Set ROI defaults from available grid points in each axis."""
        nh = len(np.unique(coords.x))
        nv = len(np.unique(coords.y))
        self.roi_h_spin.setValue(max(1, nh))
        self.roi_v_spin.setValue(max(1, nv))
        return (nv, nh)  # Return the grid shape for reference

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
            'show_centralsweep'     : self.blade_central_check.isChecked(),
            'show_bladecenter'      : self.position_center_check.isChecked(),
            'show_xbpmpositions'    : self.xbpm_calc_check.isChecked(),
            'show_bpmpositions'     : self.bpm_check.isChecked(),
            'show_xbpmpositionsraw' : self.xbpm_raw_check.isChecked(),

            'skip'                  : self.skip_spin.value(),
            'scalepolydeg'          : self.scalepolydeg.value(),
            'usebpmref'             : self.bpm_ref_check.isChecked(),
        }

        # Define variables in parameters dataclass.
        return params

    def load_beamline_data(self,
                           runtime_prm: Prm,
                           beamline_prm: BeamlinePrm,
                           grid_shape: tuple[int, int]
                           ) -> None:
        """Load parameters from BeamlinePrm and runtime Prm into the panel.

        Args:
            runtime_prm   : Runtime parameters to load.
            beamline_prm  : Beamline-specific parameters to load.
            grid_shape    : Tuple containing the vertical and horizontal
                            grid sizes.
        """
        # Block signals while programmatically reflecting loaded state.
        blockers = [QSignalBlocker(widget)
                     for widget in (
                        self.xbpmdist_spin,
                        self.roi_h_spin,
                        self.roi_v_spin,
                        self.skip_spin,
                        self.scalepolydeg,
                        self.bpm_ref_check,
                        )
                    ]   
        self.show_inputfile(runtime_prm.inputfile)
        self.set_beamline(beamline_prm.beamline)

        nv, nh = grid_shape
        self.roi_h_spin.setRange(1, nh)
        self.roi_v_spin.setRange(1, nv)
        self.roi_h_spin.setValue(min(beamline_prm.roi.sz_h, nh))
        self.roi_v_spin.setValue(min(beamline_prm.roi.sz_v, nv))
        self.xbpmdist_spin.setValue(beamline_prm.xbpmdist or 0.0)
        self.skip_spin.setValue(beamline_prm.skip)
        self.scalepolydeg.setValue(beamline_prm.scalepolydeg)
        self.bpm_ref_check.setChecked(beamline_prm.usebpmref)

        del blockers
