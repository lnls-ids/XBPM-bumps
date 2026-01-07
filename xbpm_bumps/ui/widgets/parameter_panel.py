"""Parameter input panel for XBPM analysis."""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QCheckBox, QPushButton, QFileDialog,
    QGroupBox, QDoubleSpinBox, QSpinBox
)
from PyQt5.QtCore import pyqtSignal


class ParameterPanel(QWidget):
    """Widget for inputting XBPM analysis parameters.

    Emits parametersChanged signal when any parameter is modified.
    """

    parametersChanged = pyqtSignal()  # noqa: N815

    def __init__(self, parent=None):
        """Initialize the parameter panel.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._all_checked = False  # Track toggle state
        self.setup_ui()

    def setup_ui(self):
        """Initialize the UI layout and widgets."""
        layout = QVBoxLayout(self)

        # Input files group
        files_group = self._create_files_group()
        layout.addWidget(files_group)

        # Parameters group
        params_group = self._create_parameters_group()
        layout.addWidget(params_group)

        # Analysis options group
        options_group = self._create_options_group()
        layout.addWidget(options_group)

        layout.addStretch()

    def _create_files_group(self) -> QGroupBox:
        """Create the file/directory selection group."""
        group = QGroupBox("Input Data")
        layout = QHBoxLayout()

        self.workdir_edit = QLineEdit()
        self.workdir_edit.setPlaceholderText(
            "Select working directory or data file..."
        )
        self.workdir_edit.textChanged.connect(self._on_workdir_changed)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_workdir)

        layout.addWidget(self.workdir_edit)
        layout.addWidget(browse_btn)
        group.setLayout(layout)

        return group

    def _create_parameters_group(self) -> QGroupBox:
        """Create the numerical parameters group."""
        group = QGroupBox("Parameters")
        layout = QFormLayout()

        # XBPM distance
        self.xbpmdist_spin = QDoubleSpinBox()
        self.xbpmdist_spin.setRange(0.0, 100.0)
        self.xbpmdist_spin.setValue(15.74)
        self.xbpmdist_spin.setDecimals(3)
        self.xbpmdist_spin.setSuffix(" m")
        self.xbpmdist_spin.setSpecialValueText("Auto (from beamline)")
        self.xbpmdist_spin.valueChanged.connect(self.parametersChanged.emit)
        layout.addRow("XBPM Distance:", self.xbpmdist_spin)

        # Grid step
        self.gridstep_spin = QDoubleSpinBox()
        self.gridstep_spin.setRange(0.0, 100.0)
        self.gridstep_spin.setValue(0.0)
        self.gridstep_spin.setDecimals(2)
        self.gridstep_spin.setSuffix(" µm")
        self.gridstep_spin.setSpecialValueText("Auto (infer from data)")
        self.gridstep_spin.valueChanged.connect(self.parametersChanged.emit)
        layout.addRow("Grid Step:", self.gridstep_spin)

        # Skip initial data
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 1000)
        self.skip_spin.setValue(0)
        self.skip_spin.setSuffix(" points")
        self.skip_spin.valueChanged.connect(self.parametersChanged.emit)
        layout.addRow("Skip Initial:", self.skip_spin)

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

    def _check_all_options(self):
        """Toggle all analysis option checkboxes."""
        self._all_checked = not self._all_checked
        self.bpm_check.setChecked(self._all_checked)
        self.blademap_check.setChecked(self._all_checked)
        self.central_check.setChecked(self._all_checked)
        self.center_check.setChecked(self._all_checked)
        self.xbpm_raw_check.setChecked(self._all_checked)
        self.xbpm_check.setChecked(self._all_checked)
        self.parametersChanged.emit()

    def _browse_workdir(self):
        """Open file/directory browser dialog."""
        # Try directory first, then file
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            self.workdir_edit.text() or "."
        )

        if not path:
            # Try selecting a file instead
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Data File",
                self.workdir_edit.text() or ".",
                "Data Files (*.dat *.txt);;All Files (*)"
            )

        if path:
            self.workdir_edit.setText(path)

    def _on_workdir_changed(self, _text: str):
        """Emit parametersChanged when workdir text updates."""
        self.parametersChanged.emit()

    def get_parameters(self) -> dict:
        """Extract current parameter values as a dictionary.

        Returns:
            Dictionary with parameter names and values compatible with
            ParameterBuilder.from_cli() format.
        """
        params = {
            'workdir': self.workdir_edit.text(),
            'xbpmdist': (
                self.xbpmdist_spin.value()
                if self.xbpmdist_spin.value() > 0
                else None
            ),
            'gridstep': (
                self.gridstep_spin.value()
                if self.gridstep_spin.value() > 0
                else None
            ),
            'skip': self.skip_spin.value(),
            'xbpmpositionsraw': self.xbpm_raw_check.isChecked(),
            'xbpmpositions': self.xbpm_check.isChecked(),
            'xbpmfrombpm': self.bpm_check.isChecked(),
            'showblademap': self.blademap_check.isChecked(),
            'centralsweep': self.central_check.isChecked(),
            'showbladescenter': self.center_check.isChecked(),
        }
        return params

    def set_parameters(self, params: dict):
        """Set parameter values from a dictionary.

        Args:
            params: Dictionary with parameter names and values.
        """
        # Text and numeric parameters
        if 'workdir' in params:
            self.workdir_edit.setText(params['workdir'])
        if 'xbpmdist' in params and params['xbpmdist'] is not None:
            self.xbpmdist_spin.setValue(params['xbpmdist'])
        if 'gridstep' in params and params['gridstep'] is not None:
            self.gridstep_spin.setValue(params['gridstep'])
        if 'skip' in params:
            self.skip_spin.setValue(params['skip'])

        # Boolean checkboxes - map parameter name to widget
        checkboxes = {
            'xbpmpositionsraw' : self.xbpm_raw_check,
            'xbpmpositions'    : self.xbpm_check,
            'xbpmfrombpm'      : self.bpm_check,
            'showblademap'     : self.blademap_check,
            'centralsweep'     : self.central_check,
            'showbladescenter' : self.center_check,
        }
        for param, checkbox in checkboxes.items():
            if param in params:
                checkbox.setChecked(params[param])
