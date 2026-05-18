"""Dialog for selecting beamline when multiple are detected."""

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QHBoxLayout,
)


class BeamlineSelectionDialog(QDialog):
    """Dialog to select a beamline from multiple options."""

    def __init__(self, beamlines: list, parent=None):
        """Initialize dialog with list of beamlines.

        Args:
            beamlines: List of beamline names to choose from.
            parent: Parent widget (optional).
        """
        super().__init__(parent)
        self.selected_beamline = None
        self.beamlines = beamlines
        self.setup_ui()

    def setup_ui(self):
        """Build UI with radio buttons for each beamline."""
        layout = QVBoxLayout()

        label = QLabel("Multiple beamlines found. Select one:")
        layout.addWidget(label)

        self.button_group = QButtonGroup()
        for i, beamline in enumerate(self.beamlines):
            radio = QRadioButton(beamline)
            if i == 0:
                radio.setChecked(True)
            self.button_group.addButton(radio, i)
            layout.addWidget(radio)

        buttons_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(ok_btn)
        buttons_layout.addWidget(cancel_btn)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)
        self.setWindowTitle("Select Beamline")

    def get_selection(self) -> str:
        """Return the selected beamline."""
        idx = self.button_group.checkedId()
        return self.beamlines[idx] if idx >= 0 else None
