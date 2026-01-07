"""Help dialog for XBPM analysis UI."""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
)


class HelpDialog(QDialog):
    """Simple help window with program overview and guidance.

    Displays structured text describing what the program does,
    available analyses, graphs produced, exports, and UI tips.
    """

    def __init__(self, parent=None):
        """Initialize the Help dialog window."""
        super().__init__(parent)
        self.setWindowTitle("XBPM Analysis - Help")
        self.resize(800, 600)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text.setText(self._build_help_text())
        layout.addWidget(self.text)

        btn_bar = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        btn_bar.addStretch(1)
        btn_bar.addWidget(close_btn)
        layout.addLayout(btn_bar)

    @staticmethod
    def _build_help_text() -> str:
        """Build and return the help text content."""
        return (
            """
XBPM Beam Position Analysis - Help

Overview:
- Extracts XBPM blade currents and computes beam positions using
    pairwise and cross-blade formulas.
- Supports suppression matrix correction and scaling to physical units.

Inputs & Parameters:
- Workdir/File: Source of measurements (pickle directory or text file).
- XBPM Distance: Auto-set per beamline; can be overridden.
- Grid Step, Skip: Data handling options.
- Analysis Toggles: Enable desired outputs (Blade Map, Sweeps,
    XBPM Raw/Scaled, etc.).

Analyses & Graphs:
- Blade Map: 2x2 heatmaps of blade currents (TO, TI, BI, BO).
- Central Sweeps: Responses along center lines (horizontal/vertical).
- Blades at Center: Blade currents around central region.
- XBPM Raw: Positions from XBPM without suppression; figures for
    pairwise and cross.
- XBPM Scaled: Positions with suppression + scaling; figures for
    pairwise and cross.

Figures:
- Opened in popup windows and embedded in tabs; maintained with equal
    aspect ratio.
- Layout uses constrained engine to reduce whitespace.

Exports:
- Trigger via the Export button; saves outputs based on selected
    analyses.
- Data files: blades' values and positions; central sweeps tables.
- Figures: blade_map.png, central_sweeps.png, blades_center.png, plus
    pair/cross figures for raw/scaled positions.
- Scaling factors: kx, ky, dx, dy saved for pair and cross
    (raw/scaled).

UI Tips:
- Use 'Run Analysis' after setting workdir; select beamline if
    prompted.
- 'Export…' writes data and figures to chosen prefix.
- 'All' button in parameters toggles all analysis options.
- XBPM distance updates automatically when beamline changes.

Notes:
- Logging messages appear in the Console tab.
- Figures use consistent DPI and aspect; colorbars sized to fit.
"""
        )
