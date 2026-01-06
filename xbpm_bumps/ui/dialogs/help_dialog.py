from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt


class HelpDialog(QDialog):
    """Simple help window with program overview and guidance.

    Displays structured text describing what the program does,
    available analyses, graphs produced, exports, and UI tips.
    """

    def __init__(self, parent=None):
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

    def _build_help_text(self) -> str:
        return (
            "XBPM Beam Position Analysis - Help\n\n"
            "Overview:\n"
            "- Extracts XBPM blade currents and computes beam positions using\n"
            "  pairwise and cross-blade formulas.\n"
            "- Supports suppression matrix correction and scaling to physical units.\n\n"
            "Inputs & Parameters:\n"
            "- Workdir/File: Source of measurements (pickle directory or text file).\n"
            "- XBPM Distance: Auto-set per beamline; can be overridden.\n"
            "- Grid Step, Skip: Data handling options.\n"
            "- Analysis Toggles: Enable desired outputs (Blade Map, Sweeps, XBPM Raw/Scaled, etc.).\n\n"
            "Analyses & Graphs:\n"
            "- Blade Map: 2x2 heatmaps of blade currents (TO, TI, BI, BO).\n"
            "- Central Sweeps: Responses along center lines (horizontal/vertical).\n"
            "- Blades at Center: Blade currents around central region.\n"
            "- XBPM Raw: Positions from XBPM without suppression; figures for pairwise and cross.\n"
            "- XBPM Scaled: Positions with suppression + scaling; figures for pairwise and cross.\n\n"
            "Figures:\n"
            "- Opened in popup windows and embedded in tabs; maintained with equal aspect ratio.\n"
            "- Layout uses constrained engine to reduce whitespace.\n\n"
            "Exports:\n"
            "- Trigger via the Export button; saves outputs based on selected analyses.\n"
            "- Data files: blades' values and positions; central sweeps tables.\n"
            "- Figures: blade_map.png, central_sweeps.png, blades_center.png,\n"
            "  plus pair/cross figures for raw/scaled positions.\n"
            "- Scaling factors: kx, ky, dx, dy saved for pair and cross (raw/scaled).\n\n"
            "UI Tips:\n"
            "- Use 'Run Analysis' after setting workdir; select beamline if prompted.\n"
            "- 'Export…' writes data and figures to chosen prefix.\n"
            "- 'All' button in parameters toggles all analysis options.\n"
            "- XBPM distance updates automatically when beamline changes.\n\n"
            "Notes:\n"
            "- Logging messages appear in the Console tab.\n"
            "- Figures use consistent DPI and aspect; colorbars sized to fit.\n"
        )
