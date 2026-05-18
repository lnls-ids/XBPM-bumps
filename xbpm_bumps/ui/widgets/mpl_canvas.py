"""Reusable Matplotlib canvas widget with toolbar."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT


class MatplotlibCanvas(QWidget):
    """Widget wrapping a Matplotlib figure, canvas, and toolbar."""

    def __init__(self, parent=None, width: float = 16, height: float = 4, dpi: int = 100):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)

        # Let the canvas expand to fill available space
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Ensure the canvas area stays reasonably wide by default
        self.setMinimumSize(800, 400)

    def clear(self) -> None:
        """Clear axes and refresh display."""
        self.ax.clear()
        self.canvas.draw_idle()
