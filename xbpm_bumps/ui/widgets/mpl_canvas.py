"""Reusable Matplotlib canvas widget with toolbar."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT


class MatplotlibCanvas(QWidget):
    """Widget wrapping a Matplotlib figure, canvas, and toolbar."""

    def __init__(self, parent=None, width: float = 5, height: float = 4, dpi: int = 100):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear(self) -> None:
        """Clear axes and refresh display."""
        self.ax.clear()
        self.canvas.draw_idle()
