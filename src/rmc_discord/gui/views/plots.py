"""Matplotlib helpers shared by the views (color limits as in NeuXtalViz)."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.figure import Figure
from matplotlib.transforms import Affine2D
from qtpy.QtWidgets import QVBoxLayout, QWidget

CMAPS = {"Sequential": "viridis", "Rainbow": "turbo", "Binary": "binary", "Diverging": "bwr"}
LIMITS = ["Min/Max", "μ±3×σ", "Q₃/Q₁±1.5×IQR"]


def color_limits(values, method):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0, 1.0
    if method == "μ±3×σ":
        lo, hi = v.mean() - 3 * v.std(), v.mean() + 3 * v.std()
    elif method == "Q₃/Q₁±1.5×IQR":
        q1, q3 = np.percentile(v, [25, 75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    else:
        lo, hi = v.min(), v.max()
    lo, hi = max(lo, v.min()), min(hi, v.max())
    if hi <= lo:
        hi = lo + 1e-12 + abs(lo) * 1e-6
    return float(lo), float(hi)


def norm(values, method, scale, symmetric=False):
    lo, hi = color_limits(values, method)
    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    if scale == "Log":
        if symmetric:
            return SymLogNorm(linthresh=max(1e-12, 1e-3 * hi), vmin=lo, vmax=hi)
        v = values[np.isfinite(values) & (values > 0)]
        lo = max(lo, v.min() if v.size else 1e-12)
        return LogNorm(vmin=lo, vmax=max(hi, lo * 10))
    return Normalize(vmin=lo, vmax=hi)


class PlotWidget(QWidget):
    """Figure canvas with the navigation toolbar."""

    def __init__(self, parent=None, toolbar=True):
        super().__init__(parent)
        self.canvas = FigureCanvas(Figure(constrained_layout=True))
        self.fig = self.canvas.figure
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, stretch=1)

    def draw(self):
        self.canvas.draw_idle()


def metric_mesh(ax, image, edges_a, edges_b, skew, aspect, **kwargs):
    """
    pcolormesh in projection coordinates with the true reciprocal metric:
    the ``b`` axis is sheared by ``skew`` and the aspect set so circles of
    constant :math:`|Q|` are round.
    """
    T = Affine2D(np.array([[1.0, skew, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    mesh = ax.pcolormesh(edges_a, edges_b, image.T, transform=T + ax.transData, shading="flat", **kwargs)
    xs = [edges_a[0] + skew * edges_b[0], edges_a[0] + skew * edges_b[-1],
          edges_a[-1] + skew * edges_b[0], edges_a[-1] + skew * edges_b[-1]]
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(edges_b[0], edges_b[-1])
    ax.set_aspect(aspect)
    return mesh, T
