"""Recalculation application view."""

import numpy as np
import pyvista as pv
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

COMPONENTS = {"Full model": "model", "Background": "background", "Occupancy": "occupancy",
              "Size effect": "size effect", "Displacement": "displacement", "Spin": "spin"}
from rmc_discord.gui.views.base_view import BaseView, scalar_bar, set_icon
from rmc_discord.gui.views.plots import CMAPS, LIMITS, PlotWidget, metric_mesh, norm


def _line(text, low=-1e4, high=1e4):
    line = QLineEdit(text)
    line.setValidator(QDoubleValidator(low, high, 5, notation=QDoubleValidator.StandardNotation))
    return line


class RecalculationView(BaseView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tab_widget = QTabWidget(self)
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Model of a data plane from the refined correlations")
        grid = QGridLayout(box)
        self.quantity_combo = QComboBox(self)
        grid.addWidget(QLabel("Show:"), 0, 0)
        grid.addWidget(self.quantity_combo, 0, 1, 1, 2)
        self.normal_combo = QComboBox(self)
        self.normal_combo.addItems(["Axis 1/2", "Axis 1/3", "Axis 2/3"])
        self.slice_line = _line("0.0")
        self.bragg_line = _line("0.12", 0, 0.5)
        grid.addWidget(self.normal_combo, 1, 0)
        grid.addWidget(QLabel("Slice:"), 1, 1)
        grid.addWidget(self.slice_line, 1, 2)
        grid.addWidget(QLabel("Bragg exclusion (rlu):"), 1, 3)
        grid.addWidget(self.bragg_line, 1, 4, 1, 2)
        self.q_min_line = _line("0", 0, 100)
        self.q_max_line = _line("100", 0, 100)
        grid.addWidget(QLabel("|Q| min:"), 2, 0)
        grid.addWidget(self.q_min_line, 2, 1)
        grid.addWidget(QLabel("|Q| max:"), 2, 2)
        grid.addWidget(self.q_max_line, 2, 3)
        self.recalc_button = QPushButton("Recalculate", self)
        set_icon(self.recalc_button, "wave-square")
        self.info = QLabel("")
        self.info.setWordWrap(True)
        grid.addWidget(self.info, 3, 0, 1, 5)
        grid.addWidget(self.recalc_button, 3, 5)
        layout.addWidget(box)

        draw = QHBoxLayout()
        self.scale_combo = QComboBox(self)
        self.scale_combo.addItems(["Linear", "Log"])
        self.clim_combo = QComboBox(self)
        self.clim_combo.addItems(LIMITS)
        self.clim_combo.setCurrentIndex(2)
        self.cbar_combo = QComboBox(self)
        self.cbar_combo.addItems(list(CMAPS))
        for w in (self.scale_combo, self.clim_combo, self.cbar_combo):
            draw.addWidget(w)
        draw.addStretch(1)
        layout.addLayout(draw)
        self.plot = PlotWidget(self)
        layout.addWidget(self.plot, stretch=1)
        self.tab_widget.addTab(tab, "Recalculation")
        self.add_tools(self.tab_widget)

    def connect_recalculate(self, f):
        self.recalc_button.clicked.connect(f)

    def connect_style(self, f):
        for combo in (self.scale_combo, self.clim_combo, self.cbar_combo):
            combo.currentIndexChanged.connect(f)

    def set_quantities(self, channels):
        """Full model, background, and each refined channel."""
        current = self.quantity_combo.currentData()
        self.quantity_combo.clear()
        for name, key in COMPONENTS.items():
            if key in ("model", "background") or key in channels:
                self.quantity_combo.addItem(name, key)
        keys = [self.quantity_combo.itemData(i) for i in range(self.quantity_combo.count())]
        if current in keys:
            self.quantity_combo.setCurrentIndex(keys.index(current))

    def get_parameters(self):
        lines = (self.slice_line, self.bragg_line, self.q_min_line, self.q_max_line)
        if not all(line.hasAcceptableInput() for line in lines):
            return None
        normal = {"Axis 1/2": 2, "Axis 1/3": 1, "Axis 2/3": 0}[self.normal_combo.currentText()]
        return (self.quantity_combo.currentData(), normal, float(self.slice_line.text()),
                float(self.q_min_line.text()), float(self.q_max_line.text()), float(self.bragg_line.text()))

    def set_q_range(self, q_min, q_max):
        """Default to the |Q| range the model was refined on."""
        self.q_min_line.setText(f"{q_min:g}")
        self.q_max_line.setText(f"{min(q_max, 100.0):g}")

    def set_result_info(self, text):
        self.info.setText(text)

    def draw_planes(self, res):
        fig = self.plot.fig
        fig.clear()
        if res is None:
            self.plot.draw()
            return
        scale, limits = self.scale_combo.currentText(), self.clim_combo.currentText()
        cmap = CMAPS[self.cbar_combo.currentText()]
        nrm = norm(res["data"], limits, scale)
        skew, aspect = res["transform"]
        names = res["names"]
        axes = []
        for i, (key, title) in enumerate((("data", "data"), ("model", res.get("title", "model")))):
            ax = fig.add_subplot(1, 2, i + 1, sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
            mesh, _ = metric_mesh(ax, res[key], *res["edges"], skew, aspect, cmap=cmap, norm=nrm, rasterized=True)
            ax.set_title(f"{title}, {names[res['normal']]} = {round(res['value'], 6) + 0.0:.3g}")
            ax.set_xlabel(names[res["axes"][0]])
            if i == 0:
                ax.set_ylabel(names[res["axes"][1]])
            axes.append(ax)
        fig.colorbar(mesh, ax=axes, shrink=0.7)
        self.plot.draw()

    def draw_surface(self, res):
        self.clear_scene()
        if res is not None:
            Q = res["Q"]
            grid = pv.StructuredGrid(Q[..., 0], Q[..., 1], Q[..., 2])
            grid["I"] = res["model"].ravel(order="F")
            lo, hi = np.nanpercentile(res["model"], [2, 98])
            self.plotter.add_mesh(grid, scalars="I", cmap=CMAPS[self.cbar_combo.currentText()], clim=(lo, hi),
                                  nan_opacity=0.0, scalar_bar_args=scalar_bar("recalculated"))
        self.reset_scene()
