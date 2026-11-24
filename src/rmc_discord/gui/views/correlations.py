"""Correlations application view."""

import numpy as np
import pyvista as pv
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator, QIntValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rmc_discord.gui.models.correlations import NAMES
from rmc_discord.gui.models.display import add_spheres
from rmc_discord.gui.views.base_view import BaseView, row, scalar_bar, set_icon, short
from rmc_discord.gui.views.plots import PlotWidget


def _integer(text):
    line = QLineEdit(text)
    line.setValidator(QIntValidator(-99, 99))
    line.setMaximumWidth(48)
    return line


def _bar(h):
    return f"{h}" if h >= 0 else f"\\bar{{{-h}}}"


def _miller(v, brackets="()"):
    return "$" + brackets[0] + "".join(_bar(int(x)) for x in v) + brackets[1] + "$"


def _line(text, low=0, high=1e4):
    line = QLineEdit(text)
    line.setValidator(QDoubleValidator(low, high, 4, notation=QDoubleValidator.StandardNotation))
    return line


class CorrelationsView(BaseView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recip_box.setChecked(False)
        tools = QWidget(self)
        layout = QVBoxLayout(tools)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._init_channel_box())
        self.tab_widget = QTabWidget(self)
        self._init_slice_tab()
        self._init_shells_tab()
        layout.addWidget(self.tab_widget, stretch=1)
        self.add_tools(tools)

    def _init_channel_box(self):
        box = QGroupBox("Refined correlations")
        rows = QVBoxLayout(box)
        self.quantity_combo = QComboBox(self)
        self.normalize_box = QCheckBox("Correlation coefficients (occupancy)", self)
        self.normalize_box.setChecked(True)
        self.scale_line = short(QLineEdit(""), 110)
        self.scale_line.setPlaceholderText("arbitrary units")
        self.scale_line.setValidator(QDoubleValidator(0.0, 1e300, 6, notation=QDoubleValidator.ScientificNotation))
        self.scale_line.setToolTip("Difference data fix only scale x <S.S>; enter the scale (e.g. from a fit of the "
                                   "low-temperature volume, where the unit self term sets it) to show <S.S>.")
        self.scale_label = QLabel("Scale:")
        rows.addLayout(row(QLabel("Channel:"), self.quantity_combo, self.normalize_box, self.scale_label,
                           self.scale_line))
        self.set_difference(False)
        self.pair_combo = QComboBox(self)
        self.pair_combo.addItem("All pairs")
        self.r_min_line = short(_line("0"), 60)
        self.r_max_line = short(_line("20"), 60)
        self.save_button = QPushButton("HDF5", self)
        self.save_button.setToolTip("Save the correlation table (HDF5).")
        set_icon(self.save_button, "floppy-disk")
        self.csv_button = QPushButton("CSV", self)
        self.csv_button.setToolTip("Write the shells (CSV).")
        set_icon(self.csv_button, "file-csv")
        export = QHBoxLayout()
        export.addWidget(self.save_button)
        export.addWidget(self.csv_button)
        rows.addLayout(row(QLabel("Pairs:"), self.pair_combo, QLabel("r from"), self.r_min_line, QLabel("to"),
                           self.r_max_line, QLabel("Å")))
        info = QHBoxLayout()
        self.info = QLabel("Refine the correlations first (Refinement).")
        self.info.setWordWrap(True)
        info.addWidget(self.info, stretch=1)
        info.addLayout(export)
        rows.addLayout(info)
        return box

    def _init_shells_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.marker_spin = short(_line("1.0", 0.01, 100), 60)
        self.marker_spin.setToolTip("Sphere size of the 3-D view.")
        layout.addLayout(row(QLabel("3-D sphere size:"), self.marker_spin))
        splitter = QSplitter(Qt.Vertical)
        self.shell_plot = PlotWidget(self)
        splitter.addWidget(self.shell_plot)
        self.shell_table = QTableWidget(0, 5)
        self.shell_table.setHorizontalHeaderLabels(["Pair", "Distance (Å)", "Value", "Error", "Vectors"])
        self.shell_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.shell_table.verticalHeader().setVisible(False)
        self.shell_table.setAlternatingRowColors(True)
        splitter.addWidget(self.shell_table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)
        self.tab_widget.addTab(tab, "Shells")

    def _init_slice_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.h_line, self.k_line, self.l_line = (_integer(v) for v in ("0", "0", "1"))
        self.d_line = short(_line("0", -1e4, 1e4), 60)
        self.tol_line = short(_line("0.01", 0, 10), 60)
        self.slice_size_line = short(_line("40", 0.1, 1e4), 60)
        layout.addLayout(row(QLabel("Plane ("), self.h_line, self.k_line, self.l_line, QLabel(") · [uvw] ="),
                             self.d_line, QLabel("±"), self.tol_line, QLabel("   marker size:"),
                             self.slice_size_line))
        note = QLabel("Coincident vectors of different site pairs are averaged.")
        layout.addWidget(note)
        self.slice_plot = PlotWidget(self)
        layout.addWidget(self.slice_plot, stretch=1)
        self.tab_widget.addTab(tab, "Slice")

    # ------------------------------------------------------------------

    def connect_channel(self, f):
        self.quantity_combo.currentIndexChanged.connect(f)
        self.normalize_box.clicked.connect(f)
        self.scale_line.editingFinished.connect(f)

    def connect_slice(self, f):
        for line in (self.h_line, self.k_line, self.l_line, self.d_line, self.tol_line, self.slice_size_line):
            line.editingFinished.connect(f)

    def get_slice(self):
        lines = (self.h_line, self.k_line, self.l_line, self.d_line, self.tol_line, self.slice_size_line)
        if not all(line.hasAcceptableInput() for line in lines):
            return None
        hkl = [int(line.text()) for line in lines[:3]]
        return hkl, float(self.d_line.text()), float(self.tol_line.text()), float(self.slice_size_line.text())

    def get_scale(self):
        text = self.scale_line.text().strip()
        return float(text) if text and self.scale_line.hasAcceptableInput() and float(text) > 0 else None

    def set_difference(self, difference):
        self.scale_label.setVisible(difference)
        self.scale_line.setVisible(difference)

    def set_occupancy(self, occupancy):
        """Normalization to correlation coefficients applies to occupancies only."""
        self.normalize_box.setVisible(occupancy)

    def connect_filter(self, f):
        self.pair_combo.currentIndexChanged.connect(f)
        for line in (self.r_min_line, self.r_max_line, self.marker_spin):
            line.editingFinished.connect(f)

    def connect_save(self, f):
        self.save_button.clicked.connect(f)

    def connect_csv(self, f):
        self.csv_button.clicked.connect(f)

    def set_quantities(self, quantities):
        self.quantity_combo.blockSignals(True)
        current = self.quantity_combo.currentData()
        self.quantity_combo.clear()
        for q in quantities:
            self.quantity_combo.addItem(NAMES.get(q, q), q)
        if current in quantities:
            self.quantity_combo.setCurrentIndex(quantities.index(current))
        self.quantity_combo.blockSignals(False)

    def get_channel(self):
        return self.quantity_combo.currentData(), self.normalize_box.isChecked()

    def set_compute_info(self, text):
        self.info.setText(text)

    def set_pairs(self, names):
        self.pair_combo.blockSignals(True)
        self.pair_combo.clear()
        self.pair_combo.addItems(["All pairs"] + names)
        self.pair_combo.blockSignals(False)

    def get_filter(self):
        lines = (self.r_min_line, self.r_max_line, self.marker_spin)
        if not all(line.hasAcceptableInput() for line in lines):
            return None
        return self.pair_combo.currentText(), float(self.r_min_line.text()), float(self.r_max_line.text()), \
            float(self.marker_spin.text())

    # ------------------------------------------------------------------

    def draw_shells(self, shells, quantity, normalized):
        fig = self.shell_plot.fig
        fig.clear()
        ax = fig.add_subplot(111)
        if shells is not None:
            for label in sorted(set(shells["label"])):
                k = shells["label"] == label
                ax.errorbar(shells["distance"][k], shells["value"][k], yerr=shells["error"][k], fmt="o",
                            ms=4, capsize=2, label=label)
            ax.axhline(0, color="0.5", lw=0.8)
            if len(set(shells["label"])) <= 10:
                ax.legend(fontsize=8)
        ax.set_xlabel("r (Å)")
        unit = {"occupancy": "⟨δδ⟩", "displacement": "tr V (Å²)", "spin": "⟨S·S⟩",
                "bond": "size effect along the bond (Å)"}.get(quantity, "")
        ax.set_ylabel("correlation coefficient" if normalized and quantity not in ("spin", "bond") else unit)
        self.shell_plot.draw()

        n = 0 if shells is None else len(shells["value"])
        self.shell_table.setRowCount(n)
        for i in range(n):
            values = [shells["label"][i], f"{shells['distance'][i]:.4f}", f"{shells['value'][i]:.5g}",
                      f"{shells['error'][i]:.2g}", str(shells["n"][i])]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self.shell_table.setItem(i, c, item)

    def draw_slice(self, plane, label, size):
        fig = self.slice_plot.fig
        fig.clear()
        ax = fig.add_subplot(111)
        if plane is None:
            self.slice_plot.draw()
            return
        x, y, v = plane["x"], plane["y"], plane["value"]
        m = float(np.max(np.abs(v))) if len(v) else 1.0
        m = m or 1.0
        sc = ax.scatter(x, y, c=v, s=size, cmap="bwr", vmin=-m, vmax=m, edgecolors="0.3", linewidths=0.3,
                        zorder=2)
        fig.colorbar(sc, ax=ax, label=label)
        ax.plot([0], [0], "k+", ms=8, zorder=3)
        extent = max(np.max(np.abs(np.r_[x, y])) if len(x) else 1.0, 1.0)
        axes = np.asarray(plane["axes"], float)
        axes = axes * min(1.0, 0.8 * extent / np.linalg.norm(axes, axis=1).max())  # directions only, inside the view
        for u, (ax_x, ax_y) in zip(plane["directions"], axes):
            ax.annotate("", xy=(ax_x, ax_y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="0.2", lw=1),
                        zorder=1)
            ax.text(ax_x, ax_y, " " + _miller(u, "[]"), fontsize=9, color="0.2")
        ax.set_xlim(-1.08 * extent, 1.08 * extent)
        ax.set_ylim(-1.08 * extent, 1.08 * extent)
        ax.set_aspect("equal")
        ax.grid(True, color="0.9", zorder=0)
        u1 = plane["directions"][0]
        ax.set_xlabel(f"along {_miller(u1, '[]')} (Å)")
        ax.set_ylabel("perpendicular, in plane (Å)")
        d = plane["d"]
        ax.set_title(f"{_miller(plane['hkl'].astype(int))}$\\cdot$[uvw] = {d:g}   ({len(v)} vectors)", fontsize=10)
        self.slice_plot.draw()

    def draw_map(self, r, value, size):
        self.clear_scene()
        if r is not None and len(r):
            m = np.max(np.abs(value)) or 1.0
            add_spheres(self.plotter, r, 0.5 * size * np.sqrt(np.abs(value) / m), values=value, cmap="bwr",
                        clim=(-m, m), scalar_bar_args=scalar_bar("correlation"))
            self.plotter.add_mesh(pv.Sphere(radius=0.15 * size), color="black")
        self.reset_scene()
