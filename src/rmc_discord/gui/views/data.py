"""Data application view."""

import numpy as np
import pyvista as pv
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
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

from rmc_discord.gui.views.base_view import BaseView, scalar_bar, set_icon
from rmc_discord.gui.views.plots import CMAPS, LIMITS, PlotWidget, metric_mesh, norm


def _line(text, low=-1e6, high=1e6, decimals=5):
    line = QLineEdit(text)
    line.setValidator(QDoubleValidator(low, high, decimals, notation=QDoubleValidator.StandardNotation))
    return line


class DataView(BaseView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tab_widget = QTabWidget(self)
        self._init_load_tab()
        self._init_slice_tab()
        self.add_tools(self.tab_widget)

    def _init_load_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        files_box = QGroupBox("Reduced data (MDHistoWorkspace NeXus)")
        grid = QGridLayout(files_box)
        self.file_lines, self.browse_buttons = {}, {}
        for i, (key, name) in enumerate([("data", "Data:"), ("norm", "Normalization:"), ("subtract", "Subtract:")]):
            line = QLineEdit()
            line.setPlaceholderText({"data": "intensity, or counts with a normalization",
                                     "norm": "optional (data are then counts)",
                                     "subtract": "optional, e.g. a paramagnetic or high-T volume"}[key])
            button = QPushButton("Browse", self)
            set_icon(button, "folder-open")
            self.file_lines[key], self.browse_buttons[key] = line, button
            grid.addWidget(QLabel(name), i, 0)
            grid.addWidget(line, i, 1)
            grid.addWidget(button, i, 2)
        self.load_button = QPushButton("Load NXS", self)
        set_icon(self.load_button, "file-import")
        grid.addWidget(self.load_button, 3, 2)
        self.data_info = QLabel("")
        self.data_info.setWordWrap(True)
        grid.addWidget(self.data_info, 3, 0, 1, 2)
        layout.addWidget(files_box)

        sample_box = QGroupBox("Sampling at the supercell's commensurate points")
        grid = QGridLayout(sample_box)
        self.q_min_line = _line("0.0", 0, 100)
        self.q_max_line = _line("100.0", 0, 100)
        grid.addWidget(QLabel("|Q| min:"), 0, 0)
        grid.addWidget(self.q_min_line, 0, 1)
        grid.addWidget(QLabel("|Q| max:"), 0, 2)
        grid.addWidget(self.q_max_line, 0, 3)
        grid.addWidget(QLabel("Å⁻¹"), 0, 4)
        self.bragg_lines = [_line(v, 0, 10) for v in ("0.15", "0.15", "0.15")]
        grid.addWidget(QLabel("Bragg radius h, k, l:"), 1, 0)
        for i, line in enumerate(self.bragg_lines):
            grid.addWidget(line, 1, 1 + i)
        grid.addWidget(QLabel("rlu"), 1, 4)
        self.allowed_box = QCheckBox("Mask Allowed Reflections Only", self)
        self.rings_box = QCheckBox("Mask Powder Rings", self)
        grid.addWidget(self.allowed_box, 2, 0, 1, 2)
        grid.addWidget(self.rings_box, 2, 2, 1, 2)
        self.f_sys_line = _line("0.03", 0, 1)
        grid.addWidget(QLabel("Relative error floor:"), 3, 0)
        grid.addWidget(self.f_sys_line, 3, 1)
        self.sample_button = QPushButton("Sample", self)
        set_icon(self.sample_button, "braille")
        grid.addWidget(self.sample_button, 3, 3)
        self.sample_info = QLabel("")
        self.sample_info.setWordWrap(True)
        grid.addWidget(self.sample_info, 4, 0, 1, 5)
        layout.addWidget(sample_box)
        layout.addStretch(1)

        self.tab_widget.addTab(tab, "Load && Sample")

    def _init_slice_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        self.normal_combo = QComboBox(self)
        self.normal_combo.addItems(["Axis 1/2", "Axis 1/3", "Axis 2/3"])
        self.slice_line = _line("0.0")
        self.thickness_line = _line("0.1", 0.0001, 100)
        controls.addWidget(self.normal_combo)
        controls.addWidget(QLabel("Slice:"))
        controls.addWidget(self.slice_line)
        controls.addWidget(QLabel("Thickness:"))
        controls.addWidget(self.thickness_line)
        layout.addLayout(controls)

        draw = QHBoxLayout()
        self.scale_combo = QComboBox(self)
        self.scale_combo.addItems(["Linear", "Log"])
        self.clim_combo = QComboBox(self)
        self.clim_combo.addItems(LIMITS)
        self.clim_combo.setCurrentIndex(2)
        self.cbar_combo = QComboBox(self)
        self.cbar_combo.addItems(list(CMAPS))
        self.points_box = QCheckBox("Show Sampled Points", self)
        for w in (self.scale_combo, self.clim_combo, self.cbar_combo, self.points_box):
            draw.addWidget(w)
        layout.addLayout(draw)

        self.slice_plot = PlotWidget(self)
        layout.addWidget(self.slice_plot, stretch=1)
        self.tab_widget.addTab(tab, "Slice")

    # ------------------------------------------------------------------

    def connect_browse(self, f):
        for key, button in self.browse_buttons.items():
            button.clicked.connect(lambda _=False, k=key: f(k))

    def connect_load(self, f):
        self.load_button.clicked.connect(f)

    def connect_sample(self, f):
        self.sample_button.clicked.connect(f)

    def connect_slice(self, f):
        for combo in (self.normal_combo, self.scale_combo, self.clim_combo, self.cbar_combo):
            combo.currentIndexChanged.connect(f)
        self.slice_line.editingFinished.connect(f)
        self.thickness_line.editingFinished.connect(f)
        self.points_box.clicked.connect(f)

    def nxs_file_dialog(self):
        return self.open_file_dialog("Load NXS", "NeXus files (*.nxs *.h5)")

    def set_file(self, key, filename):
        self.file_lines[key].setText(filename)

    def get_files(self):
        return tuple(self.file_lines[k].text().strip() or None for k in ("data", "norm", "subtract"))

    def set_data_info(self, text):
        self.data_info.setText(text)

    def set_sample_info(self, text):
        self.sample_info.setText(text)

    def get_sampling(self):
        lines = [self.q_min_line, self.q_max_line, *self.bragg_lines, self.f_sys_line]
        if not all(line.hasAcceptableInput() for line in lines):
            return None
        q_min, q_max = float(self.q_min_line.text()), float(self.q_max_line.text())
        bragg = tuple(float(line.text()) for line in self.bragg_lines)
        return (q_min, q_max, bragg, self.allowed_box.isChecked(), self.rings_box.isChecked(),
                float(self.f_sys_line.text()))

    def get_slice(self):
        if not (self.slice_line.hasAcceptableInput() and self.thickness_line.hasAcceptableInput()):
            return None
        normal = {"Axis 1/2": 2, "Axis 1/3": 1, "Axis 2/3": 0}[self.normal_combo.currentText()]
        return normal, float(self.slice_line.text()), float(self.thickness_line.text())

    def get_slice_style(self):
        return (self.scale_combo.currentText(), self.clim_combo.currentText(),
                CMAPS[self.cbar_combo.currentText()], self.points_box.isChecked())

    # ------------------------------------------------------------------

    def draw_slice(self, image, edges_a, edges_b, transform, labels, title, points=None):
        scale, limits, cmap, show_points = self.get_slice_style()
        fig = self.slice_plot.fig
        fig.clear()
        ax = fig.add_subplot(111)
        skew, aspect = transform
        mesh, T = metric_mesh(ax, image, edges_a, edges_b, skew, aspect, cmap=cmap,
                              norm=norm(image, limits, scale), rasterized=True)
        if show_points and points is not None and len(points):
            step = max(1, len(points) // 50000)
            ax.scatter(points[::step, 0], points[::step, 1], s=0.5, c="k", alpha=0.3, transform=T + ax.transData)
        fig.colorbar(mesh, ax=ax, shrink=0.8)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title)
        self.slice_plot.draw()

    def draw_cloud(self, Q, values):
        self.clear_scene()
        cloud = pv.PolyData(np.ascontiguousarray(Q))
        cloud["log10 I"] = values
        scale, limits, cmap, _ = self.get_slice_style()
        self.plotter.add_mesh(cloud, scalars="log10 I", cmap=cmap, point_size=2, render_points_as_spheres=False,
                              clim=tuple(np.percentile(values, [2, 98])), scalar_bar_args=scalar_bar("log10 I"))
        self.reset_scene()
