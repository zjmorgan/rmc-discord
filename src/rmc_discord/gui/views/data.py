"""Data application view."""

import numpy as np
import pyvista as pv
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QSpinBox,
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

from rmc_discord.gui.views.base_view import BaseView, row, scalar_bar, set_icon, short
from rmc_discord.gui.views.plots import CMAPS, LIMITS, PlotWidget, color_limits, metric_mesh, norm




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
        grid = QGridLayout(files_box)  # a true table: label, path, browse
        self.file_lines, self.browse_buttons = {}, {}
        files = [("data", "Data:"), ("norm", "Normalization:"), ("subtract", "Background:"),
                 ("subtract_norm", "Background norm.:")]
        for i, (key, name) in enumerate(files):
            line = QLineEdit()
            line.setPlaceholderText({"data": "intensity, or counts with a normalization",
                                     "norm": "optional (data are then counts)",
                                     "subtract": "optional: subtracted, e.g. empty can or a high-T volume",
                                     "subtract_norm": "optional: normalization of the background counts"}[key])
            button = QPushButton("Browse", self)
            set_icon(button, "folder-open")
            self.file_lines[key], self.browse_buttons[key] = line, button
            grid.addWidget(QLabel(name), i, 0)
            grid.addWidget(line, i, 1)
            grid.addWidget(button, i, 2)
        self.load_button = QPushButton("Load NXS", self)
        set_icon(self.load_button, "file-import")
        self.data_info = QLabel("")
        self.data_info.setWordWrap(True)
        self.background_combo = QComboBox(self)
        self.background_combo.addItems(["Auto", "Normalized: data/norm − bkg", "Counts: (data − bkg)/norm"])
        self.background_combo.setToolTip("How a background without its own normalization is subtracted. Auto "
                                         "compares its scale with the data counts and the normalized data. With a "
                                         "background normalization, data/norm − bkg/bkg_norm is used.")
        grid.addLayout(row(QLabel("Background is:"), self.background_combo), 4, 1)
        grid.addWidget(self.data_info, 5, 0, 1, 2)
        grid.addWidget(self.load_button, 5, 2)
        layout.addWidget(files_box)

        sample_box = QGroupBox("Observations (measured bins; equivalent bins merged)")
        rows = QVBoxLayout(sample_box)
        self.q_min_line = short(_line("0.0", 0, 100))
        self.q_max_line = short(_line("100.0", 0, 100))
        self.bragg_line = short(_line("0.15", 0, 10))
        self.rings_box = QCheckBox("Mask powder rings", self)
        rows.addLayout(row(QLabel("|Q| from"), self.q_min_line, QLabel("to"), self.q_max_line, QLabel("Å⁻¹"),
                            QLabel("   Bragg radius:"), self.bragg_line, QLabel("Å⁻¹"), self.rings_box))
        self.merge_combo = QComboBox(self)
        self.merge_combo.addItems(["Mean of Equivalents", "One per Orbit (symmetrized data)", "None"])
        self.extra_line = QLineEdit()
        self.extra_line.setPlaceholderText("extra Laue operators, e.g. y, x, z")
        self.extra_line.setToolTip("Data symmetry beyond the structure's (e.g. from merohedral twinning).")
        rows.addLayout(row(QLabel("Merge:"), self.merge_combo, QLabel("Extra symmetry:"), self.extra_line,
                            stretch=False))
        self.f_sys_line = short(_line("0.03", 0, 1))
        self.held_line = short(_line("0.1", 0, 0.9))
        self.sample_button = QPushButton("Sample", self)
        set_icon(self.sample_button, "braille")
        rows.addLayout(row(QLabel("Relative error floor:"), self.f_sys_line, QLabel("   Held out:"),
                            self.held_line, end=self.sample_button))
        self.sample_info = QLabel("")
        self.sample_info.setWordWrap(True)
        rows.addWidget(self.sample_info)
        layout.addWidget(sample_box)

        view_box = QGroupBox("3-D view")
        rows = QVBoxLayout(view_box)
        self.view_combo = QComboBox(self)
        self.view_combo.addItems(["Data planes", "Observations", "Observations and masks"])
        self.view_combo.setToolTip("Planes of the loaded volume (positions on the Slice tab), or the sampled "
                                   "observations (the merged, symmetry-reduced bins) with the masked bins.")
        self.color_combo = QComboBox(self)
        self.color_combo.addItems(["Intensity", "Held out"])
        rows.addLayout(row(QLabel("Show:"), self.view_combo, QLabel("Color:"), self.color_combo))
        self.point_spin = QSpinBox(self)
        self.point_spin.setRange(1, 20)
        self.point_spin.setValue(3)
        self.max_points_spin = QSpinBox(self)
        self.max_points_spin.setRange(1_000, 5_000_000)
        self.max_points_spin.setSingleStep(50_000)
        self.max_points_spin.setValue(300_000)
        self.max_points_spin.setGroupSeparatorShown(True)
        rows.addLayout(row(QLabel("Point size:"), self.point_spin, QLabel("Up to"), self.max_points_spin,
                            QLabel("points per layer")))
        layout.addWidget(view_box)
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
        self.points_box.setToolTip("Mark the commensurate points kept for the refinement (black dots).")
        self.masked_box = QCheckBox("Show Masked Points", self)
        self.masked_box.setToolTip("Mark commensurate points removed by the Bragg mask (red ×) and the "
                                   "powder-ring mask (orange +); sample the data first.")
        for w in (self.scale_combo, self.clim_combo, self.cbar_combo, self.points_box, self.masked_box):
            draw.addWidget(w)
        layout.addLayout(draw)

        planes = QHBoxLayout()
        planes.addWidget(QLabel("3-D planes at:"))
        self.plane_lines = [_line("0.0") for _ in range(3)]
        self.plane_labels = []
        for k, line in enumerate(self.plane_lines):
            label = QLabel(f"u{k + 1}")
            self.plane_labels.append(label)
            planes.addWidget(label)
            planes.addWidget(line)
        layout.addLayout(planes)

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

    def connect_view(self, f):
        self.view_combo.currentIndexChanged.connect(f)
        self.color_combo.currentIndexChanged.connect(f)
        self.point_spin.editingFinished.connect(f)
        self.max_points_spin.editingFinished.connect(f)

    def get_view(self):
        """``(mode, color, point size, max points)``, mode in {"planes", "observations", "masks"}."""
        mode = ("planes", "observations", "masks")[self.view_combo.currentIndex()]
        color = ("intensity", "held out")[self.color_combo.currentIndex()]
        return mode, color, self.point_spin.value(), self.max_points_spin.value()

    def connect_slice(self, f):
        for combo in (self.normal_combo, self.scale_combo, self.clim_combo, self.cbar_combo):
            combo.currentIndexChanged.connect(f)
        self.slice_line.editingFinished.connect(f)
        self.thickness_line.editingFinished.connect(f)
        self.points_box.clicked.connect(f)
        self.masked_box.clicked.connect(f)
        for line in self.plane_lines:
            line.editingFinished.connect(f)

    def set_plane_names(self, names):
        for label, name in zip(self.plane_labels, names):
            label.setText(name)

    def get_planes(self):
        if not all(line.hasAcceptableInput() for line in self.plane_lines):
            return None
        return [float(line.text()) for line in self.plane_lines]

    def nxs_file_dialog(self):
        return self.open_file_dialog("Load NXS", "NeXus files (*.nxs *.h5)")

    def set_file(self, key, filename):
        self.file_lines[key].setText(filename)

    def get_files(self):
        return tuple(self.file_lines[k].text().strip() or None for k in ("data", "norm", "subtract", "subtract_norm"))

    def get_background_mode(self):
        return ("auto", "after", "before")[self.background_combo.currentIndex()]

    def set_data_info(self, text):
        self.data_info.setText(text)

    def set_sample_info(self, text):
        self.sample_info.setText(text)

    def get_sampling(self):
        """|Q| range, Bragg radius, rings, error floor, merge mode, extra operators, held-out fraction."""
        lines = [self.q_min_line, self.q_max_line, self.bragg_line, self.f_sys_line, self.held_line]
        if not all(line.hasAcceptableInput() for line in lines):
            return None
        merge = {0: "mean", 1: "first", 2: None}[self.merge_combo.currentIndex()]
        extra = tuple(op.strip() for op in self.extra_line.text().split(";") if op.strip())
        return (float(self.q_min_line.text()), float(self.q_max_line.text()), float(self.bragg_line.text()),
                self.rings_box.isChecked(), float(self.f_sys_line.text()), merge, extra,
                float(self.held_line.text()))

    def get_slice(self):
        if not (self.slice_line.hasAcceptableInput() and self.thickness_line.hasAcceptableInput()):
            return None
        normal = {"Axis 1/2": 2, "Axis 1/3": 1, "Axis 2/3": 0}[self.normal_combo.currentText()]
        return normal, float(self.slice_line.text()), float(self.thickness_line.text())

    def show_masked(self):
        return self.masked_box.isChecked()

    def get_slice_style(self):
        return (self.scale_combo.currentText(), self.clim_combo.currentText(),
                CMAPS[self.cbar_combo.currentText()], self.points_box.isChecked())

    # ------------------------------------------------------------------

    def draw_slice(self, image, edges_a, edges_b, transform, labels, title, points=None, masked=None):
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
        if masked:
            for key, marker, color in (("bragg", "x", "red"), ("rings", "+", "orange")):
                p = masked.get(key)
                if p is not None and len(p):
                    ax.scatter(p[:, 0], p[:, 1], s=8, marker=marker, c=color, lw=0.6, transform=T + ax.transData,
                               label=f"{key} mask ({len(p):,})")
            if any(len(p) for p in masked.values() if p is not None):
                ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(mesh, ax=ax, shrink=0.8)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title)
        self.slice_plot.draw()

    def draw_planes(self, planes):
        """Three planes of the volume in Cartesian reciprocal space; NaN bins are transparent."""
        scale, limits, cmap, _ = self.get_slice_style()
        values = []
        for _, image in planes:
            image = np.array(image, dtype=float)
            if scale == "Log":
                with np.errstate(divide="ignore", invalid="ignore"):
                    image = np.where(image > 0, np.log10(image), np.nan)
            values.append(image)
        both = np.concatenate([v.ravel() for v in values])
        clim = color_limits(both, limits)
        title = "log10 I" if scale == "Log" else "I"
        self.clear_scene()
        for (points, _), image in zip(planes, values):
            grid = pv.StructuredGrid(*(points[..., k][..., None] for k in range(3)))
            grid.cell_data[title] = image.ravel(order="F")
            self.plotter.add_mesh(grid, scalars=title, cmap=cmap, clim=clim, nan_opacity=0.0, lighting=False,
                                  scalar_bar_args=scalar_bar(title))
        self.reset_scene()

    def draw_cloud(self, layers, point_size=3):
        """Observation layers from :meth:`DataModel.cloud`: scalar-colored points or solid-colored masks."""
        self.clear_scene()
        _, limits, cmap, _ = self.get_slice_style()
        for layer in layers:
            cloud = pv.PolyData(np.ascontiguousarray(layer["points"], dtype=float))
            if "scalars" in layer:
                values = layer["scalars"]
                cloud[layer["name"]] = values
                clim = layer.get("clim") or color_limits(values, limits)
                self.plotter.add_mesh(cloud, scalars=layer["name"], cmap=layer.get("cmap", cmap), clim=clim,
                                      point_size=point_size, render_points_as_spheres=False, lighting=False,
                                      scalar_bar_args=scalar_bar(layer["name"]))
            else:
                self.plotter.add_mesh(cloud, color=layer["color"], point_size=point_size,
                                      render_points_as_spheres=False, lighting=False, label=layer["name"])
        self.reset_scene()
