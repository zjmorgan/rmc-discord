"""
Base view: 3D visualization with camera controls, information tabs, a
console, and a status bar (layout of NeuXtalViz's ``NeuXtalVizWidget``).
Applications add their tool tabs to the right with :meth:`add_tools`.
"""

import os
import pathlib

import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from pyvistaqt import QtInteractor
from qtpy.QtCore import QSettings, Qt, Signal
from qtpy.QtGui import QColor, QDoubleValidator, QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rmc_discord.gui import theme
from rmc_discord.gui.utilities import ThreadPool, Worker


icon, set_icon = theme.icon, theme.set_icon


def button(text, icon_name=None, tooltip=None, parent=None):
    b = QPushButton(text, parent)
    if icon_name:
        set_icon(b, icon_name)
    if tooltip:
        b.setToolTip(tooltip)
    return b


def decorate_combo(combo):
    """Item icons and a width that fits the longest item (as in NeuXtalViz)."""

    digit = combo.count() > 0 and all(combo.itemText(i).isdigit() for i in range(combo.count()))
    names = list(combo.property("fa_items") or [])
    for i in range(combo.count()):
        if i >= len(names):
            names.append("hashtag" if digit else "minus")
        combo.setItemIcon(i, icon(names[i]))
    combo.setProperty("fa_items", names)
    combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
    fm = combo.fontMetrics()
    width = max((fm.horizontalAdvance(combo.itemText(i)) for i in range(combo.count())), default=0)
    if width:
        combo.setMinimumWidth(width + combo.iconSize().width() + 48)


class BaseView(QWidget):
    cam_ready = Signal()

    _SETTINGS = ("rmc-discord", "rmc-discord")

    def __init__(self, parent=None):
        super().__init__(parent)

        self.save_button = button("Save Screenshot", "floppy-disk", "Save a screenshot of the current 3D view.", self)
        self.reset_button = button("Reset View", "house", "Reset the view to the default orientation.", self)
        self.reset_button.clicked.connect(self.reset_view)
        self.camera_button = button("Reset Camera", "camera", "Reset the camera position.", self)
        self.camera_button.clicked.connect(self.reset_camera)

        self.theme_combo = QComboBox(self)
        self.theme_combo.addItems(theme.PLOT_THEMES)
        self.theme_combo.setCurrentText(theme.current)
        self.theme_combo.setToolTip('Select the 3D view theme. "dark" also switches the application to dark mode.')
        decorate_combo(self.theme_combo)
        self.theme_combo.currentTextChanged.connect(theme.set_theme)
        theme.notifier.changed.connect(self.update_theme)

        self.recip_box = QCheckBox("Toggle Reciprocal Lattice", self)
        self.recip_box.setChecked(True)
        self.recip_box.setToolTip("Show reciprocal (a*, b*, c*) or real (a, b, c) axes.")

        self.axes_box = QCheckBox("Show Axes", self)
        self.axes_box.setChecked(True)
        self.axes_box.setToolTip("Show or hide the coordinate axes in the plot.")
        self.axes_box.clicked.connect(self.show_axes)

        self.proj_box = QCheckBox("Enable Parallel Projection", self)
        self.proj_box.setChecked(True)
        self.proj_box.setToolTip("Toggle parallel projection for the 3D view.")
        self.proj_box.clicked.connect(self.change_projection)

        self.cons_box = QCheckBox("Expand Console", self)
        self.cons_box.setChecked(True)
        self.cons_box.setToolTip("Show or hide console output.")
        self.cons_box.stateChanged.connect(self.toggle_console)

        self.frame = QFrame()
        self.plotter = QtInteractor(self.frame)

        layout = QHBoxLayout()
        vis_layout = QVBoxLayout()

        camera_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        for w in (self.save_button, self.reset_button, self.camera_button, self.theme_combo):
            left_layout.addWidget(w)
        for w in (self.recip_box, self.axes_box, self.proj_box, self.cons_box):
            right_layout.addWidget(w)

        camera_layout.addLayout(left_layout)
        camera_layout.addWidget(self._init_view_tab())
        camera_layout.addLayout(right_layout)

        vis_layout.addLayout(camera_layout)
        vis_layout.addWidget(self.plotter.interactor)
        vis_layout.setStretch(1, 1)
        vis_layout.addWidget(self._init_info_tab())

        self.console = QPlainTextEdit(readOnly=True)
        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(9)
        self.console.setFont(font)
        self.console.setMaximumBlockCount(5000)
        self.console.setMaximumHeight(140)
        vis_layout.addWidget(self.console)

        self.stop_process_button = button("Stop Process", "circle-stop", "Stop the current process.", self)
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready!")
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.addPermanentWidget(self.stop_process_button)
        vis_layout.addWidget(self.status_bar)

        layout.addLayout(vis_layout, stretch=1)
        self.setLayout(layout)

        self.camera_position = None
        self.T = None
        self.threadpool = ThreadPool()
        self.plotter.enable_parallel_projection()
        self.plotter.set_background(pv.global_theme.background)

    # ------------------------------------------------------------------
    # layout

    def add_tools(self, tabs, stretch=1):
        """Add the application's tool tabs to the right of the view."""
        for combo in tabs.findChildren(QComboBox):
            decorate_combo(combo)
        self.layout().addWidget(tabs, stretch=stretch)

    def _init_view_tab(self):
        view_tab = QTabWidget()

        self.view_combo = QComboBox(self)
        self.view_combo.addItems(["[hkl]", "[uvw]"])
        self.view_combo.setToolTip("Select axis notation for the view direction.")
        self.view_combo.currentIndexChanged.connect(self.update_labels)

        self.viewup_combo = QComboBox(self)
        self.viewup_combo.addItems(["[hkl]", "[uvw]"])
        self.viewup_combo.setToolTip("Select axis notation for the up direction.")
        self.viewup_combo.currentIndexChanged.connect(self.update_labels)
        decorate_combo(self.view_combo)
        decorate_combo(self.viewup_combo)

        validator = QDoubleValidator(-100, 100, 5, notation=QDoubleValidator.StandardNotation)

        ordinal = ("First", "Second", "Third")
        self.axis_lines = [QLineEdit() for _ in range(3)]
        self.axisup_lines = [QLineEdit() for _ in range(3)]
        for i, t in enumerate("hkl"):
            for lines, what in ((self.axis_lines, "view"), (self.axisup_lines, "up")):
                lines[i].setValidator(validator)
                lines[i].setPlaceholderText(t)
                lines[i].setToolTip(f"{ordinal[i]} component of the {what} direction (e.g., {t} or {'uvw'[i]}).")
        self.axis_labels = [QLabel(t, self) for t in "hkl"]
        self.axisup_labels = [QLabel(t, self) for t in "hkl"]

        self.manual_button = button("View Axis", "right-long",
                                    "Set the view direction using the specified axis components.", self)
        self.manualup_button = button("View Up Axis", "up-long",
                                      "Set the up direction using the specified axis components.", self)

        def direction(text, view):
            b = button(text, "right-long", f"View along the {text} direction.", self)
            b.clicked.connect(view)
            return b

        self.px_button = direction("+Qx", self.view_yz)
        self.py_button = direction("+Qy", self.view_zx)
        self.pz_button = direction("+Qz", self.view_xy)
        self.mx_button = direction("-Qx", self.view_zy)
        self.my_button = direction("-Qy", self.view_xz)
        self.mz_button = direction("-Qz", self.view_yx)

        self.a_star_button = button("a*", None, "Align view along the a* reciprocal axis.", self)
        self.b_star_button = button("b*", None, "Align view along the b* reciprocal axis.", self)
        self.c_star_button = button("c*", None, "Align view along the c* reciprocal axis.", self)
        self.a_button = button("a", None, "Align view along the a real axis.", self)
        self.b_button = button("b", None, "Align view along the b real axis.", self)
        self.c_button = button("c", None, "Align view along the c real axis.", self)

        directions_layout = QGridLayout()
        for i, b in enumerate([self.px_button, self.py_button, self.pz_button,
                               self.a_star_button, self.b_star_button, self.c_star_button]):
            directions_layout.addWidget(b, 0, i)
        for i, b in enumerate([self.mx_button, self.my_button, self.mz_button,
                               self.a_button, self.b_button, self.c_button]):
            directions_layout.addWidget(b, 1, i)

        manual_layout = QGridLayout()
        for i in range(3):
            manual_layout.addWidget(self.axis_labels[i], 0, i, Qt.AlignCenter)
            manual_layout.addWidget(self.axis_lines[i], 1, i)
            manual_layout.addWidget(self.axisup_labels[i], 0, 4 + i, Qt.AlignCenter)
            manual_layout.addWidget(self.axisup_lines[i], 1, 4 + i)
        manual_layout.addWidget(self.view_combo, 0, 3)
        manual_layout.addWidget(self.manual_button, 1, 3)
        manual_layout.addWidget(self.viewup_combo, 0, 7)
        manual_layout.addWidget(self.manualup_button, 1, 7)

        self.rotate_step_line = QLineEdit("5.0", self)
        self.rotate_step_line.setValidator(QDoubleValidator(-360.0, 360.0, 2,
                                                            notation=QDoubleValidator.StandardNotation))
        self.rotate_step_line.setToolTip("Rotation step in degrees.")
        self.rotate_ccw_button = button("Roll CCW", "rotate-left",
                                        "Roll the view counter-clockwise about the viewing axis.", self)
        self.rotate_cw_button = button("Roll CW", "rotate-right", "Roll the view clockwise about the viewing axis.",
                                       self)
        self.elev_up_button = button("Elevate Up", "arrow-up", "Tilt the camera up by the step angle.", self)
        self.elev_down_button = button("Elevate Down", "arrow-down", "Tilt the camera down by the step angle.", self)
        self.az_left_button = button("Azimuth Left", "arrow-left", "Rotate the camera left by the step angle.", self)
        self.az_right_button = button("Azimuth Right", "arrow-right", "Rotate the camera right by the step angle.",
                                      self)
        self.camera_pos_line = QLineEdit(self)
        self.camera_pos_line.setReadOnly(True)
        self.camera_pos_line.setToolTip("Current camera roll, elevation, and azimuth in degrees.")

        rotate_layout = QGridLayout()
        rotate_layout.addWidget(self.rotate_ccw_button, 0, 0)
        rotate_layout.addWidget(self.rotate_cw_button, 1, 0)
        rotate_layout.addWidget(self.elev_up_button, 0, 1)
        rotate_layout.addWidget(self.elev_down_button, 1, 1)
        rotate_layout.addWidget(self.az_left_button, 0, 2)
        rotate_layout.addWidget(self.az_right_button, 1, 2)
        rotate_layout.addWidget(QLabel("Step [°]", self), 0, 3, Qt.AlignCenter)
        rotate_layout.addWidget(self.rotate_step_line, 1, 3)
        rotate_layout.addWidget(QLabel("Camera [°]", self), 0, 4)
        rotate_layout.addWidget(self.camera_pos_line, 1, 4)

        for name, grid in (("Direction View", directions_layout), ("Manual View", manual_layout),
                           ("Rotate View", rotate_layout)):
            outer = QVBoxLayout()
            outer.addLayout(grid)
            outer.addStretch(1)
            tab = QWidget()
            tab.setLayout(outer)
            view_tab.addTab(tab, name)

        self._init_axis_icons()
        return view_tab

    def _init_axis_icons(self):
        """Red, green, blue arrows on the a, b, c (and a*, b*, c*) buttons."""

        paraview = theme.current == "paraview"
        colors = [QColor("red"), QColor("yellow" if paraview else "green"), QColor("green" if paraview else "blue")]
        for color, real, recip in zip(colors, (self.a_button, self.b_button, self.c_button),
                                      (self.a_star_button, self.b_star_button, self.c_star_button)):
            real.setIcon(icon("right-long", color=color))
            recip.setIcon(icon("right-long", color=color))

    def _init_info_tab(self):
        info_tab = QTabWidget()

        names = ["a:", "b:", "c:", "α:", "β:", "γ:"]
        self.lattice_lines = [QLineEdit() for _ in names]
        lattice_layout = QGridLayout()
        for i, (name, line) in enumerate(zip(names, self.lattice_lines)):
            line.setReadOnly(True)
            lattice_layout.addWidget(QLabel(name, self), i // 3, 2 * (i % 3))
            lattice_layout.addWidget(line, i // 3, 2 * (i % 3) + 1)
        lattice_layout.addWidget(QLabel("Å"), 0, 6)
        lattice_layout.addWidget(QLabel("°"), 1, 6)

        names = ["N₁:", "N₂:", "N₃:", "Group:", "Atoms:", "Sites:"]
        self.supercell_lines = [QLineEdit() for _ in names]
        supercell_layout = QGridLayout()
        for i, (name, line) in enumerate(zip(names, self.supercell_lines)):
            line.setReadOnly(True)
            supercell_layout.addWidget(QLabel(name, self), i // 3, 2 * (i % 3))
            supercell_layout.addWidget(line, i // 3, 2 * (i % 3) + 1)

        lattice_tab = QWidget()
        lattice_tab.setLayout(lattice_layout)
        supercell_tab = QWidget()
        supercell_tab.setLayout(supercell_layout)
        info_tab.addTab(lattice_tab, "Lattice Parameters")
        info_tab.addTab(supercell_tab, "Supercell")
        return info_tab

    # ------------------------------------------------------------------
    # theme

    def update_theme(self, name=None):
        """Follow the application theme (set from any view's theme box)."""

        if self.theme_combo.currentText() != theme.current:
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentText(theme.current)
            self.theme_combo.blockSignals(False)
        self.plotter.set_background(pv.global_theme.background)
        self.show_axes()
        self._init_axis_icons()
        self.plotter.render()
        self._refresh_mpl_canvases()

    def _refresh_mpl_canvases(self):
        """Re-apply the matplotlib style to figures that already exist."""

        rc = plt.rcParams
        for canvas in self.findChildren(FigureCanvasQTAgg):
            figure = canvas.figure
            figure.set_facecolor(rc["figure.facecolor"])
            for ax in figure.axes:
                ax.set_facecolor(rc["axes.facecolor"])
                for spine in ax.spines.values():
                    spine.set_color(rc["axes.edgecolor"])
                ax.tick_params(colors=rc["xtick.color"], labelcolor=rc["axes.labelcolor"])
                ax.xaxis.label.set_color(rc["axes.labelcolor"])
                ax.yaxis.label.set_color(rc["axes.labelcolor"])
                ax.title.set_color(rc["text.color"])
                legend = ax.get_legend()
                if legend is not None:
                    legend.get_frame().set_facecolor(rc["axes.facecolor"])
                    legend.get_frame().set_edgecolor(rc["axes.edgecolor"])
                    for text in legend.get_texts():
                        text.set_color(rc["text.color"])
            canvas.draw_idle()

    # ------------------------------------------------------------------
    # workers, console, and status

    def start_worker_pool(self, worker):
        worker.signals.output.connect(self.append_to_console)
        self.threadpool.start_worker_pool(worker)

    def worker(self, task):
        return Worker(task)

    def append_to_console(self, text):
        text = text.rstrip("\n")
        if text:
            self.console.appendPlainText(text)

    def toggle_console(self, state):
        self.console.setVisible(bool(state))

    def set_info(self, status):
        self.status_bar.showMessage(status)

    def set_step(self, progress):
        if isinstance(progress, str):
            self.progress_bar.setFormat(progress)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setFormat("%p%")
            self.progress_bar.setValue(int(progress))

    def show_error(self, message, details=None):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Error")
        box.setText(message)
        if details:
            box.setDetailedText(details)
        box.exec_() if hasattr(box, "exec_") else box.exec()

    def set_lattice_parameters(self, a, b, c, alpha, beta, gamma):
        for line, v, fmt in zip(self.lattice_lines, (a, b, c, alpha, beta, gamma), ["{:.5f}"] * 3 + ["{:.3f}"] * 3):
            line.setText(fmt.format(v))

    def set_supercell_info(self, N, group, n_atoms, n_sites):
        values = [*(str(n) for n in N), str(group), str(n_atoms), str(n_sites)]
        for line, v in zip(self.supercell_lines, values):
            line.setText(v)

    # ------------------------------------------------------------------
    # connections

    def connect_stop_process(self, stop):
        self.stop_process_button.clicked.connect(stop)

    def connect_manual_axis(self, view_manual):
        self.manual_button.clicked.connect(view_manual)

    def connect_manual_up_axis(self, view_manual):
        self.manualup_button.clicked.connect(view_manual)

    def connect_reciprocal_axes(self, view_a_star, view_b_star, view_c_star):
        self.a_star_button.clicked.connect(view_a_star)
        self.b_star_button.clicked.connect(view_b_star)
        self.c_star_button.clicked.connect(view_c_star)

    def connect_real_axes(self, view_a, view_b, view_c):
        self.a_button.clicked.connect(view_a)
        self.b_button.clicked.connect(view_b)
        self.c_button.clicked.connect(view_c)

    def connect_rotate(self, ccw, cw):
        self.rotate_ccw_button.clicked.connect(ccw)
        self.rotate_cw_button.clicked.connect(cw)

    def connect_elev(self, up, down):
        self.elev_up_button.clicked.connect(up)
        self.elev_down_button.clicked.connect(down)

    def connect_az(self, left, right):
        self.az_left_button.clicked.connect(left)
        self.az_right_button.clicked.connect(right)

    def connect_camera_ready(self, calculate):
        self.cam_ready.connect(calculate)

    def connect_save_screenshot(self, save_screenshot):
        self.save_button.clicked.connect(save_screenshot)

    def connect_reciprocal_real_compass(self, change_lattice):
        self.recip_box.clicked.connect(change_lattice)

    # ------------------------------------------------------------------
    # camera

    def change_projection(self):
        if self.proj_box.isChecked():
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()

    def reset_view(self, negative=False):
        self.plotter.reset_camera()
        self.plotter.view_isometric(negative=negative)
        self.camera_position = self.plotter.camera_position
        self.cam_ready.emit()

    def reset_camera(self):
        self.plotter.reset_camera()
        self.cam_ready.emit()

    def clear_scene(self):
        self.plotter.clear_plane_widgets()
        self.plotter.clear_actors()
        if self.camera_position is not None:
            self.camera_position = self.plotter.camera_position

    def reset_scene(self):
        if self.camera_position is not None:
            self.plotter.camera_position = self.camera_position
        else:
            self.reset_view()
        self.cam_ready.emit()

    def get_camera_state(self):
        cam = self.plotter.camera
        return (np.array(cam.position, dtype=float), np.array(cam.focal_point, dtype=float),
                np.array(cam.up, dtype=float))

    def set_camera_state(self, position, focal_point, up):
        cam = self.plotter.camera
        cam.position = tuple(position)
        cam.focal_point = tuple(focal_point)
        cam.up = tuple(up)
        self.plotter.render()
        self.cam_ready.emit()

    def get_rotate_step(self):
        if self.rotate_step_line.hasAcceptableInput():
            return float(self.rotate_step_line.text())
        return 5.0

    def get_camera_roll_direction(self):
        return self.plotter.camera.roll, self.plotter.camera.direction

    def update_camera_display(self, roll, elevation, azimuth):
        self.camera_pos_line.setText("{:.1f},{:.1f},{:.1f}".format(roll, elevation, azimuth))

    # ------------------------------------------------------------------
    # files

    def _dialog_dir(self):
        return QSettings(*self._SETTINGS).value("file_dialog_last_dir", "")

    def _remember_dir(self, filename):
        if filename:
            path = os.path.dirname(filename)
            parts = pathlib.Path(path).parts
            if len(parts) > 4:
                path = str(pathlib.Path(*parts[:4]))
            QSettings(*self._SETTINGS).setValue("file_dialog_last_dir", path)
        return filename

    def save_screenshot(self, filename):
        self.plotter.screenshot(filename)

    def save_screenshot_file_dialog(self):
        return self.save_file_dialog("Save PNG file", "PNG files (*.png)")

    def open_file_dialog(self, title, filters, directory=""):
        filename, _ = QFileDialog.getOpenFileName(self, title, directory or self._dialog_dir(), filters,
                                                  options=QFileDialog.DontUseNativeDialog)
        return self._remember_dir(filename)

    def save_file_dialog(self, title, filters, directory=""):
        filename, _ = QFileDialog.getSaveFileName(self, title, directory or self._dialog_dir(), filters,
                                                  options=QFileDialog.DontUseNativeDialog)
        return self._remember_dir(filename)

    # ------------------------------------------------------------------
    # axes

    def set_transform(self, T):
        if T is not None:
            self.T = T
            self.show_axes()

    def reciprocal_lattice(self):
        return self.recip_box.isChecked()

    def axes_show(self):
        return self.axes_box.isChecked()

    def show_axes(self):
        if not self.axes_show():
            self.plotter.hide_axes()
        elif self.T is not None:
            t = pv._vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    t.SetElement(i, j, self.T[i, j])
            if self.reciprocal_lattice():
                actor = self.plotter.add_axes(xlabel="a*", ylabel="b*", zlabel="c*")
            else:
                actor = self.plotter.add_axes(xlabel="a", ylabel="b", zlabel="c")
            actor.SetUserMatrix(t)

    def view_vector(self, vecs):
        if len(vecs) == 2:
            self.plotter.view_vector(vecs[0], np.cross(vecs[0], vecs[1]))
        else:
            self.plotter.view_vector(vecs)
        self.cam_ready.emit()

    def view_up_vector(self, vec):
        self.plotter.set_viewup(vec)
        self.cam_ready.emit()

    def _view(self, vector, up):
        self.plotter.view_vector(vector, up)
        self.cam_ready.emit()

    def view_xy(self):
        self._view([0, 0, 1], [0, 1, 0])

    def view_yz(self):
        self._view([1, 0, 0], [0, 1, 0])

    def view_zx(self):
        self._view([0, 1, 0], [0, 0, 1])

    def view_yx(self):
        self._view([0, 0, -1], [0, 1, 0])

    def view_zy(self):
        self._view([-1, 0, 0], [0, 1, 0])

    def view_xz(self):
        self._view([0, -1, 0], [0, 0, 1])

    def update_labels(self):
        letters = "hkl" if self.view_combo.currentText() == "[hkl]" else "uvw"
        for label, line, t in zip(self.axis_labels, self.axis_lines, letters):
            label.setText(t)
            line.setPlaceholderText(t)
        letters = "hkl" if self.viewup_combo.currentText() == "[hkl]" else "uvw"
        for label, line, t in zip(self.axisup_labels, self.axisup_lines, letters):
            label.setText(t)
            line.setPlaceholderText(t)

    def _indices(self, combo, lines):
        if all(line.hasAcceptableInput() for line in lines):
            return combo.currentText(), np.array([float(line.text()) for line in lines])

    def get_manual_axis_indices(self):
        return self._indices(self.view_combo, self.axis_lines)

    def get_manual_axis_up_indices(self):
        return self._indices(self.viewup_combo, self.axisup_lines)
