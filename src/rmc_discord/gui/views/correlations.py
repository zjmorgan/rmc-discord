"""Correlations application view."""

import numpy as np
import pyvista as pv
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
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
from rmc_discord.gui.views.base_view import BaseView, set_icon
from rmc_discord.gui.views.plots import PlotWidget


def _line(text, low=0, high=1e4):
    line = QLineEdit(text)
    line.setValidator(QDoubleValidator(low, high, 4, notation=QDoubleValidator.StandardNotation))
    return line


class CorrelationsView(BaseView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recip_box.setChecked(False)
        self.tab_widget = QTabWidget(self)
        self._init_compute_tab()
        self._init_shells_tab()
        self.add_tools(self.tab_widget)

    def _init_compute_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Correlations")
        grid = QGridLayout(box)
        self.quantity_combo = QComboBox(self)
        self.symmetrize_box = QCheckBox("Symmetrize (space group)", self)
        self.symmetrize_box.setChecked(True)
        self.normalize_box = QCheckBox("Normalize (coefficients)", self)
        self.normalize_box.setChecked(True)
        grid.addWidget(QLabel("Quantity:"), 0, 0)
        grid.addWidget(self.quantity_combo, 0, 1)
        grid.addWidget(self.symmetrize_box, 0, 2)
        grid.addWidget(self.normalize_box, 0, 3)
        layout.addWidget(box)

        box = QGroupBox("Configurations (averaged; spread gives the error bars)")
        v = QVBoxLayout(box)
        self.current_box = QCheckBox("Current configuration", self)
        self.current_box.setChecked(True)
        v.addWidget(self.current_box)
        self.state_list = QListWidget(self)
        v.addWidget(self.state_list)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add Configurations", self)
        set_icon(self.add_button, "plus")
        self.remove_button = QPushButton("Remove", self)
        set_icon(self.remove_button, "trash")
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        v.addLayout(buttons)
        layout.addWidget(box, stretch=1)

        run = QHBoxLayout()
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.compute_button = QPushButton("Compute", self)
        set_icon(self.compute_button, "calculator")
        run.addWidget(self.info, stretch=1)
        run.addWidget(self.compute_button)
        layout.addLayout(run)

        box = QGroupBox("Export")
        h = QHBoxLayout(box)
        self.save_button = QPushButton("Save Table (HDF5)", self)
        set_icon(self.save_button, "floppy-disk")
        self.csv_button = QPushButton("Write Shells (CSV)", self)
        set_icon(self.csv_button, "file-csv")
        h.addWidget(self.save_button)
        h.addWidget(self.csv_button)
        h.addStretch(1)
        layout.addWidget(box)
        self.tab_widget.addTab(tab, "Compute")

    def _init_shells_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.pair_combo = QComboBox(self)
        self.pair_combo.addItem("All pairs")
        self.r_min_line = _line("0")
        self.r_max_line = _line("20")
        self.marker_spin = _line("1.0", 0.01, 100)
        controls.addWidget(QLabel("Pair:"))
        controls.addWidget(self.pair_combo, stretch=1)
        controls.addWidget(QLabel("r min:"))
        controls.addWidget(self.r_min_line)
        controls.addWidget(QLabel("r max (Å):"))
        controls.addWidget(self.r_max_line)
        controls.addWidget(QLabel("3D size:"))
        controls.addWidget(self.marker_spin)
        layout.addLayout(controls)

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

    # ------------------------------------------------------------------

    def connect_compute(self, f):
        self.compute_button.clicked.connect(f)

    def connect_add(self, f):
        self.add_button.clicked.connect(f)

    def connect_remove(self, f):
        self.remove_button.clicked.connect(f)

    def connect_filter(self, f):
        self.pair_combo.currentIndexChanged.connect(f)
        for line in (self.r_min_line, self.r_max_line, self.marker_spin):
            line.editingFinished.connect(f)

    def connect_save(self, f):
        self.save_button.clicked.connect(f)

    def connect_csv(self, f):
        self.csv_button.clicked.connect(f)

    def states_dialog(self):
        from qtpy.QtWidgets import QFileDialog

        files, _ = QFileDialog.getOpenFileNames(self, "Add configurations", "", "NumPy archives (*.npz)",
                                                options=QFileDialog.DontUseNativeDialog)
        return files

    def set_quantities(self, quantities):
        self.quantity_combo.clear()
        for q in quantities:
            self.quantity_combo.addItem(NAMES[q], q)

    def get_compute(self):
        return (self.quantity_combo.currentData(), self.symmetrize_box.isChecked(),
                self.normalize_box.isChecked(), self.current_box.isChecked())

    def set_states(self, names):
        self.state_list.clear()
        self.state_list.addItems(names)

    def selected_state(self):
        return self.state_list.currentRow()

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
        unit = {"occupancy": "⟨δδ⟩", "displacement": "⟨u·u⟩ (Å²)", "spin": "⟨S·S⟩"}.get(quantity, "")
        ax.set_ylabel("correlation coefficient" if normalized and quantity != "spin" else unit)
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

    def draw_map(self, r, value, size):
        self.clear_scene()
        if r is not None and len(r):
            m = np.max(np.abs(value)) or 1.0
            add_spheres(self.plotter, r, 0.5 * size * np.sqrt(np.abs(value) / m), values=value, cmap="bwr",
                        clim=(-m, m), scalar_bar_args={"title": "correlation"})
            self.plotter.add_mesh(pv.Sphere(radius=0.15 * size), color="black")
        self.reset_scene()
