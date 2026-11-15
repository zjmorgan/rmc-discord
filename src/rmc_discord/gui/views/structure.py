"""Structure application view."""

import numpy as np
import pyvista as pv
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rmc_discord.gui.models.display import add_spheres
from rmc_discord.gui.views.base_view import BaseView, set_icon

SITE_COLUMNS = ["Use", "Label", "Type", "Occupancy", "x", "y", "z", "Uiso (Å²)", "Mult."]
KIND_COLUMNS = ["Site", "Kind", "Type", "Fraction", "Count", "b (fm)", "μ (μB)", "Atoms"]


def _item(text, editable=False, align=Qt.AlignCenter):
    item = QTableWidgetItem(text)
    item.setTextAlignment(align)
    flags = item.flags()
    item.setFlags(flags | Qt.ItemIsEditable if editable else flags & ~Qt.ItemIsEditable)
    return item


def _table(columns):
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.Stretch)
    header.setMinimumSectionSize(60)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    return table


class StructureView(BaseView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recip_box.setChecked(False)

        self.tab_widget = QTabWidget(self)
        self._init_structure_tab()
        self._init_supercell_tab()
        self.add_tools(self.tab_widget)

    # ------------------------------------------------------------------

    def _init_structure_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        file_layout = QHBoxLayout()
        self.load_cif_button = QPushButton("Load CIF", self)
        set_icon(self.load_cif_button, "folder-open")
        self.cif_line = QLineEdit()
        self.cif_line.setReadOnly(True)
        file_layout.addWidget(self.load_cif_button)
        file_layout.addWidget(self.cif_line, stretch=1)
        layout.addLayout(file_layout)

        lattice_box = QGroupBox("Lattice (edit to use the data lattice)")
        grid = QGridLayout(lattice_box)
        validator = QDoubleValidator(0.0, 1000.0, 6, notation=QDoubleValidator.StandardNotation)
        self.cell_lines = []
        for i, (name, unit) in enumerate(zip(["a", "b", "c", "α", "β", "γ"], ["Å"] * 3 + ["°"] * 3)):
            line = QLineEdit()
            line.setValidator(validator)
            self.cell_lines.append(line)
            grid.addWidget(QLabel(f"{name}:"), i // 3, 3 * (i % 3))
            grid.addWidget(line, i // 3, 3 * (i % 3) + 1)
            grid.addWidget(QLabel(unit), i // 3, 3 * (i % 3) + 2)
        self.space_group_line = QLineEdit()
        self.space_group_line.setReadOnly(True)
        grid.addWidget(QLabel("Space group:"), 2, 0, 1, 2)
        grid.addWidget(self.space_group_line, 2, 2, 1, 7)
        layout.addWidget(lattice_box)

        sites_box = QGroupBox("Asymmetric unit (check the disordered sites; occupancies are editable)")
        sites_layout = QVBoxLayout(sites_box)
        self.site_table = _table(SITE_COLUMNS)
        sites_layout.addWidget(self.site_table)
        layout.addWidget(sites_box, stretch=1)

        display_layout = QHBoxLayout()
        self.display_combo = QComboBox(self)
        self.display_combo.addItems(["Unit Cell", "Supercell"])
        self.scale_spin = QDoubleSpinBox(self)
        self.scale_spin.setRange(0.1, 3.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setValue(1.0)
        self.inactive_box = QCheckBox("Show Other Sites", self)
        self.inactive_box.setChecked(True)
        display_layout.addWidget(QLabel("Display:"))
        display_layout.addWidget(self.display_combo)
        display_layout.addWidget(QLabel("Atom Scale:"))
        display_layout.addWidget(self.scale_spin)
        display_layout.addWidget(self.inactive_box)
        display_layout.addStretch(1)
        layout.addLayout(display_layout)

        self.tab_widget.addTab(tab, "Average Structure")

    def _init_supercell_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        disorder_box = QGroupBox("Disorder")
        form = QFormLayout(disorder_box)
        self.groups_line = QLineEdit()
        self.groups_line.setPlaceholderText("center: members; e.g. C1: O7; C2: O8")
        self.ions_line = QLineEdit()
        self.ions_line.setPlaceholderText("label: ion; e.g. Fe1: Fe3+; Mn1: Mn3+")
        self.moments_line = QLineEdit()
        self.moments_line.setPlaceholderText("ion or label: μB; e.g. Fe3+: 5.92; Mn3+: 4.90")
        form.addRow("Rigid groups:", self.groups_line)
        form.addRow("Magnetic ions:", self.ions_line)
        form.addRow("Moments:", self.moments_line)
        layout.addWidget(disorder_box)

        size_box = QGroupBox("Supercell")
        grid = QGridLayout(size_box)
        self.N_spins = []
        for i, name in enumerate(["N₁:", "N₂:", "N₃:"]):
            spin = QSpinBox(self)
            spin.setRange(1, 200)
            spin.setValue(8)
            self.N_spins.append(spin)
            grid.addWidget(QLabel(name), 0, 2 * i)
            grid.addWidget(spin, 0, 2 * i + 1)
        self.seed_spin = QSpinBox(self)
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(1)
        self.displace_box = QCheckBox("Initial Displacements from ADPs", self)
        self.displace_box.setChecked(True)
        self.build_button = QPushButton("Build Supercell", self)
        set_icon(self.build_button, "cubes")
        grid.addWidget(QLabel("Seed:"), 1, 0)
        grid.addWidget(self.seed_spin, 1, 1)
        grid.addWidget(self.displace_box, 1, 2, 1, 3)
        grid.addWidget(self.build_button, 1, 5)
        layout.addWidget(size_box)

        kinds_box = QGroupBox("Occupants of the active sites")
        kinds_layout = QVBoxLayout(kinds_box)
        self.kind_table = _table(KIND_COLUMNS)
        kinds_layout.addWidget(self.kind_table)
        layout.addWidget(kinds_box, stretch=1)

        self.tab_widget.addTab(tab, "Supercell")

    # ------------------------------------------------------------------
    # connections

    def connect_load_cif(self, f):
        self.load_cif_button.clicked.connect(f)

    def connect_build(self, f):
        self.build_button.clicked.connect(f)

    def connect_display(self, f):
        self.display_combo.currentIndexChanged.connect(f)
        self.scale_spin.valueChanged.connect(f)
        self.inactive_box.clicked.connect(f)
        self.site_table.itemChanged.connect(f)

    def load_cif_file_dialog(self):
        return self.open_file_dialog("Load CIF", "CIF files (*.cif *.mcif)")

    # ------------------------------------------------------------------
    # values

    def set_cif_file(self, filename):
        self.cif_line.setText(filename)

    def set_cell(self, cell, space_group):
        for line, v in zip(self.cell_lines, cell):
            line.setText(f"{v:.6g}")
        self.space_group_line.setText(space_group)

    def get_cell(self):
        if all(line.hasAcceptableInput() for line in self.cell_lines):
            return tuple(float(line.text()) for line in self.cell_lines)

    def set_sites(self, rows, active=()):
        self.site_table.blockSignals(True)
        self.site_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            use = _item("")
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            use.setCheckState(Qt.Checked if row["label"] in active else Qt.Unchecked)
            self.site_table.setItem(i, 0, use)
            self.site_table.setItem(i, 1, _item(row["label"]))
            self.site_table.setItem(i, 2, _item(row["symbol"]))
            self.site_table.setItem(i, 3, _item(f"{row['occupancy']:.4g}", editable=True))
            for c, key in enumerate("xyz"):
                self.site_table.setItem(i, 4 + c, _item(f"{row[key]:.5f}"))
            self.site_table.setItem(i, 7, _item(f"{row['U']:.4f}"))
            self.site_table.setItem(i, 8, _item(str(row["multiplicity"])))
        self.site_table.blockSignals(False)

    def get_active_labels(self):
        return [self.site_table.item(i, 1).text() for i in range(self.site_table.rowCount())
                if self.site_table.item(i, 0).checkState() == Qt.Checked]

    def get_occupancies(self):
        occ = {}
        for i in range(self.site_table.rowCount()):
            try:
                occ[self.site_table.item(i, 1).text()] = float(self.site_table.item(i, 3).text())
            except ValueError:
                pass
        return occ

    def get_disorder_text(self):
        return self.groups_line.text(), self.ions_line.text(), self.moments_line.text()

    def get_supercell_parameters(self):
        return (tuple(s.value() for s in self.N_spins), self.seed_spin.value(), self.displace_box.isChecked())

    def set_kinds(self, rows):
        self.kind_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            values = [row["site"], row["kind"], row["symbol"], f"{row['fraction']:.4f}", str(row["count"]),
                      f"{row['b']:.3f}", f"{row['moment']:.2f}", str(row["group"])]
            for c, v in enumerate(values):
                self.kind_table.setItem(i, c, _item(v))

    def get_display(self):
        return self.display_combo.currentText(), self.scale_spin.value(), self.inactive_box.isChecked()

    def set_display(self, mode):
        self.display_combo.setCurrentText(mode)

    # ------------------------------------------------------------------
    # drawing

    def draw_atoms(self, style, edges, scale=1.0, show_inactive=True):
        self.clear_scene()
        r, rgb, radius, active = style["r"], style["rgb"], style["radius"] * scale, style["active"]
        for mask, opacity in ((active, 1.0), (~active, 0.25)):
            if not np.any(mask) or (opacity < 1 and not show_inactive):
                continue
            add_spheres(self.plotter, r[mask], radius[mask], rgb=rgb[mask], opacity=opacity)
        lines = pv.line_segments_from_points(edges.reshape(-1, 3))
        self.plotter.add_mesh(lines, color="black", line_width=1)
        self.reset_scene()
