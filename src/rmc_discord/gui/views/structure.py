"""Structure application view."""

import numpy as np
import pyvista as pv
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rmc_discord.gui.models.display import add_spheres
from rmc_discord.gui.views.base_view import BaseView, decorate_combo, row, set_icon

SITE_COLUMNS = ["Use", "Label", "Type", "Occupancy", "x", "y", "z", "Uiso (Å²)", "Mult."]
MODEL_COLUMNS = ["Channel", "Sites", "Orbits", "Coefficients", "Free"]
MAGNETIC_COLUMNS = ["Use", "Label", "Ion", "State", "μeff", "g"]



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
        self._init_model_tab()
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
        self.display_combo.addItems(["Unit Cell", "Model Sites"])
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

    def _init_model_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        channel_box = QGroupBox("Correlations refined on the checked sites")
        rows = QVBoxLayout(channel_box)
        self.occupancy_box = QCheckBox("Occupancy", self)
        self.occupancy_box.setChecked(True)
        self.size_effect_box = QCheckBox("Size Effect", self)
        self.displacement_box = QCheckBox("Displacement", self)
        rows.addLayout(row(self.occupancy_box, self.size_effect_box, self.displacement_box))
        self.restrict_combo = QComboBox(self)
        self.restrict_combo.addItems(["Any direction", "Along axis [uvw]", "In plane (hkl)"])
        self.restrict_combo.setToolTip("Restrict size-effect and displacement directions (e.g. along a chain or "
                                       "within a layer); symmetry-equivalent pairs use the equivalent direction.")
        self.direction_line = QLineEdit()
        self.direction_line.setPlaceholderText("e.g. 0 0 1")
        self.direction_line.setMaximumWidth(110)
        self.direction_line.setEnabled(False)
        self.restrict_combo.currentIndexChanged.connect(lambda i: self.direction_line.setEnabled(i > 0))
        rows.addLayout(row(QLabel("Displacements:"), self.restrict_combo, self.direction_line))
        layout.addWidget(channel_box)

        magnetic_box = QGroupBox("Magnetic correlations")
        rows = QVBoxLayout(magnetic_box)
        self.magnetic_combo = QComboBox(self)
        self.magnetic_combo.addItems(["None", "Isotropic", "Tensor"])
        self.difference_box = QCheckBox("Difference data (self term cancels)", self)
        rows.addLayout(row(QLabel("Model:"), self.magnetic_combo, self.difference_box))
        # plain widgets in a grid, not cell widgets of a QTableWidget: editing a spin box
        # embedded in a table cell crashed Qt on Wayland when focus moved on (Tab)
        self.magnetic_grid_widget = QWidget()
        self.magnetic_grid = QGridLayout(self.magnetic_grid_widget)
        self.magnetic_grid.setContentsMargins(4, 4, 4, 4)
        for c, name in enumerate(MAGNETIC_COLUMNS):
            header = QLabel(f"<b>{name}</b>")
            self.magnetic_grid.addWidget(header, 0, c)
        self.magnetic_grid.setColumnStretch(len(MAGNETIC_COLUMNS), 1)
        self.magnetic_grid.setRowStretch(1, 1)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.magnetic_grid_widget)
        scroll.setToolTip("Sites with tabulated magnetic form factors. The state sets the effective moment "
                          "(spin only for d ions, Hund's rules for f ions); edit it to override.")
        rows.addWidget(scroll, stretch=1)
        self._magnetic_rows = []
        self._states = {}
        self.magnetic_note = QLabel("Load a CIF to list the possible magnetic sites.")
        rows.addWidget(self.magnetic_note)
        layout.addWidget(magnetic_box, stretch=1)

        range_box = QGroupBox("Pairs")
        self.r_max_spin = QDoubleSpinBox(self)
        self.r_max_spin.setRange(1.0, 200.0)
        self.r_max_spin.setDecimals(2)
        self.r_max_spin.setValue(10.0)
        self.r_max_spin.setSuffix(" Å")
        self.cluster_spin = QDoubleSpinBox(self)
        self.cluster_spin.setRange(0.0, 10.0)
        self.cluster_spin.setDecimals(2)
        self.cluster_spin.setValue(1.5)
        self.cluster_spin.setSuffix(" Å")
        self.build_button = QPushButton("Build Model", self)
        set_icon(self.build_button, "diagram-project")
        box_layout = QVBoxLayout(range_box)
        box_layout.addLayout(row(QLabel("Pair range:"), self.r_max_spin, QLabel("Cluster distance:"),
                                  self.cluster_spin, stretch=True, end=self.build_button))
        layout.addWidget(range_box)

        model_box = QGroupBox("Coefficients of the model")
        model_layout = QVBoxLayout(model_box)
        self.model_table = _table(MODEL_COLUMNS)
        model_layout.addWidget(self.model_table)
        layout.addWidget(model_box, stretch=1)

        self.tab_widget.addTab(tab, "Pair Model")

    # magnetic sites

    def set_magnetic_sites(self, rows):
        """Rows ``dict(label, ions, ion)`` of the sites that can carry a moment."""
        grid = self.magnetic_grid
        for row_widgets in self._magnetic_rows:
            for w in row_widgets.values():
                if isinstance(w, QComboBox):
                    w.model().blockSignals(True)  # no decoration callbacks while it is destroyed
                grid.removeWidget(w)
                w.setParent(None)
                w.deleteLater()
        self._magnetic_rows, self._states = [], {}
        grid.setRowStretch(len(rows) + 1, 1)
        for i, entry in enumerate(rows):
            use = QCheckBox(self.magnetic_grid_widget)
            use.setChecked(True)
            label = QLabel(entry["label"], self.magnetic_grid_widget)
            ion = QComboBox(self.magnetic_grid_widget)
            ion.addItems(entry["ions"])
            ion.setCurrentText(entry["ion"])
            state = QComboBox(self.magnetic_grid_widget)
            moment = QDoubleSpinBox(self.magnetic_grid_widget)
            moment.setRange(0.0, 20.0)
            moment.setDecimals(2)
            moment.setSuffix(" μB")
            g = QDoubleSpinBox(self.magnetic_grid_widget)
            g.setRange(0.1, 3.0)
            g.setDecimals(3)
            widgets = dict(use=use, label=label, ion=ion, state=state, moment=moment, g=g)
            for c, w in enumerate(widgets.values()):
                grid.addWidget(w, i + 1, c)
            self._magnetic_rows.append(widgets)
            ion.currentTextChanged.connect(lambda _, i=i: self._fill_states(i))
            state.currentIndexChanged.connect(lambda _, i=i: self._apply_state(i))
            moment.valueChanged.connect(lambda _, i=i: self._custom_state(i))
            g.valueChanged.connect(lambda _, i=i: self._custom_state(i))
            self._fill_states(i)
            decorate_combo(ion)
            decorate_combo(state)
        self.magnetic_note.setVisible(not rows)
        if not rows:
            self.magnetic_note.setText("No site has a tabulated magnetic form factor.")

    def _widgets(self, i):
        w = self._magnetic_rows[i]
        return w["ion"], w["state"], w["moment"], w["g"]

    def _fill_states(self, i):
        from rmc_discord.crystal.moments import spin_states

        ion, state, _, _ = self._widgets(i)
        states = spin_states(ion.currentText())
        self._states[i] = states  # plain item texts in the combo: no Python objects as Qt item data
        state.blockSignals(True)
        state.clear()
        state.addItems([st.label for st in states] + ["custom"])
        state.blockSignals(False)
        state.setCurrentIndex(0)
        self._apply_state(i)

    def _apply_state(self, i):
        _, state, moment, g = self._widgets(i)
        k = state.currentIndex()
        states = self._states.get(i, [])
        if not 0 <= k < len(states):
            return
        st = states[k]
        for w, v in ((moment, st.moment), (g, st.g)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

    def _custom_state(self, i):
        _, state, _, _ = self._widgets(i)
        state.blockSignals(True)
        state.setCurrentIndex(state.count() - 1)
        state.blockSignals(False)

    def set_magnetic_ion(self, label, ion, state=0, moment=None):
        """Choose the ion (and state index or explicit moment) of a magnetic site row."""
        for i, w in enumerate(self._magnetic_rows):
            if w["label"].text() == label:
                w["ion"].setCurrentText(ion)
                w["state"].setCurrentIndex(state)
                if moment is not None:
                    w["moment"].setValue(moment)
                return
        raise KeyError(label)

    def get_magnetic(self):
        """``({label: ion}, {ion: moment}, {ion: g})`` of the checked magnetic sites."""
        ions, moments, g = {}, {}, {}
        for w in self._magnetic_rows:
            if not w["use"].isChecked():
                continue
            ion = w["ion"].currentText()
            for table, value in ((moments, w["moment"].value()), (g, w["g"].value())):
                if ion in table and not np.isclose(table[ion], value):
                    raise ValueError(f"{ion} has two different moments or g factors; use one per ion")
                table[ion] = value
            ions[w["label"].text()] = ion
        return ions, moments, g

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
        for i, entry in enumerate(rows):
            use = _item("")
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            use.setCheckState(Qt.Checked if entry["label"] in active else Qt.Unchecked)
            self.site_table.setItem(i, 0, use)
            self.site_table.setItem(i, 1, _item(entry["label"]))
            self.site_table.setItem(i, 2, _item(entry["symbol"]))
            self.site_table.setItem(i, 3, _item(f"{entry['occupancy']:.4g}", editable=True))
            for c, key in enumerate("xyz"):
                self.site_table.setItem(i, 4 + c, _item(f"{entry[key]:.5f}"))
            self.site_table.setItem(i, 7, _item(f"{entry['U']:.4f}"))
            self.site_table.setItem(i, 8, _item(str(entry["multiplicity"])))
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

    def get_model_parameters(self):
        """Channels, magnetic model, pair range, cluster distance, (axis, plane), difference flag."""
        channels = dict(occupancy=self.occupancy_box.isChecked(), size_effect=self.size_effect_box.isChecked(),
                        displacement=self.displacement_box.isChecked())
        magnetic = {"None": None, "Isotropic": "isotropic", "Tensor": "tensor"}[self.magnetic_combo.currentText()]
        mode = self.restrict_combo.currentIndex()
        axis = plane = None
        if mode:
            values = tuple(float(v) for v in self.direction_line.text().replace(",", " ").split())
            if len(values) != 3 or not any(values):
                raise ValueError("the displacement direction needs three components, not all zero")
            axis, plane = (values, None) if mode == 1 else (None, values)
        return (channels, magnetic, self.r_max_spin.value(), self.cluster_spin.value(), (axis, plane),
                self.difference_box.isChecked())

    def set_model_rows(self, rows):
        self.model_table.setRowCount(len(rows))
        for i, entry in enumerate(rows):
            values = [entry["channel"], entry["sites"], str(entry["orbits"]), str(entry["coefficients"]), str(entry["free"])]
            for c, v in enumerate(values):
                self.model_table.setItem(i, c, _item(v))

    def get_display(self):
        return self.display_combo.currentText(), self.scale_spin.value(), self.inactive_box.isChecked()

    def set_display(self, mode):
        self.display_combo.setCurrentText(mode)

    # ------------------------------------------------------------------
    # drawing

    def draw_atoms(self, style, edges, scale=1.0, show_inactive=True):
        self.clear_scene()
        r, rgb, radius, active = style["r"], style["rgb"], style["radius"] * scale, style["active"]
        if not show_inactive:
            r, rgb, radius = r[active], rgb[active], radius[active]
        else:
            # Faded toward the background rather than translucent:
            # translucent point sprites render black.
            background = np.array(self.plotter.background_color.int_rgb, dtype=float)
            rgb = np.where(active[:, None], rgb, 0.3 * rgb + 0.7 * background).astype(np.uint8)
        if len(r):
            add_spheres(self.plotter, r, radius, rgb=rgb)
        lines = pv.line_segments_from_points(edges.reshape(-1, 3))
        self.plotter.add_mesh(lines, color="black", line_width=1)
        self.reset_scene()
