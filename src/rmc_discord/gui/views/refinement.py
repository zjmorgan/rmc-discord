"""Refinement application view."""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator, QIntValidator
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
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rmc_discord.engine.refinement import MOVES
from rmc_discord.gui.models.display import draw_configuration
from rmc_discord.gui.views.base_view import BaseView, set_icon
from rmc_discord.gui.views.plots import PlotWidget

MOVE_NAMES = {"swap": "Swap", "spin": "Spin", "displace": "Displace", "swap_displace": "Swap + Displace",
              "exchange": "Exchange"}


def _held_name(name):
    """Column/legend name of a held-out set ("held" is the only one)."""

    return "Held-out" if name == "held" else f"Held {name}"


def _float(text="", low=-1e12, high=1e12, decimals=8, blank=False):
    line = QLineEdit(text)
    line.setValidator(QDoubleValidator(low, high, decimals, notation=QDoubleValidator.ScientificNotation))
    line._blank = blank
    return line


def _int(text="", low=0, high=2**31 - 1, blank=False):
    line = QLineEdit(text)
    line.setValidator(QIntValidator(low, high))
    line._blank = blank
    return line


def _value(line, cast=float):
    text = line.text().strip()
    if not text:
        if getattr(line, "_blank", False):
            return None
        raise ValueError("missing value")
    if not line.hasAcceptableInput():
        raise ValueError(f"invalid value {text!r}")
    return cast(text)


class RefinementView(BaseView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recip_box.setChecked(False)
        self.tab_widget = QTabWidget(self)
        self._init_setup_tab()
        self._init_anneal_tab()
        self.add_tools(self.tab_widget)

    # ------------------------------------------------------------------

    def _init_setup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Scattering")
        grid = QGridLayout(box)
        self.nuclear_box = QCheckBox("Nuclear", self)
        self.magnetic_box = QCheckBox("Magnetic", self)
        self.radiation_combo = QComboBox(self)
        self.radiation_combo.addItems(["Neutron", "X-ray"])
        self.offset_combo = QComboBox(self)
        self.offset_combo.addItems(["None", "Paramagnetic"])
        grid.addWidget(self.nuclear_box, 0, 0)
        grid.addWidget(self.magnetic_box, 0, 1)
        grid.addWidget(QLabel("Radiation:"), 0, 2)
        grid.addWidget(self.radiation_combo, 0, 3)
        grid.addWidget(QLabel("Offset:"), 0, 4)
        grid.addWidget(self.offset_combo, 0, 5)
        layout.addWidget(box)

        box = QGroupBox("Moves (relative weights; 0 disables)")
        grid = QGridLayout(box)
        self.weight_lines = {}
        for i, key in enumerate(MOVES):
            line = _float("0", 0, 1e6)
            self.weight_lines[key] = line
            grid.addWidget(QLabel(MOVE_NAMES[key] + ":"), i // 3, 2 * (i % 3))
            grid.addWidget(line, i // 3, 2 * (i % 3) + 1)
        self.penalty_line = _float("10", 0, 1e9)
        self.step_line = _float("0.5", 1e-4, 10)
        self.cone_line = _float("60", 1, 180)
        grid.addWidget(QLabel("ADP penalty:"), 2, 0)
        grid.addWidget(self.penalty_line, 2, 1)
        grid.addWidget(QLabel("Step (× ADP):"), 2, 2)
        grid.addWidget(self.step_line, 2, 3)
        grid.addWidget(QLabel("Spin cone (°):"), 2, 4)
        grid.addWidget(self.cone_line, 2, 5)
        grid.addWidget(QLabel("Displacive sites:"), 3, 0, Qt.AlignTop)
        self.displacive_list = QListWidget(self)
        self.displacive_list.setMaximumHeight(90)
        grid.addWidget(self.displacive_list, 3, 1, 1, 5)
        layout.addWidget(box)

        box = QGroupBox("Model")
        grid = QGridLayout(box)
        self.background_combo = QComboBox(self)
        self.background_combo.addItems(["Radial", "Constant"])
        self.knot_line = _float("0.5", 0.01, 100)
        self.symmetrize_line = QLineEdit()
        self.symmetrize_line.setPlaceholderText("extra operators averaged over, e.g. k,h,l")
        self.seed_line = _int("1")
        grid.addWidget(QLabel("Background:"), 0, 0)
        grid.addWidget(self.background_combo, 0, 1)
        grid.addWidget(QLabel("Knot spacing (Å⁻¹):"), 0, 2)
        grid.addWidget(self.knot_line, 0, 3)
        grid.addWidget(QLabel("Symmetrize:"), 1, 0)
        grid.addWidget(self.symmetrize_line, 1, 1, 1, 3)
        grid.addWidget(QLabel("Seed:"), 0, 4)
        grid.addWidget(self.seed_line, 0, 5)
        layout.addWidget(box)

        box = QGroupBox("Validation")
        grid = QGridLayout(box)
        self.held_line = _float("0.1", 0, 0.9)
        self.n_held_line = _int("", 1, blank=True)
        self.n_held_line.setPlaceholderText("or a number per set")
        self.strata_combo = QComboBox(self)
        self.strata_combo.addItems(["None", "Planes"])
        self.plane_axis_combo = QComboBox(self)
        self.plane_axis_combo.addItems(["h", "k", "l"])
        self.plane_axis_combo.setCurrentIndex(2)
        self.planes_line = QLineEdit()
        self.planes_line.setPlaceholderText("|index| of the planes, e.g. 0.8, 2.85, 4.9")
        self.plane_width_line = _float("0.1", 0, 10)
        self.ratio_line = _float("", 0, 1e6, blank=True)
        self.ratio_line.setPlaceholderText("all")
        grid.addWidget(QLabel("Held-out fraction:"), 0, 0)
        grid.addWidget(self.held_line, 0, 1)
        grid.addWidget(self.n_held_line, 0, 2, 1, 2)
        grid.addWidget(QLabel("Strata:"), 1, 0)
        grid.addWidget(self.strata_combo, 1, 1)
        grid.addWidget(QLabel("Axis:"), 1, 2)
        grid.addWidget(self.plane_axis_combo, 1, 3)
        grid.addWidget(QLabel("Planes:"), 2, 0)
        grid.addWidget(self.planes_line, 2, 1, 1, 3)
        grid.addWidget(QLabel("Half width:"), 3, 0)
        grid.addWidget(self.plane_width_line, 3, 1)
        grid.addWidget(QLabel("Pool off:on ratio:"), 3, 2)
        grid.addWidget(self.ratio_line, 3, 3)
        layout.addWidget(box)

        init_layout = QHBoxLayout()
        self.init_info = QLabel("")
        self.init_info.setWordWrap(True)
        self.init_button = QPushButton("Initialize Refinement", self)
        set_icon(self.init_button, "gears")
        init_layout.addWidget(self.init_info, stretch=1)
        init_layout.addWidget(self.init_button)
        layout.addLayout(init_layout)
        layout.addStretch(1)
        self.tab_widget.addTab(tab, "Setup")

    def _init_anneal_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Schedule (blank temperature: calibrate; blank adapt/floor: off)")
        grid = QGridLayout(box)
        self.schedule_lines = [
            ("Rounds:", _int("60", 1)), ("Sweeps/round:", _float("4", 1e-3)), ("Cooling:", _float("0.92", 0, 1)),
            ("Temperature:", _float("", 0, blank=True)), ("Batch points:", _int("100000", 1, blank=True)),
            ("Sweeps/batch:", _float("2", 1e-3)), ("Check every:", _int("5", 1)),
            ("Adapt to:", _float("", 0, 1, blank=True)), ("Scale floor:", _float("0.25", 0, 10, blank=True)),
        ]
        for i, (name, line) in enumerate(self.schedule_lines):
            grid.addWidget(QLabel(name), i // 3, 2 * (i % 3))
            grid.addWidget(line, i // 3, 2 * (i % 3) + 1)
        layout.addWidget(box)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run", self)
        set_icon(self.run_button, "play")
        self.stop_button = QPushButton("Stop", self)
        set_icon(self.stop_button, "stop")
        self.stop_button.setEnabled(False)
        self.save_config_button = QPushButton("Save Configuration", self)
        set_icon(self.save_config_button, "floppy-disk")
        self.load_config_button = QPushButton("Load Configuration", self)
        set_icon(self.load_config_button, "folder-open")
        for b in (self.run_button, self.stop_button, self.save_config_button, self.load_config_button):
            buttons.addWidget(b)
        layout.addLayout(buttons)

        splitter = QSplitter(Qt.Vertical)
        self.progress_plot = PlotWidget(self)
        splitter.addWidget(self.progress_plot)
        self.record_table = QTableWidget(0, 0)
        self.record_table.verticalHeader().setVisible(False)
        self.record_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.record_table.setAlternatingRowColors(True)
        splitter.addWidget(self.record_table)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)
        self.tab_widget.addTab(tab, "Anneal")

    # ------------------------------------------------------------------
    # connections

    def connect_initialize(self, f):
        self.init_button.clicked.connect(f)

    def connect_run(self, f):
        self.run_button.clicked.connect(f)

    def connect_stop(self, f):
        self.stop_button.clicked.connect(f)

    def connect_save_configuration(self, f):
        self.save_config_button.clicked.connect(f)

    def connect_load_configuration(self, f):
        self.load_config_button.clicked.connect(f)

    def set_running(self, running):
        self.run_button.setEnabled(not running)
        self.init_button.setEnabled(not running)
        self.load_config_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    # ------------------------------------------------------------------
    # values

    def set_settings(self, st, sites):
        self.nuclear_box.setChecked(st.nuclear)
        self.magnetic_box.setChecked(st.magnetic)
        self.offset_combo.setCurrentText("Paramagnetic" if st.offset == "paramagnetic" else "None")
        for key, line in self.weight_lines.items():
            line.setText(f"{st.weights.get(key, 0.0):g}")
        self.displacive_list.clear()
        for j, label in enumerate(sites):
            item = QListWidgetItem(label)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if j in st.displacive else Qt.Unchecked)
            self.displacive_list.addItem(item)

    def get_settings(self):
        from rmc_discord.gui.models.refinement import Settings

        weights = {k: _value(line) for k, line in self.weight_lines.items()}
        displacive = [i for i in range(self.displacive_list.count())
                      if self.displacive_list.item(i).checkState() == Qt.Checked]
        planes = None
        if self.strata_combo.currentText() == "Planes":
            positions = [float(v) for v in self.planes_line.text().replace(";", ",").split(",") if v.strip()]
            if not positions:
                raise ValueError("give the plane positions")
            planes = (self.plane_axis_combo.currentIndex(), positions, _value(self.plane_width_line))
        symmetrize = [op.strip() for op in self.symmetrize_line.text().split(";") if op.strip()]
        return Settings(
            nuclear=self.nuclear_box.isChecked(), magnetic=self.magnetic_box.isChecked(),
            radiation="xray" if self.radiation_combo.currentText() == "X-ray" else "neutron",
            offset="paramagnetic" if self.offset_combo.currentText() == "Paramagnetic" else None,
            weights=weights, displacive=displacive, penalty=_value(self.penalty_line), step=_value(self.step_line),
            cone=_value(self.cone_line),
            background="radial" if self.background_combo.currentText() == "Radial" else None,
            knot_spacing=_value(self.knot_line), symmetrize=symmetrize, held_out=_value(self.held_line),
            n_held=_value(self.n_held_line, int), planes=planes, ratio=_value(self.ratio_line),
            seed=_value(self.seed_line, int),
        )

    def get_schedule_values(self):
        casts = [int, float, float, float, int, float, int, float, float]
        return [_value(line, cast) for (_, line), cast in zip(self.schedule_lines, casts)]

    def set_init_info(self, text):
        self.init_info.setText(text)

    def config_save_dialog(self):
        return self.save_file_dialog("Save configuration", "NumPy archives (*.npz)")

    def config_load_dialog(self):
        return self.open_file_dialog("Load configuration", "NumPy archives (*.npz)")

    # ------------------------------------------------------------------
    # drawing

    def draw_records(self, records, baseline):
        fig = self.progress_plot.fig
        fig.clear()
        if not records:
            self.progress_plot.draw()
            return
        rounds = [r["round"] + 1 for r in records]
        ax = fig.add_subplot(211)
        ax.plot(rounds, [r["score"] for r in records], "o-", color="#1a73e8", label="score")
        best = int(np.argmin([r["score"] for r in records]))
        ax.plot(rounds[best], records[best]["score"], "*", ms=14, color="#ea4335", label="best")
        ax.axhline(1.0, color="0.6", ls="--", lw=1)
        ax.set_ylabel("held-out χ² / background only")
        ax.legend(loc="upper right", fontsize=8)
        ax2 = fig.add_subplot(212, sharex=ax)
        for name in records[0]["held"]:
            ax2.plot(rounds, [r["held"][name] for r in records], "o-", ms=3, label=_held_name(name).lower())
        ax2.plot(rounds, [r["pool"] for r in records], "s-", ms=3, color="0.4", label="pool (free scale)")
        ax2.set_xlabel("round")
        from matplotlib.ticker import MaxNLocator

        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_ylabel("reduced χ²")
        ax2.legend(loc="upper right", fontsize=8)
        self.progress_plot.draw()

        names = list(records[0]["held"])
        moves = list(records[0]["acceptance"])
        columns = ["Round", "T", "Pool χ²", "Scale"] + [_held_name(n) for n in names] + ["Score"] + \
                  [f"Acc. {MOVE_NAMES.get(m, m)}" for m in moves]
        self.record_table.setColumnCount(len(columns))
        self.record_table.setHorizontalHeaderLabels(columns)
        self.record_table.setRowCount(len(records))
        for i, r in enumerate(records):
            values = [str(r["round"] + 1), f"{r['temperature']:.3g}", f"{r['pool']:.4g}", f"{r['scale']:.3g}"] + \
                     [f"{r['held'][n]:.4g}" for n in names] + [f"{r['score']:.4f}"] + \
                     [f"{r['acceptance'].get(m, np.nan):.2f}" for m in moves]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self.record_table.setItem(i, c, item)
        self.record_table.scrollToBottom()

    def draw_configuration(self, style, edges):
        self.clear_scene()
        draw_configuration(self.plotter, style, edges)
        self.reset_scene()
