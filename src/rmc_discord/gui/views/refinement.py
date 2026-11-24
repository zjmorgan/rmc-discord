"""Refinement application view."""

import numpy as np
from matplotlib.colors import Normalize
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
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rmc_discord.gui.models.refinement import Settings
from rmc_discord.gui.views.base_view import BaseView, set_icon
from rmc_discord.gui.views.plots import CMAPS, LIMITS, PlotWidget, color_limits, metric_mesh, norm

BACKGROUNDS = {"Radial (|Q| hat functions)": "radial", "Polynomial in |Q|²": "polynomial", "None": None}
RECORD_COLUMNS = ["Fit", "Channels", "r max (Å)", "Rank", "χ² train", "χ² held-out", "Scale", "Time (s)"]


def _float(text="", low=-1e12, high=1e12, decimals=8):
    line = QLineEdit(text)
    line.setValidator(QDoubleValidator(low, high, decimals, notation=QDoubleValidator.ScientificNotation))
    return line


class RefinementView(BaseView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recip_box.setChecked(False)
        self.tab_widget = QTabWidget(self)
        self._init_setup_tab()
        self._init_results_tab()
        self._init_compare()
        self.add_tools(self.tab_widget)

    # ------------------------------------------------------------------

    def _init_setup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Background")
        grid = QGridLayout(box)
        self.background_combo = QComboBox(self)
        self.background_combo.addItems(list(BACKGROUNDS))
        self.background_combo.setToolTip("Smooth background refined with the correlations.")
        self.knot_line = _float("0.5", 1e-3, 100)
        self.knot_line.setToolTip("Knot spacing of the radial hat functions (Å⁻¹).")
        self.degree_line = QLineEdit("1")
        self.degree_line.setValidator(QIntValidator(0, 10))
        self.degree_line.setToolTip("Degree of the polynomial in |Q|².")
        grid.addWidget(QLabel("Basis:"), 0, 0)
        grid.addWidget(self.background_combo, 0, 1, 1, 3)
        grid.addWidget(QLabel("Knot spacing (Å⁻¹):"), 1, 0)
        grid.addWidget(self.knot_line, 1, 1)
        grid.addWidget(QLabel("Degree:"), 1, 2)
        grid.addWidget(self.degree_line, 1, 3)
        layout.addWidget(box)

        box = QGroupBox("Least squares")
        grid = QGridLayout(box)
        self.ridge_line = _float("0", 0, 1)
        self.ridge_line.setToolTip("Tikhonov damping relative to the mean diagonal of the normal matrix "
                                   "(e.g. 1e-6 for nearly collinear orbits); 0 for none.")
        self.fit_button = QPushButton("Refine Correlations", self)
        set_icon(self.fit_button, "play")
        grid.addWidget(QLabel("Relative ridge:"), 0, 0)
        grid.addWidget(self.ridge_line, 0, 1)
        grid.addWidget(self.fit_button, 0, 2)
        self.fit_info = QLabel("")
        self.fit_info.setWordWrap(True)
        grid.addWidget(self.fit_info, 1, 0, 1, 3)
        layout.addWidget(box)
        layout.addStretch(1)
        self.tab_widget.addTab(tab, "Setup")

    def _init_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Vertical)
        self.progress_plot = PlotWidget(self)
        splitter.addWidget(self.progress_plot)
        self.record_table = QTableWidget(0, len(RECORD_COLUMNS))
        self.record_table.setHorizontalHeaderLabels(RECORD_COLUMNS)
        self.record_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.record_table.verticalHeader().setVisible(False)
        splitter.addWidget(self.record_table)
        layout.addWidget(splitter)
        self.tab_widget.addTab(tab, "Fits")

    def _init_compare(self):
        """Observed and calculated intensity on a slice of the data volume (replaces the 3-D view)."""
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.compare_normal_combo = QComboBox(self)
        self.compare_normal_combo.addItems(["Axis 1/2", "Axis 1/3", "Axis 2/3"])
        self.compare_normal_combo.setToolTip("Plane of the slice (projection axes of the data volume).")
        self.compare_value_line = _float("0", -1e6, 1e6)
        self.compare_value_line.setToolTip("Position of the slice along the remaining axis (r.l.u.).")
        self.compare_thickness_line = _float("0.01", 1e-6, 100)
        self.compare_thickness_line.setToolTip("Slab thickness (r.l.u.); points within half of it are shown.")
        self.compare_scale_combo = QComboBox(self)
        self.compare_scale_combo.addItems(["Linear", "Log"])
        self.compare_clim_combo = QComboBox(self)
        self.compare_clim_combo.addItems(LIMITS)
        self.compare_clim_combo.setCurrentIndex(2)
        self.compare_cmap_combo = QComboBox(self)
        self.compare_cmap_combo.addItems(list(CMAPS))
        for w in (QLabel("Slice:"), self.compare_normal_combo, QLabel("at"), self.compare_value_line,
                  QLabel("thickness"), self.compare_thickness_line, self.compare_scale_combo,
                  self.compare_clim_combo, self.compare_cmap_combo):
            controls.addWidget(w)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.compare_plot = PlotWidget(self)
        layout.addWidget(self.compare_plot, stretch=1)
        self.replace_scene(widget)

    def get_compare(self):
        lines = (self.compare_value_line, self.compare_thickness_line)
        if not all(line.hasAcceptableInput() for line in lines):
            return None
        normal = {"Axis 1/2": 2, "Axis 1/3": 1, "Axis 2/3": 0}[self.compare_normal_combo.currentText()]
        return normal, float(self.compare_value_line.text()), float(self.compare_thickness_line.text())

    def connect_compare(self, f):
        for combo in (self.compare_normal_combo, self.compare_scale_combo, self.compare_clim_combo,
                      self.compare_cmap_combo):
            combo.currentIndexChanged.connect(f)
        for line in (self.compare_value_line, self.compare_thickness_line):
            line.editingFinished.connect(f)

    def draw_comparison(self, out, title=""):
        fig = self.compare_plot.fig
        fig.clear()
        if out is None:
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No measured bins in this slice" if title else "Refine the correlations",
                    ha="center", va="center", transform=ax.transAxes)
            self.compare_plot.draw()
            return
        scale = self.compare_scale_combo.currentText()
        limits = self.compare_clim_combo.currentText()
        cmap = CMAPS[self.compare_cmap_combo.currentText()]
        ea, eb = out["edges"]
        names = [out["names"][a] for a in out["axes"]]
        shared = norm(np.concatenate([out["obs"].ravel(), out["calc"].ravel()]), limits, scale)
        lo, hi = color_limits(out["residual"], limits)
        m = max(abs(lo), abs(hi), 1e-12)
        panels = [("observed", out["obs"], cmap, shared), ("calculated", out["calc"], cmap, shared),
                  ("(obs − calc) / σ", out["residual"], "bwr", Normalize(-m, m))]
        for k, (name, image, cm, nm) in enumerate(panels):
            ax = fig.add_subplot(1, 3, k + 1)
            mesh, _ = metric_mesh(ax, image, ea, eb, *out["transform"], cmap=cm, norm=nm, rasterized=True)
            fig.colorbar(mesh, ax=ax, shrink=0.7, location="bottom")
            ax.set_title(name, fontsize=10)
            ax.set_xlabel(names[0])
            if k == 0:
                ax.set_ylabel(names[1])
        fig.suptitle(title, fontsize=10)
        self.compare_plot.draw()

    # ------------------------------------------------------------------
    # connections and values

    def connect_fit(self, f):
        self.fit_button.clicked.connect(f)

    def set_running(self, running):
        self.fit_button.setEnabled(not running)

    def set_settings(self, st):
        name = next(k for k, v in BACKGROUNDS.items() if v == st.background)
        self.background_combo.setCurrentText(name)
        self.knot_line.setText(f"{st.knot_spacing:g}")
        self.degree_line.setText(str(st.degree))
        self.ridge_line.setText(f"{st.ridge:g}")

    def get_settings(self):
        lines = (self.knot_line, self.degree_line, self.ridge_line)
        if not all(line.hasAcceptableInput() for line in lines):
            return None
        return Settings(background=BACKGROUNDS[self.background_combo.currentText()],
                        knot_spacing=float(self.knot_line.text()), degree=int(self.degree_line.text()),
                        ridge=float(self.ridge_line.text()))

    def set_fit_info(self, text):
        self.fit_info.setText(text)

    def draw_records(self, records):
        fig = self.progress_plot.fig
        fig.clear()
        ax = fig.add_subplot(111)
        if records:
            x = np.arange(1, len(records) + 1)
            ax.plot(x, [r["chi2_test"] for r in records], "o-", color="#1a73e8", label="held-out")
            ax.plot(x, [r["chi2_train"] for r in records], "s--", ms=4, color="0.4", label="train")
            best = int(np.nanargmin([r["chi2_test"] for r in records])) if any(
                np.isfinite(r["chi2_test"]) for r in records) else 0
            ax.plot(x[best], records[best]["chi2_test"], "*", ms=14, color="#ea4335", label="best")
            ax.axhline(1.0, color="0.6", ls="--", lw=1)
            from matplotlib.ticker import MaxNLocator

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("fit")
        ax.set_ylabel("reduced χ²")
        self.progress_plot.draw()

        self.record_table.setRowCount(len(records))
        for i, r in enumerate(records):
            values = [str(i + 1), r["channels"], f"{r['r_max']:g}", str(r["free"]), f"{r['chi2_train']:.4g}",
                      f"{r['chi2_test']:.4g}", f"{r['scale']:.4g}", f"{r['seconds']:.1f}"]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self.record_table.setItem(i, c, item)
        self.record_table.scrollToBottom()
