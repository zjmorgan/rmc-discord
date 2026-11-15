"""
Main window.

Applications are the steps of the workflow (Structure, Data, Refinement,
Correlations, Recalculation), listed in a sidebar in order; a step is
enabled once the steps it needs are done. The Applications menu offers the
same choice, as in NeuXtalViz.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")


def prefer_gpu():
    """
    Use the GPU for OpenGL under WSL.

    WSL's Mesa falls back to software rendering (llvmpipe), which makes the
    3D views crawl; its d3d12 driver renders on the Windows GPU, preferring
    an NVIDIA adapter if there is one. Variables already set are kept; set
    ``GALLIUM_DRIVER=llvmpipe`` to force software rendering.
    """
    import platform

    if "microsoft" not in platform.release().lower() or "GALLIUM_DRIVER" in os.environ:
        return
    drivers = ("/usr/lib/x86_64-linux-gnu/dri/d3d12_dri.so", "/usr/lib/wsl/lib/libd3d12.so")
    if all(os.path.exists(f) for f in drivers):
        os.environ["GALLIUM_DRIVER"] = "d3d12"
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")


prefer_gpu()

import pyvista
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from rmc_discord import __version__
from rmc_discord.gui import theme
from rmc_discord.gui.models.correlations import CorrelationsModel
from rmc_discord.gui.models.recalculation import RecalculationModel
from rmc_discord.gui.models.data import DataModel
from rmc_discord.gui.models.refinement import RefinementModel
from rmc_discord.gui.models.structure import StructureModel
from rmc_discord.gui.presenters.correlations import CorrelationsPresenter
from rmc_discord.gui.presenters.recalculation import RecalculationPresenter
from rmc_discord.gui.presenters.data import DataPresenter
from rmc_discord.gui.presenters.refinement import RefinementPresenter
from rmc_discord.gui.presenters.structure import StructurePresenter
from rmc_discord.gui.session import Session
from rmc_discord.gui.views.correlations import CorrelationsView
from rmc_discord.gui.views.recalculation import RecalculationView
from rmc_discord.gui.views.data import DataView
from rmc_discord.gui.views.refinement import RefinementView
from rmc_discord.gui.views.structure import StructureView


ICON = Path(__file__).resolve().parents[3] / "disorder" / "graphical" / "logo.png"


class Placeholder(QWidget):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)


class MainWindow(QMainWindow):
    STEPS = ["Structure", "Data", "Refinement", "Correlations", "Recalculation"]
    ICONS = ["atom", "cube", "fire", "diagram-project", "wave-square"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"rmc-discord {__version__}")
        if ICON.exists():
            self.setWindowIcon(QIcon(str(ICON)))

        self.session = Session()

        self.structure_view = StructureView(self)
        self.structure = StructurePresenter(self.structure_view, StructureModel(self.session))
        self.data_view = DataView(self)
        self.data = DataPresenter(self.data_view, DataModel(self.session))
        self.refinement_view = RefinementView(self)
        self.refinement = RefinementPresenter(self.refinement_view, RefinementModel(self.session))
        self.correlations_view = CorrelationsView(self)
        self.correlations = CorrelationsPresenter(self.correlations_view, CorrelationsModel(self.session))
        self.recalculation_view = RecalculationView(self)
        self.recalculation = RecalculationPresenter(self.recalculation_view, RecalculationModel(self.session))
        self.views = [self.structure_view, self.data_view, self.refinement_view, self.correlations_view,
                      self.recalculation_view]

        self.stack = QStackedWidget()
        for view in self.views:
            self.stack.addWidget(view)
        for name in self.STEPS[len(self.views):]:
            self.stack.addWidget(Placeholder(f"{name}: in progress", self))

        self.workflow = QListWidget()
        self.workflow.setObjectName("workflow")
        self.workflow.setFixedWidth(180)
        self.workflow.setIconSize(QSize(16, 16))
        for i, name in enumerate(self.STEPS):
            item = QListWidgetItem(f"{i + 1}  {name}")
            item.setSizeHint(QSize(170, 40))
            self.workflow.addItem(item)
        self.workflow.currentRowChanged.connect(self.stack.setCurrentIndex)

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.workflow)
        layout.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central)

        file_menu = self.menuBar().addMenu("File")
        quit_action = QAction("Quit", self)
        self.quit_action = quit_action
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        app_menu = self.menuBar().addMenu("Applications")
        self.app_actions = []
        for i, name in enumerate(self.STEPS):
            action = QAction(name, self)
            action.triggered.connect(lambda _=False, i=i: self.workflow.setCurrentRow(i))
            app_menu.addAction(action)
            self.app_actions.append(action)

        help_menu = self.menuBar().addMenu("Help")
        about = QAction("About", self)
        self.about_action = about
        about.triggered.connect(self.about)

        help_menu.addAction(about)

        for signal in (self.session.crystal_changed, self.session.supercell_changed,
                       self.session.sample_changed, self.session.refinement_changed,
                       self.session.correlations_changed, self.session.recalculated):
            signal.connect(self.update_steps)
        theme.notifier.changed.connect(self.update_steps)
        self.update_steps()
        self.workflow.setCurrentRow(0)

    def ready(self):
        s = self.session
        return [True, s.crystal is not None, s.sample is not None and s.supercell is not None,
                s.supercell is not None,
                s.supercell is not None and bool(s.correlations) and s.volume is not None]

    def done(self):
        s = self.session
        return [s.supercell is not None, s.sample is not None, s.result is not None,
                bool(s.correlations), self.recalculation.model.result is not None]

    def update_steps(self):
        for i, (ready, done) in enumerate(zip(self.ready(), self.done())):
            item = self.workflow.item(i)
            flags = item.flags()
            item.setFlags(flags | Qt.ItemIsEnabled if ready else flags & ~Qt.ItemIsEnabled)
            item.setText(f"{i + 1}  {self.STEPS[i]}")
            item.setIcon(theme.icon("circle-check", color="#2e7d32") if done else theme.icon(self.ICONS[i]))
            item.setToolTip("Done" if done else ("Ready" if ready else "Needs the earlier steps"))
            self.app_actions[i].setEnabled(ready)
            self.app_actions[i].setIcon(theme.icon(self.ICONS[i]))
        self.quit_action.setIcon(theme.icon("right-from-bracket"))
        self.about_action.setIcon(theme.icon("circle-info"))

    def closeEvent(self, event):
        model = self.refinement.model
        if model.running:
            model.stop()
            self.refinement_view.threadpool.waitForDone(600_000)
        for view in self.views:
            view.plotter.close()
        super().closeEvent(event)

    def about(self):
        QMessageBox.about(self, "rmc-discord",
                          f"rmc-discord {__version__}\n\nReverse Monte Carlo refinement of diffuse "
                          "scattering from single crystals.\n\nInterface after NeuXtalViz.")


def gui():
    app = QApplication.instance() or QApplication(sys.argv)
    theme.apply(app)
    window = MainWindow()
    window.resize(1600, 950)
    window.show()
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())


if __name__ == "__main__":
    gui()
