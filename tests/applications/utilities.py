"""
Drive the graphical interface through a tutorial and take screenshots.

As in NeuXtalViz, a scenario clicks through the application and saves
window screenshots that the documentation shows. Here the window runs on
Qt's offscreen platform (no display or Xvfb needed): the widgets are
grabbed, and each visible 3D view is rendered by an off-screen PyVista
plotter with the same actors, lights, and camera and painted in place.

Usage (from the repository root)::

    python tests/applications/bixbyite.py
    python tests/applications/carbonate.py

Data are under ``$DIFFUSE_DATA_ROOT`` (as for the examples); the PNGs go to
``docs/source/``.
"""

import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qtpy.QtCore import QPoint, QTimer  # noqa: E402
from qtpy.QtGui import QImage, QPainter  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

DATA = Path(os.environ.get("DIFFUSE_DATA_ROOT", Path.home() / "git/discordant-parsimony/local_data"))
DOCS = ROOT / "docs" / "source"
SIZE = (1600, 1000)
HIGHLIGHT = "background-color: #2e7d32; color: white;"


def process(app, seconds=0.0):
    """Let Qt deliver pending events for (at least) ``seconds``."""

    end = time.monotonic() + seconds
    while True:
        app.processEvents()
        if time.monotonic() >= end:
            break
        time.sleep(0.02)


def wait(app, window, timeout=3600):
    """Wait for every view's workers to finish and their signals to land."""

    start = time.monotonic()
    process(app, 0.1)
    while any(view.threadpool.activeThreadCount() for view in window.views):
        if time.monotonic() - start > timeout:
            raise TimeoutError("task did not finish")
        process(app, 0.1)
    process(app, 0.3)


def click(app, window, button, wait_for=True):
    """Click a button (highlighted while the task runs) and wait for it."""

    button.setStyleSheet(HIGHLIGHT)
    button.click()
    if wait_for:
        wait(app, window)


def _render(interactor):
    """The image a PyVista Qt interactor would show, rendered off screen."""

    import pyvista as pv

    w, h = interactor.width(), interactor.height()
    off = pv.Plotter(off_screen=True, window_size=(w, h))
    off.set_background(interactor.background_color)
    for name, actor in interactor.renderer.actors.items():
        off.add_actor(actor, name=name, reset_camera=False)
    off.remove_all_lights()
    for light in interactor.renderer.lights:
        off.add_light(light.copy())
    off.camera = interactor.camera.copy()
    widget = getattr(interactor.renderer, "axes_widget", None)
    if widget is not None and widget.GetEnabled():
        off.add_orientation_widget(interactor.renderer.axes_actor)
    image = off.screenshot(return_img=True)
    for name in list(off.renderer.actors):
        off.remove_actor(name, render=False)
    off.close()
    return np.ascontiguousarray(image[:, :, :3])


def screenshot(app, window, name):
    """Save the window (3D views included) as ``docs/source/<name>.png``."""

    from pyvistaqt import QtInteractor

    process(app, 0.3)
    shot = window.grab().toImage()
    painter = QPainter(shot)
    for interactor in window.findChildren(QtInteractor):
        if not interactor.isVisible() or interactor.visibleRegion().isEmpty():
            continue
        image = _render(interactor)
        h, w, _ = image.shape
        painter.drawImage(interactor.mapTo(window, QPoint(0, 0)), QImage(image.data, w, h, 3 * w,
                                                                         QImage.Format_RGB888))
    painter.end()
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / f"{name}.png"
    shot.save(str(path), "png")
    print(f"saved {path.relative_to(ROOT)}", flush=True)
    for widget in window.findChildren(object):
        if hasattr(widget, "styleSheet") and widget.styleSheet() == HIGHLIGHT:
            widget.setStyleSheet("")


def _echo(append):
    """Also print a view's console output, to follow long tasks."""

    def wrapped(text):
        append(text)
        if text.strip():
            sys.__stdout__.write(text.rstrip("\n") + "\n")
            sys.__stdout__.flush()

    return wrapped


def run_qt_scenario(scenario):
    """Open the main window, run ``scenario(app, window)``, and exit."""

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)

    from rmc_discord.gui import theme
    from rmc_discord.gui.app import MainWindow

    theme.set_theme("default", app)
    window = MainWindow()
    window.resize(*SIZE)
    window.show()

    errors = []
    for view in window.views:  # no modal dialogs: fail the scenario instead
        view.show_error = lambda message, details=None: errors.append((message, details))
        view.append_to_console = _echo(view.append_to_console)

    result = {"rc": 1}

    def _wrapped():
        try:
            scenario(app, window, errors)
            check(errors)
            result["rc"] = 0
        except BaseException:
            traceback.print_exc()
        finally:
            process(app, 0.5)
            app.exit(result["rc"])

    QTimer.singleShot(0, _wrapped)
    rc = app.exec()
    # Skip Qt/VTK destructors at exit (they can crash offscreen).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


def check(errors):
    """Raise the first error a task reported."""

    if errors:
        message, details = errors[0]
        raise RuntimeError(f"{message}\n{details or ''}")
