"""
Look of NeuXtalViz: QDarkStyle (Spyder's theme), light or dark to match the
system, with PyVista and matplotlib themes chosen together.

Selecting the "dark" plot theme in any view switches the whole application
to dark; :data:`notifier` relays the change to every view.
"""

import matplotlib.pyplot as plt
import pyvista as pv
import qdarkstyle
import qtawesome as qta
from qdarkstyle.dark.palette import DarkPalette
from qdarkstyle.light.palette import LightPalette
from qtpy.QtCore import QObject, Signal
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QApplication

PLOT_THEMES = ("default", "document", "dark", "paraview")

MPL_STYLE = {"default": "default", "document": "default", "dark": "dark_background", "paraview": "Solarize_Light2"}


class _Notifier(QObject):
    changed = Signal(str)


notifier = _Notifier()
current = "default"


def _sidebar(p):
    """Workflow sidebar rules in the colors of a QDarkStyle palette."""

    return f"""
#workflow {{ background-color: {p.COLOR_BACKGROUND_1}; border: none;
             border-right: 1px solid {p.COLOR_BACKGROUND_4}; outline: 0; }}
#workflow::item {{ padding: 10px 12px; border-radius: 4px; margin: 2px 4px; color: {p.COLOR_TEXT_1}; }}
#workflow::item:selected {{ background-color: {p.COLOR_ACCENT_2}; color: {p.COLOR_TEXT_1}; }}
#workflow::item:hover:!selected {{ background-color: {p.COLOR_BACKGROUND_2}; }}
#workflow::item:disabled {{ color: {p.COLOR_DISABLED}; }}
"""


def palette():
    return DarkPalette if current == "dark" else LightPalette


def is_dark():
    return current == "dark"


def set_theme(theme, app=None):
    """Apply a plot theme and the matching application style sheet."""

    global current
    if theme not in PLOT_THEMES:
        theme = "default"
    changed = theme != current
    current = theme
    app = app or QApplication.instance()
    pv.set_plot_theme(theme)
    plt.style.use(MPL_STYLE[theme])
    if app is not None:
        p = palette()
        app.setStyleSheet(qdarkstyle.load_stylesheet(palette=p) + _sidebar(p))
        app.setProperty("ui_dark", theme == "dark")
        _refresh_icons(app)
    if changed:
        notifier.changed.emit(theme)


def icon(name, **kwargs):
    """Font Awesome 6 (solid) icon in the theme's text color."""

    p = palette()
    kwargs.setdefault("color", p.COLOR_TEXT_1)
    kwargs.setdefault("color_disabled", p.COLOR_DISABLED)
    return qta.icon(f"fa6s.{name}", **kwargs)


def set_icon(widget, name):
    """Give a button (or action) a themed icon that follows theme changes."""

    widget.setProperty("fa_icon", name)
    widget.setIcon(icon(name))


def _refresh_icons(app):
    for w in app.allWidgets():
        name = w.property("fa_icon")
        if name:
            w.setIcon(icon(name))
        names = w.property("fa_items")
        if names:
            for i, n in enumerate(names):
                if i < w.count():
                    w.setItemIcon(i, icon(n))


def apply(app):
    """Theme a new QApplication: dark if the system palette is dark."""

    dark = app.palette().color(QPalette.Window).lightness() < 128
    set_theme("dark" if dark else "default", app)
