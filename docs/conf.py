# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

# -- Project information -------------------------------------------------

project = "rmc-discord"
copyright = "Zachary Morgan"
author = "Zachary Morgan"

# -- General configuration -----------------------------------------------

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]
root_doc = "index"

# Mantid is installed separately (conda); its readers import it lazily, but
# mock it so the API pages build without it.
autodoc_mock_imports = ["mantid"]
autodoc_member_order = "bysource"
add_module_names = False
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output ---------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "rmc-discord"
html_permalinks_icon = "#"
html_show_sourcelink = False
html_copy_source = True
html_static_path = ["_static"]
html_favicon = "_static/icon.svg"

html_theme_options = {
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/zjmorgan/rmc-discord",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "external_links": [
        {
            "url": "https://doi.org/10.1107/S1600576721010141",
            "name": "Paper",
        },
        {
            "url": "https://neutrons.github.io/NeuXtalViz-tools/",
            "name": "NeuXtalViz",
        },
    ],
    "show_nav_level": 2,
}

# -- Source links ----------------------------------------------------------


def linkcode_resolve(domain, info):
    baseurl = "https://github.com/zjmorgan/rmc-discord/blob/master/src/{}.py"
    if domain != "py" or not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    try:
        mod = importlib.import_module(info["module"])
    except Exception:
        return None
    if os.path.basename(getattr(mod, "__file__", "")) == "__init__.py":
        filename += "/__init__"
    url = baseurl.format(filename)
    obj = mod
    for attr in info["fullname"].split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return url
    try:
        lines, start = inspect.getsourcelines(inspect.unwrap(obj))
    except (TypeError, OSError):
        return url
    return f"{url}#L{start}-L{start + len(lines) - 1}"


def skip_qt_members(app, what, name, obj, skip, options):
    try:
        from qtpy.QtCore import QMetaObject, Signal
    except ImportError:
        return skip
    if isinstance(obj, (Signal, QMetaObject)):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_qt_members)
