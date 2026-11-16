Installation
============

rmc-discord needs Python 3.11 or newer. A dedicated conda environment
from `conda-forge <https://conda-forge.org/>`_ is the easiest route, since
reading reduced neutron data (MDHistoWorkspace NeXus files) needs
`Mantid <https://www.mantidproject.org/>`_, which is installed with conda.

Conda environment
-----------------

Create and activate an environment with Mantid:

.. code-block:: bash

   conda create -n rmc-discord -c mantid -c conda-forge python=3.11 mantid
   conda activate rmc-discord

Get the source and install it with the graphical interface:

.. code-block:: bash

   git clone https://github.com/zjmorgan/rmc-discord.git
   cd rmc-discord
   pip install -e ".[gui]"

The ``gui`` extra adds qtpy, pyvistaqt, QDarkStyle, and QtAwesome; the
rest (NumPy, SciPy, Numba, h5py, PyCifRW, Matplotlib, PyVista) are core
dependencies.

Running the program
-------------------

Launch the graphical interface with

.. code-block:: bash

   rmc-discord-gui

or ``python -m rmc_discord.gui``. The refinement kernels are compiled by
Numba the first time they run and are parallel; the number of threads is
set by ``NUMBA_NUM_THREADS``, e.g.

.. code-block:: bash

   NUMBA_NUM_THREADS=16 rmc-discord-gui

The interface follows the system's light or dark appearance; the plot theme
menu in each view switches between the default, document, dark, and
ParaView looks.

.. note::

   Under Windows Subsystem for Linux, Mesa renders OpenGL in software by
   default, which makes large supercells slow to rotate. rmc-discord then
   uses Mesa's ``d3d12`` driver, which renders on the Windows GPU
   (preferring an NVIDIA adapter). Set ``GALLIUM_DRIVER=llvmpipe`` to force
   software rendering.

Scripts
-------

Everything the interface does is available from Python. The scripts in
``examples/`` refine the two tutorial datasets without the interface, e.g.

.. code-block:: bash

   DIFFUSE_DATA_ROOT=/path/to/data python examples/bixbyite.py

Running the tests
-----------------

.. code-block:: bash

   pip install -e ".[gui,test]"
   python -m pytest

The interface tests run on Qt's offscreen platform and are skipped if the
``gui`` extra is missing.

Building this documentation
---------------------------

.. code-block:: bash

   conda install -c conda-forge sphinx pydata-sphinx-theme numpydoc
   sphinx-apidoc -o docs/source src/rmc_discord
   sphinx-build -b html docs docs/_build/html

The tutorial screenshots are made by driving the interface through each
tutorial (data under ``$DIFFUSE_DATA_ROOT``):

.. code-block:: bash

   python tests/applications/bixbyite.py
   python tests/applications/carbonate.py

The windows are drawn on Qt's offscreen platform, so no display is needed;
the PNGs are written to ``docs/source/``.
