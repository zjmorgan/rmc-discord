rmc-discord documentation
=========================

*discord* (noun): a lack of harmony or agreement.

rmc-discord is a reverse Monte Carlo (RMC) program for refining
correlated disorder in single crystals against their diffuse scattering.
It provides:

- A graphical user interface (GUI) for the whole workflow: average
  structure, data, refinement, correlations, and recalculation
- Occupational, displacive, and magnetic disorder, alone or combined
- Refinement against neutron or X-ray volumes on a supercell's
  commensurate points, with minibatches and held-out validation
- Pair correlations and recalculated diffuse scattering
- Step-by-step tutorials

Core libraries and technologies:

- Python 3
- NumPy, SciPy, and Numba for scientific computation
- Mantid for reading reduced neutron data
- qtpy for the cross-platform GUI, styled as in NeuXtalViz
- PyVista for 3D visualization
- Matplotlib for plotting
- Sphinx for documentation

Citation
--------

If you use rmc-discord in your work, please cite:

Z. J. Morgan, H. D. Zhou, B. C. Chakoumakos, and F. Ye, "*rmc-discord*:
reverse Monte Carlo refinement of diffuse scattering and correlated
disorder from single crystals," *J. Appl. Cryst.* **54**\ (6), 1867-1885
(2021). https://doi.org/10.1107/S1600576721010141

.. code-block:: bibtex

   @article{morgan2021rmcdiscord,
     author  = {Morgan, Zachary J. and Zhou, Haidong D. and Chakoumakos, Bryan C. and Ye, Feng},
     title   = {{rmc-discord}: reverse {Monte Carlo} refinement of diffuse scattering and correlated disorder from single crystals},
     journal = {Journal of Applied Crystallography},
     volume  = {54},
     number  = {6},
     pages   = {1867--1885},
     year    = {2021},
     doi     = {10.1107/S1600576721010141}
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/installation
   source/rmc_discord.tutorials
   source/background
   source/modules
