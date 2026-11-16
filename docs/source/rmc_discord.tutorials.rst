rmc-discord tutorials
=====================

Each tutorial goes through the five applications of the interface in
order, listed in the sidebar of the main window:

1. **Structure**: load the average structure from a CIF, choose the
   disordered sites, and build the supercell.
2. **Data**: load the reduced diffuse scattering volume and sample it at
   the supercell's commensurate points.
3. **Refinement**: choose the disorder and moves, hold out points for
   validation, and anneal.
4. **Correlations**: compute pair correlations of the refined
   configuration.
5. **Recalculation**: recalculate a plane of the diffuse scattering from
   the correlations and compare it with the data.

An application becomes available once the ones it needs are done; a check
mark shows the finished ones. The settings match the scripts in
``examples/``, which do the same refinements without the interface.

.. toctree::
   :maxdepth: 2
   :caption: CORELLI Tutorials:

   tutorial_magnetic_CORELLI_Bixbyite
   tutorial_occupational_CORELLI_Carbonate
