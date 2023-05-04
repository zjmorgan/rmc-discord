.. rmc-discord documentation master file

===========
rmc-discord
===========

Reverse Monte Carlo refinement of diffuse scattering and correlated disorder from single crystals.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   modules/index
   classes/index
   about/index

Disorder
========

The `disorder` package contains all of the tools and utilities for rmc-discord.
It is split into various modules each with functions and classes required to perform a reverse Monte Carlo refinement.

+++++++++++
Conventions
+++++++++++

In dealing with crystallographic and scattering data, several conventions are chosen out of convenience.
In crystal coordinates, the real space vector is given by

.. math:: \boldsymbol{r}=u\boldsymbol{a}+v\boldsymbol{b}+w\boldsymbol{c}

whereas the reciprocal lattice vector is

.. math:: \frac{\boldsymbol{G}}{2\pi}=h\boldsymbol{a^\ast}+h\boldsymbol{b^\ast}+k\boldsymbol{c^\ast}.

Cartesian axes are chosen such that the first crystal axis coincides with the first orthonormal axis.
The second orthogonal axis is in the plane of the first two crystal axes.
Finally, the third forms a right-hand perpendicular triplet.
The symbols :math:`x`, :math:`y`, and :math:`z` are reserved for the Cartesian system.
