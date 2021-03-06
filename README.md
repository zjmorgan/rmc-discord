# **rmc-discord**

*R*everse *M*onte *C*arlo refinement of *di*ffuse *sc*attering and
*cor*related *d*isorder from single crystals

<p align="center">
<img src="docs/banner.svg" alt="Disorder" width="512">
<br />
</p>

[*rmc-discord*](https://zjmorgan.github.io/rmc-discord/) is an atomistic program for the analysis of single crystal diffuse scattering from disordered materials capable of extracting short-range magnetic, occupational, and displacive correlations.

There are two required inputs:
- CIF file of the average structure
- NeXus file of the scattering data

There are several outputs:
- CIF file of disordered structure
- CSV file of spherically averaged pair correlations
- VTK file of three-dimensional pair correlations
- VTK file of recalculated diffuse scattering pattern

The program is written in Python with underlying performance-critical extensions written in Cython and includes a PyQt5 GUI.
