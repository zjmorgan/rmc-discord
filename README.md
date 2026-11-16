# **rmc-discord**

*R*everse *M*onte *C*arlo refinement of *di*ffuse *sc*attering and
*cor*related *d*isorder from single crystals

![](https://github.com/zjmorgan/rmc-discord/blob/master/docs/banner.svg)

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

### Citation

If you use rmc-discord in your work, please cite:

Z. J. Morgan, H. D. Zhou, B. C. Chakoumakos, and F. Ye, "*rmc-discord*: reverse Monte Carlo refinement of diffuse scattering and correlated disorder from single crystals," *J. Appl. Cryst.* **54**(6), 1867-1885 (2021). [https://doi.org/10.1107/S1600576721010141](https://doi.org/10.1107/S1600576721010141)

```bibtex
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
```
