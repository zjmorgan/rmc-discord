# **rmc-discord**

*R*everse *M*onte *C*arlo refinement of *di*ffuse *sc*attering and
*cor*related *d*isorder from single crystals

![](https://github.com/zjmorgan/rmc-discord/blob/master/docs/banner.svg)

[*rmc-discord*](https://zjmorgan.github.io/rmc-discord/) analyzes single crystal diffuse scattering from disordered materials and extracts short-range magnetic, occupational, and displacive correlations.

Pair correlations are refined directly: the diffuse intensity is linear in symmetry-reduced pair correlations, which are fit by constrained least squares against the measured bins, with exact bin averaging, a smooth background, and held-out validation. There is no supercell and no Monte Carlo. The channels are:
- occupancy covariances, with within-cluster values, sum rules and rigid units fixed by the average structure
- size effect (occupation-displacement) and displacement-displacement correlations
- magnetic correlations, isotropic or as symmetric spin tensors, including mixed magnetic sites

There are two required inputs:
- CIF file of the average structure
- NeXus file of the scattering data

There are several outputs:
- HDF5 table of the refined pair correlations with uncertainties
- CSV file of correlation shells
- three-dimensional maps of the correlations and the recalculated diffuse scattering

The program is written in Python and includes a Qt GUI. The earlier reverse Monte Carlo engine (`rmc_discord.engine`) is kept in the repository but no longer used by the GUI; its example scripts are in `examples/legacy_rmc/`.

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
