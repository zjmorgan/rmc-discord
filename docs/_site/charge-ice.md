# **Charge ice**

Metastable cubic water ice phase $$\mathrm{I_c}$$ exhibits disorder of hydrogen protons such that only two sites are occupied and two sites are unoccupied on each tetrahedra. Using the forward Monte Carlo method, the Hamiltonian

\\[E=-J\sum_{\langle i,j\rangle}\sigma_i\sigma_j\\]

is used to describe the nearest neighbor interactions where the interaction strength is less than zero ($$J<0$$). The occupancy parameter $$\sigma$$ indicates either the presence of a hydrogen atom ($$\sigma=+1$$) or vacancy ($$\sigma=-1$$).

<p align="center">
<img src="standard-cubic.svg" alt="Standard-cubic" width="64" style="vertical-align:bottom">
<img src="pyrochlore-occ.png" alt="Pyrochlore-occupational" width="256">
<br />
Example of pyrochlore lattice with occupational disorder
</p>

The occupancy-pair correlations of an $$8\times8\times8$$ supercell are obtained from a forward Monte Carlo simulation. The first and second nearest neighbors are clearly negative.

<p align="center">
<img src="pyrochlore-occupancy-correlations.svg" alt="Pyrochlore-occupancy correlations" width="512">
<br />
Occupancy-pair correlations
</p>

The diffuse scattering intensity is calculated over a range of -6 to 6 in each $$h$$-, $$k$$-, and $$l$$-direction with a bin size of 0.04 in each dimension. Averaging is done over 20 independent forward Monte Carlo simulations to improve the statistics. A slice in the $$(h,h,l)$$-plane is also calculated from -4 to 4 in the $$[h,h,0]$$-direction -6 to 6 in the $$[0,0,l]$$-direction.

<p align="center">
<img src="pyrochlore-intensity-occ.svg" alt="Pyrochlore-occupancy intensity" width="512">
<br />
Diffuse scattering intensity
</p>
