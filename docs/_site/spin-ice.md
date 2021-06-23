# **Spin ice**

The pyrochlore lattice decorated with magnetic moments and ferromagnetic nearest neighbor interactions is a simple example of a three-dimensional geometrically frustrated system. Using the forward Monte Carlo method, the Hamiltonian

\\[E=-J\sum_{\langle i,j\rangle}\pmb{S}_i\cdot\pmb{S}_j\\]

is used to describe the nearest neighbor ferromagnetic interactions where the interaction strength is greater than zero ($$J>0$$). The spin vectors $$\pmb{S}$$ of the $$\mathrm{Ho^{3+}}$$ ions are restricted to be oriented either in or out of the vertex of each tetrahedra which corresponds to the $$\langle111\rangle$$ family of directions.

<p align="center">
<img src="standard-cubic.svg" alt="Standard-cubic" width="64" style="vertical-align:bottom">
<img src="pyrochlore-mag.png" alt="Pyrochlore-magnetic" width="256">
<br />
Example of pyrochlore lattice with magnetic disorder
</p>

The spin-pair correlations of an $$8\times8\times8$$ supercell are obtained from a forward Monte Carlo simulation. The first nearest neighbors are clearly ferromagnetic while the second are antiferromagnetic.

<p align="center">
<img src="pyrochlore-spin-correlations.svg" alt="Pyrochlore-magnetic correlations" width="512">
<br />
Spin-pair correlations
</p>

The diffuse scattering intensity is calculated over a range of -6 to 6 in each $$h$$-, $$k$$-, and $$l$$-direction with a bin size of 0.04 in each dimension. Averaging is done over 20 independent forward Monte Carlo simulations to improve the statistics. A slice in the $$(h,h,l)$$-plane is also calculated from -4 to 4 in the $$[h,h,0]$$-direction -6 to 6 in the $$[0,0,l]$$-direction.

<p align="center">
<img src="pyrochlore-intensity-mag.svg" alt="Pyrochlore-magnetic intensity" width="512">
<br />
Diffuse scattering intensity
</p>
