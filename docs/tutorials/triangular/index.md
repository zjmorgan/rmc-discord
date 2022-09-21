# **Triangular -- occupational**

The primitive triangular lattice is perhaps the simplest example of a geometrically frustrated lattice. Using the forward Monte Carlo method, the Hamiltonian

\\[E=-J\sum_{\langle i,j\rangle}\sigma_i\sigma_j\\]

is used to describe the nearest neighbor interactions where the interaction strength is less than zero ($$J<0$$). The occupancy parameter $$\sigma$$ indicates either the presence of an atom ($$\sigma=+1$$) or vacancy ($$\sigma=-1$$).

<p align="center">
<img src="triangular-occupational-visualize.svg" alt="Triangular-occupational" width="512">
<br />
Example of triangular lattice with occupational disorder
</p>

The occupancy-pair correlations of an $$16\times16\times16$$ supercell are obtained from a forward Monte Carlo simulation. The first nearest neighbors are clearly negatively correlated while the second are positively correlated. In the case of third nearest neighbors, the correlation is also negative.

<p align="center">
<img src="triangular-occupancy-correlations.svg" alt="Triangular-occupational correlations" width="512">
<br />
Occupancy-pair correlations
</p>

The diffuse scattering intensity is calculated over a range of -2 to 2 in each $$h$$-, $$k$$-, and $$l$$-direction with a bin size of 0.02 in each dimension. Averaging is done over 20 independent forward Monte Carlo simulations to improve the statistics.

<p align="center">
<img src="triangular-intensity.svg" alt="Triangular-magnetic intensity" width="512">
<br />
Diffuse scattering intensity
</p>

## **RMC refinement**

Setup, run, and analyze a refinement with occupational disorder.

### **Crystal tab**

Create a supercell for nonmagnetic refinement.
* Click on *Load CIF file*, navigate to the `tutorials/triangular/` directory, and locate the `triangular.cif` file.
* Create a supercell with size $$N_1=8$$, $$N_2=8$$, and $$N_3=8$$.

<p align="center">
<img src="triangular-gui-crystal.png" alt="Triangular GUI crystal tab" width="640">
<br />
Crystal tab
</p>

### **Intensity tab**

Preprocess intensity obtained from forward Monte Carlo.
* Download the [`triangular-occupational.nxs`](http://dl.dropboxusercontent.com/s/pezwdptghinjslj/triangular-occupational.nxs?dl=0) file.
* Click on *Load NeXus file* and locate the `triangular-occupational.nxs` file.
* Under the *Rebin* tab, uncheck the integer box and change the step size to *0.08* in each $$h$$-, $$k$$-, and $$l$$-direction.
* Under the *Crop* tab, change the $$h$$-, $$k$$-, and $$l$$-range from *0* to *2*.

<p align="center">
<img src="triangular-gui-intensity.png" alt="Triangular GUI intensity tab" width="640">
<br />
Intensity tab
</p>

### **Refinement tab**

Perform the refinement.
* Change the number of RMC *Cycles* to *100*.
* Change the filter size to *1.0* pixel in each $$h$$-, $$k$$-, and $$l$$-direction.
* Change the temperature prefactor to *1.00e+05* and decay constant to *1.00e-04*.
* Click on run and save refinement file.

<p align="center">
<img src="triangular-gui-refinement.png" alt="Triangular GUI refinement tab" width="640">
<br />
Refinement tab
</p>

### **Correlations tab**

Calculate three-dimensional occupancy-pair correlations.
* Under the *Three-dimensional* tab, change *Fraction* to *0.22*.
* Click *Calculate*.
* Change *Linear* scaling to *Logarithmic*.

<p align="center">
<img src="triangular-gui-correlations.png" alt="Triangular GUI correlations tab" width="640">
<br />
Correlations tab
</p>

### **Recalculation tab**

Recalculate the intensity over the initial reciprocal space volume.
* In the table, change $h$, $k$ and $l$ filter size to *4.0*.
* Change the *Laue* symmetry to *cif* which is inferred from the loaded `triangular.cif` file.
* Click *Calculate*.

<p align="center">
<img src="triangular-gui-recalculation.png" alt="Triangular GUI recalculation tab" width="640">
<br />
Recalculation tab
</p>

* Navigate to *File->Save* to save all results into the refinement file.
