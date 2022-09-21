# **Honeycomb -- displacive**

The honeycomb lattice is another example that may be geometrically frustrated. Using the forward Monte Carlo method, the Hamiltonian

\\[E=K\sum_{\langle i,j\rangle}\big(\|\pmb{u}\_i-\pmb{u}\_j\|-d_{ij}\big)^2\\]

is used to describe the nearest neighbor interactions where the spring constant is great than zero ($$K>0$$) and equilibrium distance parameter is $$d_{ij}=-2\|\pmb{u}\|$$. Displacement of each atom is fixed with magnitude $$u$$ and are free to be oriented in any direction.

<p align="center">
<img src="honeycomb-displacive-visualize.svg" alt="honeycomb-displacive" width="512">
<br />
Example of honeycomb lattice with displacive disorder
</p>

The displacement-pair correlations of an $$12\times12\times12$$ supercell are obtained from a forward Monte Carlo simulation. The first nearest neighbors are clearly negatively correlated while the second are positively correlated. In the case of third nearest neighbors, the correlation is also negative.

<p align="center">
<img src="honeycomb-displacement-correlations.svg" alt="honeycomb-displacive correlations" width="512">
<br />
Displacement-pair correlations
</p>

The diffuse scattering intensity is calculated over a range of 4 to 4 in each $$h$$-, $$k$$-, and $$l$$-direction with a bin size of 0.025 in each dimension. Averaging is done over 20 independent forward Monte Carlo simulations to improve the statistics.

<p align="center">
<img src="honeycomb-intensity.svg" alt="honeycomb-magnetic intensity" width="512">
<br />
Diffuse scattering intensity
</p>

## **RMC refinement**

Setup, run, and analyze a refinement with displacive disorder.

### **Crystal tab**

Create a supercell for nonmagnetic refinement.
* Click on *Load CIF file*, navigate to the `tutorials/honeycomb/` directory, and locate the `honeycomb.cif` file.
* Create a supercell with size $$N_1=8$$, $$N_2=8$$, and $$N_3=8$$.

<p align="center">
<img src="honeycomb-gui-crystal.png" alt="Honeycomb GUI crystal tab" width="640">
<br />
Crystal tab
</p>

### **Intensity tab**

Preprocess intensity obtained from forward Monte Carlo.
* Download the [`honeycomb-displacive.nxs`](http://dl.dropboxusercontent.com/s/2zeiw6r5t309eml/honeycomb-displacive.nxs?dl=0) file.
* Click on *Load NeXus file* and locate the `honeycomb-displacive.nxs` file.
* Under the *Rebin* tab, change the step size to *0.125* in each $$h$$-, $$k$$-, and $$l$$-direction.
* Under the *Crop* tab, change the $$h$$-, $$k$$-, and $$l$$-range from *0* to *4*.

<p align="center">
<img src="honeycomb-gui-intensity.png" alt="Honeycomb GUI intensity tab" width="640">
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
<img src="honeycomb-gui-refinement.png" alt="Honeycomb GUI refinement tab" width="640">
<br />
Refinement tab
</p>

### **Correlations tab**

Calculate three-dimensional displacement-pair correlations.
* Under the *Three-dimensional* tab, change *Fraction* to *0.16*.
* Click *Calculate*.
* Change *Linear* scaling to *Logarithmic*.

<p align="center">
<img src="honeycomb-gui-correlations.png" alt="Honeycomb GUI correlations tab" width="640">
<br />
Correlations tab
</p>

### **Recalculation tab**

Recalculate the intensity over the initial reciprocal space volume.
* In the table, change $h$, $k$ and $l$ filter size to *4.0*.
* Change the *Laue* symmetry to *cif* which is inferred from the loaded `honeycomb.cif` file.
* Click *Calculate*.

<p align="center">
<img src="honeycomb-gui-recalculation.png" alt="Honeycomb GUI recalculation tab" width="640">
<br />
Recalculation tab
</p>

* Navigate to *File->Save* to save all results into the refinement file.
