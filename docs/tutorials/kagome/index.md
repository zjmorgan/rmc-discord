# **Kagome -- magnetic**

The classical Heisenberg kagome antiferromagnet is a simple example of a magnetic system with a geometrically frustrated lattice. Using the forward Monte Carlo method, the Hamiltonian

\\[E=-J\sum_{\langle i,j\rangle}\pmb{S}_i\cdot\pmb{S}_j\\]

is used to describe the nearest neighbor antiferromagnetic interactions where the interaction strength is less than zero ($$J<0$$). The spin vectors $$\pmb{S}$$ of the $$\mathrm{Ho^{3+}}$$ ions are free to be oriented in any direction.

<p align="center">
<img src="kagome-magnetic-visualize.svg" alt="Kagome-magnetic" width="512">
<br />
Example of kagome lattice with magnetic disorder
</p>

The spin-pair correlations of an $$8\times8\times8$$ supercell are obtained from a forward Monte Carlo simulation. The first nearest neighbors are clearly antiferromagnetic while the second and third are ferromagnetic with the second being more strongly correlated than the third.

<p align="center">
<img src="kagome-spin-correlations.svg" alt="Kagome-magnetic correlations" width="512">
<br />
Spin-pair correlations
</p>

The diffuse scattering intensity is calculated over a range of -6 to 6 in each $$h$$-, $$k$$-, and $$l$$-direction with a bin size of 0.04 in each dimension. Averaging is done over 20 independent forward Monte Carlo simulations to improve the statistics.

<p align="center">
<img src="kagome-intensity.svg" alt="Kagome-magnetic intensity" width="512">
<br />
Diffuse scattering intensity
</p>

## **RMC refinement**

Setup, run, and analyze a refinement with magnetic disorder.

### **Crystal tab**

Create a supercell for magnetic refinement.
* Click on *Load CIF file*, navigate to the `tutorials/kagome/` directory, and locate the `kagome.cif` file.
* Change the refinement from *Neutron nonmagnetic* to *Neutron magnetic*.
* Create a supercell with size $$N_1=8$$, $$N_2=8$$, and $$N_3=8$$.
* Change the atom to a $$\mathrm{Ho3+}$$ ion.

<p align="center">
<img src="kagome-gui-crystal.png" alt="Kagome GUI crystal tab" width="640">
<br />
Crystal tab
</p>

### **Intensity tab**

Preprocess intensity obtained from forward Monte Carlo.
* Download the [`kagome-magnetic.nxs`](http://dl.dropboxusercontent.com/s/e61m0myhcca40x7/kagome-magnetic.nxs?dl=0) file.
* Click on *Load NeXus file* and locate the `kagome-magnetic.nxs` file.
* Under the *Rebin* tab, change the step size to *0.2* in each $$h$$-, $$k$$-, and $$l$$-direction.
* Under the *Crop* tab, change the $$h$$-, $$k$$-, and $$l$$-range from *0* to *6*.

<p align="center">
<img src="kagome-gui-intensity.png" alt="Kagome GUI intensity tab" width="640">
<br />
Intensity tab
</p>

### **Refinement tab**

Perform the refinement.
* Change the number of RMC *Cycles* to *100*.
* Change the filter size to *1.0* pixel in each $$h$$-, $$k$$-, and $$l$$-direction.
* Change the temperature prefactor to *1.00e+01* and decay constant to *1.00e-04*.
* Click on run and save refinement file.

<p align="center">
<img src="kagome-gui-refinement.png" alt="Kagome GUI refinement tab" width="640">
<br />
Refinement tab
</p>

### **Correlations tab**

Calculate three-dimensional spin-pair correlations.
* Under the *Three-dimensional* tab, change *Fraction* to *0.1* and *Tolerance* to *1e-3*.
* Click *Calculate*.
* Change *Linear* scaling to *Logarithmic*.

<p align="center">
<img src="kagome-gui-correlations.png" alt="Kagome GUI correlations tab" width="640">
<br />
Correlations tab
</p>

### **Recalculation tab**

Recalculate the intensity over the initial reciprocal space volume.
* In the table, change $h$, $k$ and $l$ filter size to *4.0*.
* Change the *Laue* symmetry to *cif* which is inferred from the loaded `kagome.cif` file.
* Click *Calculate*.

<p align="center">
<img src="kagome-gui-recalculation.png" alt="Kagome GUI recalculation tab" width="640">
<br />
Recalculation tab
</p>

* Navigate to *File->Save* to save all results into the refinement file.
