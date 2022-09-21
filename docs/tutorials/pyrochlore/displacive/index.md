# **Displacement ice**

Analogous to spin ice, atomic displacements along the vertices of the tetrahedra of a pyrochlore lattice can also be frustrated. Using the forward Monte Carlo method, the Hamiltonian

\\[E=K\sum_{\langle i,j\rangle}(\pmb{u}\_i-\pmb{u}\_j)^2\\]

is used to describe the nearest neighbor interactions where the spring constant is great than zero ($$K>0$$). The displacement vectors $$\pmb{u}$$ of the $$\mathrm{Ho}$$ atoms are restricted to allow atomic displacements either in or out of the vertex of each tetrahedra which corresponds to the $$\langle111\rangle$$ family of directions.

<p align="center">
<img src="standard-cubic.svg" alt="Standard-cubic" width="64" style="vertical-align:bottom">
<img src="pyrochlore-dis.png" alt="Pyrochlore-displacive" width="256">
<br />
Example of pyrochlore lattice with displacive disorder
</p>

The displacement-pair correlations of an $$8\times8\times8$$ supercell are obtained from a forward Monte Carlo simulation. The first nearest neighbors are clearly positive while the second are negative.

<p align="center">
<img src="pyrochlore-displacement-correlations.svg" alt="Pyrochlore-displacement correlations" width="512">
<br />
Displacement-pair correlations
</p>

The diffuse scattering intensity is calculated over a range of -6 to 6 in each $$h$$-, $$k$$-, and $$l$$-direction with a bin size of 0.04 in each dimension. Averaging is done over 20 independent forward Monte Carlo simulations to improve the statistics. A slice in the $$(h,h,l)$$-plane is also calculated from -4 to 4 along $$(h,h,0)$$ and -6 to 6 along $$(0,0,l)$$.

<p align="center">
<img src="pyrochlore-intensity-dis.svg" alt="Pyrochlore-displacement intensity" width="512">
<br />
Diffuse scattering intensity
</p>

## **RMC refinement**

Setup, run, and analyze a refinement with displacive disorder.

### **Crystal tab**

Create a supercell for magnetic refinement.
* Click on *Load CIF file*, navigate to the `tutorials/pyrochlore/` directory, and locate the `pyrochlore.cif` file.
* Create a supercell with size $$N_1=4$$, $$N_2=4$$, and $$N_3=4$$.

<p align="center">
<img src="pyrochlore-displacive-gui-crystal.png" alt="Pyrochlore GUI crystal tab" width="640">
<br />
Crystal tab
</p>

### **Intensity tab**

Preprocess intensity obtained from forward Monte Carlo.
* Download the [`pyrochlore-displacive.nxs`](http://dl.dropboxusercontent.com/s/waons3ako68ywra/pyrochlore-displacive.nxs?dl=00) file.
* Click on *Load NeXus file* and locate the `pyrochlore-displacive.nxs` file.
* Under the *Rebin* tab, change the step size to *0.2* in each $$h$$-, $$k$$-, and $$l$$-direction.
* Under the *Crop* tab, change the $$h$$-, $$k$$-, and $$l$$-range from *0* to *6*.

<p align="center">
<img src="pyrochlore-displacive-gui-intensity.png" alt="Pyrochlore GUI intensity tab" width="640">
<br />
Intensity tab
</p>

### **Refinement tab**

Perform the refinement.
* Change the number of RMC *Cycles* to *100*.
* Change the filter size to *1.0* pixel in each $$h$$-, $$k$$-, and $$l$$-direction.
* Change the temperature prefactor to *1.00e+03* and decay constant to *1.00e-04*.
* Click on run and save refinement file.

<p align="center">
<img src="pyrochlore-displacive-gui-refinement.png" alt="Pyrochlore GUI refinement tab" width="640">
<br />
Refinement tab
</p>

### **Correlations tab**

Calculate three-dimensional displacement-pair correlations.
* Under the *Three-dimensional* tab, change $h$, $k$ and $l$ to *1*.
* Change *Fraction* to *0.2*.
* Click *Calculate*.
* Change *Linear* scaling to *Logarithmic*.

<p align="center">
<img src="pyrochlore-displacive-gui-correlations.png" alt="Pyrochlore GUI correlations tab" width="640">
<br />
Correlations tab
</p>

### **Recalculation tab**

Recalculate the intensity over new axes.
* In the table, change $h$ and $k$ size to *201* and $l$ size to *301*.
* Change $h$ and $k$ min to *-4* and $l$ min to *-6*.
* Change $h$ and $k$ max to *4* and $l$ max to *6*.
* Change $h$, $k$ and $l$ filter size to *3.0*.
* Change the *Axes* to *(hh0), (-kk0), (00l)*.
* Change the *Laue* symmetry to *cif* which is inferred from the loaded `pyrochlore.cif` file.
* Click *Calculate*.

<p align="center">
<img src="pyrochlore-displacive-gui-recalculation.png" alt="Pyrochlore GUI recalculation tab" width="640">
<br />
Recalculation tab
</p>

* Navigate to *File->Save* to save all results into the refinement file.
