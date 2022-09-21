# **Performance**

The program is designed to take advantage of multi-threaded parallelism. Taking the pyrochlore examples, a speedup curve is generated for each of the three disorder types.

<p align="center">
<img src="speedup.svg" alt="Speedup curve for the three disorder types" width="512">
</p>

Fitting each data set to the theoretical speedup curves gives a metric of the parallel performance.

* *Magnetic* 86.5%
* *Occupational* 80.1%
* *Displacive* 88.1%

This metric indicates the percentage of the refinement the benefits from the parallelism.
