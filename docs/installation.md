# **Installation**

The basic requirement is to have a Python environment with version >= 3.6. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a good choice if one is not available.

Current supported operating systems:
- Ubuntu 20.04
- Windows 2019
- MacOS 10.15

The recommended method for installing *rmc-discord* is to utilize the Python Package Index [(PyPI)](https://pypi.org/project/rmc-discord/).

```bash
pip install rmc-discord
```

This will build the package and make the main executable `rmc-discord` available. IPython may also need to be installed via `pip install ipython`.

### **Windows installation**
Encountering the error `error: Microsoft Visual C++ 14.0 is required` when building on Windows with Anaconda or Miniconda may require [Visual Studio](https://visualstudio.microsoft.com/downloads/).

1.	Navigate to [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/) and download the *Community* version.
2.	Open *Visual Studio Installer*, click *Modify* on the installed version, check *Desktop development with C++*, and install the default tools shown by clicking *Modify*.
3.	Relaunch Anaconda or Miniconda and try `pip install rmc-discord` again.

Alternatively, the program can be build directly from source.

## **Build from source**

Download a copy of the source directory from the [repository](https://github.com/zjmorgan/rmc-discord).

```bash
git clone https://github.com/zjmorgan/rmc-discord.git
```

After installing all dependencies, the repository is built with `setup.py`.

```bash
python setup.py build_ext --inplace
```

The package can then be installed into the Python environment.

```bash
pip install -e .
```

The required dependencies are listed below:

- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Cython](https://cython.org/)
- [PyCifRW](https://www.iucr.org/resources/cif/software/pycifrw)
- [nexusformat](https://github.com/nexpy/nexusformat)
- [PyVista](https://www.pyvista.org/)

The tests can also be optionally run.

```bash
python -m unittest
```

## **Running the program**

The program utilizes [OpenMP](https://www.openmp.org/) parallelism and simply typing `rmc-discord` in the terminal will launch the program with up to 8 threads.

The number of threads can also be manually adjusted by passing in an integer argument to the program.

```bash
rmc-discord 4
```

Specifying too many threads may lead to unexpected slowdowns.
