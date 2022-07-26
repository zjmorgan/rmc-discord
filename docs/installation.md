# **Installation**

The basic requirement is to have a Python environment with version >= 3.6. It is best to use a dedicated Python environment to install required dependencies. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a good choice if one is not available.

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

Encountering the error `ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject` may require upgrading NumPy. Type `pip install numpy -U` to resolve the issue.

Alternatively, the program can be build directly from source.

## **Build from source**

Download a copy of the source directory from the [repository](https://github.com/zjmorgan/rmc-discord).

```bash
git clone https://github.com/zjmorgan/rmc-discord.git
```

Create a conda environment for rmc-discord.

```bash
 conda create --name rmc-discord
```

Activate the rmc-discord environment.

```bash
conda activate rmc-discord
```

Next, install all dependencies by navigating to the cloned rmc-discord repository and updating with `environment.yml`.

```bash
conda env update --file environment.yml --name rmc-discord
```

After installing all dependencies, the repository is built with `setup.py`.

```bash
python setup.py build_ext --inplace
```

The package can then be installed into the Python environment.

```bash
pip install -e .
```

## **Running the program**

The program utilizes [OpenMP](https://www.openmp.org/) parallelism and simply typing `rmc-discord` in the terminal will launch the program with up to 8 threads.

The number of threads can also be manually adjusted by passing in an integer argument to the program.

```bash
rmc-discord 4
```

Specifying too many threads may lead to unexpected slowdowns.

## **Running the tests**

The tests can also be optionally run using `pytest`.

```bash
python -m pytest --pyargs disorder
```
