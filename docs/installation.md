# **Installation**

The recommended method for installing *rmc-discord* is to utilize the Python Package Index (PyPI).

```bash
pip install rmc-discord
```

This will build the package and make the main executable available. Alternatively, the program can be build directly from source.

## **Build from source**

Download a copy of the source directory from the [repository](https://github.com/zjmorgan/rmc-discord).

```bash
git clone git@github.com:zjmorgan/rmc-discord.git
```

After installing all dependencies, the repository is built with `setup.py`.

```bash
python3 setup.py build_ext --inplace
```

The required dependencies are listed below:

- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Cython](https://cython.org/)
- [PyCifRW](https://www.iucr.org/resources/cif/software/pycifrw)
- [nexusformat](https://github.com/nexpy/nexusformat)
- [PyVista](https://www.pyvista.org/)
