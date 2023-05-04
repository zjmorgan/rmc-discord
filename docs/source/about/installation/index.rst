Installation
============

Download a copy of the source directory from the [repository](https://github.com/zjmorgan/rmc-discord).

```bash
git clone https://github.com/zjmorgan/rmc-discord.git
```

+++++
Conda
+++++

Create a conda environment for rmc-discord.

``conda create --name rmc-discord``

Activate the rmc-discord environment.

``conda activate rmc-discord``

Next, install all dependencies by navigating to the cloned rmc-discord repository and updating with `environment.yml`.

``conda env update --file environment.yml --name rmc-discord``

After installing all dependencies, the repository is built with `setup.py`.

``python setup.py build_ext --inplace``

The package can then be installed into the Python environment.

``pip install -e .``
