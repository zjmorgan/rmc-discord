"""Cython extensions of the legacy ``disorder`` package.

Package metadata, dependencies, and versioning are in ``pyproject.toml``.
"""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

if sys.platform == 'win32':
    compile_openmp = ['/openmp']
    link_openmp = []
elif sys.platform == 'darwin':
    compile_openmp = ['-Xpreprocessor', '-fopenmp', '-lomp,',
                      '-I"$(brew --prefix libomp)/include"',
                      '-L"$(brew --prefix libomp)/lib"']
    link_openmp = ['-lomp']
else:
    compile_openmp = ['-fopenmp']
    link_openmp = ['-fopenmp']

modules = {
    'disorder.diffuse.refinement': 'disorder/diffuse/refinement.pyx',
    'disorder.diffuse.filters': 'disorder/diffuse/filters.pyx',
    'disorder.diffuse.powder': 'disorder/diffuse/powder.pyx',
    'disorder.correlation.functions': 'disorder/correlation/functions.pyx',
    'disorder.diffuse.monocrystal': 'disorder/diffuse/monocrystal.pyx',
    'disorder.diffuse.simulation': 'disorder/diffuse/simulation.pyx',
}

extensions = [
    Extension(
        name=name,
        sources=[source],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=[np.get_include()],
        language='c++' if name == 'disorder.diffuse.simulation' else None,
    )
    for name, source in modules.items()
]

setup(ext_modules=cythonize(extensions))
