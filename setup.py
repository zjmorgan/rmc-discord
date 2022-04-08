import setuptools
setuptools.dist.Distribution().fetch_build_eggs(['Cython>=0.15.1',
                                                 'numpy>=1.10'])

from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

import numpy as np

import sys, re

if (sys.platform == 'win32'):
    compile_openmp = ['/openmp']
    link_openmp = ['/openmp']
elif (sys.platform == 'darwin'):
    compile_openmp = ['-Xpreprocessor','-fopenmp','-lomp,',
                     '-I"$(brew --prefix libomp)/include"',
                     '-L"$(brew --prefix libomp)/lib"']
    link_openmp = ['-lomp']
else:
    compile_openmp = ['-fopenmp']
    link_openmp = ['-fopenmp']

np_include_dir = [np.get_include()]

with open('README.md', 'r') as fh:
    long_description = fh.read()

ver = open('disorder/version.py', "rt").read()
version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", ver, re.M).group(1)

ext = '.pyx' if USE_CYTHON else '.c'

ext_modules = [
    Extension(
        'disorder.diffuse.refinement',
        ['disorder/diffuse/refinement'+ext],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=np_include_dir
    ),
    Extension(
        'disorder.diffuse.filters',
        ['disorder/diffuse/filters'+ext],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=np_include_dir
    ),
    Extension(
        'disorder.diffuse.powder',
        ['disorder/diffuse/powder'+ext],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=np_include_dir
    ),
    Extension(
        'disorder.diffuse.refinement',
        ['disorder/diffuse/refinement'+ext],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=np_include_dir
    ),
    Extension(
        'disorder.correlation.functions',
        ['disorder/correlation/functions'+ext],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=np_include_dir
    ),
    Extension(
        'disorder.diffuse.monocrystal',
        ['disorder/diffuse/monocrystal'+ext],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=np_include_dir
    ),
    Extension(
        'disorder.diffuse.simulation',
        ['disorder/diffuse/simulation'+ext],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=np_include_dir,
        language='c++'
    )
]

if (USE_CYTHON):
    cmdclass = {'build_ext': build_ext}
else:
    cmdclass = { }

setuptools.setup(
    name='rmc-discord',
    version=version,
    author='Zachary Morgan',
    author_email='morganzj@ornl.gov',
    description='Reverse Monte Carlo refinement of diffuse scattering and '+\
                'correlated disorder from single crystals',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/zjmorgan/rmc-discord',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pycifrw',
        'nexusformat',
        'pyvista',
        'pyqt5',
        'ipython'
    ],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    entry_points={
        'console_scripts': [
            'rmc-discord=disorder.application:run',
        ],
    },
    package_data={
        'disorder': ['material/*.csv',
                     'graphical/*.ui',
                     'graphical/*.png',
                     'diffuse/*.pxd',
                     'diffuse/*.pyx',
                     'material/*.pxd',
                     'material/*.pyx',
                     'tests/diffuse/*.pyx',
                     'tests/data/*'],
    },
    zip_safe=False
)
