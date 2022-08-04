from setuptools import setup, find_packages, Extension
from distutils.command.build import build as build_orig

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

class build(build_orig):

    def finalize_options(self):
        super().finalize_options()
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        for extension in self.distribution.ext_modules:
            extension.include_dirs.append(numpy.get_include())
        from Cython.Build import cythonize
        self.distribution.ext_modules = cythonize(
                                            self.distribution.ext_modules,
                                            language_level=3
                                        )

with open('README.md', 'r') as fh:
    long_description = fh.read()

ver = open('disorder/version.py', "rt").read()
version = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", ver, re.M).group(1)

extensions = [
    Extension(
        name='disorder.diffuse.refinement',
        sources=['disorder/diffuse/refinement.pyx'],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=[]
    ),
    Extension(
        name='disorder.diffuse.filters',
        sources=['disorder/diffuse/filters.pyx'],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=[]
    ),
    Extension(
        name='disorder.diffuse.powder',
        sources=['disorder/diffuse/powder.pyx'],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=[]
    ),
    Extension(
        'disorder.diffuse.refinement',
        sources=['disorder/diffuse/refinement.pyx'],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=[]
    ),
    Extension(
        name='disorder.correlation.functions',
        sources=['disorder/correlation/functions.pyx'],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=[]
    ),
    Extension(
        name='disorder.diffuse.monocrystal',
        sources=['disorder/diffuse/monocrystal.pyx'],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=[]
    ),
    Extension(
        name='disorder.diffuse.simulation',
        sources=['disorder/diffuse/simulation.pyx'],
        extra_compile_args=compile_openmp,
        extra_link_args=link_openmp,
        include_dirs=[],
        language='c++'
    )
]

setup(
    name='rmc-discord',
    version=version,
    author='Zachary Morgan',
    author_email='morganzj@ornl.gov',
    description='Reverse Monte Carlo refinement of diffuse scattering and '+\
                'correlated disorder from single crystals',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/zjmorgan/rmc-discord',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    setup_requires=[
        'numpy',
        'cython',
    ],
    install_requires=[
        'numpy',
        'cython',
        'scipy',
        'matplotlib',
        'pycifrw',
        'nexusformat',
        'pyvista',
        'pyqt5',
        'ipython',
    ],
    ext_modules=extensions,
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
    cmdclass={"build": build},
    zip_safe=False
)
