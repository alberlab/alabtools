#!/usr/bin/env python
from setuptools import setup, Extension, find_packages
# from distutils.core import setup, Extension  # old way, replaced by setuptools
from Cython.Distutils import build_ext

import numpy
import sys

# Add include and library directories from conda envs for swig.
std_include = [sys.prefix + '/include', sys.prefix + '/Library/include']
std_library = [sys.prefix + '/lib', sys.prefix + '/Library/lib']

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


cmdclass = {}
install_requires = [
    'numpy>=1.9',
    'scipy>=0.16',
    'pandas>=0.17',
    'h5py>=2.5',
    'matplotlib>=1.5',
    'Cython>=0.25',
    'tqdm',
    'six',
    'alphashape>=1.3.1',
    'trimesh>=3.21.5'
]
tests_require = [
    'mock'
]

extras_require = {
    'docs': [
        'Sphinx>=1.1',
    ]
}

extensions = [
    Extension("alabtools.numutils", ["alabtools/numutils.pyx"]),
    # Extension("alabtools._cmtools", ["alabtools/cmtools/cmtools.i", "alabtools/cmtools/cmtools.cpp"],
    #           swig_opts=['-c++'],
    #           language="c++",
    #           include_dirs=[numpy_include]+std_include,
    #           extra_compile_args=["-fopenmp"],
    #           extra_link_args=["-fopenmp"]
    #          )
]

# if '--no-geotools' not in sys.argv:  # gives compile issues in local machines
if False:
    extensions.append(
        Extension("alabtools._geotools", ["alabtools/geotools/geotools.i", "alabtools/geotools/geotools.cpp"],
                  swig_opts=['-c++'],
                  language="c++",
                  include_dirs=[numpy_include]+std_include,
                  library_dirs = std_library,
                  libraries=["CGAL", "mpfr", "gmp"],
                  extra_compile_args=["-fopenmp"],
                  extra_link_args=["-fopenmp"]
                  )
    )

clscripts = [
    'bin/triplets-compact',
    'bin/triplets-compute',
    'bin/triplets-extract'
]

cmdclass.update({'build_ext': build_ext})
setup(
    name='alabtools',
    version='1.1.0',
    author='Nan Hua, Francesco Musella',
    author_email='nhua@usc.edu',
    url='https://github.com/alberlab/alabtools',
    description='Alber lab toolbox',
    cmdclass=cmdclass,
    # packages=['alabtools'],  # old way, replaced by find_packages()
    packages=find_packages(),
    package_data={'alabtools': ['genomes/*', 'config/*']},
    install_requires=install_requires,
    # tests_require=tests_require,
    # extras_require=extras_require,
    scripts=clscripts,
    ext_modules=extensions,
    include_dirs=[numpy_include]+std_include
)
