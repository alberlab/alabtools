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
python_requires = '>=3.11'
install_requires = [
    'numpy>=1.23,<3',
    'scipy>=1.10',
    'pandas>=2.0',
    'h5py>=3.8',
    'matplotlib>=3.7',
    'ipyparallel>=8.0',
    'tqdm>=4.60',
    'six>=1.16',
    'cooler>=0.8.11',
    'pyBigWig>=0.3.22',
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
    Extension(
        "alabtools.numutils",
        ["alabtools/numutils.pyx"],
        extra_compile_args=["-std=gnu99"],
    ),
    Extension("alabtools._cmtools", ["alabtools/cmtools/cmtools.i", "alabtools/cmtools/cmtools.cpp"],
              swig_opts=['-c++'],
              language="c++",
              include_dirs=[numpy_include]+std_include,
             )
]

clscripts = [
    'bin/triplets-compact',
    'bin/triplets-compute',
    'bin/triplets-extract'
]

cmdclass.update({'build_ext': build_ext})
setup(
    name='alabtools',
    version='1.1.29',
    author='Nan Hua, Francesco Musella',
    author_email='nhua@usc.edu',
    url='https://github.com/alberlab/alabtools',
    description='Alber lab toolbox',
    cmdclass=cmdclass,
    # packages=['alabtools'],  # old way, replaced by find_packages()
    packages=find_packages(),
    package_data={'alabtools': ['genomes/*', 'config/*']},
    python_requires=python_requires,
    install_requires=install_requires,
    # tests_require=tests_require,
    # extras_require=extras_require,
    scripts=clscripts,
    ext_modules=extensions,
    include_dirs=[numpy_include]+std_include
)
