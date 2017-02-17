#!/usr/bin/env python
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

install_requires = [
    'numpy>=1.9', 
    'scipy>=0.16', 
    'pandas>=0.17',
    'h5py>=2.5', 
    'matplotlib>=1.5',
    'cython',
    'cooler'
]
tests_require = [
    'mock'
]


extras_require = {
    'docs': [
        'Sphinx>=1.1', 
    ]
}
setup(
        name = 'alabtools', 
        version = '1.0.0', 
        author = 'Nan Hua', 
        author_email = 'nhua@usc.edu', 
        url = 'https://github.com/alberlab/alabtools', 
        description = 'Alber lab toolbox',
        packages=['alabtools'],
        package_data={'src' : ['genomes/*']},
        install_requires=install_requires,
        tests_require=tests_require,
        extras_require=extras_require,
        ext_modules=cythonize("alabtools/numutils.pyx"),
        include_dirs=[numpy.get_include()]
)
