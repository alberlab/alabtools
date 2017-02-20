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
    Extension("alabtools.numutils", ["src/numutils.pyx"])
]

setup(
        name = 'alabtools', 
        version = '1.0.0', 
        author = 'Nan Hua', 
        author_email = 'nhua@usc.edu', 
        url = 'https://github.com/alberlab/alabtools', 
        description = 'Alber lab toolbox',
        packages=['alabtools'],
        package_data={'alabtools' : ['genomes/*']},
        package_dir={'alabtools': 'src'},
        install_requires=install_requires,
        tests_require=tests_require,
        extras_require=extras_require,
        ext_modules=cythonize(extensions),
        include_dirs=[numpy.get_include()]
)
