#!/usr/bin/env python
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

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
    Extension("alabtools.numutils", ["src/numutils.pyx"]),
    Extension("alabtools._cmtools", ["src/cmtools/cmtools.i","src/cmtools/cmtools.cpp"],
              swig_opts=['-c++'],
              language="c++",
              include_dirs = [numpy_include],
              extra_compile_args=["-fopenmp"],
              extra_link_args=["-fopenmp"]
             ),
    Extension("alabtools._geotools", ["src/geotools/geotools.i","src/geotools/geotools.cpp"],
              swig_opts=['-c++'],
              language="c++",
              include_dirs = [numpy_include],
              extra_compile_args=["-lCGAL","-lmpfr","-lgmp"],
              extra_link_args=["-lCGAL","-lmpfr","-lgmp"]
             )
]
cmdclass.update({'build_ext': build_ext})
setup(
        name = 'alabtools', 
        version = '1.0.0', 
        author = 'Nan Hua', 
        author_email = 'nhua@usc.edu', 
        url = 'https://github.com/alberlab/alabtools', 
        description = 'Alber lab toolbox',
        cmdclass = cmdclass,
        packages=['alabtools'],
        package_data={'alabtools' : ['genomes/*']},
        package_dir={'alabtools': 'src'},
        install_requires=install_requires,
        tests_require=tests_require,
        extras_require=extras_require,
        ext_modules=extensions,
        include_dirs=[numpy.get_include()]
)
