# alabtools

Package for working with the data in Frank Alber's lab at the University of California Los Angeles for 3D genome analysis and modeling.

See also our Integrative Genome Modeling (IGM) software package: https://github.com/alberlab/igm.git.

## Installation

ATTENTION: This package works only on Linux systems. It has been tested on CentOS 7.9.2009 (Core).

Make sure you have conda installed (https://docs.conda.io/en/latest/miniconda.html) and that you have added the conda-forge channel (https://conda-forge.org/). The conda version used for building the package is 22.9.0.

Create a conda environment with a Python 3.7 version. We recommend the 3.7.3.
```bash
conda create -n alab python=3.7.3
conda activate alab
```

Then, install the CGAL 4.14 version. It might be necessary to specify the conda-forge channel:
```bash
conda install cgal=4.14
```
or, if it doesn't work:
```bash
conda install -c conda-forge cgal=4.14
```
ATTENTION: cgal 4.14 requires a version of Python less than 3.8. Newer versions of cgal lead to compile errors, as the syntax has changed.

Then, install the following packages with conda:
```bash
conda install pandas swig hdf5 h5py numpy scipy tornado ipyparallel cloudpickle
```

Then install another set of packages with pip. These could be installed with conda too, but in our experience it's better to use pip. The pip version used for building the package is 20.0.2.

To avoid a warning message, it might be useful to upgrade the jupyter-core package:
```bash
pip install --upgrade jupyter-core
```

Then perform the following installations:
```bash
pip install cython matplotlib scikit-learn cooler alphashape mrcfile pyBigWig
```

Finally, install the alabtools package:
```bash
pip install git+https://github.com/alberlab/alabtools.git
```
