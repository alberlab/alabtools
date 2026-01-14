# alabtools

Package for working with the data in Frank Alber's lab at the University of California Los Angeles for 3D genome analysis and modeling.

See also our Integrative Genome Modeling (IGM) software package: https://github.com/alberlab/igm.git.

## Installation

ATTENTION: This package should work on both Linux and MacOS systems. It has been tested on CentOS 7.9.2009 (Core) and on Mac Tahoe 26.2.

Make sure you have conda installed (https://docs.conda.io/en/latest/miniconda.html) and that you have added the conda-forge channel (https://conda-forge.org/). The conda version used for building the package is 22.9.0.

Create a conda environment with a Python 3.11 version.
```bash
conda create -n alab python=3.11 -y
conda activate alab
```

Then, install the following packages with conda forge:
```bash
conda install pandas swig hdf5 h5py numpy scipy tornado ipyparallel cloudpickle
```

Then, install the following packages with conda forge:
```bash
conda install -c conda-forge \
    numpy \
    scipy \
    pandas \
    h5py \
    hdf5 \
    matplotlib \
    cython \
    swig \
    ipyparallel \
    cloudpickle \
    tqdm \
    six \
    -y
```

Then install another set of packages with pip. These could be installed with conda too, but in our experience it's better to use pip. The pip version used for building the package is 20.0.2.

To avoid a warning message, it might be useful to upgrade the jupyter-core package:
```bash
pip install --upgrade jupyter-core
```

Then perform the following installations:
```bash
pip install cooler pyBigWig
```

Finally, install the alabtools package:
```bash
pip install git+https://github.com/alberlab/alabtools.git
```
