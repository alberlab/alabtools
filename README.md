# alabtools

Package for working with the data in Frank Alber's lab at the University of California Los Angeles for 3D genome analysis and modeling.
See also our Integrative Genome Modeling (IGM) software package: https://github.com/alberlab/igm.git.

## Installation

ATTENTION: This package should work on both Linux and MacOS systems. It has been tested on CentOS 7.9.2009 (Core) and on Mac Tahoe 26.2.

Make sure you have conda installed (https://docs.conda.io/en/latest/miniconda.html) and that you have added the conda-forge channel (https://conda-forge.org/).
We recommend using conda ≥ 24. Older versions may stall during installation of some packages.

This package relies on some C++ codes, requiring a C++ compiler installed on your system. If you're using an old GCC version (e.g. GCC 4.x),
please consider updating it if you encounter compilation errors during installation. Although we haven't tested it systematically, GCC > 9 should work fine.

Create a conda environment with a Python 3.11 version.
```bash
conda create -n alab python=3.11 -y
conda activate alab
```

Then, install the following packages with conda forge:
```bash
conda install -c conda-forge \
    numpy<2.4 \
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

NOTE 1: The numpy version must be less than 2.4, otherwise some scripts will raise errors.
NOTE 2: If you're having touble installing these packages simultaneously, try installing them one by one or in smaller batches. 

Then install another set of packages with pip:
```bash
pip install cooler pyBigWig
```

Finally, install the alabtools package:
```bash
pip install git+https://github.com/alberlab/alabtools.git
```

If the install fails with build-isolation / NumPy build errors (common on older HPC systems):
```bash
pip install git+https://github.com/alberlab/alabtools.git --no-build-isolation

```
