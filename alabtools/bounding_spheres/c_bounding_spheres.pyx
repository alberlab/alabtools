# distutils: language = c++

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern from "bounding_spheres_functions.h":
    void bounding_spheres(float* crd, float* radii, int n_bead, int n_struct, float* results)

@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def cc_bounding_spheres(
        np.ndarray[float, ndim=3] crds, 
        np.ndarray[float, ndim=1] radii):
    '''
    Compute minimum bounding spheres for a group of beads.
    Parameters
    ----------
    crds : np.ndarray[float]
        A nbeads x nstruct x 3 coordinates vector
    radii : np.ndarray or iterable
        Radii of the beads[float]
    Returns
    -------
    bsx : np.ndarray[float]
        nstruct x 3 coordinates of the center of the bounding spheres
    bsr : np.ndarray[float]
        radii of the bounding spheres
    '''
    cdef int n_struct = crds.shape[1]
    cdef int n_bead = crds.shape[0]

    cdef np.ndarray[float, ndim=2] results = np.zeros((n_struct, 4), dtype=np.float32)

    bounding_spheres(&crds[0,0,0], &radii[0], n_bead, n_struct, &results[0, 0])

    bscenters = results[:, :3]
    bsradii = results[:, 3]

    return bscenters, bsradii
