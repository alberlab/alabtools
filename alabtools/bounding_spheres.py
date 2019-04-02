import numpy as np
from ._bounding_spheres import cc_bounding_spheres

def compute_bounding_spheres(crds, radii):
    # convert the data to float numpy array
    crds = np.array(crds, dtype=np.float32, order='C')
    radii = np.array(radii, dtype=np.float32, order='C')
    return cc_bounding_spheres(crds, radii)

