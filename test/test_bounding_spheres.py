import numpy as np
from alabtools.bounding_spheres import compute_bounding_spheres

def test_bounding_sphere():
    x = np.random.random((10, 100, 3))
    r = np.random.random(10) * 0.1
    bsx, bsr = compute_bounding_spheres(x, r)

    for struct_coords, center, R in zip(x.swapaxes(0, 1), bsx, bsr):
        closest = np.inf
        for bead_coord, bead_radius in zip(struct_coords, r):
            surf_dist = R - np.linalg.norm(bead_coord - center) - bead_radius
            # each bead should be inside the sphere, so the distance cannot be
            # negative. We use float precision, so set the tolerance to 1e6
            assert (surf_dist > -1e6)
            closest = min(closest, surf_dist)
        # also, to be a minimum bounding sphere, at least one bead should touch
        # the surface of the bounding sphere.
        assert closest < 1e6

