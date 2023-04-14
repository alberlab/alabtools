import numpy as np

def flatten_coordinates(crd):
    """Flattens the coordinates keeping track of the indices.
    Used to go from a 2D array to a 1D array and back.
    Also removes NaNs
    
    Parameters
    ----------
    crd: np.array(ndomain_chrom, nspot_max, 3)
    
    Returns
    ----------
    crd_flat: np.array(ndomain_chrom_nonan * nspot_max, 3)
    
    idx: np.array(ndomain_chrom_nonan * nspot_max, 2)
        Indices of the original array.
    """
    # Create a meshgrid of indices
    ii, jj, = np.meshgrid(np.arange(crd.shape[0]), np.arange(crd.shape[1]))
    # stack indices together vertically, e.g.
    #   [[0 0]
    #    [1 0]
    #    [2 0]]
    idx = np.vstack((ii.flatten(), jj.flatten())).T
    crd_flat = crd[idx[:,0], idx[:,1], :]
    # Removes NaNs from crd_flat and idx
    crd_flat_nonan = crd_flat[~np.isnan(crd_flat).any(axis=1)]
    idx_nonan = idx[~np.isnan(crd_flat).any(axis=1)]
    return crd_flat_nonan, idx_nonan

def compute_centroids(pts, lbl):
    """Computes centroids and spreads of a set of points with their clustering labels.

    Args:
        pts (np.array(n,3)): 3D coordinates of points to cluster
        lbl (np.array(n,)): labels of the clusters

    Returns:
        ctr: dict of np.array(3,): list of centroids of each cluster
        spd: dict of np.array(3,): list of spreads of each cluster
    """
    ctr, spd = {}, {}  # dict of centroid and spreads for each cluster
    for l in np.unique(lbl):
        if l == 0:
            continue
        pts_l = pts[lbl == l]
        ctr[l] = np.nanmean(pts_l, axis=0)  # Compute the centroids
        spd[l] = np.nanstd(pts_l, axis=0)  # Compute the standard deviation of the points
    return ctr, spd
