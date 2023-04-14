import numpy as np
from functools import partial
from functools import partialmethod
import sys
import os
import warnings
from scipy.spatial import distance
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from ..ctfile import CtFile
from .phaser import Phaser


class TakeiPhaser(Phaser):
    """
    A class that implements Takei's phasing method.
    
    It implements a combination of Ward and Spectral clustering.
    
    Reference: https://www.nature.com/articles/s41586-020-03126-2
    
    """
    
    def __init__(self, cfg):
        
        super().__init__(cfg)

        self.parallel_task = partial(self._parallel_phasing,
                                     cfg = self.cfg,
                                     temp_dir = self.temp_dir)

    @staticmethod
    def _parallel_phasing(cellID, cfg, temp_dir):
        """Applies phasing to a single cell.
        Uses Ward and Spectral clustering to identify the two clusters of the chromosome.

        Args:
            cellID (str)

        Returns:
            None
        """
        
        # get ct_name, temp_dir, st, ot from cfg
        try:
            ct_name = cfg['ct_name']
        except KeyError:
            "ct_name not found in cfg."
        try:
            st = cfg['additional_parameters']['st']
            ot = cfg['additional_parameters']['ot']
        except KeyError:
            "st and ot not found in cfg['additional_parameters']."
        
        # open ct file
        ct = CtFile(ct_name, 'r')
                        
        # initialize phasing labels of the cell to 0
        cell_phase = np.zeros((ct.ndomain, ct.nspot_max))

        # loop over chromosomes
        for chrom in ct.genome.chroms:
            
            # I can also read it from cfg!               
            ncluster = 2 if chrom != 'chrX' and chrom != 'chrY' else 1
            
            # get coordinates for chromosome/cell
            crd = ct.coordinates[ct.get_cellnum(cellID),
                                 ct.index.chromstr==chrom,
                                 0, :, :]  # np.array(ndomain_chrom, nspot_max, 3)
            
            # initialize phasing labels to 0
            phs = np.zeros((crd.shape[0], crd.shape[1]))
            
            # flatten coordinates and remove NaNs
            # crd_flat_nonan: np.array(nspot, 3)
            crd_flat_nonan, idx_nonan = flatten_coordinates(crd)
            
            # if there are not enough spots to phase, skip
            nspot = crd_flat_nonan.shape[0]
            if nspot <= ncluster:  # not enough spots to phase
                warnings.warn(f"Cell # {cellID} has {nspot} spots on chromosome {chrom},\
                              too few to phase. Skipping.")
                continue
            
            # phase
            if ncluster == 1:  # no need to cluster
                phs_flat_nonan = np.ones(crd_flat_nonan.shape[0])
            else:  # apply Ward and/or Spectral clustering
                phs_flat_nonan = clustering('ward', crd_flat_nonan, ncluster)
                if not are_separated(crd_flat_nonan, phs_flat_nonan, st):
                    phs_flat_nonan = clustering('spectral', crd_flat_nonan, ncluster)
            
            # remove outliers
            phs_flat_nonan_corrected = remove_outliers(crd_flat_nonan, phs_flat_nonan, ot)
            
            # if the phase labels are skipping an integer (e.g. 0, 2 - missing 1),
            # map them to increasing integers (e.g. 0, 1)
            if np.max(phs_flat_nonan_corrected) != len(np.unique(phs_flat_nonan_corrected)) - 1:
                phs_flat_nonan_corrected_cp = np.copy(phs_flat_nonan_corrected)
                for i, lbl in enumerate(np.unique(phs_flat_nonan_corrected)):
                    phs_flat_nonan_corrected[phs_flat_nonan_corrected_cp == lbl] = i
                del phs_flat_nonan_corrected_cp
            
            # assign phasing labels to original coordinates
            for w, ij in enumerate(idx_nonan):
                i, j = ij
                phs[i,j] = phs_flat_nonan_corrected[w]
            
            # fill in the phasing labels in the global array
            cell_phase[ct.index.chromstr == chrom, :] = phs
        
        # save cell_phase
        out_name = os.path.join(temp_dir, f'{cellID}.npy')
        np.save(out_name, cell_phase)
        
        return out_name



def remove_outliers(pts, lbl, ot):
        """Removes outliers from the set of points.
        Outliers are identified as points that are too far from the centroid of their cluster.

        Args:
            pts (np.array(n,3)): 3D coordinates of points to cluster
            lbl (np.array(n,)): labels of the clusters
            ot (float): outlier threshold

        Returns:
            lbl_crrect (np.array(n,)): labels of the clusters with outliers removed (i.e. set to 0)
        """
        ctr, spd = compute_centroids(pts, lbl)
        lbl_correct = lbl.copy()
        for i in range(len(lbl)):
            p = pts[i]
            l = lbl[i]
            if np.linalg.norm(p - ctr[l]) > ot * np.linalg.norm(spd[l]):
                lbl_correct[i] = 0
        return lbl_correct
    
def are_separated(pts, lbl, st):
    """Checks if the two clusters are separated enough.

    Args:
        pts (np.array(n,3)): 3D coordinates of points to cluster
        lbl (np.array(n,)): labels of the clusters
        st (float): separation threshold

    Returns:
        bool: True if the clusters are separated enough, False otherwise
    """
    if len(np.unique(lbl)) <= 1:
        return True
    ctr, spd = compute_centroids(pts, lbl)
    for i, l1 in enumerate(ctr.keys()):
        for j, l2 in enumerate(ctr.keys()):
            if j <= i:
                continue
            ctr1, spd1 = ctr[l1], spd[l1]
            ctr2, spd2 = ctr[l2], spd[l2]
            # Separation criterion:
            # the distance between the centroids is smaller than the sum of the spreads
            # multiplied by the factor st
            if np.linalg.norm(ctr1 - ctr2) <= st * (np.linalg.norm(spd1) + np.linalg.norm(spd2)):
                return False
    return True

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

def clustering(method, pts, ncluster):
    """Clusters the coordinates.
    method indicates the clustering algorithm.
    Supported: 'ward' and 'spectral'.

    Args:
        method (str): method used to cluster the coordinates
        pts (np.array(n,3)): 3D coordinates of points to cluster
        ncluster (int): number of clusters

    Raises:
        ValueError: method not supported

    Returns:
        lbl (np.array(n,)): labels of the clusters
    """
    if method == 'ward':
        clt = AgglomerativeClustering(n_clusters=ncluster, linkage='ward').fit(pts)
    elif method == 'spectral':
        dist_mat = distance.cdist(pts, pts, 'euclidean')
        # The affinity matrix is needed for the Spectral Clustering
        # This transformation is advised by sklearn (https://scikit-learn.org/dev/modules/clustering.html)
        beta = 1  # This parameter can be tuned, but 1 seems to give good results
        aff_mat = np.exp(- beta * dist_mat / dist_mat.std())
        clt = SpectralClustering(n_clusters=ncluster, affinity='precomputed').fit(aff_mat)
    else:
        raise ValueError("method not supported.")
    lbl = clt.labels_ + 1  # labels are 1 or 2
    return lbl

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
    