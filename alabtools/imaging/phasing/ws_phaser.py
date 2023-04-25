import numpy as np
from functools import partial
from functools import partialmethod
import sys
import os
import warnings
from scipy.spatial import distance
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from alabtools.imaging import CtFile
from alabtools.imaging.utils_imaging import flatten_coordinates
from .phaser import Phaser, reorder_spots, phase_cell_coordinates


class WSPhaser(Phaser):
    """
    
    It implements a combination of Ward and Spectral clustering,
    as described in Takei et al. Nature (2021). 
    
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
        
        # get ct_name, ncluster, st, ot from cfg
        try:
            ct_name = cfg['ct_name']
        except KeyError:
            "ct_name not found in cfg."
        try:
            ncluster = cfg['ncluster']
        except KeyError:
            "ncluster not found in cfg."
        try:
            st = cfg['additional_parameters']['st']
            ot = cfg['additional_parameters']['ot']
        except KeyError:
            "st and ot not found in cfg['additional_parameters']."
        
        # open ct file
        ct = CtFile(ct_name, 'r')
        
        # assert ncluster is correct and fill it autosomes if '#' is present
        assert isinstance(ncluster, dict), "ncluster must be a dictionary."
        for chrom in ncluster.keys():
            assert chrom == '#' or chrom in ct.genome.chroms,\
                "Invalid chromosome name in ncluster."
        if '#' in ncluster.keys():
            for chrom in ct.genome.chroms:
                try:  # check if chrom is an autosome (e.g. chr1, chr2, ...)
                    chrom_num = chrom.split('chr')[1]
                    chrom_num = int(chrom_num)
                except ValueError:  # if not, continue
                    continue
                else:  # if so, fill in ncluster[chrom]
                    ncluster[chrom] = ncluster['#']
                        
        # initialize phasing labels of the cell to 0
        cell_phase = np.zeros((ct.ndomain, ct.nspot_max),
                              dtype=np.int32)  # np.array(ndomain, nspot_max)
        
        # get coordinates for cell
        cell_coordinates = ct.coordinates[ct.get_cellnum(cellID),
                                          :, 0, :, :]  # np.array(ndomain, nspot_max, 3)

        # loop over chromosomes
        for chrom in ct.genome.chroms:
            
            # get coordinates for chromosome/cell
            crd = cell_coordinates[ct.index.chromstr==chrom,
                                   :, :]  # np.array(ndomain_chrom, nspot_max, 3)
            
            # initialize phasing labels to 0
            phs = np.zeros((crd.shape[0], crd.shape[1]),
                           dtype=np.int32)  # np.array(ndomain_chrom, nspot_max)
            
            # flatten coordinates
            # crd_flat: np.array(ndomain_chrom*nspot_max, 3)
            crd_flat, idx_flat = flatten_coordinates(crd)
            
            # remove nan coordinates
            # crd_flat_nonan: np.array(nspot, 3)
            crd_flat_nonan = crd_flat[~np.isnan(crd_flat).any(axis=1)]
            idx_flat_nonan = idx_flat[~np.isnan(crd_flat).any(axis=1)]
            
            # if there are not enough spots to phase, skip
            nspot = crd_flat_nonan.shape[0]
            if nspot <= ncluster[chrom]:  # not enough spots to phase
                warnings.warn(f"Cell # {cellID} has {nspot} spots on chromosome {chrom},\
                              too few to phase. Skipping.")
                continue
            
            # phase
            if ncluster[chrom] == 1:  # no need to cluster
                phs_flat_nonan = np.ones(crd_flat_nonan.shape[0])
            else:  # apply Ward and/or Spectral clustering
                phs_flat_nonan = clustering('ward', crd_flat_nonan, ncluster[chrom])
                if not are_separated(crd_flat_nonan, phs_flat_nonan, st):
                    phs_flat_nonan = clustering('spectral', crd_flat_nonan, ncluster[chrom])
            
            # remove outliers
            phs_flat_nonan_noout = remove_outliers(crd_flat_nonan, phs_flat_nonan, ot)
            # the function remove_outliers doesn't change the shape of the input,
            # but rather sets the outliers to 0. So the index doesn't change.
            # I keep the same variable name for clarity.
            idx_flat_nonan_noout = idx_flat_nonan
            
            # if the phase labels are skipping an integer (e.g. 0, 2 - missing 1),
            # map them to increasing integers (e.g. 0, 1)
            if len(np.unique(phs_flat_nonan_noout)) != ncluster[chrom] + 1:
                phs_flat_nonan_noout_cp = np.copy(phs_flat_nonan_noout)
                for i, lbl in enumerate(np.unique(phs_flat_nonan_noout)):
                    if lbl == 0:
                        continue
                    phs_flat_nonan_noout[phs_flat_nonan_noout_cp == lbl] = i + 1
                del phs_flat_nonan_noout_cp
            
            # assign phasing labels to original coordinates
            for w, ij in enumerate(idx_flat_nonan_noout):
                i, j = ij
                phs[i,j] = phs_flat_nonan_noout[w]
            
            # fill in the phasing labels in the global array
            cell_phase[ct.index.chromstr == chrom, :] = phs
        
        # phase cell coordinates
        cell_coordinates_phased = phase_cell_coordinates(cell_coordinates,
                                                         cell_phase,
                                                         ncopy_max=2)
        
        # reorder spots
        cell_coordinates_phased = reorder_spots(cell_coordinates_phased)
        
        # save cell_phase
        out_name = os.path.join(temp_dir, f'{cellID}.npy')
        np.save(out_name, cell_coordinates_phased)
        
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
    