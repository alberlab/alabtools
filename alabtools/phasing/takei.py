import numpy as np
from scipy.spatial import distance
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from .phaser import Phaser


class TakeiPhaser(Phaser):
    """
    A class that implements Takei's phasing method.
    
    It implements a combination of Ward and Spectral clustering.
    
    Reference: https://www.nature.com/articles/s41586-020-03126-2
    
    """
    
    def __init__(self, ct, ncopy=2, st=1.2, ot=2.5):
        
        super().__init__(ct)
        
        self.nchrom = len(self.ct.genome.chroms)
        
        if isinstance(ncopy, int):
            assert ncopy >= 2, "ncopy must be >=1"
            assert ncopy % 2 == 0, "ncopy must be a multiple of 2"
            self.ncopy = ncopy * np.ones((self.ct.ncell, self.nchrom), dtype=int)
            self.ncopy[:, self.ct.genome.chroms == 'chrX'] = ncopy / 2
            self.ncopy[:, self.ct.genome.chroms == 'chrY'] = ncopy / 2
        elif isinstance(ncopy, np.ndarray):
            assert ncopy.shape == (self.ct.ncell, self.nchrom), "ncopy shape must be (ncell, nchrom)"
            assert ncopy.dtype == int, "ncopy must be an integer array"
            assert np.all(ncopy >= 1), "ncopy must be >=1"
            self.ncopy = ncopy
        else:
            raise ValueError("ncopy must be either an int or an array (ncell, nchrom) of int")
        
        assert isinstance(st, float), "sep must be a float"
        assert isinstance(ot, float), "out must be a float"
        self.st = st  # separation threshold
        self.ot = ot  # outlier threshold

    def _apply_chrom_phasing(self, crd, dom, cellnum, chrnum):
        """Applies phasing to a single chromosome.
        Uses Ward and Spectral clustering to identify the two clusters of the chromosome.

        Args:
            crd (np.array(ndomain_chrom, nspot_max, 3)): coordinates of spots on the chromosome
            dom (Index object): index of domains on the chromosome
            cellnum (int): # of cell
            chrnum (int): # of chromosome

        Returns:
            phs (np.array(ndomain_chrom, nspot_max)): phasing labels of spots on the chromosome.
                Can be 0, 1, 2, ... depending on the number of copies.
                Spots with phs=0 are to be considered outliers.
        """
        
        n = self.ncopy[cellnum, chrnum]  # number of copies of chrom in cellID
        
        phs = np.zeros((crd.shape[0], crd.shape[1]))  # initialize phasing labels to 0
        
        nspot = crd.shape[0] * crd.shape[1]
        if nspot <= n:  # not enough spots to phase
            return phs
        
        crd_flat_nonan, idx_nonan = self._flatten_coordinates(crd)
        if n == 1:  # only one copy, no need to cluster
            phs_flat_nonan = np.ones(crd_flat_nonan.shape[0])
        else:
            phs_flat_nonan = self._clustering('ward', crd_flat_nonan, n)
            if not self.are_separated(crd_flat_nonan, phs_flat_nonan):
                phs_flat_nonan = self._clustering('spectral', crd_flat_nonan, n)
        
        phs_flat_nonan_corrected = self.remove_outliers(crd_flat_nonan, phs_flat_nonan)  # remove outliers
        
        for w, ij in enumerate(idx_nonan):
            i, j = ij
            phs[i,j] = phs_flat_nonan_corrected[w]  # assign phasing labels to original coordinates
                
        return phs
    
    def remove_outliers(self, pts, lbl):
        """Removes outliers from the set of points.
        Outliers are identified as points that are too far from the centroid of their cluster.

        Args:
            pts (np.array(n,3)): 3D coordinates of points to cluster
            lbl (np.array(n,)): labels of the clusters

        Returns:
            lbl_crrect (np.array(n,)): labels of the clusters with outliers removed (i.e. set to 0)
        """
        ctr, spd = self._compute_centroids(pts, lbl)
        lbl_correct = lbl.copy()
        for i in range(len(lbl)):
            p = pts[i]
            l = lbl[i]
            if np.linalg.norm(p - ctr[l]) > self.ot * np.linalg.norm(spd[l]):
                lbl_correct[i] = 0
        return lbl_correct
    
    def are_separated(self, pts, lbl):
        """Checks if the two clusters are separated enough.

        Args:
            pts (np.array(n,3)): 3D coordinates of points to cluster
            lbl (np.array(n,)): labels of the clusters

        Returns:
            bool: True if the clusters are separated enough, False otherwise
        """
        if len(np.unique(lbl)) <= 1:
            return True
        ctr, spd = self._compute_centroids(pts, lbl)
        for i, l1 in enumerate(ctr.keys()):
            for j, l2 in enumerate(ctr.keys()):
                if j <= i:
                    continue
                ctr1, spd1 = ctr[l1], spd[l1]
                ctr2, spd2 = ctr[l2], spd[l2]
                # Separation criterion:
                # the distance between the centroids is smaller than the sum of the spreads
                # multiplied by a factor self.st
                if np.linalg.norm(ctr1 - ctr2) <= self.st * (np.linalg.norm(spd1) + np.linalg.norm(spd2)):
                    return False
        return True
    
    @staticmethod
    def _compute_centroids(pts, lbl):
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
    
    @staticmethod
    def _clustering(method, pts, ncluster):
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

    @staticmethod
    def _flatten_coordinates(crd):
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
    