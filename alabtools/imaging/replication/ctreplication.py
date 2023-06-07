import os
import sys
import tempfile
import warnings
from functools import partial
import numpy as np
import pickle
from alabtools.utils import Genome, Index
from alabtools.imaging import CtFile, CtEnvelope
from alabtools.parallel import Controller
from . import cellcycle
from . import bernoulli

class CtRep(object):
    """
    A class for cell-cycle replication analysis from CT data.
    
    Saved and loaded as a pickle file with .ctrep extension.
    
    Attributes:
        ...
    
    Datasets:
        ...
    """
    
    def __init__(self, filename, mode='r'):
        
        # assert the input filename
        assert isinstance(filename, str), "The input filename must be a string."
        assert filename.endswith('.ctrep'), "The input filename must end with .ctrep."
        
        # assert the input mode
        assert mode in ['r', 'w'], "The input mode must be 'r' or 'w'."
        
        # set the filename and mode attributes
        self.filename = filename
        self.mode = mode
        
        if mode == 'r':
            self.load()
        
        if mode == 'w':
            self.ncell = None
            self.ndomain = None
            self.ncopy_max = None
            self.cell_labels = None
            self.genome = None
            self.index = None
            self.cycle = None
            self.nraw = None
            self.rho = None
            self.nu = None
            self.efficiency = None
            self.n = None
    
    def load(self):
        """Loads a CtRep from a pickle file.
        """
        
        with open(self.filename, 'rb') as f:
            loaded_ctrep = pickle.load(f)
        
        # Throw warnings if the loaded CtRep has missing attributes
        if not hasattr(loaded_ctrep, 'ncell'):
            warnings.warn("Loaded CtRep has no attribute 'ncell'.")
        if not hasattr(loaded_ctrep, 'ndomain'):
            warnings.warn("Loaded CtRep has no attribute 'ndomain'.")
        if not hasattr(loaded_ctrep, 'ncopy_max'):
            warnings.warn("Loaded CtRep has no attribute 'ncopy_max'.")
        if not hasattr(loaded_ctrep, 'nraw'):
            warnings.warn("Loaded CtRep has no attribute 'nraw'.")
        if not hasattr(loaded_ctrep, 'cell_labels'):
            warnings.warn("Loaded CtRep has no attribute 'cell_labels'.")
        if not hasattr(loaded_ctrep, 'genome'):
            warnings.warn("Loaded CtRep has no attribute 'genome'.")
        if not hasattr(loaded_ctrep, 'index'):
            warnings.warn("Loaded CtRep has no attribute 'index'.")
        if not hasattr(loaded_ctrep, 'cycle'):
            warnings.warn("Loaded CtRep has no attribute 'cycle'.")
        if not hasattr(loaded_ctrep, 'rho'):
            warnings.warn("Loaded CtRep has no attribute 'rho'.")
        if not hasattr(loaded_ctrep, 'nu'):
            warnings.warn("Loaded CtRep has no attribute 'nu'.")
        if not hasattr(loaded_ctrep, 'efficiency'):
            warnings.warn("Loaded CtRep has no attribute 'efficiency'.")
        if not hasattr(loaded_ctrep, 'n'):
            warnings.warn("Loaded CtRep has no attribute 'n'.")
        
        # update the attributes of the current object
        self.__dict__.update(loaded_ctrep.__dict__)
    
    def save(self):
        """Saves a CtRep to a pickle file.
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(self, f)
    
    def read_ct(self, ct=None, ctenv=None):
        """Reads data from a CtFile or CtEnvelope.

        Args:
            ct (CtFile): Input CtFile.
            ctenv (CtEnvelope): Input CtEnvelope.
        """
        
        # Assert the input
        if ct is not None:
            assert isinstance(ct, CtFile), "The input ct must be a CtFile."
        if ctenv is not None:
            assert isinstance(ctenv, CtEnvelope), "The input ctenv must be a CtEnvelope."
        
        # If both ct and ctenv are provided, check if they are consistent
        if ct is not None and ctenv is not None:
            assert ct.ncell == ctenv.ncell,\
                "ct.ncell = {} != ctenv.ncell = {}".format(ct.ncell, ctenv.ncell)
        
        # Read data from CtFile if provided
        if ct is not None:
            self.ncell = ct.ncell
            self.ndomain = ct.ndomain
            self.ncopy_max = ct.ncopy_max
            self.cell_labels = ct.cell_labels
            self.genome = ct.genome
            self.index = ct.index
            self.nraw = ct.nspot
        
        # Read data from CtEnvelope if provided
        if ctenv is not None:
            self.ncell = ctenv.ncell
            self.cell_labels = ctenv.cell_labels
            self.volume = ctenv.volume
    
    def run_cellcycle(self, cfg):
        """Runs the cell-cycle imputation algorithm.
        
        Finds the best G1/S/G2 segmentation of the cells:
            1) cells are sorted by increasing volume,
            2) Iteratively segments the lowest X% of cells in G1,
               the highest Y% in G2, and the remaining in S,
            3) For each segmentation, the spot counts are normalized by
                a bias (estimated from the G1 and G2 cells),
            4) The Pearson correlation between the experimental RT and the
                simulated one is computed with the normalized count,
            5) The segmentation with the highest correlation is selected.
        
        The function computes the following data, stored in the object:
            - the cell-cycle state of each cell (self.cycle)
            - the continuous normalized spot counts (self.rho)
            - the discretized normalized spot counts (self.nu)
        
        The function also returns the Pearson correlation between the
        experimental RT and the simulated one for the best segmentation.

        Args:
            cfg (dict or json): Configuration file.

        Returns:
            r (float): Pearson correlation between the experimental RT
                       and the simulated one for the best segmentation.
        """
        
        # Check that cfg is a dictionary
        assert isinstance(cfg, dict), "The input cfg must be a dictionary."
        
        # Check that the required keys are present in cfg
        required_keys = ['parallel', 'rt_bedfile', 'assembly']
        for key in required_keys:
            assert key in cfg.keys(), "The input cfg must have the key '{}'.".format(key)
        
        # create a temporary directory to store nodes' results
        temp_dir = tempfile.mkdtemp(dir=os.getcwd())
        sys.stdout.write("Temporary directory for nodes' results: {}\n".format(temp_dir))
        
        # create a Controller
        controller = Controller(cfg)
        
        # Read the RT data and assert that Index matches
        rt_bedfile = cfg['rt_bedfile']
        assembly = cfg['assembly']
        rt = Index(rt_bedfile, genome=Genome(assembly))
        assert assembly == self.genome.assembly,\
            "Assembly provided in configuration file doesn't match the one in the CtRep."
        assert rt == self.index,\
            "Index from RT BedGraph doesn't match the one in the CtRep."
        
        # compute all the possible G1/G2 segmentations
        segmentation = []
        # (assuming that G1 (and G2, separately) can have at most half of the cells)
        for ncell_g1 in range(1, int(self.ncell / 2) + 1):
            for ncell_g2 in range(1, int(self.ncell / 2) + 1):
                segmentation.append([ncell_g1, ncell_g2])
        segmentation = np.array(segmentation)
        nsegment = segmentation.shape[0]
        # save the segmentation to a temporary file
        np.save(os.path.join(temp_dir, 'segmentation.npy'), segmentation)
        
        # Save the data needed for the parallel and reduce tasks to temporary files
        np.save(os.path.join(temp_dir, 'nraw.npy'), self.nraw)
        np.save(os.path.join(temp_dir, 'volume.npy'), self.volume)
        
        # set the parallel and reduce tasks
        parallel_task = partial(cellcycle.parallel_function,
                                cfg=cfg,
                                temp_dir=temp_dir)
        reduce_task = cellcycle.reduce_function

        # run the parallel and reduce tasks
        r, cycle = controller.map_reduce(parallel_task,
                                         reduce_task,
                                         args=np.arange(nsegment))
        
        # Delete the temporary directory and its contents
        os.system('rm -r {}'.format(temp_dir))
        
        # Update the attributes of the current object
        self.cycle = cycle
        self.rho = cellcycle.normalize_bias(self.nraw, self.cycle)
        self.nu = nu = np.round(self.rho).astype(int)
        self.nu[nu > 2] = 2
                
        return r

    def replication_probability(self):
        """Imputes the probability of replication in each cell.
        
        The probability of replication is estimated to be proportional
        to the volume of the cell is the S phase.

        Returns:
            pr (np.array(ncell), dtype=float): Probability of replication of each cell.
        """
        
        assert self.volume is not None,\
            "The volume of each cell is not available."
        assert self.cycle is not None,\
            "The cell-cycle state of each cell must be computed first."
        
        # Find largest volume in G1 (lower bound for S)
        v_min = np.max(self.volume[self.cycle == 0])
        # Find smallest volume in G2 (upper bound for S)
        v_max = np.min(self.volume[self.cycle == 2])
        
        # Compute the probability of replication
        pr = (self.volume - v_min) / (v_max - v_min)
        pr[pr < 0] = 0
        pr[pr > 1] = 1
        
        # Set the probability to 0 and 1 for G1 and G2 cells
        pr[self.cycle == 0] = 0
        pr[self.cycle == 2] = 1
        
        return pr
    
    def compute_fraction(self):
        """Computes the fraction of domains with 0, 1 or 2 copies in each cell.
        
        Sex chromosomes are excluded from the computation.

        Returns:
            f (np.array(3, ncell), dtype=float): Fraction of domains with 0, 1 or 2 copies.
        """
        
        # Assert that the required data is available
        assert self.cycle is not None,\
            "The cell-cycle state of each cell must be computed first."
        assert self.nu is not None,\
            "The discretized normalized spot counts are not available."
        
        # Set sex chromosome domains to NaN (might screw up the computation)
        nu_cp = self.nu.copy().astype(float)
        for sex_chrom in ['chrX', 'chrY']:
            nu_cp[:, self.index.chromstr == sex_chrom, :] = np.nan
        
        # Count the fraction of domains with 0, 1 or 2 copies in each cell
        f0 = np.nanmean(nu_cp == 0, axis=(1, 2))  # np.array(ncell)
        f1 = np.nanmean(nu_cp == 1, axis=(1, 2))
        f2 = np.nanmean(nu_cp == 2, axis=(1, 2))
        
        f = np.array([f0, f1, f2])  # np.array(3, ncell)
        
        return f
    
    def impute_efficiency(self, method):
        """Imputes the efficiency of replication in each cell.
        
        Uses the method specified in the input.

        Args:
            method (str): Method to use for imputing the efficiency.

        Returns:
            None
        """
        
        # Assert that the method is acceptable
        acceptable_methods = ['bernoulli']
        assert method in acceptable_methods,\
            "The input method must be one of the following: {}".format(acceptable_methods)
        
        if method == 'bernoulli':
            
            # Assert that the required data is available
            assert self.cycle is not None,\
                "The cell-cycle state of each cell must be computed first."
            assert self.volume is not None,\
                "The volume of each cell is not available."
            assert self.nu is not None,\
                "The discretized normalized spot counts are not available."
            
            # Compute the fractions of domains with 0, 1 or 2 copies
            f = self.compute_fraction()
            
            # Compute the replication probability
            pr = self.replication_probability()
            
            # Impute the efficiency with the Bernoulli model
            efficiency, costs = bernoulli.efficiency_optimization(f, pr)
            
            # Update the attributes of the current object
            self.pr = pr
            self.efficiency_cost_residual = costs
            self.efficiency = efficiency
    
    def impute_replication(self, method):
        
        # Assert that the method is acceptable
        acceptable_methods = ['bernoulli_maxlikelihood']
        assert method in acceptable_methods,\
            "The input method must be one of the following: {}".format(acceptable_methods)
        
        if method == 'bernoulli_maxlikelihood':
                
            # Assert that the required data is available
            assert self.nu is not None,\
                "The discretized normalized spot counts are not available."
            assert self.efficiency is not None,\
                "The efficiency of replication is not available."
            
            # Compute the replication probability
            pr = self.replication_probability()
            
            # Impute the replication with the Bernoulli model
            n, lkl_max = bernoulli.likelihood_maximization_n(self.nu, self.efficiency, pr)
            
            # Update the attributes of the current object
            self.n = n
            self.likelihood_max = lkl_max
    
    def compute_rt(self, mat, isolate_s=True):
        """Computes the RT from an input matrix.
        
        Only S cells are considered for the computation.

        Returns:
            rt (np.array(ndomain), dtype=float): RT of each domain.
        """
        
        # Check that the input matrix has the correct shape
        assert mat.shape == (self.ncell, self.ndomain, self.ncopy_max),\
            "The input matrix has the wrong shape."
        # Check that cycle has been computed
        if isolate_s:
            assert self.cycle is not None,\
                "The cell-cycle state of each cell must be computed first."
        
        # Isolate the S data
        if isolate_s:
            mat_cp = mat[self.cycle == 1, :, :]
        else:
            mat_cp = mat.copy()
        
        # Compute the RT
        rt = np.nanmean(mat_cp, axis=(0, 2))  # np.array(ndomain)
        
        return rt
    
    def sort_by_volume(self, mat, isolate_s=True):
        """Sorts the matrix by increasing volume,
        and transforms it into a haploid matrix where copies
        of the same cell are concatenated contiguously.

        Args:
            mat (np.array(ncell, ndomain, ncopy_max)): Input matrix.
            isolate_s (bool): If True, only S cells are considered.

        Returns:
            mat_srt_hap (np.array(ncell_s * ncopy_max, ndomain)):
                        Haploid matrix ranked by increasing volume.
            volume_srt (np.array(ncell_s)):
                        Volume of each cell, ranked from smallest to largest.
        """
        
        # Assert the input
        assert mat.shape == (self.ncell, self.ndomain, self.ncopy_max),\
            "The input matrix has the wrong shape."
        # Assert that the volume has been computed
        assert self.volume is not None,\
            "The volume of each cell is not available."
        if isolate_s:
            # Assert that cycle has been computed
            assert self.cycle is not None,\
                "The cell-cycle state of each cell must be computed first."
        
        # Isolate S cells
        if isolate_s:
            volume_srt = self.volume[self.cycle == 1]
            mat_srt = mat[self.cycle == 1, :, :]
        else:
            volume_srt = self.volume.copy()
            mat_srt = mat.copy()
        
        # Sort the cells by increasing volume
        mat_srt = mat_srt[np.argsort(volume_srt), :, :]
        volume_srt = volume_srt[np.argsort(volume_srt)]
        
        # Reshape the matrix to a 2D array (ncell_s * ncopy_max, ndomain)
        ncell_s = int(np.sum(self.cycle == 1))
        mat_srt_hap = np.zeros((ncell_s * self.ncopy_max, self.ndomain))
        for cell in range(ncell_s):
            for copy in range(self.ncopy_max):
                mat_srt_hap[cell * self.ncopy_max + copy, :] = mat_srt[cell, :, copy]
        
        return mat_srt_hap, volume_srt
        