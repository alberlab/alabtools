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
    
    def compute_efficiency(self):
        """TODO: Docstring for compute_efficiency and improve readability.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        
        # Check that the required attributes are present
        if self.cycle is None:
            raise ValueError("The cell-cycle state of each cell is not available.")
        if self.nu is None:
            raise ValueError("The normalized spot counts are not available.")
        
        # Compute the fraction of zeros in each cell
        f0 = np.mean(self.nu == 0, axis=(1, 2))
        assert f0.shape == (self.ncell,)
        
        # Initialize the efficiency array
        efficiency = np.zeros(self.ncell)
        
        # Compute the efficiency for G1 cells
        efficiency[self.cycle == 0] = 1 - f0[self.cycle == 0]
        
        # Compute the efficiency for G2 cells
        efficiency[self.cycle == 2] = 1 - f0[self.cycle == 2] ** 0.5
        
        # Compute the efficiency for S cells
        # We first need to estimate the probability that a domain is replicated in S
        # We assume this probability is linear with the volume of the cell
        v_min = np.min(self.volume[self.cycle == 1])  # minimum volume of S cells
        v_max = np.max(self.volume[self.cycle == 1])  # maximum volume of S cells
        pr = (self.volume - v_min) / (v_max - v_min)  # probability of replication
        # Now we use the following equation for the efficiency:
        #  e = (1 + pr - √((1 - pr)^2 + 4 pr f0)) / 2pr
        numerator1 = 1 + pr
        numerator2 = ((1 - pr)**2 + 4 * pr * f0) ** 0.5
        denominator = 2 * pr
        equation = (numerator1 - numerator2) / denominator
        efficiency[self.cycle == 1] = equation[self.cycle == 1]
        
        return efficiency
        