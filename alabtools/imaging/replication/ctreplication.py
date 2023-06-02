import os
import sys
import tempfile
from functools import partial
import numpy as np
import pickle
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
        
        if mode == 'rw':
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
        
        assert hasattr(loaded_ctrep, 'nraw'), "Loaded CtRep has no attribute 'nraw'."
        ### TODO: add more attributes
        
        # update the attributes of the current object
        # (every object has a __dict__ attribute, which is a dictionary of its attributes.
        # In this way, we can update the attributes of the current object with the attributes
        # of the loaded object)
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
        