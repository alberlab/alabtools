import numpy as np
import pickle
import warnings
from alabtools.imaging import CtFile, CtEnvelope

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
        