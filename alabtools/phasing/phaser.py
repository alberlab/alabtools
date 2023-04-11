import numpy as np
from functools import partial
from functools import partialmethod
import sys
import os
sys.path.append(os.path.abspath('..'))
from alabtools.imaging import CtFile
from alabtools.utils import Genome, Index
from alabtools.parallel import Controller


class Phaser(object):
    """
    A class that describes a general Phasing object.
    
    Works on CtFiles.
    
    Attributes
    ---------
    ct: CtFile instance
    
    """
    
    def __init__(self, ct, controller_cfg, *args):
        
        if isinstance(ct, CtFile):
            self.ct = ct
        elif isinstance(ct, str) and ct[-3:] == '.ct':
            self.ct = CtFile(ct, 'r')
        else:
            raise ValueError("ct not in supported format.")
        
        # assert that controller_cfg has a get method
        assert hasattr(controller_cfg, 'get'), "controller_cfg must have a get method."
        self.controller = Controller(controller_cfg)
        sys.stdout.write("Using a {} controller.\n".format(type(self.controller)))
        
        self.phase = np.zeros((self.ct.ncell, self.ct.ndomain, self.ct.nspot_max))
        
        self.task = partial(self._apply_cell_phasing, *args)
    
    def phasing(self):
        
        assert self.ct.ncopy_max == 1, "Data is already phased."
        
        # I am getting the error
        #   raise TypeError("h5py objects cannot be pickled")
        #   TypeError: h5py objects cannot be pickled
        # this is because the controller is trying to pickle the ct object,
        # which contains an h5py file. See this link:
        #   https://dannyvanpoucke.be/parallel-python-classes-pickle/
        
        # One option would be to read the ct file into the _apply_cell_phasing function,
        # and I think something like this is what they did in IGM, but I have to double check.
        # I think it's recommended to make the parallelized functions static methods.
        
        # Another option, similar to the first one but probably more refined,
        # would be to give the coord, domain (as a tuple (chrstr, start, end), not Index),
        # as static attributes of the _apply_cell_phasing function, as is done in IGM:
        #   serial_function = partial(self.__class__.task,
        #                             cfg=self.cfg,
        #                             tmp_dir=self.tmp_dir)
        
        # Another possibility maybe could be to save only the coordinates,
        # the cell labels, and so on, and hopefully - since we don't store h5py objects -
        # the controller will be able to pickle the data.
                
        cell_phases = self.controller.map(self.task, np.arange(self.ct.ncell))
        
        for cellnum in range(self.ct.ncell):
            self.phase[cellnum, :, :] = cell_phases[cellnum]
    
    @staticmethod
    def _apply_cell_phasing(cellnum, *args):
        pass
    