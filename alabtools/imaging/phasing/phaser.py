import numpy as np
from functools import partial
import sys
import os
import tempfile
from ..ctfile import CtFile
from ...parallel import Controller


class Phaser(object):
    """
    A class that describes a general Phasing object.
    
    Works on CtFiles.
    
    Attributes
    ---------
    ct: CtFile instance
    
    """
    
    def __init__(self, cfg, *args):
        
        # Save the configuration file
        assert hasattr(cfg, 'get'), "configuration must have a get method."
        self.cfg = cfg
        
        # Create a controller
        self.controller = Controller(cfg)
        sys.stdout.write("Using a {} controller.\n".format(type(self.controller)))
        
        # Read ct name and open the file
        try:
            ct_name = os.path.join(os.getcwd(), cfg.get('ct_name'))
            self.ct_name = ct_name
        except KeyError:
            raise KeyError("ct_name must be specified in the configuration file.")
        try:
            self.ct = CtFile(self.ct_name, 'r')
        except IOError:
            raise IOError("{} must be a valid CtFile.".format(self.ct_name))
        
        # Initialize the phase array as None
        self.phase = None
        
        # Create a temporary directory for the controller, make sure it doesn't exist
        self.temp_dir = tempfile.mkdtemp(dir=os.getcwd())
        sys.stdout.write("Nodes' results will be saved in {}.\
                          The directory will be removed after phasing.\n".format(self.temp_dir))
        
        # Create the parallel task (to be overwritten by the child classes)
        self.parallel_task = partial(self._parallel_phasing, *args)
        # Create the reduce task
        self.reduce_task = partial(self._reduce_phasing,
                                   cfg=self.cfg)
    
    def phasing(self):
        """Performs the phasing on the CtFile.
        Uses the parallel_task and reduce_task saved as attributes.
        Creates a temporary directory for the nodes, and deletes it after the phasing.
        """
        
        assert self.ct.ncopy_max == 1, "CtFile already phased."
        
        # Create a temporary directory for the controller
        os.mkdir(self.temp_dir)
        
        # Parallelize the phasing, and save the results in the temporary directory
        self.phase = self.controller.map_reduce(self.parallel_task,
                                                self.reduce_task,
                                                args=self.ct.cell_labels)
        
        # Delete the temporary directory and its contents
        os.system('rm -r {}'.format(self.temp_dir))
    
    @staticmethod
    def _parallel_phasing(cellnum, *args):
        """To be overwritten by the child classes.
        """
        raise NotImplementedError("This method must be overwritten by the child classes.")
    
    @staticmethod
    def _reduce_phasing(out_names, cfg):
        """Reduce the results from child nodes' output files into a single array.

        Args:
            out_names (list): list of output file names from child nodes
            cell_labels (np.array(ncell,)): list of cell labels
            phase_shape (tuple): shape of the phase array

        Returns:
            phase (np.array(ncell, ndomains, nspot_max): phase array
        """
        
        # get ct_name, temp_dir from cfg
        try:
            ct_name = cfg['ct_name']
        except KeyError:
            "ct_name not found in cfg."
        
        # open ct file
        ct = CtFile(ct_name, 'r')
        
        # initialize phase array
        phase = np.zeros((ct.ncell, ct.ndomain, ct.nspot_max), dtype=int)
        
        # Read the results from the temporary directory and store them in the phase array
        for out_name in out_names:
            
            # the files are stored in a temporary directory with a path:
            #   /temp_dir/cellID.npy
            cellID = os.path.basename(out_name).split('.')[0]  # get cellID from file name
            cellnum = ct.get_cellnum(cellID)  # get cell number from cellID
            
            # read cell phase from file
            try:
                phs = np.load(out_name)  # np.array(ndomain, nspot_max)
            except IOError:
                "File {} not found.".format(out_name)
            
            # fill in the phase array
            phase[cellnum, :, :] = phs
    
    
    def separate_alleles(self, out_name):
        
        if self.phase is None:
            raise ValueError("Phasing must be performed before allele separation.")
        
        # initialize the phased coordinates array
        coordinates_phased = np.copy(self.ct.coordinates)  # np.array(ncell, ndomain, ncopy_max, nspot_max, 3)
        
        # loop over the phase labels
        for lbl in np.unique(self.phase):
            
            # The spots with phase label 0, i.e. outliers, are ignored
            if lbl == 0:
                continue
            
            # create a mask for the current phase label
            lbl_mask = self.phase == lbl  # np.array(ncell, ndomain, nspot_max)
            
            # expand the mask to match the shape of the coordinates array
            lbl_mask_exp = np.expand_dims(lbl_mask, axis=2)  # np.array(ncell, ndomain, 1, nspot_max)
            lbl_mask_exp = np.expand_dims(lbl_mask_exp, axis=4)  # np.array(ncell, ndomain, 1, nspot_max, 1)
            lbl_mask_exp = np.repeat(lbl_mask_exp, repeats=3, axis=4)  # np.array(ncell, ndomain, 1, nspot_max, 3)
            
            # check that the mask is correct
            for i in range(3):
                assert np.array_equal(lbl_mask[:, :, :], lbl_mask_exp[:, :, 0, :, i]),\
                    "Mask for coordinates is not correct."
            
            # set the coordinates to nan where the mask is False
            coordinates_phased[lbl_mask_exp] = np.nan
        
        # create a new CtFile instance with the phased coordinates
        if out_name is None:
            raise ValueError("out_name must be specified.")
        ct_phased = CtFile(out_name, 'w')
        ct_phased.set_manually(coordinates_phased,
                               self.ct.genome,
                               self.ct.index,
                               self.ct.cell_labels)
        
        return ct_phased
    