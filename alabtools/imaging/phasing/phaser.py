import numpy as np
from functools import partial
import sys
import os
import tempfile
from alabtools.imaging import CtFile
from alabtools.parallel import Controller


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
        
        # Read ct name (appending working directory) and open the CtFile
        try:
            ct_name = os.path.join(os.getcwd(), cfg.get('ct_name'))
            self.ct_name = ct_name
        except KeyError:
            raise KeyError("ct_name must be specified in the configuration file.")
        try:
            self.ct = CtFile(self.ct_name, 'r')
        except IOError:
            raise IOError("{} must be a valid CtFile.".format(self.ct_name))
        
        # Create a temporary directory for nodes' results
        self.temp_dir = tempfile.mkdtemp(dir=os.getcwd())
        sys.stdout.write("Nodes' results will be saved in {}."\
                         "\nThe directory will be removed after phasing.\n".format(self.temp_dir))
        
        # Create the parallel task (to be overwritten by the child classes)
        self.parallel_task = partial(self._parallel_phasing, *args)
        # Create the reduce task
        self.reduce_task = partial(self._reduce_phasing,
                                   cfg=self.cfg)
    
    def run(self, out_name=None):
        """Run the phasing.
        Uses the parallel_task and reduce_task saved as attributes.
        Creates a temporary directory for the nodes' results, and deletes it after reducing.
        """
        
        assert self.ct.ncopy_max == 1, "CtFile already phased."
        
        sys.stdout.write("Performing phasing...\n")
                
        # Parallelize the phasing, and save the results in the temporary directory
        coordinates_phsd = self.controller.map_reduce(self.parallel_task,
                                                      self.reduce_task,
                                                      args=self.ct.cell_labels)
        
        # Delete the temporary directory and its contents
        os.system('rm -r {}'.format(self.temp_dir))
        
        # Create a CtFile with the phasing results
        if out_name is None:
            out_name = self.ct_name.replace('.ct', '_phased.ct')
        assert isinstance(out_name, str), "out_name must be a string."
        assert out_name.endswith('.ct'), "out_name must end with .ct."
        ct_phased = CtFile(out_name, 'w')
        ct_phased.set_manually(coordinates_phsd,  # set data manually
                               self.ct.genome,
                               self.ct.index,
                               self.ct.cell_labels)
        
        return ct_phased
    
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
        
        # get ct_name from cfg
        try:
            ct_name = cfg['ct_name']
        except KeyError:
            "ct_name not found in cfg."
        
        # open ct file
        ct = CtFile(ct_name, 'r')
        
        # check that the output size is correct
        assert len(out_names) == ct.ncell, "Number of output files does not match number of cells."
        
        # list of coordinates, outputs from the child nodes
        crd_list = [None for _ in range(ct.ncell)]
        
        # Read the results from the temporary directory and store them in the phase array
        for out_name in out_names:
            
            # the files are stored in a temporary directory with a path:
            #   /temp_dir/cellID.npy
            cellID = os.path.basename(out_name).split('.')[0]  # get cellID from file name
            cellnum = ct.get_cellnum(cellID)  # get cell number from cellID
            
            # read coordinates from temp file
            try:
                crd = np.load(out_name)  # np.array(ndomain, ncpoy_max, nspot_max, 3)
            except IOError:
                "File {} not found.".format(out_name)
            
            # store coordinates in the list
            crd_list[cellnum] = crd
        
        # stack the list of coordinates into a single array
        coordinates_phsd = np.stack(crd_list, axis=0)  # np.array(ncell, ndomain, ncopy_max, nspot_max, 3)
        for cellnum in range(ct.ncell):
            assert np.array_equal(coordinates_phsd[cellnum, :, :, :, :], crd_list[cellnum],
                                  equal_nan=True), "Coordinates do not match."
                
        return coordinates_phsd



# Auxiliary functions

def phase_cell_coordinates(crd, phs, ncopy_max):
    """Phases the coordinates of a single cell, creating a new array with multiple copy labels.
    
    This function is called the parallel task in the child classes.

    Args:
        crd (np.array(ndomain, nspot_max, 3), np.float32)
        phs (np.array(ndomain, nspot_max), np.int32)
    Returns:
        crd_phased (np.array(ndomain, ncopy_max, nspot_max, 3), np.float32)
    """
    
    # get attributes
    ndomain, nspot_max, _ = crd.shape
        
    # initialize the phased coordinates array
    crd_phsd = np.full((ndomain, ncopy_max, nspot_max, 3),
                       np.nan)  # np.array(ndomain, ncopy_max, nspot_max, 3)
    
    # loop over the phase labels
    for cp in np.unique(phs):
        
        # The spots with phase label 0, i.e. outliers, are ignored
        if cp == 0:
            continue
        
        # create a mask for the current phase label
        cp_mask = phs == cp  # np.array(ndomain, nspot_max) of bool
        
        # extend the mask to match the shape of the coordinates array
        cp_mask_ext = np.expand_dims(cp_mask, axis=2)  # np.array(ndomain, nspot_max, 1)
        cp_mask_ext = np.repeat(cp_mask_ext, repeats=3, axis=2)  # np.array(ndomain, nspot_max, 3)
                
        # check that the mask is correct
        for i in range(3):
            assert np.array_equal(cp_mask[:, :], cp_mask_ext[:, :, i]),\
                "Mask for coordinates is not correct."
        
        # create copy of crd that is nan outside the mask
        crd_cp = np.full(crd.shape, np.nan)  # np.array(ndomain, nspot_max, 3)
        crd_cp[cp_mask_ext] = crd[cp_mask_ext]
        
        # create a copy of crd_phsd that is not-nan for the current copy
        # this is required to avoid shape mismatch when updating crd_phsd
        crd_phsd_cp = np.full(crd_phsd.shape, np.nan)  # np.array(ndomain, ncopy_max, nspot_max, 3)
        crd_phsd_cp[:, cp-1, :, :] = crd_cp[:, :, :]
        
        # update the phased coordinates
        crd_phsd[~np.isnan(crd_phsd_cp)] = crd_phsd_cp[~np.isnan(crd_phsd_cp)]
        
    return crd_phsd
    

def reorder_spots(crd):
    """Reorders the coordinate spots of a single cell by putting the NaNs at the end.
    
    This function is called in the parallel task in the child classes.
    
    Args:
        crd (np.array(ndomain, ncopy_max, nspot_max, 3), np.float32)
    
    Returns:
        crd_reord (np.array(ndomain, ncopy_max, nspot_max, 3), np.float32)
    """
    
    ndomain, ncopy_max, nspot_max, _ = crd.shape
    
    crd_reord = np.copy(crd)
    
    for d in range(ndomain):
        for c in range(ncopy_max):
            crd_reord[d, c, :, :] = crd[d, c, np.argsort(np.isnan(crd[d, c, :, 0])), :]
    
    return crd_reord
    
    