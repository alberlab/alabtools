import numpy as np
from functools import partial
import sys
import os
import tempfile
import pickle
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
        coordinates_phsd, intensity_phsd = self.controller.map_reduce(self.parallel_task,
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
                               self.ct.cell_labels,
                               intensity=intensity_phsd)
        # sort and trim
        ct_phased.sort_copies()
        ct_phased.sort_spots()
        ct_phased.trim()
        
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
            cfg (dict): configuration dictionary
        Returns:
            coordinates_phsd (np.array(ncell, ndomain, ncopy_max, nspot_max, 3), np.float32)
            intensity_phsd (np.array(ncell, ndomain, ncopy_max, nspot_max), np.float32) or None
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
        
        # list of coordinates/intensity, outputs from the child nodes
        crd_list = [None for _ in range(ct.ncell)]
        if 'intensity' in ct:
            lum_list = [None for _ in range(ct.ncell)]
        
        # Read the results from the temporary directory and store them in the phase array
        for out_name in out_names:
            
            # the files are stored in a temporary directory with a path:
            #   /temp_dir/cellID.pickle
            cellID = os.path.basename(out_name).split('.')[0]  # get cellID from file name
            cellnum = ct.get_cellnum(cellID)  # get cell number from cellID
            
            # open the file
            with open(out_name, 'rb') as f:
                data = pickle.load(f)
                crd = data['cell_coordinates_phased']
                if 'intensity' in ct:
                    lum = data['cell_intensity_phased']
            
            # store coordinates in the list
            crd_list[cellnum] = crd
            if 'intensity' in ct:
                lum_list[cellnum] = lum
        
        # stack the list of coordinates into a single array
        coordinates_phsd = np.stack(crd_list, axis=0)  # np.array(ncell, ndomain, ncopy_max, nspot_max, 3)
        for cellnum in range(ct.ncell):
            assert np.array_equal(coordinates_phsd[cellnum, :, :, :, :], crd_list[cellnum],
                                  equal_nan=True), "Coordinates do not match."
        # stack the list of intensity into a single array
        if 'intensity' in ct:
            intensity_phsd = np.stack(lum_list, axis=0)  # np.array(ncell, ndomain, ncopy_max, nspot_max)
            for cellnum in range(ct.ncell):
                assert np.array_equal(intensity_phsd[cellnum, :, :, :], lum_list[cellnum],
                                      equal_nan=True), "Intensity does not match."
        else:
            intensity_phsd = None
                
        return coordinates_phsd, intensity_phsd



# Auxiliary functions

def phase_cell_data(arr, phs, ncopy_max):
    """Phases the data (coordinates, intensity) of a single cell,creating a new array with multiple copy labels.
    
    This function is called the parallel task in the child classes.

    Args:
        arr (np.array(ndomain, nspot_max, (3)), the (3) is optional
        phs (np.array(ndomain, nspot_max), np.int32)
    Returns:
        arr_phased (np.array(ndomain, ncopy_max, nspot_max, (3)), np.float32)
    """
    # initialize the phased array
    # add a dimension ncopy_max in the second position to the shape of arr
    phsd_shape = arr.shape[:1] + (ncopy_max,) + arr.shape[1:]
    arr_phsd = np.full(phsd_shape, np.nan)  # np.array(ndomain, ncopy_max, nspot_max, (3))
    # loop over the phase labels
    for cp in np.unique(phs):
        # The spots with phase label 0, i.e. outliers, are ignored
        if cp == 0:
            continue
        # create a mask for the current phase label
        cp_mask = phs == cp  # np.array(ndomain, nspot_max) of bool
        # extend the mask to match the shape of the coordinates array
        if len(arr.shape) == 3:
            cp_mask = np.expand_dims(cp_mask, axis=2)  # np.array(ndomain, nspot_max, 1)
            cp_mask = np.repeat(cp_mask, repeats=3, axis=2)  # np.array(ndomain, nspot_max, 3) 
        # create copy of arr that is nan for other copies (outside the mask)
        arr_cp = np.copy(arr)
        arr_cp[~cp_mask] = np.nan
        # create a copy of arr_phsd that is nan for other copies
        # this is required to avoid shape mismatch when updating arr_phsd
        arr_phsd_cp = np.full(arr_phsd.shape, np.nan)  # np.array(ndomain, ncopy_max, nspot_max, (3))
        if len(arr.shape) == 2:
            arr_phsd_cp[:, cp-1, :] = arr_cp[:, :]
        elif len(arr.shape) == 3:
            arr_phsd_cp[:, cp-1, :, :] = arr_cp[:, :, :]
        # update the phased data
        arr_phsd[~np.isnan(arr_phsd_cp)] = arr_phsd_cp[~np.isnan(arr_phsd_cp)]
    return arr_phsd
    

def reorder_spots(arr):
    """Reorders the spots of a single cell data by putting the NaNs at the end.
    
    This function is called in the parallel task in the child classes.
    
    Args:
        arr (np.array(ndomain, ncopy_max, nspot_max, (3)), the (3) is optional
    Returns:
        arr_srt (np.array(ndomain, ncopy_max, nspot_max, (3))
    """
    ndomain, ncopy_max = arr.shape[0], arr.shape[1]
    arr_srt = np.copy(arr)
    for d in range(ndomain):
        for c in range(ncopy_max):
            if len(arr.shape) == 3:
                arr_srt[d, c] = arr[d, c, np.argsort(np.isnan(arr[d, c, :]))]
            elif len(arr.shape) == 4:
                arr_srt[d, c] = arr[d, c, np.argsort(np.isnan(arr[d, c, :, 0])), :]
    return arr_srt
    
    