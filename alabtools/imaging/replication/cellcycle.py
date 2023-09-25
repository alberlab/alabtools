import os
import sys
import time
import numpy as np
import h5py
from scipy.stats import pearsonr
from alabtools.utils import Genome, Index

def safe_h5py_read(hdf5, log_file):
    """Safe h5py read.
    
    Reads the data from the temporary files with h5py.
    
    If the reading fails, it waits 10 seconds and tries again.
    If it fails again, it raises an error.

    Args:
        hdf5 (h5py.File): h5py file object.
        log_file (str): Path to the log file.

    Returns:
        segmentation (np.array(nsegment, 2), dtype=int): segmentation array.
        chromstr (np.array(nspot), dtype='U10'): chromosome array.
        nraw (np.array(nspot, ndomain, ncopy_max), dtype=int): raw single-cell spot counts.
        volume (np.array(nspot), dtype=float): cell volume array.
    """
    
    # Read the data from the temporary files with h5py
    try:
        segmentation = hdf5['segmentation'][:]
        chromstr = hdf5['chromstr'][:].astype('U10')
        nraw = hdf5['nraw'][:]
        volume = hdf5['volume'][:]
    except:
        # If the reading fails, wait 10 seconds and try again
        time.sleep(10)
        try:
            segmentation = hdf5['segmentation'][:]
            chromstr = hdf5['chromstr'][:].astype('U10')
            nraw = hdf5['nraw'][:]
            volume = hdf5['volume'][:]
        except:
            # If it fails again, raise an error
            log_file.write('Error reading the temporary files with h5py\n')
            raise ValueError("Error reading the temporary files with h5py")
    
    return segmentation, chromstr, nraw, volume

def parallel_function(segmentID, cfg, temp_dir):
    """Parallel function for cell cycle imputation.
    
    It computes the Pearson correlation coefficient between the
    simulated RT signal and the experimental one for a given
    G1/S/G2 segmentation.
    
    Saves the cell-cycle state array and the Pearson correlation.
    
    The data is saved as a Dictionary (with pickle) in the temporary
    directory.

    Args:
        segmentID (int): Segmentation ID (index of the segmentation
                                          in the segmentation array)
        cfg (dict): Configuration dictionary.
        temp_dir (str): Temporary directory where the data is stored.

    Returns:
        out_name (str): Name of the output file.
    """

    # Create a log file to write progress
    log_file = os.path.join(temp_dir, '{}_log.txt'.format(segmentID))
    f = open(log_file, 'w')
    f.write('Starting parallel function\n')
    
    # Read the data from the temporary files with h5py
    with h5py.File(os.path.join(temp_dir, 'data_for_nodes.hdf5'), 'r') as hdf5:
        segmentation, chromstr, nraw, volume = safe_h5py_read(hdf5, log_file)
    f.write('Files read\n')
    
    # Number of cells in G1 and G2
    ncell_g1, ncell_g2 = segmentation[segmentID]
    ncell_g1, ncell_g2 = int(ncell_g1), int(ncell_g2)
    ncell = nraw.shape[0]
    
    # Define the cell cycle array:
    #   0: G1 (first ncells_g1 cells with the smallest volume)
    #   2: G2 (last ncells_g2 cells with the largest volume)
    #   1: S (all the other cells in between)
    cycle = np.ones(ncell, dtype=int)
    cycle[:ncell_g1] = 0
    cycle[(ncell - ncell_g2):] = 2
    # The cell cycle array is sorted by volume (low to high)
    # Sort the cell cycle array back to the original order
    cycle = cycle[np.argsort(np.argsort(volume))]
    f.write('cycle defined\n')
    
    # Normalize the spots matrix (rho matrix)
    rho = normalize_bias(nraw, cycle)
    f.write('bias normalized\n')
    
    # Isolate the S phase submatrix
    rho_s = rho[cycle == 1, :, :]
    
    # Compute the simulated RT signal
    rt_sim = np.nansum(rho_s, axis=(0, 2))
    f.write('rt_sim computed\n')
    
    # Read the experimental RT signal
    rt_bedfile = cfg['rt_bedfile']
    assembly = cfg['assembly']
    rt_exp_idx = Index(rt_bedfile, genome=Genome(assembly))
    f.write('rt_exp loaded\n')
    try:
        rt_exp = rt_exp_idx.track0
    except:
        raise ValueError("{} must be a BedGraph, \
            with a single track and no header".format(rt_bedfile))
    
    # Remove chrX and chrY
    rt_sim_nosex = rt_sim[np.logical_and(chromstr != 'chrX', chromstr != 'chrY')]
    # rt_exp.pop_chromosome('chrX')  TODO: remove comment when version is updated
    # rt_exp.pop_chromosome('chrY')
    f.write('rt_sim_nosex computed\n')
    
    # Compute the Pearson correlation coefficient
    r = clean_pearsonr(rt_sim_nosex, rt_exp)
    f.write('r computed\n')
    
    # Save the cycle as a compressed numpy array
    out_name = os.path.join(temp_dir, '{}.npz'.format(segmentID))
    np.savez_compressed(out_name, cycle=cycle)
    f.write('cycle saved\n')
    
    # Close the log file
    f.close()
    
    # Free memory
    del segmentation, nraw, volume, cycle, rho, rho_s, rt_sim, rt_exp, rt_exp_idx
    
    return r, out_name

def reduce_function(parallel_returns):
    """Reduce function for cell cycle imputation.
    
    Determines the best segmentation based on largest
    Pearson correlation coefficient from all possible segmentations.
    
    Returns the best cycle and Pearson correlation.

    Args:
        parallel_returns (list): List of the parallel returns.

    Returns:
        r_best (float): Pearson correlation coefficient.
        cycle_best (np.array(ncell), dtype=int): Cell cycle array.
    """
    
    sys.stdout.write('Starting reduce function\n')
    
    r_best = 0
    cycle_best = None
    
    # Read the segment IDs and Pearson correlation coefficients
    r_values, out_names = [], []
    for parallel_return in parallel_returns:
        r, out_name = parallel_return
        r_values.append(r)
        out_names.append(out_name)
    r_values = np.array(r_values)
    out_names = np.array(out_names)
    
    # Find the best r and its corresponding segmentation
    r_best = np.max(r_values)
    out_name_best = out_names[np.argmax(r_values)]
    
    # Read the cell cycle array of the best segmentation
    cycle_best = np.load(out_name_best)['cycle']
    
    return r_best, cycle_best


def clean_pearsonr(x, y):
    """Pearson correlation coefficient, ignoring NaNs and Infs.

    Args:
        x (np.array(n), dtype=float): first input array.
        y (np.array(n), dtype=float): second input array.
    
    Returns:
        r (float): Pearson correlation coefficient.
    """
    
    # Convert Infs to NaNs
    x[np.isinf(x)] = np.nan
    y[np.isinf(y)] = np.nan
    
    # Remove NaNs (from both arrays)
    mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    # Compute Pearson correlation coefficient
    r = pearsonr(x, y)[0]
    
    return r

def normalize_bias(nraw, cycle):
    """Normalize the bias in the raw spots counts.

    Args:
        nraw (np.array(ncell, ndomain, ncopy_max), dtype=int): raw single-cell spot counts.
        cycle (np.array(ndomain), dtype=int): cell cycle (G1=0, S=1 or G2=2) array.

    Returns:
        rho (np.array(ncell, ndomain, ncopy_max), dtype=float): normalized single-cell spot counts.
                                                                It is a float signal.
    """
    
    # If cycle doesn't have G1 or G2 cells, throw an error
    if not np.any(cycle == 0) or not np.any(cycle == 2):
        raise ValueError("cycle must have G1 and G2 cells")
    
    # Assert that the input arrays have the correct shape
    ncell, ndomain, _ = nraw.shape
    assert cycle.shape[0] == ncell,\
        "nraw and cycle must have the same number of cells"
    
    # Isolate G1 and G2 raw spots    
    nraw_g1 = nraw[cycle == 0, :, :]
    nraw_g2 = nraw[cycle == 2, :, :]
    
    # Compute the bias arrays
    # Since the cells in G1 and G2 are not replicating,
    # variation in the total number of spots is due noise or bias.
    # If we see that a domain has systematically more/less spots than others in G1 or G2,
    # we can assume that this is due to bias and not noise
    # (for example GC rich domains are detected more likely than AT rich domains).
    # Therefore, we can estimate the bias by computing the total number of spots
    # in each domain in G1 and G2.
    bias_g1 = np.nansum(nraw_g1, axis=(0, 2))  # np.array(ndomain)
    bias_g2 = np.nansum(nraw_g2, axis=(0, 2))
    
    # Rescale the bias arrays to have mean 1
    bias_g1 = bias_g1 / np.nanmean(bias_g1)
    bias_g2 = bias_g2 / np.nanmean(bias_g2)
    
    # Set the total bias as mean of the G1 and G2 biases
    bias = (bias_g1 + bias_g2) / 2
    
    # If bias_g1 has NaNs, set the bias as bias_g2 and vice versa
    bias[np.isnan(bias_g1)] = bias_g2[np.isnan(bias_g1)]
    bias[np.isnan(bias_g2)] = bias_g1[np.isnan(bias_g2)]
    
    # Rescale the bias to have mean 1
    # (again, since NaNs could have screwed up the mean)
    bias = bias / np.nanmean(bias)
    
    # Reshape the bias array to be able to broadcast it
    bias = np.reshape(bias, (1, ndomain, 1))  # np.array(1, ndomain, 1)
    
    # Compute the normalized spots matrix
    rho = np.copy(nraw)
    rho = rho / bias
    
    return rho
