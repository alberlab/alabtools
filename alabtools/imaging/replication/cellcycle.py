import os
import numpy as np
import pickle
from scipy.stats import pearsonr
from alabtools.utils import Genome, Index

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
    
    # Read the data from the temporary files
    # Number of cells in G1 and G2
    segmentation = np.load(os.path.join(temp_dir, 'segmentation.npy'))
    ncell_g1, ncell_g2 = segmentation[segmentID]
    ncell_g1, ncell_g2 = int(ncell_g1), int(ncell_g2)
    # Raw number of spots
    nraw = np.load(os.path.join(temp_dir, 'nraw.npy'))
    ncell = nraw.shape[0]
    # Cell nuclei volumes
    volume = np.load(os.path.join(temp_dir, 'volume.npy'))
    
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
    
    # Normalize the spots matrix (rho matrix)
    rho = normalize_bias(nraw, cycle)
    
    # Isolate the S phase submatrix
    rho_s = rho[cycle == 1, :, :]
    
    # Compute the simulated RT signal
    rt_sim = np.nansum(rho_s, axis=(0, 2))
    
    # Read the experimental RT signal
    rt_bedfile = cfg['rt_bedfile']
    assembly = cfg['assembly']
    rt_exp_idx = Index(rt_bedfile, genome=Genome(assembly))
    try:
        rt_exp = rt_exp_idx.track0
    except:
        raise ValueError("{} must be a BedGraph, \
            with a single track and no header".format(rt_bedfile))
    
    # Compute the Pearson correlation coefficient
    r = clean_pearsonr(rt_sim, rt_exp)
    
    # Save the data in a dictionary
    out_name = os.path.join(temp_dir, '{}.pkl'.format(segmentID))
    with open(out_name, 'wb') as f:
        pickle.dump({'cycle': cycle, 'r': r}, f)
    
    # Free memory
    del segmentation, nraw, volume, cycle, rho, rho_s, rt_sim, rt_exp, rt_exp_idx
    
    return out_name

def reduce_function(out_names):
    """Reduce function for cell cycle imputation.
    
    Determines the best segmentation based on largest
    Pearson correlation coefficient from all possible segmentations.
    
    Returns the best cycle and Pearson correlation.

    Args:
        out_names (list): List of the output files.

    Returns:
        r_best (float): Pearson correlation coefficient.
        cycle_best (np.array(ncell), dtype=int): Cell cycle array.
    """
    
    r_best = 0
    cycle_best = None
    
    for out_name in out_names:
        # Load the data from the dictionary
        with open(out_name, 'rb') as f:
            data = pickle.load(f)
        
        # If the Pearson correlation is larger than the current best,
        # update the current best
        if data['r'] > r_best:
            r_best = data['r']
            cycle_best = data['cycle']
    
    # If cycle_best is None, raise an error
    if cycle_best is None:
        raise ValueError("Something went wrong (cycle_best is None")
    
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
