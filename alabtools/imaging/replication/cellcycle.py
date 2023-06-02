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
        segmentID (int): segmentation ID (index of the segmentation
                                          in the segmentation array)
        cfg (_type_): _description_
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
    # Sort the cell cycle array back to the original order
    cycle = cycle[np.argsort(np.argsort(volume))]
    
    # Normalize the spots matrix (rho matrix)
    rho = normalize(nraw, cycle)
    
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
        raise ValueError("{} must be a bedfile\
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
        cfg (dict): Configuration dictionary.
        tempdir (str): Temporary directory where the data is stored.

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
    idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x = x[idx]
    y = y[idx]
    
    # Compute Pearson correlation coefficient
    r = pearsonr(x, y)[0]
    
    return r

def normalize(nspot, cycle):
    """Computes the normalized spots matrix.

    Args:
        nspot (np.array(ncell, ndomain, ncopy_max), dtype=int): single-cell spots matrix.
        cycle (np.array(ndomain), dtype=int): cell cycle (G1=0, S=1 or G2=2) array.

    Returns:
        nspot_norm (np.array(ncell, ndomain, ncopy_max), dtype=float): normalized single-cell spots matrix.
    """
    
    # If cycle doesn't have G1 or G2 cells, throw an error
    if not np.any(cycle == 0) or not np.any(cycle == 2):
        raise ValueError("cycle must have G1 and G2 cells")
    
    # Assert that the input arrays have the correct shape
    ncell, ndomain, _ = nspot.shape
    assert cycle.shape[0] == ncell,\
        "nspot and cycle must have the same number of cells"
    
    # Isolate G1 and G2 submatrices    
    nspot_g1 = nspot[cycle == 0, :, :]
    nspot_g2 = nspot[cycle == 2, :, :]
    
    # Compute the bias arrays
    # Since the cells in G1 and G2 are not replicating,
    # variation in the total number of spots is due noise or bias.
    # If we see that a domain has systematically more/less spots than other in G1 or G2,
    # we can assume that this is due to bias and not noise
    # (for example GC rich domains are detected more likely than AT rich domains).
    # Therefore, we can estimate the bias by computing the total number of spots
    # in each domain in G1 and G2.
    bias_g1 = np.nansum(nspot_g1, axis=(0, 2))  # np.array(ndomain)
    bias_g2 = np.nansum(nspot_g2, axis=(0, 2))
    
    # Compute the normalized spots matrix
    nspot_norm = np.copy(nspot)
    
    # nspot_norm is a 3D array (ncell, ndomain, ncopy_max)
    # bias_g1 and bias_g2 are 1D arrays (ndomain)
    # I want to divide each element of nspot_norm by the corresponding element of bias_g1 or bias_g2
    bias_g1 = np.reshape(bias_g1, (1, ndomain, 1))  # np.array(1, ndomain, 1)
    bias_g2 = np.reshape(bias_g2, (1, ndomain, 1))
    
    # Normalize the biases
    bias_g1 = bias_g1 / np.nanmean(bias_g1)
    bias_g2 = bias_g2 / np.nanmean(bias_g2)
    
    # Set the bias as the sum and normalize again
    bias = bias_g1 + bias_g2
    bias = bias / 2
    
    # Normalize the spots matrix
    nspot_norm = nspot_norm / bias
    
    return nspot_norm
