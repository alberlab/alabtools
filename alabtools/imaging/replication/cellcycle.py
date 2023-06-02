import os
import numpy as np
import pickle
from scipy.stats import pearsonr

def parallel_function(segmentID, cfg, temp_dir):
    """Parallel function for computing the simulated RT signal
    for a G1/S/G2 cell cycle segmentation.
    
    Saves the cell-cycle state array and the Pearson correlation
    between the simulated RT signal and the experimental one.
    
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
    
    # Define the cell cycle state array:
    #   0: G1 (first ncells_g1 cells with the smallest volume)
    #   2: G2 (last ncells_g2 cells with the largest volume)
    #   1: S (all the other cells in between)
    cycle_state = np.ones(ncell, dtype=int)
    cycle_state[:ncell_g1] = 0
    cycle_state[(ncell - ncell_g2):] = 2
    # Sort the cell cycle state array back to the original order
    cycle_state = cycle_state[np.argsort(np.argsort(volume))]
    
    # Normalize the spots matrix (rho matrix)
    rho = normalize(nraw, cycle_state)
    
    # Isolate the S phase submatrix
    rho_s = rho[cycle_state == 1, :, :]
    
    # Compute the simulated RT signal
    rt_sim = np.nansum(rho_s, axis=(0, 2))
    
    # Compute the Pearson correlation coefficient
    ### TODO: compute the Pearson correlation coefficient
    ###      (read RT and genome from the cfg file,
    ###       use Index to read file)
    r = 0
    
    # Save the data in a dictionary
    out_name = os.path.join(temp_dir, '{}.pkl'.format(segmentID))
    with open(out_name, 'wb') as f:
        pickle.dump({'cycle_state': cycle_state, 'r': r}, f)
    
    # Close numpy arrays
    del segmentation, nraw, volume, cycle_state, rho, rho_s, rt_sim
    
    return out_name

def reduce_function(out_names, cfg, tempdir):
    # This function is executed on the master node
    return None


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

def normalize(nspot, cycle_state):
    """Computes the normalized spots matrix.

    Args:
        nspot (np.array(ncell, ndomain, ncopy_max), dtype=int): single-cell spots matrix.
        cycle_state (np.array(ndomain), dtype=int): cell cycle state (G1, S or G2) for each domain.
                                                    G1=0, S=1, G2=2.

    Returns:
        nspot_norm (np.array(ncell, ndomain, ncopy_max), dtype=float): normalized single-cell spots matrix.
    """
    
    # If cycle_state doesn't have G1 or G2 cells, throw an error
    if not np.any(cycle_state == 0) or not np.any(cycle_state == 2):
        raise ValueError("cycle_state must have G1 and G2 cells")
    
    # Assert that the input arrays have the correct shape
    ncell, ndomain, _ = nspot.shape
    assert cycle_state.shape[0] == ncell,\
        "nspot and cycle_state must have the same number of cells"
    
    # Isolate G1 and G2 submatrices    
    nspot_g1 = nspot[cycle_state == 0, :, :]
    nspot_g2 = nspot[cycle_state == 2, :, :]
    
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
