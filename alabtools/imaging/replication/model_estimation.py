import os
import numpy as np
from scipy.optimize import minimize
import h5py
from . import bernoulli

def mle(rho, only_eff):
    """Computes the maximum likelihood estimates of the parameters
    of the replication model in a single cell."""
    nu = np.round(rho).astype(int)
    if only_eff:
        nu[nu > 2] = 2
    def log_likelihood(x):
        log_lkl = np.zeros(nu.shape)
        for nu_val in np.unique(nu):
            log_lkl[nu == nu_val] = np.log(bernoulli.compute_pi_prime(nu_val, x[0], 1., x[1], phased=True, only_eff=only_eff))
        log_lkl[np.isinf(log_lkl)] = np.nan
        return - np.nansum(log_lkl)
    res = minimize(log_likelihood, x0=[0.5, 0.5], bounds=[(0.0001, 0.9999), (0.0001, 0.9999)])
    eps = res.x[0]
    pr = res.x[1]
    return eps, pr

def parallel_function(cellID, temp_dir):
    
    # Read the data from the temporary file
    with h5py.File(os.path.join(temp_dir, 'data_for_nodes.hdf5'), 'r') as hdf5:
        cell_labels = hdf5['cell_labels'][:].astype('U10')
        cellnum = np.where(cell_labels == cellID)[0][0]
        rho = hdf5['rho'][cellnum, :, :].flatten()
        rho = rho[~np.isnan(rho)]
        pr = hdf5['pr'][cellnum]
    
    # Compute the MLE estimates
    eps = mle(rho, pr, only_eff=True)
    return eps, cellID

def reduce_function(parallel_returns, cell_labels):
    eps_arr = np.zeros(len(cell_labels)).astype(float)
    for parallel_return in parallel_returns:
        eps, cellID = parallel_return
        cellnum = np.where(cell_labels == cellID)[0][0]
        eps_arr[cellnum] = eps
    return eps_arr
