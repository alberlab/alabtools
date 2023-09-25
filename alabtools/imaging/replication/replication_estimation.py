import os
import numpy as np
import h5py
from scipy.optimize import minimize
from . import bernoulli

def sliding_window_sum(x, size):
    """Computes the sliding window sum of a 1D array.
    Returns an array y of the same size as x such that
    y[i] = sum(x[i-(size-1)/2:i+(size-1)/2]).
    """
    assert size % 2 == 1, "size must be an odd integer"
    s = (size - 1) // 2
    y = np.full(x.shape, np.nan)
    for i in range(s, len(x) - s):
        y[i] = np.sum(x[i-s:i+s+1])
    return y

def mle(nu, eff, size, only_eff):
    # If all the values are NaN, return n=1
    if np.all(nu == 0):
        return np.ones(nu.shape).astype(int)
    # Compute the PIs with n=1 and n=2
    pi1 = bernoulli.compute_pi(nu, int(1 * size), eff, 1., only_eff=only_eff)
    pi2 = bernoulli.compute_pi(nu, int(2 * size), eff, 1., only_eff=only_eff)
    # Maximize the log likelihood
    n = np.ones(nu.shape).astype(int)
    n[pi2 > pi1] = 2  # where both are NaN, n=1
    return n

def impute_param(rho):
    nu = np.round(rho).astype(int)
    nu[nu > 2] = 2
    def minus_log_likelihood(x):
        eff, pr = x[0], x[1]
        log_lkl = np.zeros(nu.shape)
        for x in np.unique(nu):
            log_lkl[nu == x] = np.log((1 - pr) * bernoulli.binomial(x, 1, eff) + pr * bernoulli.binomial(x, 2, eff))
        log_lkl[np.isinf(log_lkl)] = np.nan
        return - np.nansum(log_lkl)
    res = minimize(minus_log_likelihood, x0=[0.5, 0.5], bounds=[(0.0001, 0.9999), (0.0001, 0.9999)])
    eff = res.x[0]
    pr = res.x[1]
    return eff, pr

def impute_n(rho, eff):
    nu = np.round(rho).astype(int)
    nu[nu > 2] = 2
    def log_likelihood(n):
        log_lkl = np.zeros(nu.shape)
        for x in np.unique(nu):
            log_lkl[nu == x] = np.log(bernoulli.binomial(x, n, eff))
        log_lkl[np.isinf(log_lkl)] = np.nan
        return np.nansum(log_lkl)
    lkl_n1 = log_likelihood(1)
    lkl_n2 = log_likelihood(2)
    if lkl_n1 >= lkl_n2:
        return 1
    else:
        return 2

def get_n(rho, win1, win2):
    assert win1 % 2 == 1, "win1 must be an odd integer"
    assert win2 % 2 == 1, "win2 must be an odd integer"
    assert win2 > win1, "win2 must be larger than win1"
    s1 = (win1 - 1) // 2
    s2 = (win2 - 1) // 2
    n = np.ones(rho.shape).astype(int)
    eff = np.full(rho.shape, np.nan)
    pr = np.full(rho.shape, np.nan)
    for i in range(s2, len(rho) - s2):
        rho_i_win1 = rho[i-s1:i+s1+1]
        rho_i_win2 = rho[i-s2:i+s2+1]
        eff_i, pr_i = impute_param(rho_i_win2)
        n_i = impute_n(rho_i_win1, eff_i)
        n[i] = n_i
        eff[i] = eff_i
        pr[i] = pr_i
    return n, eff, pr

def parallel_function(cellID, cfg, temp_dir):
    # Read parameters from the config file
    win1 = cfg['window_size_1']
    win2 = cfg['window_size_2']
    # Read the data from the temporary file
    with h5py.File(os.path.join(temp_dir, 'data_for_nodes.hdf5'), 'r') as hdf5:
        cell_labels = hdf5['cell_labels'][:].astype('U10')
        cellnum = np.where(cell_labels == cellID)[0][0]
        rho = hdf5['rho'][cellnum, :, :]
        efficiency = hdf5['efficiency'][cellnum]
        pr = hdf5['pr'][cellnum]
    # Initialize n
    ndomain, ncopy = rho.shape
    n = np.ones(rho.shape).astype(int)
    efficiency = np.ones(rho.shape).astype(float)
    prob_rep = np.ones(rho.shape).astype(float)
    # Compute the MLE estimates
    for copy in range(ncopy):
        n_copy, eff_copy, prob_rep_copy = get_n(rho[:, copy], win1, win2)
        n[:, copy] = n_copy
        efficiency[:, copy] = eff_copy
        prob_rep[:, copy] = prob_rep_copy
    # Save the results in a temporary file
    np.savez_compressed(os.path.join(temp_dir, '{}_n.npz'.format(cellID)), n=n)
    np.savez_compressed(os.path.join(temp_dir, '{}_efficiency.npz'.format(cellID)), efficiency=efficiency)
    np.savez_compressed(os.path.join(temp_dir, '{}_prob_rep.npz'.format(cellID)), prob_rep=prob_rep)
    return None

def reduce_function(returns, cell_labels, rho_shape, temp_dir):
    # Initialize n
    n = np.ones(rho_shape).astype(int)
    eff = np.ones(rho_shape).astype(float)
    prob_rep = np.ones(rho_shape).astype(float)
    # Loop over the temporary files
    for cellnum, cellID in enumerate(cell_labels):
        # Read the data from the temporary file
        with np.load(os.path.join(temp_dir, '{}_n.npz'.format(cellID))) as data:
            n_cell = data['n']
            assert n_cell.shape == rho_shape[1:], "Cell {} has the wrong shape for n".format(cellID)
            n[cellnum, :, :] = n_cell
        with np.load(os.path.join(temp_dir, '{}_efficiency.npz'.format(cellID))) as data:
            eff_cell = data['efficiency']
            assert eff_cell.shape == rho_shape[1:], "Cell {} has the wrong shape for efficiency".format(cellID)
            eff[cellnum, :, :] = eff_cell
        with np.load(os.path.join(temp_dir, '{}_prob_rep.npz'.format(cellID))) as data:
            prob_rep_cell = data['prob_rep']
            assert prob_rep_cell.shape == rho_shape[1:], "Cell {} has the wrong shape for prob_rep".format(cellID)
            prob_rep[cellnum, :, :] = prob_rep_cell
    return n, eff, prob_rep
