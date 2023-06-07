import numpy as np
from scipy.optimize import minimize
from functools import partial

def theoretical_fractions(eps, pr):
    """Computes the theoretical fractions of events
    with 0, 1, and 2 counts.

    Args:
        eps (float): Detection efficiency.
        pr (float): Probability of replication.

    Returns:
        pi0 (float): Theoretical fraction of events with 0 counts.
        pi1 (float): Theoretical fraction of events with 1 count.
        pi2 (float): Theoretical fraction of events with 2 counts.
    """
    
    pi0 = (1 - pr) * (1 - eps) + pr * (1 - eps) ** 2
    pi1 = (1 - pr) * eps + pr * 2 * (1 - eps) * eps
    pi2 = pr * eps ** 2
    
    return pi0, pi1, pi2

def efficiency_cost_function(eps, f0, f1, f2, pr):
    """Computes the cost function for the estimation
    of the detection efficiency.

    Args:
        eps (float): Detection efficiency.
        f0 (float): Fraction of events with 0 counts.
        f1 (float): Fraction of events with 1 count.
        f2 (float): Fraction of events with 2 counts.
        pr (float): Probability of replication.

    Returns:
        cost (float): Cost function.
    """
    
    # Compute the theoretical fractions
    pi0, pi1, pi2 = theoretical_fractions(eps, pr)
    
    # Compute the cost function
    cost = ((pi0 - f0) ** 2 +
            (pi1 - f1) ** 2 +
            (pi2 - f2) ** 2) ** 0.5
    
    return cost

def efficiency_optimization(f, pr):
    """Computes the detection efficiency for each cell
    by minimizing the cost function.

    Args:
        f (np.array((3, ncell), dtype=float)):
                    Fraction of events with 0, 1, and 2 counts.
        pr (np.array((ncell), dtype=float)):
                    Probability of replication.

    Returns:
        efficiency (np.array((ncell), dtype=float)):
                    Detection efficiency for each cell.
        cost (np.array((ncell), dtype=float)):
                    Minimized cost function for each cell.
    """
    
    # Check input
    assert f.shape[0] == 3, "f must be a 3xN array."
    assert f.shape[1] == len(pr), "f and pr must have the same length."
    ncell = f.shape[1]
    
    # Initialize the list of efficiencies and of cost residuals
    efficiency = []
    cost_residual = []
    
    # Loop over cells to impute the efficiency
    for cell in range(ncell):
        f0, f1, f2 = f[:, cell]
        cost = partial(efficiency_cost_function,
                       f0=f0, f1=f1, f2=f2, pr=pr[cell])
        res = minimize(cost, x0=0.5, method='Nelder-Mead')
        efficiency.append(res.x[0])
        cost_residual.append(res.fun)
    
    # Convert to numpy arrays
    efficiency = np.array(efficiency)
    cost_residual = np.array(cost_residual)
    
    return efficiency, cost_residual

def likelihood_maximization_n(nu, efficiency, pr):
    
    # Take data dimensions
    ncell, ndomain, ncopy_max = nu.shape
    
    assert efficiency.shape == (ncell,),\
        "efficiency must be a 1D array of length ncell."
    assert pr.shape == (ncell,),\
        "pr must be a 1D array of length ncell."
    
    # Reshape efficiency and pr to match the dimensions of nu
    efficiency = np.expand_dims(efficiency, axis=(1, 2))  # np.array(ncell, 1, 1)
    efficiency = np.repeat(efficiency, repeats=ndomain, axis=1)  # np.array(ncell, ndomain, 1)
    efficiency = np.repeat(efficiency, repeats=ncopy_max, axis=2)  # np.array(ncell, ndomain, nspot_max)
    for cell in range(ncell):
        assert np.all(efficiency[cell, :, :] == efficiency[cell, 0, 0]),\
            "Broadcasting of efficiency failed."
    pr = np.expand_dims(pr, axis=(1, 2))  # np.array(ncell, 1, 1)
    pr = np.repeat(pr, repeats=ndomain, axis=1)  # np.array(ncell, ndomain, 1)
    pr = np.repeat(pr, repeats=ncopy_max, axis=2)  # np.array(ncell, ndomain, nspot_max)
    for cell in range(ncell):
        assert np.all(pr[cell, :, :] == pr[cell, 0, 0]),\
            "Broadcasting of pr failed."
    
    # Initialize the likelihood of no replication (n=1)
    # and the likelihood of replication (n=2)
    lkl_1 = np.zeros((ncell, ndomain, ncopy_max), dtype=float)
    lkl_2 = np.zeros((ncell, ndomain, ncopy_max), dtype=float)
    
    # Compute the likelihood of no replication (n=1)
    lkl_1[nu == 0] = (1 - efficiency[nu == 0])
    lkl_1[nu == 1] = efficiency[nu == 1]
    lkl_1[nu == 2] = 0
    lkl_1 = lkl_1 * (1 - pr)
    
    # Compute the likelihood of replication (n=2)
    lkl_2[nu == 0] = (1 - efficiency[nu == 0]) ** 2
    lkl_2[nu == 1] = 2 * efficiency[nu == 1] * (1 - efficiency[nu == 1])
    lkl_2[nu == 2] = efficiency[nu == 2] ** 2
    lkl_2 = lkl_2 * pr
    
    # Maximize the likelihood: choose n=1 if lkl_1 > lkl_2
    n = np.ones((ncell, ndomain, ncopy_max), dtype=int)
    n[lkl_1 < lkl_2] = 2
    
    # Return the maximum likelihood
    lkl_max = np.copy(lkl_1)
    lkl_max[lkl_1 < lkl_2] = lkl_2[lkl_1 < lkl_2]
    
    return n, lkl_max
