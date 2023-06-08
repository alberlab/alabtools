import numpy as np
from scipy.optimize import minimize
from functools import partial

def compute_pi(eff, phased):
    """Computes the probability of observing nu given n.

    Args:
        eff (float): Detection efficiency.
        phased (bool): Whether the data is phased or not.
    
    Returns:
        pi (np.array((n_nu, n_n)), dtype=float):
                    Probability of observing nu given n.
    """
    
    # Check input
    assert isinstance(eff, float), "eff must be a float."
    assert eff >= 0 and eff <= 1, "eff must be between 0 and 1."
    assert isinstance(phased, bool), "phased must be a boolean."
    
    if phased:
        # There are 2 possible values of n (1, 2)
        # and 3 possible values of nu (0, 1, 2)
        pi = np.zeros((3, 2), dtype=float)
        # n = 1
        pi[0, 0] = (1 - eff)  # nu=0, n=1
        pi[1, 0] = eff  # nu=1, n=1
        pi[2, 0] = 0  # nu=2, n=1
        # n = 2
        pi[0, 1] = (1 - eff) ** 2  # nu=0, n=2
        pi[1, 1] = 2 * eff * (1 - eff)  # nu=1, n=2
        pi[2, 1] = eff ** 2  # nu=2, n=2
    
    if not phased:
        # There are 3 possible values of n (2, 3, 4)
        # and 5 possible values of nu (0, 1, 2, 3, 4)
        pi = np.zeros((5, 3), dtype=float)
        # n = 2
        pi[0, 0] = (1 - eff) ** 2  # nu=0, n=2
        pi[1, 0] = 2 * eff * (1 - eff)  # nu=1, n=2
        pi[2, 0] = eff ** 2  # nu=2, n=2
        pi[3, 0] = 0  # nu=3, n=2
        pi[4, 0] = 0  # nu=4, n=2
        # n = 3
        pi[0, 1] = (1 - eff) ** 3  # nu=0, n=3
        pi[1, 1] = 3 * eff * (1 - eff) ** 2  # nu=1, n=3
        pi[2, 1] = 3 * eff ** 2 * (1 - eff)  # nu=2, n=3
        pi[3, 1] = eff ** 3  # nu=3, n=3
        pi[4, 1] = 0  # nu=4, n=3
        # n = 4
        pi[0, 2] = (1 - eff) ** 4  # nu=0, n=4
        pi[1, 2] = 4 * eff * (1 - eff) ** 3  # nu=1, n=4
        pi[2, 2] = 6 * eff ** 2 * (1 - eff) ** 2  # nu=2, n=4
        pi[3, 2] = 4 * eff ** 3 * (1 - eff)  # nu=3, n=4
        pi[4, 2] = eff ** 4  # nu=4, n=4
    
    return pi

def compute_phi(eff, pr, phased):
    """Computes the expected fraction of domains
    with nu counts given the replication probability pr.
    
    Note that pr = 0 for G1 and pr = 1 for G2.

    Args:
        eff (float): Detection efficiency.
        pr (float): Probability of replication.
        phased (bool): Whether the data is phased or not.
    
    Returns:
        phi (np.array((n_nu)), dtype=float):
                    Expected fraction of domains with nu counts.
    """
    
    # Check input
    assert isinstance(eff, float), "eff must be a float."
    assert eff >= 0 and eff <= 1, "eff must be between 0 and 1."
    assert isinstance(pr, float), "pr must be a float."
    assert pr >= 0 and pr <= 1, "pr must be between 0 and 1."
    assert isinstance(phased, bool), "phased must be a boolean."
    
    # Compute the probability of observing nu given n
    pi = compute_pi(eff, phased)
    
    if phased:
        # There are 3 possible values of nu (0, 1, 2)
        phi = np.zeros(3, dtype=float)
        for nu in range(3):
            phi[nu] = (1 - pr) * pi[nu, 0] + pr * pi[nu, 1]
    
    if not phased:
        # There are 5 possible values of nu (0, 1, 2, 3, 4)
        phi = np.zeros(5, dtype=float)
        for nu in range(5):
            phi[nu] = (1 - pr) ** 2 * pi[nu, 0] + \
                        2 * pr * (1 - pr) * pi[nu, 1] +\
                           pr ** 2 * pi[nu, 2]
    
    return phi

def efficiency_cost_function(eff, f, pr, phased):
    """Computes the cost function for the estimation
    of the detection efficiency.

    Args:
        eff (float): Detection efficiency.
        f (np.array, dtype=float): Observed fraction of events with nu counts.
        pr (float): Probability of replication.
        phased (bool): Whether the data is phased or not.

    Returns:
        cost (float): Cost function.
    """

    # Check the input
    if isinstance(eff, list) or isinstance(eff, np.ndarray):
        assert len(eff) == 1, "eff must be a scalar."
        eff = eff[0]
    assert isinstance(eff, float), "eff must be a float."
    assert eff >= 0 and eff <= 1, "eff must be between 0 and 1."
    assert isinstance(phased, bool), "phased must be a boolean."
    if phased:
        assert f.shape == (3,), "f must have 3 rows."
    else:
        assert f.shape == (5,), "f must have 5 rows."
    assert isinstance(pr, float), "pr must be a float."
    assert pr >= 0 and pr <= 1, "pr must be between 0 and 1."
    
    # Compute the theoretical fractions
    phi = compute_phi(eff, pr, phased)
    assert phi.shape == f.shape, "pi and f must have the same shape."
    
    # Compute the cost function
    cost = ((f - phi) ** 2).sum() ** 0.5
    
    return cost

def efficiency_optimization(f, pr, phased):
    """Computes the detection efficiency for each cell
    by minimizing the cost function.

    Args:
        f (np.array((3, ncell), dtype=float)):
                    Fraction of events with 0, 1, and 2 counts.
        pr (np.array((ncell), dtype=float)):
                    Probability of replication.
        phased (bool): Whether the data is phased or not.

    Returns:
        efficiency (np.array((ncell), dtype=float)):
                    Detection efficiency for each cell.
        cost (np.array((ncell), dtype=float)):
                    Minimized cost function for each cell.
    """
    
    # Check input
    assert isinstance(phased, bool), "phased must be a boolean."
    if phased:
        assert f.shape[0] == 3, "f must have 3 rows."
    else:
        assert f.shape[0] == 5, "f must have 5 rows."
    assert f.shape[1] == len(pr), "f and pr must have the same length."
    
    # Take the number of cells
    ncell = f.shape[1]
    
    # Initialize the list of efficiencies and of cost residuals
    efficiency = []
    cost_residual = []
    
    # Loop over cells to impute the efficiency
    for cell in range(ncell):
        cost = partial(efficiency_cost_function,
                       f=f[:, cell],
                       pr=pr[cell],
                       phased=phased)
        res = minimize(cost, x0=0.5, bounds=[(0, 1)])
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
