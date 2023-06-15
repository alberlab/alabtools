import numpy as np
from scipy.special import binom
from scipy.optimize import minimize

def compute_pi(nu, n, eps):
    """Computes the probability of observing nu given n
    only in the case of a detection efficiency.
    
    If multiple arguments are arrays, they must have the same shape.

    Args:
        nu (int or np.array((n), dtype=int)):
                Number of observed spots.
        n (int or np.array((n), dtype=int)):
                Number of copies.
        eps (float or np.array((n), dtype=float)):
                Detection efficiency.

    Returns:
        pi (float or np.array((n), dtype=float)):
                Probability of observing nu given n.
    """
    
    # Compute the probability of observing nu given n
    pi = binom(n, nu) * eps ** nu * (1 - eps) ** (n - nu)
    
    return pi

def compute_phi(eps, pr, nu_max, phased):
    """Computes the theoretical fractions of events with
    nu (0, 1, 2, ...) counts in a cell.

    Args:
        eps (float):
                Detection efficiency.
        pr (float):
                Probability of replication.
        nu_max (int):
                Maximum number of spots of interest.
        phased (bool):
                Whether the data is phased or not.

    Returns:
        phi (np.array((nu_max + 1), dtype=float)):
                Theoretical fractions of events with nu counts.
    """
    
    # Set the number of copies array and the respective probabilities
    if phased:
        n_arr = np.array([1, 2])
        prob_arr = np.array([1 - pr, pr])
    else:
        n_arr = np.array([2, 3, 4])
        prob_arr = np.array([(1 - pr) ** 2, 2 * pr * (1 - pr), pr ** 2])
    
    # Initialize the list of theoretical fractions
    phi = []
    
    # Compute the theoretical fractions for each nu
    for nu in range(nu_max + 1):
        phi_nu = np.sum(prob_arr * compute_pi(nu, n_arr, eps))
        phi.append(phi_nu)
    
    phi = np.array(phi)
    
    return phi

def efficiency_cost_function(eps, f, pr, phased):
    """Cost function to impute the efficiency and the noise
    from the fraction of observed events with nu (0, 1, 2, ...).

    Args:
        eps (float):
                Detection efficiency.
        f (np.array((nu_max + 1), dtype=float)):
                Observed fractions of events with nu counts.
        pr (float):
                Probability of replication.
        phased (bool):
                Whether the data is phased or not.

    Returns:
        cost (float):
                Cost value.
    """
    
    # If eps is an array-like, take the first element
    if isinstance(eps, np.ndarray) or isinstance(eps, list):
        assert len(eps) == 1, "eps must be a scalar."
        eps = eps[0]
    
    # Compute the theoretical fractions
    nu_max = len(f) - 1
    phi = compute_phi(eps, pr, nu_max, phased)
    assert phi.shape == f.shape, "pi and f must have the same shape."
    
    # Compute the cost function
    cost = np.sqrt( np.nansum( (f - phi) ** 2 ) )
    
    return cost

def efficiency_optimization(f, pr, phased):
    """Computes the detection efficiency for each cell
    by minimizing the cost function.

    Args:
        f (np.array((nu_max, ncell), dtype=float)):
                    Fraction of events with nu (0, 1, 2, ...) counts.
        pr (np.array((ncell), dtype=float)):
                    Probability of replication.
        phased (bool): Whether the data is phased or not.

    Returns:
        eps_res (np.array((ncell), dtype=float)):
                        Estimated efficiency for each cell.
        eta_res (np.array((ncell), dtype=float)):
                        Estimated noise for each cell.
        costopt_res (np.array((ncell), dtype=float)):
                        Cost function optimized for each cell.
    """
    
    # Check input
    assert isinstance(phased, bool), "phased must be a boolean."
    assert f.shape[1] == len(pr), "f and pr must have the same length."
    
    # Take the number of cells
    ncell = f.shape[1]
    
    # Initialize the list of efficiencies and of cost residuals
    eps_res = []
    costopt_res = []
    
    # Loop over cells to impute the efficiency
    for cell in range(ncell):
        cost = lambda x: efficiency_cost_function(x, f[:, cell], pr[cell], phased)
        res = minimize(cost, x0=[0.5], bounds=[(0.0001, 0.9999)])
        eps_res.append(res.x[0])
        costopt_res.append(res.fun)
    
    # Convert to numpy arrays
    eps_res = np.array(eps_res)
    costopt_res = np.array(costopt_res)
    
    return eps_res, costopt_res

def likelihood_maximization_n(nu, efficiency):
    
    # Take data dimensions
    ncell, ndomain, ncopy_max = nu.shape
    
    assert efficiency.shape == (ncell,),\
        "efficiency must be a 1D array of length ncell."
    
    
    # Reshape efficiency and pr to match the dimensions of nu
    efficiency = np.expand_dims(efficiency, axis=(1, 2))  # np.array(ncell, 1, 1)
    efficiency = np.repeat(efficiency, repeats=ndomain, axis=1)  # np.array(ncell, ndomain, 1)
    efficiency = np.repeat(efficiency, repeats=ncopy_max, axis=2)  # np.array(ncell, ndomain, nspot_max)
    for cell in range(ncell):
        assert np.all(efficiency[cell, :, :] == efficiency[cell, 0, 0]),\
            "Broadcasting of efficiency failed."
    
    # Initialize the likelihood of no replication (n=1)
    # and the likelihood of replication (n=2)
    lkl_1 = np.zeros((ncell, ndomain, ncopy_max), dtype=float)
    lkl_2 = np.zeros((ncell, ndomain, ncopy_max), dtype=float)
    
    # Compute the likelihood of no replication (n=1)
    lkl_1[nu == 0] = (1 - efficiency[nu == 0])
    lkl_1[nu == 1] = efficiency[nu == 1]
    lkl_1[nu == 2] = 0
    lkl_1 = lkl_1
    
    # Compute the likelihood of replication (n=2)
    lkl_2[nu == 0] = (1 - efficiency[nu == 0]) ** 2
    lkl_2[nu == 1] = 2 * efficiency[nu == 1] * (1 - efficiency[nu == 1])
    lkl_2[nu == 2] = efficiency[nu == 2] ** 2
    lkl_2 = lkl_2
    
    # Maximize the likelihood: choose n=1 if lkl_1 > lkl_2
    n = np.ones((ncell, ndomain, ncopy_max), dtype=int)
    n[lkl_1 < lkl_2] = 2
    
    # Return the maximum likelihood
    lkl_max = np.copy(lkl_1)
    lkl_max[lkl_1 < lkl_2] = lkl_2[lkl_1 < lkl_2]
    
    return n, lkl_max


# PRIMED PIS: INCLUDING NOISE PROBABILITY IN THE MODEL

def compute_pi_prime(nu, n, eps, eta):
    """Computes the probability of observing nu given n
    in the case of a detection efficiency and a noise.

    Args:
        nu (int):
                Number of observed spots.
        n (int):
                Number of copies.
        eps (float):
                Detection efficiency.
        eta (float):
                Noise probability.

    Returns:
        pi_p (float or np.array((n), dtype=float)):
                Probability of observing nu given n.
    """
    
    # Assert istances of input
    assert isinstance(nu, int), "nu must be an integer."
    assert isinstance(n, int), "n must be an integer."
    assert isinstance(eps, float), "eps must be a float."
    assert isinstance(eta, float), "eta must be a float."
    assert nu >= 0, "nu must be positive."
    assert n > 0, "n must be positive."
    assert eps >= 0 and eps <= 1, "eps must be between 0 and 1."
    assert eta >= 0 and eta <= 1, "eta must be between 0 and 1."
    
    # Initialize the probability to 0
    pi_p = 0
    # Loop over all possible values of nu_p
    for nu_p in range(nu + 1):
        pi = compute_pi(nu - nu_p, n, eps)
        pi_p += binom(nu, nu_p) * eta ** nu_p * (1 - eta) ** (nu - nu_p) * pi
    
    return pi_p

def compute_mean_from_pi_prime(n, eps, eta):
    """Computes the mean number of observed spots in
    the pi_prime model, fixing the number of copies.

    Args:
        n (int): Number of copies.
        eps (float): Detection efficiency.
        eta (float): Noise probability.

    Returns:
        mean_nu (float): Mean number of observed spots.
    """
    
    # Assert istances of input
    assert isinstance(n, int), "n must be an integer."
    assert isinstance(eps, float), "eps must be a float."
    assert isinstance(eta, float), "eta must be a float."
    # Assert values of input
    assert n > 0, "n must be positive."
    assert eps >= 0 and eps <= 1, "eps must be between 0 and 1."
    assert eta >= 0 and eta <= 1, "eta must be between 0 and 1."
    
    # Define the range of possible values of nu
    # (could be infinite but taking 100 is enough)
    nu_range = np.arange(100)
    
    # Compute the probability of observing each nu
    pi_p_range = []
    for nu in nu_range:
        nu = int(nu)
        pi_p = compute_pi_prime(nu, n, eps, eta)
        pi_p_range.append(pi_p)
    pi_p_range = np.array(pi_p_range)
    
    # Compute the mean number of observed spots
    mean_nu = (nu_range * pi_p_range).sum()
    
    return mean_nu