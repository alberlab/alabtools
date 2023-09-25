import numpy as np
from scipy.special import binom, factorial
from scipy.optimize import minimize, basinhopping, shgo

def binomial(nu, n, eps):
    """Computes the binomial probability of observing nu given n.
    
    If multiple arguments are arrays, they must have the same shape.

    Args:
        nu (int or np.array(dtype=int)):
                Number of observed spots.
        n (int or np.array(dtype=int)):
                Number of true spots.
        eps (float or np.array(dtype=float)):
                Detection efficiency.

    Returns:
        (float or np.array(dtype=float)):
                Probability of observing nu given n.
    """
    
    return binom(n, nu) * eps ** nu * (1 - eps) ** (n - nu)

def poisson(z, lam):
    """Computes the Poisson probability of having z false positives.

    Args:
        z (int or np.array(dtype=int)):
                Number of false positives.
        lam (float or np.array(dtype=float)):
                False positive rate.

    Returns:
        (float or np.array(dtype=float)):
                Poisson probability of having z false positives.
    """
    return lam ** z * np.exp(-lam) / factorial(z)

def gaussian(rho, mu, sigma):
    """Computes the Gaussian probability of having a
    continuous normalized count rho.

    Args:
        rho (float or np.array(dtype=float)):
                Normalized (continuous) count.
        mu (float or np.array(dtype=float)):
                Mean of the distribution.
        sigma (float or np.array(dtype=float)):
                Standard deviation of the distribution.

    Returns:
        (float or np.array(dtype=float)):
                Gaussian probability of having rho.
    """
    return np.exp(- (rho - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

def compute_pi(nu, n, eps, lam, only_eff = False):
    """Computes the probability of observing nu given n
    in the case of a detection efficiency and false positives.

    Args:
        nu (int):
                Number of observed spots.
        n (int):
                Number of copies.
        eps (float):
                Detection efficiency.
        lam (float):
                False positive rate.
        only_eff (bool):
                Whether to compute only with the efficiency or not.
                (default: False).

    Returns:
        pi_p (float or np.array((n), dtype=float)):
                Probability of observing nu given n.
    """
    
    # Initialize the probability to 0
    pi = 0
    
    if only_eff:
        pi = binomial(nu, n, eps).astype(float)
        return pi
    
    # Loop over all possible values of x1
    # for x1 in range(n + 1):
    for x1 in range(min(nu, n) + 1):
        pi_x1 = binomial(x1, n, eps).astype(float) * poisson(nu - x1, lam).astype(float)
        # pi_x1[np.logical_or(np.isnan(pi_x1), np.isinf(pi_x1))] = 0
        # pi_x1[nu < x1] = 0
        pi += pi_x1
        
    return pi

def compute_pi_prime(nu, eps, lam, pr, phased, only_eff=False):
    """Computes the probability of observing nu by conditioning
    on all replication events (given by the probability pr).
    
    Args:
        nu (int):
                Number of observed spots.
        eps (float):
                Detection efficiency.
        lam (float):
                False positive rate.
        pr (float):
                Probability of replication.
        phased (bool):
                Whether the data is phased or not.
    Returns:
        pi_p (float):
                Probability of observing nu. """
    if phased:
        pi_prime = (1 - pr) * compute_pi(nu, 1, eps, lam, only_eff) + pr * compute_pi(nu, 2, eps, lam, only_eff)
    else:
        pi_prime = (1 - pr) ** 2 * compute_pi(nu, 2, eps, lam, only_eff) +\
            2 * pr * (1 - pr) * compute_pi(nu, 3, eps, lam, only_eff) +\
                pr ** 2 * compute_pi(nu, 4, eps, lam, only_eff)
    # pi_prime[pi_prime == 0] = np.nan
    return pi_prime

def param_mle(rho, pr):
    """Computes the maximum likelihood estimates of the efficiency and the FPR
    in a single cell."""
    nu = np.round(rho).astype(int)
    def log_likelihood(x):
        log_lkl = np.zeros(nu.shape)
        for nu_val in np.unique(nu):
            log_lkl[nu == nu_val] = np.log(compute_pi_prime(nu_val, x[0], x[1], pr, phased=True))
        return - np.nansum(log_lkl)
    res = minimize(log_likelihood, x0=[0.5, 0.05], bounds=[(0.0001, 0.9999), (0.0001, 0.9999)])
    eps = res.x[0]
    lam = res.x[1]
    # res = basinhopping(log_lkl, x0=[0.5, 0.05])
    # eps = inv_reparam(res.x[0])
    # lam = inv_reparam(res.x[1])
    return eps, lam

def compute_phi(eps, lam, pr, nu_list, phased):
    """Computes the theoretical fractions of events with
    nu (0, 1, 2, ...) counts in a cell.
    Since in principle we can have an infinite number of
    nu counts, we need to specify a set to consider.

    Args:
        eps (float):
                Detection efficiency.
        lam (float):
                False positive rate.
        pr (float):
                Probability of replication.
        nu_list (list(int)):
                List of nu counts to compute the fractions for.
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
    for nu in nu_list:
        phi_nu = np.sum(prob_arr * compute_pi(nu, n_arr, eps, lam))
        phi.append(phi_nu)
    
    phi = np.array(phi)
    
    return phi

def efficiency_cost_function(eps, lam, f, nu, pr, phased):
    """Cost function to impute the efficiency and the noise
    from the fraction of observed events with nu (0, 1, 2, ...).

    Args:
        eps (float):
                Detection efficiency.
        lam (float):
                False positive rate.
        f (np.array(s, dtype=float)):
                Observed fractions of events with nu counts.
        nu (np.array(s, dtype=int)):
                List of nu counts to compute the fractions for.
        pr (float):
                Probability of replication.
        phased (bool):
                Whether the data is phased or not.

    Returns:
        cost (float):
                Cost value.
    """
    
    # If eps or lam is an array-like, take the first element
    if isinstance(eps, np.ndarray) or isinstance(eps, list):
        assert len(eps) == 1, "eps must be a scalar."
        eps = eps[0]
    if isinstance(lam, np.ndarray) or isinstance(lam, list):
        assert len(lam) == 1, "lam must be a scalar."
        lam = lam[0]
    
    # Compute the theoretical fractions
    phi = compute_phi(eps, lam, pr, nu, phased)
    assert phi.shape == f.shape, "pi and f must have the same shape."
    
    # Compute the cost function
    cost = np.sqrt( np.nansum( (f - phi) ** 2 ) )
    
    return cost

def efficiency_optimization(f, nu, pr, phased):
    """Computes the detection efficiency for each cell
    by minimizing the cost function.

    Args:
        f (np.array((s, ncell), dtype=float)):
                    Fraction of events with nu (0, 1, 2, ...) counts.
        nu (np.array((s), dtype=int)):
                    List of nu counts to compute the fractions for.
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
    assert f.shape[0] == len(nu), "f and nu must have the same length."
    assert f.shape[1] == len(pr), "f and pr must have the same length."
    
    # Take the number of cells
    ncell = f.shape[1]
    
    # Initialize the list of efficiencies, FPRs and of cost residuals
    eps_res = []
    lam_res = []
    costopt_res = []
    
    # Loop over cells to impute the efficiency
    for cell in range(ncell):
        # Use lambda to create a function of eps and lam
        cost = lambda x: efficiency_cost_function(x[0], x[1],
                                                  f[:, cell], nu,
                                                  pr[cell], phased)
        res = minimize(cost, x0=[0.5, 0.05], bounds=[(0.0001, 1.), (0., 0.9999)])
        eps_res.append(res.x[0])
        lam_res.append(res.x[1])
        costopt_res.append(res.fun)
    
    # Convert to numpy arrays
    eps_res = np.array(eps_res)
    lam_res = np.array(lam_res)
    costopt_res = np.array(costopt_res)
    
    return eps_res, lam_res, costopt_res

def likelihood_maximization_n(nu, efficiency, fpr, chromstr, w_size):
    
    # Take data dimensions
    ncell, ndomain, ncopy_max = nu.shape
    
    assert efficiency.shape == fpr.shape == (ncell,),\
        "efficiency and fpr must be 1D arrays."
    
    
    # Reshape efficiency and fpr to match the dimensions of nu
    efficiency = np.expand_dims(efficiency, axis=(1, 2))  # np.array(ncell, 1, 1)
    efficiency = np.repeat(efficiency, repeats=ndomain, axis=1)  # np.array(ncell, ndomain, 1)
    efficiency = np.repeat(efficiency, repeats=ncopy_max, axis=2)  # np.array(ncell, ndomain, nspot_max)
    fpr = np.expand_dims(fpr, axis=(1, 2))
    fpr = np.repeat(fpr, repeats=ndomain, axis=1)
    fpr = np.repeat(fpr, repeats=ncopy_max, axis=2)
    efficiency = efficiency.astype(float)
    fpr = fpr.astype(float)
    
    # Assert that the broadcasting worked
    assert efficiency.shape == nu.shape, "Broadcasting of efficiency failed."
    assert fpr.shape == nu.shape, "Broadcasting of fpr failed."
    for cell in range(ncell):
        assert np.all(efficiency[cell, :, :] == efficiency[cell, 0, 0]),\
            "Broadcasting of efficiency failed."
        assert np.all(fpr[cell, :, :] == fpr[cell, 0, 0]),\
            "Broadcasting of fpr failed."
    
    # Compute the likelihood
    lkl_1 = compute_pi(nu, 1, efficiency, fpr)
    lkl_2 = compute_pi(nu, 2, efficiency, fpr)
    
    lkl_w_1 = np.ones(lkl_1.shape)
    lkl_w_2 = np.ones(lkl_2.shape)
    
    for i in range(ndomain):
        lkl_i_1 = lkl_1[:, i, :]
        lkl_i_2 = lkl_2[:, i, :]
        for j in range(i + 1, i + w_size):
            if j == ndomain:
                break
            if chromstr[i] != chromstr[j]:
                break
            lkl_i_1 = lkl_i_1 * lkl_1[:, j, :]
            lkl_i_2 = lkl_i_2 * lkl_2[:, j, :]
        lkl_w_1[:, i, :] = lkl_i_1
        lkl_w_2[:, i, :] = lkl_i_2
    
    # Maximize the likelihood: choose n=1 if lkl_1 > lkl_2
    n = np.ones((ncell, ndomain, ncopy_max), dtype=int)
    n[lkl_w_1 < lkl_w_2] = 2
    
    # Return the maximum likelihood
    lkl_max = np.copy(lkl_1)
    lkl_max[lkl_1 < lkl_2] = lkl_2[lkl_1 < lkl_2]
    
    return n, lkl_max
