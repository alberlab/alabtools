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
    cost = ((pi0 - f0) ** 2 + (pi1 - f1) ** 2 + (pi2 - f2) ** 2) ** 0.5
    
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
