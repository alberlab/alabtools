from . import bernoulli
import numpy as np
from alabtools.utils import Genome, Index

def likelihood(nu, n, eps):
    """Compute the likelihood of the data nu given the model parameters.

    Args:
        nu (np.array, int): Number of observation.
        n (np.array, int): Number of copies.
        eps (float): Detection efficiency.

    Returns:
        lkl (float): Likelihood.
    """
    
    # Compute the observed probability of the data
    pi = bernoulli.compute_pi(nu, n, eps)
    
    # Compute the likelihood
    lkl = np.prod(pi)
    
    return lkl

def ising(n, J, mask):
    """Compute the probability of the configuration of n given the Ising model.

    Args:
        n (np.array, int): Number of copies.
        J (float): Ising model parameter.
        mask (np.array, bool): Mask to remove spurious pairs.

    Returns:
        isi (float): Ising model probability.
    """
    
    # roll the array, from (n0, n1, n2, ..., nN) to (n1, n2, ..., nN, n0)
    n_roll = np.roll(n, -1)
    
    # Multiply the arrays element-wise, (2 * n_i - 3) * (2 * n_i+1 - 3)
    isi = (2 * n - 3) * (2 * n_roll - 3)
    
    # Remove the last element, has spurious pair
    isi[-1] = 0
    
    # Remove the remaining spurious pairs using the mask
    isi[mask] = 0  # these will be the elements where chr changes
    
    # Return the exponential of the sum of the array
    isi = np.exp(J * np.sum(isi))
    
    return isi

def cost_function(nu, n, eps, J, mask):
    """Cost function that combines the likelihood and the Ising model.
    To be minimized.

    Args:
        nu (np.array, int): Number of observation.
        n (np.array, int): Number of copies.
        eps (float): Detection efficiency.
        J (float): Ising model parameter.
        mask (np.array, bool): Mask to remove spurious pairs.

    Returns:
        cost (float): Cost function.
    """
    
    # Compute the likelihood
    lkl = likelihood(nu, n, eps)
    
    # Compute the Ising model
    isi = ising(n, J, mask)
    
    # Compute the cost
    cost = -np.log(lkl) - np.log(isi)
    
    return cost

def initialize(nu, r):
    """Initialize the number of copies.

    Args:
        nu (np.array, int): Number of observation.
        r (float): Fraction of replicates.

    Returns:
        n (np.array, int): Number of copies.
    """
    
    # Compute fraction of 100% replicates from the data
    r_data = np.sum(nu == 2) / len(nu)
    # If r_obs is larger than r, throw an error
    if r_data > r:
        raise ValueError('More replicates than imposed!')
    
    # Translate the fraction to an integer
    n_r = int(r * len(nu))
    n_r_data = int(r_data * len(nu))
    if n_r > len(nu):
        n_r = len(nu)
    if n_r_data > len(nu):
        n_r_data = len(nu)
    
    # Find the indices where nu != 2
    idx = np.where(nu != 2)[0]
    # Randomly select n_r - n_r_data indices
    idx_r = np.random.choice(idx, n_r - n_r_data, replace=False)
    
    # Initialize n
    n = np.ones(len(nu))
    # Set the values of n to 2 where nu == 2
    n[nu == 2] = 2
    # Set the values of n to 2 for the randomly selected indices
    n[idx_r] = 2
    
    return n

def update(n, nu):
    """Update the number of copies.

    Args:
        n (np.array, int): Number of copies.
        nu (np.array, int): Number of observation.

    Returns:
        n_new (np.array, int): Updated number of copies.
    """
    
    # Randomly select i and j with:
    # - i != j
    # - n[i] != n[j]
    # - nu[i], nu[j] != 2
    while True:
        i, j = np.random.choice(len(n), 2, replace=False)
        if n[i] != n[j] and nu[i] != 2 and nu[j] != 2:
            break
    
    # Update the values (swap i and j)
    n_new = n.copy()
    n_new[i] = n[j]
    n_new[j] = n[i]
    
    return n_new

def accept_probability(cost, cost_new, tmp):
    """Compute the acceptance probability.

    Args:
        cost (float): Cost value.
        cost_new (float): Updated cost value.
        tmp (float): Temperature.

    Returns:
        (float): Acceptance probability.
    """
    
    if cost_new <= cost:
        return 1.
    if cost_new > cost:
        return np.exp(-(cost_new - cost) / tmp)

def simulated_annealing(nu, r, eps, J, ising_mask, tmp0, alpha, n_steps):
    """Perform the simulated annealing algorithm.

    Args:
        nu (np.array, int): Number of observation.
        r (float): Fraction of replicates.
        eps (float): Detection efficiency.
        J (float): Ising model parameter.
        ising_mask (np.array, bool): Mask to remove spurious pairs.
        tmp0 (float): Initial temperature.
        alpha (float): Temperature schedule parameter.
        n_steps (int): Number of steps.

    Returns:
        n (np.array, int): Number of copies.
        cost_list (list, float): Cost function values.
        prob_list (list, float): Acceptance probability values.
    """

    # Initialize n
    n = initialize(nu, r)
    
    # Create lists to store the cost and acceptance probability
    cost_list = list()
    prob_list = list()
    
    # Define the temperature schedule
    tmp_scd = tmp0 * alpha ** np.arange(n_steps)
    
    # Perform the algorithm
    for tmp in tmp_scd:
        # Compute the cost
        cost = cost_function(nu, n, eps, J, ising_mask)
        # Update n
        n_new = update(n, nu)
        # Compute the new cost
        cost_new = cost_function(nu, n_new, eps, J, ising_mask)
        # Accept or reject
        prob = accept_probability(cost, cost_new, tmp)
        if prob >= np.random.uniform():
            n = n_new
        # Store the cost and acceptance probability
        cost_list.append(cost)
        prob_list.append(prob) if cost_new > cost else prob_list.append(None)
    
    return n, cost_list, prob_list

def parallel_function(cellID, cfg, temp_dir):
    
    # As in cellcycle.py, I can read the data I need from the temp_dir
    # I can create an Index object if I need to
    
    return None

def reduce_function(out_names):
    
    
    return None

# Since n is a 3D array (ncell, ndomain, ncopy),
# even though we are only working on one cell,
# we still have a 2D array.
# I think the easiest thing is just to perform the
# algorithm on each copy separately in this code.
