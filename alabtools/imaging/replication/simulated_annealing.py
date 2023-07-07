import numpy as np
import pickle
import os
from . import bernoulli

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
    
    # Remove the elements where n == 0
    pi[n == 0] = np.nan
    
    # Assert that there are no pi == 0
    assert np.sum(pi == 0) == 0, 'pi == 0 encountered'
    
    # Compute the likelihood
    lkl = np.nanprod(pi)
    
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
    
    beta = 1.  # inverse temperature
    
    # roll the array, from (n0, n1, n2, ..., nN) to (n1, n2, ..., nN, n0)
    n_roll = np.roll(n, -1)
    
    # Multiply the arrays element-wise, (2 * n_i - 3) * (2 * n_i+1 - 3)
    isi = (2 * n - 3) * (2 * n_roll - 3)
    isi = isi.astype(float)
    
    # Remove the spurious pairs using the mask
    isi[mask] = np.nan
    
    # Remove the elements where n == 0
    isi[n == 0] = np.nan
    
    # Return the exponential of the sum of the array
    isi = np.exp(beta * J * np.nansum(isi))
    
    # Normalize the probability by the partition function Z
    z = 2 * (2 * np.cosh(beta * J)) ** (len(n) - 1)
    isi = isi / z
    
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
    isi = ising(n, J, mask) if J != 0 else 1.
    
    # Compute the cost
    cost = - np.log(lkl) - np.log(isi)
    
    return cost

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
    # - n[i], n[j] != 0
    # - nu[i], nu[j] != 2
    while True:
        i, j = np.random.choice(len(n), 2, replace=False)
        if n[i] != n[j] and n[i] != 0 and n[j] != 0 and nu[i] != 2 and nu[j] != 2:
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

def simulated_annealing(nu, eps, J, ising_mask, n0, tmp0, alpha, n_steps):
    """Perform the simulated annealing algorithm.

    Args:
        nu (np.array, int): Number of observation.
        eps (float): Detection efficiency.
        J (float): Ising model parameter.
        ising_mask (np.array, bool): Mask to remove spurious pairs.
        n0 (np.array, int): Initialized number of copies.
        tmp0 (float): Initial temperature.
        alpha (float): Temperature schedule parameter.
        n_steps (int): Number of steps.

    Returns:
        n (np.array, int): Number of copies.
        cost_list (list, float): Cost function values.
        prob_list (list, float): Acceptance probability values.
    """

    # Initialize n
    n = n0
    
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
    """Parallel function to perform the simulated annealing algorithm.
    
    Saves the results as pickle file in the temporary directory.

    Args:
        cellID (int): Cell ID.
        cfg (dict): Configuration dictionary.
        temp_dir (str): Temporary directory.

    Returns:
        out_name (str): Name of the temporary file.
    """
    
    # Read the data from the temporary files    
    # Normalized observed counts (nu)
    nu = np.load(os.path.join(temp_dir, 'nu.npy'))[cellID, :, :]
    # Detection efficiency
    eps = np.load(os.path.join(temp_dir, 'efficiency.npy'))[cellID]
    # Replication probability (pr)
    pr = np.load(os.path.join(temp_dir, 'pr.npy'))[cellID]
    # chromstr
    chromstr = np.load(os.path.join(temp_dir, 'chromstr.npy'))
    
    # Assert that the data is in the correct format
    ndomain, ncopy_max = nu.shape
    assert len(chromstr) == ndomain, 'len(chromstr) != ndomain'
    
    # Read parameters from the config file
    sex = cfg['sex']
    J = cfg['J']
    tmp0 = cfg['tmp0']
    alpha = cfg['alpha']
    n_steps = cfg['n_steps']
    
    # Create the mask to remove spurious pairs in the Ising model
    ising_mask = create_ising_mask(chromstr)
    
    # Initialize the data for the simulated annealing results
    n = initialize(nu, pr, chromstr, sex)
    sa_costs = []
    sa_probs = []
    
    # Create the output file name
    out_name = os.path.join(temp_dir, '{}.pkl'.format(cellID))
    
    # Perform SA only for S cells
    if pr != 0 and pr != 1:
        for i in range(ncopy_max):
            # Perform SA only if
            n1, n2 = int(np.sum(n[:, i] == 1)), int(np.sum(n[:, i] == 2))
            if pr <= n2 / (n1 + n2):
                continue
            n_i, cost_list, prob_list = simulated_annealing(nu[:, i],
                                                            eps,
                                                            J,
                                                            ising_mask,
                                                            n[:, i],
                                                            tmp0,
                                                            alpha,
                                                            n_steps)
            # Store the results
            n[:, i] = n_i
            sa_costs.append(cost_list)
            sa_probs.append(prob_list)
            # Free memory
            del n_i, cost_list, prob_list
    
    # Save the results to temporary files
    with open(out_name, 'wb') as f:
        pickle.dump({'n': n, 'sa_costs': sa_costs, 'sa_probs': sa_probs}, f)
    
    # Free memory
    del nu, eps, chromstr, ising_mask, n, sa_costs, sa_probs
    
    return out_name

def reduce_function(out_names, temp_dir):
    """Reduce function to combine the results of the simulated annealing algorithm.

    Args:
        out_names (list, str): List of temporary files.
        temp_dir (str): Temporary directory.

    Returns:
        n (np.array, int): Number of copies.
        sa_costs (dict): Cost functions of SA algorithm for all cells/copies.
        sa_probs (dict): Acceptance probabilities of SA algorithm for all cells/copies.
        
    """
    
    # Read nu from the temporary files
    nu = np.load(os.path.join(temp_dir, 'nu.npy'))
    ncell, ndomain, ncopy_max = nu.shape
    
    # Initialize the number of copies array
    n = np.zeros((ncell, ndomain, ncopy_max), dtype=int)
    
    # We store the cost and acceptance probabilities as dictionaries
    sa_costs = dict()
    sa_probs = dict()
    
    for out_name in out_names:
        # Load the data from the dictionary
        with open(out_name, 'rb') as f:
            data = pickle.load(f)
        # Get the cellID
        cellID = int(os.path.basename(out_name).split('.')[0])
        # Take the data from the dictionary
        n_cellID = data['n']
        sa_costs_cellID = data['sa_costs']
        sa_probs_cellID = data['sa_probs']
        # Store the data
        n[cellID, :, :] = n_cellID
        sa_costs[cellID] = sa_costs_cellID
        sa_probs[cellID] = sa_probs_cellID
        # Free memory
        del n_cellID, sa_costs_cellID, sa_probs_cellID
        
    return n, sa_costs, sa_probs


def create_ising_mask(chromstr):
    """Create the mask to remove spurious pairs in the Ising model,
    i.e. interactions between different chromosomes.
    
    Args:
        chromstr (np.array, str): Chromosome strings.
        
    Returns:
        mask (np.array, bool): Mask to remove spurious pairs
                               (True = remove, False = keep)
    
    """
    
    # roll the array, from (0, 1, 2, ..., N) to (1, 2, ..., N, 0) 
    chromstr_roll = np.roll(chromstr, -1)
    
    # Create the mask
    mask = chromstr != chromstr_roll
    
    return mask

def initialize(nu, pr, chromstr, sex):
    
    # Take ndomain, ncopy_max
    ndomain, ncopy_max = nu.shape
    
    # Start with a matrix of ones
    n = np.ones(nu.shape, dtype=int)  # (ndomain, ncopy_max)
    
    # If organism is male, set n = 0 for the second copy of chrX or chrY
    if sex == 'XY':
        n[np.logical_or(chromstr == 'chrX', chromstr == 'chrY'), 1] = 0
    
    # Case 1: cell in G1
    if pr == 0:
        return n
    
    # Case 2: cell in G2
    if pr == 1:
        n = 2 * n  # all domains have been replicated
        return n
    
    # Case 3: cell in S phase
    # Set the values of n to 2 where nu == 2 (100% replicated)
    n[nu == 2] = 2
    # Loop over the copies
    for i in range(ncopy_max):
        # Count how many domains have n=0, n=1, n=2
        n1, n2 = int(np.sum(n[:, i] == 1)), int(np.sum(n[:, i] == 2))
        # Translate the probability pr into number of domains
        # We have to exclude n=0, since they are not biological
        nr = int(pr * (n1 + n2))
        # Skip if there are at least nr domains already replicated
        if n2 >= nr:
            continue
        # Otherwise, transform some n=1 into n=2 (so that n2 = nr)
        # Find the indices where n == 1
        idx = np.where(n[:, i] == 1)[0]
        # Randomly select nr - n2 indices
        idx_r = np.random.choice(idx, nr - n2, replace=False)
        # Set the values of n to 2 for the randomly selected indices
        n[idx_r, i] = 2
    
    return n
