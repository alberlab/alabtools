from . import bernoulli
import numpy as np

def likelihood(nu, n, eps):
    
    # Compute the observed probability of the data
    pi = bernoulli.compute_pi(nu, n, eps)
    
    # Compute the likelihood
    lkl = np.prod(pi)
    
    return lkl

def ising(n, J, mask):
    
    # roll the array, from (n0, n1, n2, ..., nN) to (n1, n2, ..., nN, n0)
    n_roll = np.roll(n, -1)
    
    # Multiply the arrays element-wise, (2 * n_i - 3) * (2 * n_i+1 - 3)
    isi = (2 * n - 3) * (2 * n_roll - 3)
    
    # Remove the last element, has spurious pair
    isi[-1] = 0
    
    # Remove the masked elements
    isi[mask] = 0  # these will be the elements where chr changes
    
    # Return the exponential of the sum of the array
    isi = np.exp(J * np.sum(isi))
    
    return isi

def cost(nu, n, eps, J, mask):
    
    # Compute the likelihood
    lkl = likelihood(nu, n, eps)
    
    # Compute the Ising model
    isi = ising(n, J, mask)
    
    # Compute the cost
    cost = -np.log(lkl) - np.log(isi)
    
    return cost

def initialize(nu, r):
    
    # Count how many nu == 2 there are
    r_obs = np.sum(nu == 2)
    # If r_obs is larger than r, throw an error
    if r_obs > r:
        raise ValueError('More replicates than imposed!')
    
    # Find the indices where nu != 2
    idx = np.where(nu != 2)[0]
    # Randomly (r - r_obs) indices from idx
    r_idx = np.random.choice(idx, r - r_obs, replace=False)
    
    # Initialize n
    n = np.ones(len(nu))
    # Set the values of n to 2 where nu == 2
    n[nu == 2] = 2
    # Set the values of n to 2 for the randomly selected indices
    n[r_idx] = 2
    
    return n

def update(n, nu):
    
    # Randomly select i and j where nu != 2
    while True:
        i = np.random.randint(0, len(n))
        j = np.random.randint(0, len(n))
        if nu[i] != 2 and nu[j] != 2:
            break
    
    # Update the values
    n_new = n.copy()
    n_new[i] = n[j]
    n_new[j] = n[i]
    
    return n_new


# Since n is a 3D array (ncell, ndomain, ncopy),
# even though we are only working on one cell,
# we still have a 2D array.
# I think the easiest thing is just to perform the
# algorithm on each copy separately in this code.
