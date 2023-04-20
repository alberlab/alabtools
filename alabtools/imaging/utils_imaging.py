import numpy as np

def flatten_coordinates(crd):
    """
    Flattens the coordinates keeping track of the indices.
    Used to go from an N-dimensional array to a 2D array and back.
    Also removes NaNs
    
    Parameters
    ----------
    crd: np.array(n1, n2, ..., nK, 3)
    
    Returns
    ----------
    crd_flat: np.array(n1*n2*...*nK, 3)
    
    idx: np.array(n1*n2*...*nK, K)
        Indices of the original array.
    """
    # Create a meshgrid of indices
    shape = crd.shape[:-1]
    grid = np.meshgrid(*[np.arange(dim) for dim in shape], indexing='ij')
    
    # stack indices together vertically, e.g.
    #   [[0 0]
    #    [1 0]
    #    [2 0]]
    idx = np.column_stack([g.flatten() for g in grid])
    
    # Get coordinates
    crd_flat = crd[tuple(idx.T)]
    
    return crd_flat, idx
