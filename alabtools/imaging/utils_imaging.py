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


def sort_coord_by_boolmask(bool_mask, axis=-1):
    """Sort coordinates by an input boolean mask on a given axis.
    
    The False values are sorted first, then the True values.
    
    The mask should have a shape of (n1, n2, ..., nK).
    
    Usage:
        given A with same shape as mask,
            idx = sort_coord_by_boolmask(mask, axis)
            A_srt = A[idx]  <-- A sorted by mask along axis
    
    Args:
        bool_mask (np.array((n1, n2, ..., nK))): Boolean mask to sort by.
        axis (int, optional): Axis to sort on. Defaults to -1.
    
    Returns:
        idx (np.array((n1, n2, ..., nK))): Indices to sort the coordinates.
    """
    
    # Sort the coordinates by the mask (False first, then True),
    # on the given axis
    sort_subgrid = np.argsort(bool_mask, axis=axis)
    
    # Create a grid of indices to sort the coordinates
    grid = np.meshgrid(*[np.arange(i) for i in bool_mask.shape], indexing='ij')
    
    # Replace the last axis of the grid by the sorted indices
    grid[axis] = sort_subgrid
    
    # Convert the grid to a tuple of indices
    idx = tuple(grid)
    
    return idx
