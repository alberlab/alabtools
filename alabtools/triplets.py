import numpy as np
from numba import jit
# if ABSOLUTE DISTANCE is true, cutoff is interpreted as a surface
# to surface distance. If false, the surface to surface contact
# distance between beads i and j is computed as cutoff*(r_i + r_j),
# where r_i and r_j indicate radiuses. 
ABSOLUTE_DISTANCE = True
# how fine should the grid be. 1 means that space
# is separated in cells with side equal to the maximum
# cutoff. 2 means half of the cutoff, etc. You probably want to
# keep 2
N_CELL_PER_CUTOFF = 2


def buildCells(crd, cut):
    csize = cut / N_CELL_PER_CUTOFF
    origin = np.min(crd, axis=0)
    bsize = np.max(crd, axis=0) - origin + np.array( [0.1, 0.1, 0.1] )
    ncells = np.array([ int(bsize[i] / csize) + 1 for i in range(3) ])
    cells = [ [ [ list() for _ in range(ncells[2])] for _ in range(ncells[1]) ] for _ in range(ncells[0]) ]
    for i, xyz in enumerate(crd):
        x, y, z = xyz - origin
        cells[ int( x / csize ) ][ int( y / csize ) ][ int( z / csize ) ].append( i )
    return cells, ncells, bsize

@jit
def check_contacts(cci, ccj, cck, 
                   oci, ocj, ock, 
                   crd, radii, cutoff, 
                   cmap, cells, exclude_neigh=0):
    

    for i in cells [ cci ] [ ccj ] [ cck ]:
        for j in cells [ oci ] [ ocj ] [ ock ]:
            if i <= j+exclude_neigh:
                continue
            if ABSOLUTE_DISTANCE is True:
                c = np.linalg.norm( crd[i] - crd[j] ) - radii[i] - radii[j] < cutoff
            else:
                c = np.linalg.norm( crd[i] - crd[j] ) < cutoff * (radii[i] + radii[j])
            if c:
                cmap[ i ].add( j )


def cell_loop(cci, ccj, cck, 
              crd, radii, cutoff, cmap, 
              cells, ncells, exclude_neigh=0):
    
    for oci in range(cci - N_CELL_PER_CUTOFF, cci + N_CELL_PER_CUTOFF + 1):
        if oci < 0 or oci >= ncells[0]:
            continue
        for ocj in range(ccj - N_CELL_PER_CUTOFF, ccj + N_CELL_PER_CUTOFF + 1):
            if ocj < 0 or ocj >= ncells[1]:
                continue
            for ock in range(cck - N_CELL_PER_CUTOFF, cck + N_CELL_PER_CUTOFF + 1):
                if ock < 0 or ock >= ncells[2]:
                    continue
                check_contacts(cci, ccj, cck, 
                               oci, ocj, ock, 
                               crd, radii, cutoff, 
                               cmap, cells, exclude_neigh)


def buildNeighList(crd, radii, cutoff, exclude_neigh=0):
    cell_cutoff = cutoff + 2*np.max(radii)
    cells, ncells, bsize = buildCells(crd, cell_cutoff)
    cmap = [ set() for _ in range(len(crd)) ]
    for cci in range( ncells[0] ):
        for ccj in range( ncells[1] ):
            for cck in range( ncells[2] ):
                cell_loop(cci, ccj, cck, 
                          crd, radii, cutoff, cmap, 
                          cells, ncells, exclude_neigh)
    return cmap

from scipy.spatial.distance import pdist, squareform
def get_triplets(crd, radii, cutoff, exclude_neigh=None):
    n = len(crd)
    pd = squareform(pdist(crd))
    r1 = np.resize(radii, (n, n))
    r2 = r1.T
    v = pd - r1 - r2 < cutoff
    cmap = [ {j for j in np.where(v[i])[0] if j > i} for i in range(len(crd)) ]
    if exclude_neigh is not None:
        exclude_neigh(cmap)
    triplets = set()
    for i in range(len(crd)):
        for j in cmap[i]:
            for k in cmap[j]:
                if k in cmap[i]:
                    triplets.add( tuple( sorted( [ i, j, k ] ) ) )
    return triplets


def get_triplets_cell(crd, radii, cutoff, exclude_neigh=0):
    if isinstance(exclude_neigh, int):
        cmap = buildNeighList(crd, radii, cutoff, exclude_neigh)
    else:
        cmap = buildNeighList(crd, radii, cutoff)
        exclude_neigh(cmap)
    triplets = set()
    n_hapl = len(crd) // 2
    for i in range(len(crd)):
        for j in cmap[i]:
            for k in cmap[j]:
                if k in cmap[i]:
                    triplets.add( tuple( sorted( [ i % n_hapl, j % n_hapl, k % n_hapl ] ) ) )
    return triplets

def filter_only_trans(index):
    '''
    function to filter out all the intra-chromosomal contacts
    '''
    def inner(cmap):
        for i in range(len(cmap)):
            nset = set()
            for j in cmap[i]:
                if index.chrom[i] != index.chrom[j]: #or np.abs(i-j) > 10: 
                    nset.add(j)
            cmap[i] = nset
    return inner


def filter_only_cis(index, gap=0):
    '''
    function to filter out all the intra-chromosomal contacts
    '''
    def inner(cmap):
        for i in range(len(cmap)):
            nset = set()
            for j in cmap[i]:
                if index.chrom[i] == index.chrom[j] and np.abs(i-j) > gap: 
                    nset.add(j)
            cmap[i] = nset
    return inner

def filter_mixed(index, gap=0):
    def inner(cmap):
        for i in range(len(cmap)):
            nset = set()
            for j in cmap[i]:
                if index.chrom[i] != index.chrom[j] or np.abs(i-j) > gap: 
                    nset.add(j)
            cmap[i] = nset
    return inner


filters = {
    'trans': filter_only_trans,
    'cis': filter_only_cis,
    'mixed': filter_mixed
}

@jit
def get_tindex(a, b, c, n):
    a, b, c = sorted([a, b, c])
    return a*n*n + b*n + c

@jit
def unpack_tindex(tindex, n):
    c = tindex % n
    tindex -= c
    b = ( tindex % (n*n) ) // n
    tindex -= b * n
    a = tindex // (n*n)
    return a, b, c