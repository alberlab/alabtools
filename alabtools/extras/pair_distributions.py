#!/bin/env python

import numpy as np
from scipy.stats import entropy, ks_2samp
from scipy.spatial.distance import cdist

def get_distibutions(data0, data1, nbins, epsilon=1e-6):
    binmax = max(np.max(data0), np.max(data1))
    binmin = min(np.min(data0), np.min(data1)) 
    x, edges = np.histogram(
        data0, 
        range=(binmin, binmax), 
        bins=nbins, 
    )
    y, edges = np.histogram(
        data1, 
        range=(binmin, binmax), 
        bins=nbins, 
    )
    
    x = x + epsilon
    y = y + epsilon
    return x, y


def ranked_distance(odata, ndata):
    a = np.sort(odata)
    b = np.sort(ndata)
    return np.sqrt(np.sum(np.square(a-b)))/len(a) 

def get_pair_distance_statistics(hss0, hss1, i, j,
    compute=['KL'], use=None, itype='all', nbins=20, epsilon=None):
    '''
    Get Kullback-Liebler (KL) divergence, Kolmogorov Smirnoff (KS) statistics,
    and ranked vertical distance for distributions of pair distances.

    Parameters
    ----------
    hss0 : alabtools.HssFile
        first structure object. It is used as 'posterior probability' in
        KL divergence
    hss1 : alabtools.HssFile
        second structure object. Used as prior probability in KL divergence
    i : int
        first genomic segment id
    j : int
        second genomic segment id
    compute : list of str:
        specify which statistics to compute. Valid strings are:
        - 'KL' : computes Kullback-Liebler divergence
        - 'KS' : computes Kolmogorov-Smirnoff statistics
        - 'RD' : computes ranked distance value 
    use : list of str/function
        specifies how to handle the data from multiploid structures. Each item
        can be one of 'all', 'min', 'max' or a function.
        Defaults to ['all'].
        Valid string values are:
        - 'all' uses all the possible distances between region copies
        - 'min' uses only the minimum distance between region copies
        - 'max' uses only the maximum distance between region copies
        If a user defined function is passed in the list, it should take as the 
        only input a 1D array of values and return a one-dimensional array 
        of filtered values.
    itype : str
        specify the behavior for regions inside the same chromosome for 
        multiploid structures. Valid values are:
        - 'all' retains all the possible distances, including intra- and
            inter-chromosomal
        - 'cis' considers only the intra-chromosomal distances
        - 'trans' considers only the inter-chromosomal distances
    nbins:
        number of bins to histogram the distances distributions to compute
        KL divergence. Defaults to 20
    epsilon:  
        epsilon value to add to the histograms when the prior probability is 0.
        Defaults to 0.5 / n_structures (this assumes the probability is small,
        trying not to overweight those bins).

    Returns
    -------
    list of tuples : for each item in `use`, it return a tuple containing the
        results for the statistics in `compute`
    '''
    if use is None:
        use = ['all']

    nstruct = hss0.nstruct
    assert hss1.nstruct == nstruct
    
    if epsilon is None:
        epsilon = 0.5 / nstruct

    index = hss0.index

    crd0i = hss0['coordinates'][index.copy_index[i]]
    crd0j = hss0['coordinates'][index.copy_index[j]]

    crd1i = hss1['coordinates'][index.copy_index[i]]
    crd1j = hss1['coordinates'][index.copy_index[j]]

    
    # get the distance matrices
    data0 = [ 
        cdist(x, y) for x, y in zip(
            crd0i.swapaxes(0, 1), 
            crd0j.swapaxes(0, 1)
        ) 
    ]

    data1 = [ 
        cdist(x, y) for x, y in zip(
            crd1i.swapaxes(0, 1), 
            crd1j.swapaxes(0, 1)
        )
    ]

    if index.chrom[i] != index.chrom[j]:
        itype = 'all'

    if itype == 'all':
        data0 = np.array([x.ravel() for x in data0])
        data1 = np.array([x.ravel() for x in data1])

    elif itype == 'cis':
        data0 = np.array([x.diagonal() for x in data0])
        data1 = np.array([x.diagonal() for x in data1])

    elif itype == 'trans':
        data0 = np.array([x[np.tril_indices(len(x), -1)] for x in data0])
        data1 = np.array([x[np.tril_indices(len(x), -1)] for x in data1])

    results = []
    for u in use:
        
        if u == 'all':
            x0 = data0.flatten()
            x1 = data1.flatten()

        elif u == 'max':
            x0 = np.max(data0, axis=1)
            x1 = np.max(data1, axis=1)

        elif u == 'min':
            x0 = np.min(data0, axis=1)
            x1 = np.min(data1, axis=1)
        
        else:
            x0 = [ u(x) for x in data0 ]
            x1 = [ u(x) for x in data1 ]

        x0 = x0.flatten()
        x1 = x1.flatten()

        r = []
        d0, d1 = get_distibutions(x0, x1, nbins, epsilon)
        if 'KL' in compute:
            r.append( entropy(d0, qk=d1) )
        if 'KS' in compute:
            r.append(ks_2samp(x0, x1))
        if 'RD' in compute:
            r.append(ranked_distance(x0,x1))
        results.append(tuple(r))

    return results 











#     n_done = 0
#     for i in range(n_beads/2):
#         for j in range(i):
#             status = MPI.Status()
#             recv = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
#             if status.tag == 1 or status.tag == 10:
#             # tag 1 codes for initialization.
#             # tag 10 codes for requesting more data.
#                 if status.tag == 10: # data received
#                     (ii, jj) = recv[0]
#                     klvals = recv[1]
#                     fdvals = np.sqrt(recv[2])
#                     if index[ii] == index[jj]:
#                         kldiv_intra[ii, jj]   = klvals[1]
#                         fitdist_intra[ii, jj] = fdvals[1]
#                     kldiv_inter[ii, jj]   = klvals[0]
#                     fitdist_inter[ii, jj] = fdvals[0]
#                     n_done += 1

#                 MPI.COMM_WORLD.send((i, j), dest=status.source, tag=10)
            
#             now = time.clock()
#             if (now-last_print) > 10:
#                 last_print = now
#                 elapsed_time = now-start
#                 if elapsed_time != 0:
#                     speed = float(n_done)/(now-start)
#                 else:
#                     speed = 0
#                 if speed != 0:
#                     remaining_time = float(n_total-n_done)/speed
#                 else:
#                     remaining_time = None
#                 print '{:.2%} completed,   Elapsed time: {}   Time remaining: {}'.format(
#                                                     float(n_done)/n_total,
#                                                     formatted_time_interval(elapsed_time),
#                                                     formatted_time_interval(remaining_time) )
#                 sys.stdout.flush()        
#     # exit workers
#     while True:
#         status = MPI.Status()
#         # Receive input from workers.
#         recv = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
#         if status.tag == 1 or status.tag == 10:
#             if status.tag == 10: # data received
#                 (ii, jj) = recv[0]
#                 klvals = recv[1]
#                 fdvals = np.sqrt(recv[2])
#                 if index[ii] == index[jj]:
#                     kldiv_intra[ii, jj]   = klvals[1]
#                     fitdist_intra[ii, jj] = fdvals[1]
#                 kldiv_inter[ii, jj]   = klvals[0]
#                 fitdist_inter[ii, jj] = fdvals[0]
#             MPI.COMM_WORLD.send([], dest=status.source, tag=2)
#         elif status.tag == 2:
#             process_list.remove(status.source)
#         if len(process_list) == 0:
#             break
   
#     np.save('kldiv/intra_' + out_fname + '_kldiv_v5.npy', kldiv_intra)
#     np.save('fitdist/intra_' + out_fname + '_fitdist_v5.npy', fitdist_intra)

#     np.save('kldiv/inter_' + out_fname + '_kldiv_v5.npy', kldiv_inter)
#     np.save('fitdist/inter_' + out_fname + '_fitdist_v5.npy', fitdist_inter)

# else:
#     _mpi_worker()