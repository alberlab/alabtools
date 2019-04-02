import numpy as np
from alabtools import Contactmatrix, Genome, Index
from alabtools.utils import make_diploid, make_multiploid


def test_sum_copies_haploid():
    g = Genome('hg38', usechr=('1', 'X', 'Y'), silence=True)
    idx = g.bininfo(20000000)
    n = len(idx)
    m = Contactmatrix(np.random.random((n, n)), genome=g, resolution=idx)
    x = m.toarray()
    y = m.sumCopies().toarray()
    assert np.allclose(x, y)


def test_sum_copies_diploid():
    g = Genome('hg38', usechr=('1', 'X', 'Y'), silence=True)
    idx = g.bininfo(20000000)
    didx = make_diploid(idx)
    n = len(didx)
    m = Contactmatrix(np.random.random((n, n)), genome=g, resolution=didx)
    q = m.toarray()
    k = n // 2
    x = np.zeros((k, k))
    for i in [0, 1]:
        for j in [0, 1]:
            x[:, :] += q[k*i:k*(i+1)][:, k*j:k*(j+1)]
            # Note: we must not double count the diagonal for the out-of-diagonal parts
            # of intra-chromosomal contacts ( (c1a, c1b) == (c1b, c1a).T )
            if i > j:
                x[:, :] -= np.diag(np.diag(q[k*i:k*(i+1)][:, k*j:k*(j+1)]))
    y = m.sumCopies().toarray()

    assert np.allclose(np.triu(x), np.triu(y))


def test_sum_copies_multiploid():
    g = Genome('hg38', usechr=('1', 'X', 'Y'), silence=True)
    idx = g.bininfo(20000000)
    copies = [2, 1, 1]
    didx = make_multiploid(idx, [0, 1, 2], copies)
    n = len(didx)
    m = Contactmatrix(np.random.random((n, n)), genome=g, resolution=didx)
    q = m.toarray()
    k = len(idx)
    x = np.zeros((k, k))
    for ci in [0, 1, 2]:
        for cj in [0, 1, 2]:
            # for each pair of chromosomes
            i0 = idx.get_chrom_pos(ci)
            j0 = idx.get_chrom_pos(cj)
            for cx in range(copies[ci]):
                for cy in range(copies[cj]):
                    # for each pair of copies
                    i1 = didx.get_chrom_pos(ci, cx)
                    j1 = didx.get_chrom_pos(cj, cy)
                    x[np.ix_(i0, j0)] += q[np.ix_(i1, j1)]
                    # Note: we must not double count the diagonal for the out-of-diagonal parts
                    # of intra-chromosomal contacts ( (c1a, c1b) == (c1b, c1a).T )
                    if ci == cj and cx < cy:
                        x[np.ix_(i0, j0)] -= np.diag(np.diag(q[np.ix_(i1, j1)]))
    y = m.sumCopies().toarray()
    assert np.allclose(np.triu(x), np.triu(y))
