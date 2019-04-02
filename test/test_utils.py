from alabtools.utils import Genome, Index
import h5py
import os


def test_genome_io():
    g = Genome('hg38')
    with h5py.File('./teststr.h5', 'w') as f:
        g.save(f)
    with h5py.File('./teststr.h5', 'r') as f:
        g2 = Genome(f)
    os.remove('./teststr.h5')
    assert g == g2


def test_index_io():
    with open('testindex5.txt', 'w') as f:
        f.write('#comment line\n')
        f.write('chr1 0 1000 gap 0\n')
        f.write('chr3 0 1000 domain 2\n')

    # load without genome
    i = Index('testindex5.txt')
    with h5py.File('./teststr.h5', 'w') as f:
        i.save(f)
    with h5py.File('./teststr.h5', 'r') as f:
        ii = Index(f)
    assert ii == i

    # load with genome
    i = Index('testindex5.txt', genome='hg38')
    with h5py.File('./teststr.h5', 'w') as f:
        i.save(f)
    with h5py.File('./teststr.h5', 'r') as f:
        ii = Index(f)
    assert ii == i

    os.remove('./teststr.h5')
    os.remove('./testindex5.txt')


