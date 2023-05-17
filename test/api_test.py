import unittest
import random
import numpy as np
from alabtools.utils import Genome, Index
from alabtools.api import Contactmatrix

class TestContactmatrix(unittest.TestCase):
    """Test functions in Contactmatrix
    """
    
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_sort(self):
        """Test sort method in Contactmatrix.
        """
        # Set genome
        chroms = ['chr1', 'chr4', 'chr7', 'chr12', 'chr13', 'chrX']
        assembly = 'mm10'
        lengths = [random.randint(100, 200) for _ in range(len(chroms))]
        resolution = 10
        genome = Genome(assembly=assembly, chroms=chroms, lengths=lengths)
        # Set index
        index = genome.bininfo(resolution=resolution)
        # Create random symmetric matrix
        matrix = np.random.randint(0, 100, size=(len(index), len(index)))
        matrix = matrix + matrix.T - np.diag(matrix.diagonal())
        # Shuffle genome
        np.random.seed(0)
        rnd_ord = np.random.permutation(len(chroms))
        chroms_rnd = [chroms[i] for i in rnd_ord]
        lengths_rnd = [lengths[i] for i in rnd_ord]
        genome_rnd = Genome(assembly=assembly, chroms=chroms_rnd, lengths=lengths_rnd)
        # Shuffle index
        index_rnd = genome_rnd.bininfo(resolution=resolution)
        # Shuffle matrix
        order = np.zeros(len(index), dtype=int)
        for chrom in chroms_rnd:
            order[index_rnd.chromstr == chrom] = np.arange(len(index))[index.chromstr == chrom]
        matrix_rnd = np.copy(matrix)
        matrix_rnd = matrix_rnd[order, :]
        matrix_rnd = matrix_rnd[:, order]
        # Assert shuffling is correct (check intra matrices)
        for chrom in chroms:
            np.testing.assert_array_equal(matrix[index.chromstr == chrom, :]
                                                [:, index.chromstr == chrom],
                                          matrix_rnd[index_rnd.chromstr == chrom, :]
                                                    [:, index_rnd.chromstr == chrom])

        # Create Contactmatrix with shuffled genome
        hcs = Contactmatrix(mat=matrix_rnd, genome=genome_rnd,
                            resolution=resolution, usechr=('#', 'X', 'Y'))
        # Sort
        hcs.sort()
        # Assert genome
        np.testing.assert_array_equal(hcs.genome.chroms, chroms)
        np.testing.assert_array_equal(hcs.genome.lengths, lengths)
        # Assert index
        np.testing.assert_array_equal(hcs.index.chromstr, index.chromstr)
        np.testing.assert_array_equal(hcs.index.start, index.start)
        np.testing.assert_array_equal(hcs.index.end, index.end)
        # Assert matrix
        np.testing.assert_array_equal(hcs.matrix.diagonal, np.diag(matrix))
        np.testing.assert_array_equal(hcs.matrix.toarray(), matrix)


if __name__ == '__main__':
    unittest.main()
