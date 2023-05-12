import unittest
import random
import numpy as np
from alabtools.utils import Genome, Index

class TestGenome(unittest.TestCase):
    """Test Genome class"""
    
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_init_from_binary(self):
        """Test initialization of Genome from binary chromosomes.
        """
        # Define the input
        chroms = ['chr1', 'chr2', 'chr3', 'chr10']
        lengths = [2000, 1800, 1500, 1000]
        origins = [0, 10, 0, 0]
        # Transform chroms to a binary-encoded array without the 'chr'
        chroms_bnr = [chrom.split('chr')[1] for chrom in chroms]
        chroms_bnr = [bytes(chrom, 'utf-8') for chrom in chroms_bnr]
        # Initialize the Genome object
        genome = Genome(assembly='mm10', chroms=chroms_bnr,
                        lengths=lengths, origins=origins)
        # Check the results
        np.array_equal(genome.chroms, chroms)
        np.array_equal(genome.lengths, lengths)
        np.array_equal(genome.origins, origins)
    
    def test_sort(self):
        """Test sorting of the Genome.
        """
        # Define the input
        chroms = ['chr1', 'chr2', 'chr3', 'chr5', 'chr6', 'chr10', 'chr12', 'chr18', 'chrX']
        lengths = [2000 - 100 * i for i in range(len(chroms))]
        origins = [0 for _ in range(len(chroms))]
        # Shuffle the input
        idx = list(range(len(chroms)))
        random.shuffle(idx)
        chroms_shuffled = [chroms[i] for i in idx]
        lengths_shuffled = [lengths[i] for i in idx]
        origins_shuffled = [origins[i] for i in idx]
        # Initialize the Genome object
        genome = Genome(assembly='mm10', chroms=chroms_shuffled,
                        lengths=lengths_shuffled, origins=origins_shuffled)
        genome.sort()
        # Check the results
        np.testing.assert_equal(genome.chroms, chroms)
        np.testing.assert_equal(genome.lengths, lengths)
        np.testing.assert_equal(genome.origins, origins)
    

if __name__ == '__main__':
    unittest.main()
