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
    
    def test_init_from_binary_shuffled(self):
        """Test initialization of Genome object where chroms are given as
        binary-encoded strings without the 'chr' prefix and are shuffled.
        """
        # Define the input
        chroms = ['chr1', 'chr2', 'chr3', 'chr10']
        lengths = [2000, 1800, 1500, 1000]
        origins = [0, 10, 0, 0]
        # Shuffle the input
        idx = list(range(len(chroms)))
        random.shuffle(idx)
        chroms_shuffled = [chroms[i] for i in idx]
        lengths_shuffled = [lengths[i] for i in idx]
        origins_shuffled = [origins[i] for i in idx]
        # Transform chroms_shuffled to a binary-encoded array without the 'chr'
        chroms_shuffled = [chrom.split('chr')[1] for chrom in chroms_shuffled]
        chroms_shuffled = [bytes(chrom, 'utf-8') for chrom in chroms_shuffled]
        # Initialize the Genome object
        genome = Genome(assembly='mm10', chroms=chroms_shuffled,
                        lengths=lengths_shuffled, origins=origins_shuffled)
        # Check the results
        np.array_equal(genome.chroms, chroms)
        np.array_equal(genome.lengths, lengths)
        np.array_equal(genome.origins, origins)
    

if __name__ == '__main__':
    unittest.main()
