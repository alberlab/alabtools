import unittest
import random
import numpy as np
from alabtools.utils import Genome, Index, standardize_chromosomes

class TestUtils(unittest.TestCase):
    """Test functions in utils.py
    """
    
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_standardize_chromosomes(self):
        """Test standardize_chromosomes function in utils.

        Returns:
            _type_: _description_
        """
        chroms = [b'1', b'1', b'1', b'2', b'2', b'7', b'X', b'X']
        chroms_std = ['chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chr7', 'chrX', 'chrX']
        np.testing.assert_array_equal(standardize_chromosomes(chroms), chroms_std)

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
        np.testing.assert_array_equal(genome.chroms, chroms)
        np.testing.assert_array_equal(genome.lengths, lengths)
        np.testing.assert_array_equal(genome.origins, origins)
    
    def test_sort(self):
        """Test sort method in Genome.
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
        np.testing.assert_array_equal(genome.chroms, chroms)
        np.testing.assert_equal(genome.lengths, lengths)
        np.testing.assert_equal(genome.origins, origins)
    
    def test_check_sorted(self):
        # Define a sorted input
        chroms = ['chr1', 'chr2', 'chr3', 'chr5', 'chr6', 'chr10', 'chr12', 'chr18', 'chrX', 'chrM']
        lengths = [2000 - 100 * i for i in range(len(chroms))]
        origins = [0 for _ in range(len(chroms))]
        # Create the genome
        genome = Genome(assembly='mm10', chroms=chroms,
                        lengths=lengths, origins=origins)
        # Check the results
        self.assertTrue(genome.check_sorted())
        # Define a shuffled input
        chroms = ['chr1', 'chr6', 'chr13', 'chr7', 'chr2', 'chrX', 'chr12', 'chrM', 'chr18', 'chr5']
        lengths = [2000 - 100 * i for i in range(len(chroms))]
        origins = [0 for _ in range(len(chroms))]
        # Create the genome
        genome = Genome(assembly='mm10', chroms=chroms,
                        lengths=lengths, origins=origins)
        # Check the results
        self.assertFalse(genome.check_sorted())
        

class TestIndex(unittest.TestCase):
    """Test Index class"""
    
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_get_chromint(self):
        """Test initialization of Genome from binary chromosomes.
        """
        # Define the input
        chromstr = ['chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chr7', 'chrX', 'chrX']
        start = [0, 100, 200, 0, 100, 0, 0, 100]
        end = [100, 200, 300, 100, 200, 100, 100, 200]
        # Initialize the Index object
        index = Index(chrom=chromstr, start=start, end=end)
        # Check the results
        chromint = [1, 1, 1, 2, 2, 7, 100, 100]
        np.testing.assert_array_equal(index.get_chromint(), chromint)
        
    

if __name__ == '__main__':
    unittest.main()
