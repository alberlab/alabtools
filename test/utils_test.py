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
        """Test initialization of Genome from binary chromosomes."""
        chromstr, start, end, chromint = generate_domains()
        index = Index(chrom=chromstr, start=start, end=end)
        np.testing.assert_array_equal(index.get_chromint(), chromint)
    
    def test_get_index_hashmap(self):
        """Test get_index_hashmap method in Index."""
        chromstr, start, end, _ = generate_domains()
        index = Index(chrom=chromstr, start=start, end=end)
        hashmap = index.get_index_hashmap()
        for i, dom in enumerate(zip(chromstr, start, end)):
            self.assertEqual(hashmap[dom], [i])
    
    def test_sort_by_chromosome(self):
        """Test sort_by_chromosome method in Index."""
        # Generate sorted domains
        chromstr_srt, start_srt, end_srt, _ = generate_domains()
        x_srt = np.random.rand(len(chromstr_srt))
        y_srt = np.random.rand(len(chromstr_srt))
        # Shuffle the domains
        chromstr, start, end, x, y = shuffle_in_place([chromstr_srt, start_srt, end_srt, x_srt, y_srt])
        # Create the index with unsorted domains
        index = Index(chrom=chromstr, start=start, end=end)
        index.add_custom_track('x', x)
        index.add_custom_track('y', y)
        # Create sorted index
        index_sorted = index.sort_by_chromosome()
        # Test the results
        np.testing.assert_array_equal(index_sorted.chromstr, chromstr_srt)
        np.testing.assert_array_equal(index_sorted.start, start_srt)
        np.testing.assert_array_equal(index_sorted.end, end_srt)
        np.testing.assert_array_equal(index_sorted.get_custom_track('x'), x_srt)
        np.testing.assert_array_equal(index_sorted.get_custom_track('y'), y_srt)

def shuffle_in_place(arrays):
    """Shuffle a list of arrays in place."""
    for array in arrays:
        assert len(array) == len(arrays[0])
    order = np.random.permutation(len(arrays[0]))
    arrays_shuffled = []
    for array in arrays:
        arrays_shuffled.append(array[order])
    return arrays_shuffled

def generate_domains():
    chromstr = np.array(['chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chr7', 'chrX', 'chrX'])
    chromint = np.array([1, 1, 1, 2, 2, 7, 100, 100])
    start = np.array([0, 100, 200, 0, 100, 0, 0, 100])
    end = np.array([100, 200, 300, 100, 200, 100, 100, 200])
    return chromstr, start, end, chromint
    

if __name__ == '__main__':
    unittest.main()
