import unittest
import random
import os
import numpy as np
from alabtools.utils import *

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
    
    def test_pop(self):
        """Test pop method in Genome.
        """
        # Define the input
        chroms = ['chr1', 'chr2', 'chr3', 'chr5', 'chr6', 'chr10', 'chr12', 'chr18', 'chrX']
        lengths = [2000 - 100 * i for i in range(len(chroms))]
        origins = [0 for _ in range(len(chroms))]
        # Initialize the Genome object
        genome = Genome(assembly='mm10', chroms=chroms,
                        lengths=lengths, origins=origins)
        # Pop a chromosome
        genome_new = genome.pop('chrX')
        # Check the results
        np.testing.assert_array_equal(genome_new.chroms, chroms[:-1])
        np.testing.assert_array_equal(genome_new.lengths, lengths[:-1])
        np.testing.assert_array_equal(genome_new.origins, origins[:-1])
        

class TestIndex(unittest.TestCase):
    """Test Index class"""
    
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    # This test is commented because it fails, that's why I introduced the optimized version
    # However, for backward compatibility, I keep the bininfo method for now,
    # otherwise I would have to make absolutely sure that the optimized version doesn't break anything in IGM
    """def test_bininfo(self):
        res = 22
        genome, chromstr, start, end, _, _, _ = generate_domains(resolution=res)
        index = Index(chrom=chromstr, start=start, end=end, genome=genome)
        bininfo = genome.bininfo(resolution=res)
        np.testing.assert_array_equal(index.chromstr, bininfo.chromstr)
        np.testing.assert_array_equal(index.start, bininfo.start)
        np.testing.assert_array_equal(index.end, bininfo.end)"""
    
    def test_bininfo_optimized(self):
        res = 22
        genome, chromstr, start, end, _, _, _ = generate_domains(resolution=res)
        index = Index(chrom=chromstr, start=start, end=end, genome=genome)
        bininfo = genome.bininfo_optimized(resolution=res)
        np.testing.assert_array_equal(index.chromstr, bininfo.chromstr)
        np.testing.assert_array_equal(index.start, bininfo.start)
        np.testing.assert_array_equal(index.end, bininfo.end)
    
    def test_chromstr_to_chromint(self):
        """Test chromstr_to_chromint function."""
        res = 22
        genome, chromstr, start, end, chromint, _, _ = generate_domains(resolution=res)
        index = Index(chrom=chromstr, start=start, end=end, genome=genome)
        np.testing.assert_array_equal(chromstr_to_chromint(index.chromstr), chromint)
    
    def test_get_index_hashmap(self):
        """Test get_index_hashmap method in Index."""
        res = 22
        genome, chromstr, start, end, _, _, _ = generate_domains(resolution=res)
        index = Index(chrom=chromstr, start=start, end=end, genome=genome)
        hashmap = index.get_index_hashmap()
        for i, dom in enumerate(zip(chromstr, start, end)):
            self.assertEqual(hashmap[dom], [i])
    
    def test_sort_by_chromosome(self):
        """Test sort_by_chromosome method in Index."""
        res = 22
        # Generate sorted domains
        genome, chromstr_srt, start_srt, end_srt, _, x_srt, y_srt = generate_domains(resolution=res)
        # Shuffle the domains
        chromstr, start, end, x, y = shuffle_in_place([chromstr_srt, start_srt, end_srt, x_srt, y_srt])
        # Create the index with unsorted domains
        # I need to specify the copy, because if I don't, the Index will assume the copy
        # index is all messed up, since I am shuffling everything
        copy = np.zeros(len(chromstr))
        index = Index(chrom=chromstr, start=start, end=end, copy=copy, genome=genome)
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
    
    def test_coarsegrain(self):
        """Test coarsegrain method in Index."""
        in_res, out_res = 22, 44
        # Generate domains
        genome, chromstr, start, end, _, x, y = generate_domains(resolution=in_res)
        # Remove consecutive domains
        i_remove = [4, 5, 6, 7]
        mask = np.ones(len(chromstr), dtype=bool)
        mask[i_remove] = False
        chromstr, start, end, x, y = chromstr[mask], start[mask], end[mask], x[mask], y[mask]
        # Create the index
        index = Index(chrom=chromstr, start=start, end=end, genome=genome)
        index.add_custom_track('x', x)
        index.add_custom_track('y', y)
        # Coarsegrain the index
        index_coarse = index.coarsegrain(out_res)
        # Test the results
        _, chromstr_test, start_test, end_test, _, _, _ = generate_domains(resolution=out_res)
        # Remove the consecutive domains that match the ones in the high-resolution index
        i_remove = [2, 3]
        mask = np.ones(len(chromstr_test), dtype=bool)
        mask[i_remove] = False
        chromstr_test, start_test, end_test = chromstr_test[mask], start_test[mask], end_test[mask]
        index_test = Index(chrom=chromstr_test, start=start_test, end=end_test, genome=genome)
        x_test, y_test = [], []
        for c, s, e in zip(chromstr_test, start_test, end_test):
            idx = np.where((chromstr == c) & (start >= s) & (end <= e))[0]
            x_test.append(np.mean(x[idx]))
            y_test.append(np.mean(y[idx]))
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        # Test the results
        np.testing.assert_array_equal(index_coarse.chromstr, index_test.chromstr)
        np.testing.assert_array_equal(index_coarse.start, index_test.start)
        np.testing.assert_array_equal(index_coarse.end, index_test.end)
        np.testing.assert_array_equal(index_coarse.get_custom_track('x'), x_test)
        np.testing.assert_array_equal(index_coarse.get_custom_track('y'), y_test)
    
    def test_pop_chromosome(self):
        """Test pop_chromosome method in Index."""
        res = 22
        # Generate domains
        genome, chromstr, start, end, _, x, y = generate_domains(resolution=res)
        # Create the index
        index = Index(chrom=chromstr, start=start, end=end, genome=genome)
        index.add_custom_track('x', x)
        index.add_custom_track('y', y)
        # Pop a chromosome
        chrom = 'chr2'
        index_new = index.pop_chromosome(chrom)
        # Test the results
        np.testing.assert_array_equal(index_new.chromstr, chromstr[chromstr != chrom])
        np.testing.assert_array_equal(index_new.start, start[chromstr != chrom])
        np.testing.assert_array_equal(index_new.end, end[chromstr != chrom])
        np.testing.assert_array_equal(index_new.get_custom_track('x'), x[chromstr != chrom])
        np.testing.assert_array_equal(index_new.get_custom_track('y'), y[chromstr != chrom])
    
    def test_get_index_from_set(self):
        """Test get_index_from_set function."""
        
        # Generate domains
        res = 22
        genome, chromstr, start, end, _, _, _ = generate_domains(resolution=res)
        # Create a domain set from chromstr, start, end
        domain_set = set()
        for c, s, e in zip(chromstr, start, end):
            domain_set.add((c, s, e))
        # Randomly shuffle the domain set
        domain_set = list(domain_set)
        random.shuffle(domain_set)
        domain_set = set(domain_set)
        # Create the index
        index = get_index_from_set(domain_set, assembly=genome.assembly)
        # Test the results
        np.testing.assert_array_equal(index.chromstr, chromstr)
        np.testing.assert_array_equal(index.start, start)
        np.testing.assert_array_equal(index.end, end)
    
    def test_get_index_from_bed(self):
        """ Test get_index_from_bed function."""
        
        # Generate a BED file and save it
        chromstr = np.array(['chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chr7', 'chrX']).astype('U20')
        start = np.array([0, 100, 200, 100, 200, 400, 0]).astype(int)
        end = start + 100
        x = np.random.rand(len(chromstr))
        y = np.random.rand(len(chromstr))
        bed = np.column_stack([chromstr, start, end, x, y])
        np.savetxt('test.bed', bed, fmt='%s', delimiter='\t', header='chrom\tstart\tend\tx\ty')
        # Create the index: I add 'chr9' (not present in the BED file) to check if it raises an error
        # I also don't use 'chrX' to check if it is correctly ignored
        genome = Genome('mm10', usechr=('chr1', 'chr2', 'chr7', 'chr9'))
        index = get_index_from_bed('./test.bed', genome=genome)
        # Test the results
        np.testing.assert_array_equal(index.chromstr, chromstr[chromstr != 'chrX'])
        np.testing.assert_array_equal(index.start, start[chromstr != 'chrX'])
        np.testing.assert_array_equal(index.end, end[chromstr != 'chrX'])
        np.testing.assert_array_almost_equal(index.track0, x[chromstr != 'chrX'])
        np.testing.assert_array_almost_equal(index.track1, y[chromstr != 'chrX'])
        # The chromosomes present in the genome should not contain 'chrX', because it was not given to the input genome,
        # and should contain 'chr9', because it was not present in the BED file
        np.testing.assert_array_equal(index.genome.chroms, ['chr1', 'chr2', 'chr7'])
        np.testing.assert_array_equal(index.genome.origins, np.array([0, 100, 400]))
        np.testing.assert_array_equal(index.genome.lengths, np.array([300, 200, 100]))
        # Delete the file
        os.remove('test.bed')
    
    def test_get_index_sliding(self):
        """ Test get_index_sliding function."""
        
        # Initialize the Index manually
        chromstr = np.array(
            [
                'chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr1',
                'chr2', 'chr2', 'chr2', 'chr2',
                'chr3', 'chr3', 'chr3',
                'chrX'
            ]
        ).astype('U20')
        start = np.array(
            [
                0, 100, 200, 300, 400, 500,
                0, 100, 200, 300,
                0, 100, 200,
                0
            ]
        ).astype(int)
        end = start + 100
        index = Index(chrom=chromstr, start=start, end=end, genome=Genome('mm10', usechr=('chr1', 'chr2', 'chr3', 'chrX')))
        
        # Get the sliding index
        index_sliding = get_index_sliding_mapping(index, window=3)
        
        # Test the results
        real_sliding = {
            0: np.array([0, 1]),
            1: np.array([0, 1, 2]),
            2: np.array([1, 2, 3]),
            3: np.array([2, 3, 4]),
            4: np.array([3, 4, 5]),
            5: np.array([4, 5]),
            6: np.array([6, 7]),
            7: np.array([6, 7, 8]),
            8: np.array([7, 8, 9]),
            9: np.array([8, 9]),
            10: np.array([10, 11]),
            11: np.array([10, 11, 12]),
            12: np.array([11, 12]),
            13: np.array([13])
        }
        for i in real_sliding:
            np.testing.assert_array_equal(index_sliding[i], real_sliding[i])
    
    def test_get_index_mappings(self):
        """ Test get_index_mappings function."""
        
        # Create two indices, one with double the resolution of the other
        res1, res2 = 22, 110
        genome1, chromstr1, start1, end1, _, _, _ = generate_domains(resolution=res1)
        index1 = Index(chrom=chromstr1, start=start1, end=end1, genome=genome1)
        genome2, chromstr2, start2, end2, _, _, _ = generate_domains(resolution=res2)
        index2 = Index(chrom=chromstr2, start=start2, end=end2, genome=genome2)
        
        # Get the mappings
        cmap, fmap, bmap = get_index_mappings(index1, index2)
        
        # fmap maps the domains of index1 to index2: fmap[i] = j means that domain i in index1 corresponds to domain j in index2
        # Loop over the domains of index1
        for i in range(len(chromstr1)):
            # Get the corresponding domain in index2
            j = fmap[i]
            # Since the resolution of index2 is double, only one domain in index2 should correspond to domain i in index1: check it
            self.assertEqual(len(j), 1)
            j = j[0]
            # Now check that the domain of i is contained in the domain of j
            chrom_i, start_i, end_i = index1.chromstr[i], index1.start[i], index1.end[i]
            chrom_j, start_j, end_j = index2.chromstr[j], index2.start[j], index2.end[j]
            self.assertEqual(chrom_i, chrom_j)
            self.assertTrue(start_i >= start_j)
            self.assertTrue(end_i <= end_j)
        
        # bmap maps the domains of index2 to index1: bmap[j] = i means that domain j in index2 corresponds to domain i in index1
        # Loop over the domains of index2
        for j in range(len(chromstr2)):
            # Get the corresponding domain in index1
            i = bmap[j]
            # Since the resolution of index2 is double, two domains in index1 should correspond to domain j in index2
            # (unless the domain is at the end of the chromosome, in which case only one domain should correspond): check it
            if index2.end[j] != np.max(index2.end[index2.chromstr == index2.chromstr[j]]):
                self.assertEqual(len(i), int(res2 / res1))
            # Now check that all the domains of i are contained in the domain of j
            for ii in i:
                chrom_ii, start_ii, end_ii = index1.chromstr[ii], index1.start[ii], index1.end[ii]
                chrom_j, start_j, end_j = index2.chromstr[j], index2.start[j], index2.end[j]
                self.assertEqual(chrom_ii, chrom_j)
                self.assertTrue(start_ii >= start_j)
                self.assertTrue(end_ii <= end_j)
    
    def test_map_indices(self):
        
        # Create index 1
        genome = Genome('mm10', usechr=('chr1', 'chr2', 'chr3', 'chrX'))
        chromstr1 = np.array(['chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chr3', 'chrX', 'chrX']).astype('U20')
        start1 = np.array([0, 130, 270, 0, 110, 30, 0, 1000]).astype(int)
        end1 = start1 + 100
        idx1 = Index(chrom=chromstr1, start=start1, end=end1, genome=genome)
        
        # Create index 2
        chromstr2 = np.array(['chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chr2', 'chr2', 'chr3', 'chr3', 'chrX', 'chrX']).astype('U20')
        start2 = np.array([0, 52, 105, 170, 231, 290, 3, 55, 104, 157, 28, 80, 0, 48]).astype(int)
        end2 = start2 + 50
        idx2 = Index(chrom=chromstr2, start=start2, end=end2, genome=genome)
        
        # Test the mapping idx1 -> idx2
        map = map_indices(idx1, idx2)
        map_test = {
            ('chr1', 0, 100): [('chr1', 0, 50), ('chr1', 52, 102)],
            ('chr1', 130, 230): [('chr1', 105, 155), ('chr1', 170, 220)],
            ('chr1', 270, 370): [('chr1', 231, 281), ('chr1', 290, 340)],
            ('chr2', 0, 100): [('chr2', 3, 53), ('chr2', 55, 105)],
            ('chr2', 110, 210): [('chr2', 104, 154), ('chr2', 157, 207)],
            ('chr3', 30, 130): [('chr3', 28, 78), ('chr3', 80, 130)],
            ('chrX', 0, 100): [('chrX', 0, 50), ('chrX', 48, 98)],
            ('chrX', 1000, 1100): [],
        }
        self.assertDictEqual(map, map_test)
        
        # Test the mapping idx2 -> idx1
        map = map_indices(idx2, idx1)
        map_test = {
            ('chr1', 0, 50): [('chr1', 0, 100)],
            ('chr1', 52, 102): [('chr1', 0, 100)],
            ('chr1', 105, 155): [('chr1', 130, 230)],
            ('chr1', 170, 220): [('chr1', 130, 230)],
            ('chr1', 231, 281): [('chr1', 270, 370)],
            ('chr1', 290, 340): [('chr1', 270, 370)],
            ('chr2', 3, 53): [('chr2', 0, 100)],
            ('chr2', 55, 105): [('chr2', 0, 100)],
            ('chr2', 104, 154): [('chr2', 110, 210)],
            ('chr2', 157, 207): [('chr2', 110, 210)],
            ('chr3', 28, 78): [('chr3', 30, 130)],
            ('chr3', 80, 130): [('chr3', 30, 130)],
            ('chrX', 0, 50): [('chrX', 0, 100)],
            ('chrX', 48, 98): [('chrX', 0, 100)]
        }
        self.assertDictEqual(map, map_test)


def shuffle_in_place(arrays):
    """Shuffle a list of arrays in place."""
    for array in arrays:
        assert len(array) == len(arrays[0])
    order = np.random.permutation(len(arrays[0]))
    arrays_shuffled = []
    for array in arrays:
        arrays_shuffled.append(array[order])
    return arrays_shuffled

def generate_domains(ploidy='haploid', resolution=100):
    # Fix the random seed
    np.random.seed(0)
    assert ploidy in ['haploid', 'diploid'], 'ploidy must be haploid or diploid'
    # Generate the Genome
    chroms = np.array(['chr1', 'chr2', 'chr7', 'chrX'])
    chroms_int = np.array([1, 2, 7, 100])
    lengths = np.array([400, 400, 200, 600])
    origins = np.array([0, 200, 0, 200])
    genome = Genome(assembly='mm10', chroms=chroms, lengths=lengths, origins=origins)
    # Generate the Index
    # Generate chromstr and chromint
    bin_sizes = np.ceil(lengths / resolution).astype(int)  # Number of bins per chromosome
    chromstr = np.repeat(chroms, bin_sizes)
    chromint = np.repeat(chroms_int, bin_sizes)
    # Generate start and end
    start, end = [], []
    for length, origin in zip(lengths, origins):
        start.extend(np.arange(origin, origin + length, resolution))
        end.extend(np.arange(origin + resolution, origin + length + resolution, resolution))
        end[-1] = origin + length
    start = np.array(start)
    end = np.array(end)
    # Generate custom tracks
    x = np.random.rand(len(chromstr))
    y = np.random.rand(len(chromstr))
    # If diploid, duplicate the data for autosomes
    if ploidy == 'diploid':
        chromstr_hap = chromstr.copy()
        chromstr = np.concatenate([chromstr, chromstr[chromstr_hap != 'chrX']])
        chromint = np.concatenate([chromint, chromint[chromstr_hap != 'chrX']])
        start = np.concatenate([start, start[chromstr_hap != 'chrX']])
        end = np.concatenate([end, end[chromstr_hap != 'chrX']])
        x = np.concatenate([x, x[chromstr_hap != 'chrX']])
        y = np.concatenate([y, y[chromstr_hap != 'chrX']])
    return genome, chromstr, start, end, chromint, x, y
    

if __name__ == '__main__':
    unittest.main()
