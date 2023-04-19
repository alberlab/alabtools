import unittest
import os
import numpy as np
import pickle as pkl
from alabtools.imaging import CtFile
from alabtools.utils import Genome, Index

class TestCtFile(unittest.TestCase):
    """Test class for CtFile.

    Args:
        unittest (unittest.TestCase): Test class for unittest.
    """
    
    def setUp(self):
        super().setUp()
        # load the test data
        with open('test_data.pkl', 'rb') as f:
            self.data = pkl.load(f)
        self.fofct_file = 'test_fofct.csv'
    
    def test_set_from_fofct(self):
        """Test the set_from_fofct method of CtFile.
        """
        
        # create a CtFile object and set the data from fofct
        ct = CtFile('test_ct.ct', 'w')
        ct.set_from_fofct(self.fofct_file)
        
        # check the results
        self.assertTrue(ct.genome.assembly, self.data['assembly'])
        self.assertTrue(np.array_equal(ct.index.chromstr, self.data['chromstr']))
        self.assertTrue(np.array_equal(ct.index.start, self.data['start']))
        self.assertTrue(np.array_equal(ct.index.end, self.data['end']))
        self.assertTrue(ct.ncell, self.data['ncell'])
        self.assertTrue(ct.ndomain, self.data['ndomain'])
        self.assertTrue(ct.ncopy_max, self.data['ncopy_max'])
        self.assertTrue(ct.nspot_max, self.data['nspot_max'])
        self.assertTrue(ct.nspot_tot, self.data['nspot_tot'])
        self.assertTrue(ct.ntrace_tot, self.data['ntrace_tot'])
        self.assertTrue(np.array_equal(ct.ncopy, self.data['ncopy']))
        self.assertTrue(np.array_equal(ct.nspot, self.data['nspot']))
        self.assertTrue(np.allclose(ct.coordinates, self.data['coordinates'], equal_nan=True))
        
        # close and delete the file
        ct.close()
        os.remove('test_ct.ct')
    
    def test_merge(self):
        """Test the merge method of CtFile.
        """
        
        # create two CtFile objects and merge them
        ct1 = CtFile('test_ct1.ct', 'w')
        ct1.set_from_fofct(self.fofct_file)
        ct2 = CtFile('test_ct2.ct', 'w')
        ct2.set_from_fofct(self.fofct_file)
        ct = ct1.merge(ct2, 'test_ct_merged.ct', tag1='1', tag2='2')
        
        # check the results
        self.assertTrue(ct.genome.assembly, self.data['assembly'])
        self.assertTrue(np.array_equal(ct.index.chromstr, self.data['chromstr']))
        self.assertTrue(np.array_equal(ct.index.start, self.data['start']))
        self.assertTrue(np.array_equal(ct.index.end, self.data['end']))
        self.assertTrue(ct.ncell, self.data['ncell'] * 2)
        self.assertTrue(ct.ndomain, self.data['ndomain'])
        self.assertTrue(ct.ncopy_max, self.data['ncopy_max'])
        self.assertTrue(ct.nspot_max, self.data['nspot_max'])
        self.assertTrue(ct.nspot_tot, self.data['nspot_tot'] * 2)
        self.assertTrue(ct.ntrace_tot, self.data['ntrace_tot'] * 2)
        self.assertTrue(np.array_equal(ct.ncopy, np.concatenate((self.data['ncopy'],
                                                                 self.data['ncopy']),
                                                                axis=0)))
        self.assertTrue(np.array_equal(ct.nspot, np.concatenate((self.data['nspot'],
                                                                 self.data['nspot']),
                                                                axis=0)))
        self.assertTrue(np.allclose(ct.coordinates, np.concatenate((self.data['coordinates'],
                                                                    self.data['coordinates']),
                                                                   axis=0),
                           equal_nan=True))
        
        # close and delete the files
        ct1.close()
        ct2.close()
        ct.close()
        os.remove('test_ct1.ct')
        os.remove('test_ct2.ct')
        os.remove('test_ct_merged.ct')
    
    def test_set_manually(self):
        """Test the manual setting of CtFile.
        """
        
        # open a CtFile object and set the data manually
        ct = CtFile('test_ct.ct', 'w')
        genome = Genome(self.data['assembly'], usechr=np.unique(self.data['chromstr']))
        index = Index(chrom=self.data['chromstr'],
                      start=self.data['start'],
                      end=self.data['end'],
                      genome=genome)
        ct.set_manually(self.data['coordinates'], genome, index)
        
        # check the results
        self.assertTrue(ct.genome.assembly, self.data['assembly'])
        self.assertTrue(np.array_equal(ct.index.chromstr, self.data['chromstr']))
        self.assertTrue(np.array_equal(ct.index.start, self.data['start']))
        self.assertTrue(np.array_equal(ct.index.end, self.data['end']))
        self.assertTrue(ct.ncell, self.data['ncell'])
        self.assertTrue(ct.ndomain, self.data['ndomain'])
        self.assertTrue(ct.ncopy_max, self.data['ncopy_max'])
        self.assertTrue(ct.nspot_max, self.data['nspot_max'])
        self.assertTrue(ct.nspot_tot, self.data['nspot_tot'])
        self.assertTrue(ct.ntrace_tot, self.data['ntrace_tot'])
        self.assertTrue(np.array_equal(ct.ncopy, self.data['ncopy']))
        self.assertTrue(np.array_equal(ct.nspot, self.data['nspot']))
        self.assertTrue(np.allclose(ct.coordinates, self.data['coordinates'], equal_nan=True))
        ct.close()
        os.remove('test_ct.ct')
    
    def test_trimming(self):
        """Test the trim method of CtFile for trimming.
        First we inclulde a number of NaN columns in copies and spots of the coordinates.
        Then we create a CtFile object and trim the spots.
        """
        
        ncopy_totrim = 2  # number of NaN columns to trim for copies
        nspot_totrim = 3  # number of NaN columns to trim for spots
        
        # include NaN columns in copies
        coordinates_totrim = np.concatenate((self.data['coordinates'],
                                            np.full((self.data['ncell'],
                                                    self.data['ndomain'],
                                                    ncopy_totrim,
                                                    self.data['nspot_max'], 3), np.nan)),
                                            axis=2)
        # include NaN columns in spots
        coordinates_totrim = np.concatenate((coordinates_totrim,
                                            np.full((self.data['ncell'],
                                                    self.data['ndomain'],
                                                    self.data['ncopy_max']+ncopy_totrim,
                                                    nspot_totrim, 3), np.nan)),
                                            axis=3)

        # create CtFile object and set data manually
        ct = CtFile('test_ct.ct', 'w')
        genome = Genome(self.data['assembly'], usechr=np.unique(self.data['chromstr']))
        index = Index(chrom=self.data['chromstr'],
                      start=self.data['start'],
                      end=self.data['end'],
                      genome=genome)
        ct.set_manually(coordinates_totrim, genome, index)  # use coordinates with NaN columns
        
        # trim the spots
        ct.trim()
        
        # check the results
        self.assertTrue(ct.genome.assembly, self.data['assembly'])
        self.assertTrue(np.array_equal(ct.index.chromstr, self.data['chromstr']))
        self.assertTrue(np.array_equal(ct.index.start, self.data['start']))
        self.assertTrue(np.array_equal(ct.index.end, self.data['end']))
        self.assertTrue(ct.ncell, self.data['ncell'])
        self.assertTrue(ct.ndomain, self.data['ndomain'])
        self.assertTrue(ct.ncopy_max, self.data['ncopy_max'])
        self.assertTrue(ct.nspot_max, self.data['nspot_max'])
        self.assertTrue(ct.nspot_tot, self.data['nspot_tot'])
        self.assertTrue(ct.ntrace_tot, self.data['ntrace_tot'])
        self.assertTrue(np.array_equal(ct.ncopy, self.data['ncopy']))
        self.assertTrue(np.array_equal(ct.nspot, self.data['nspot']))
        self.assertTrue(np.allclose(ct.coordinates, self.data['coordinates'], equal_nan=True))
        
        # close and remove the CtFile
        ct.close()
        os.remove('test_ct.ct')
        
    

if __name__ == '__main__':
    unittest.main()
