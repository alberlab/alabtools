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
        # load data with pickle
        with open('test_data.pkl', 'rb') as f:
            self.data = pkl.load(f)
        self.fofct_file = 'test_fofct.csv'
    
    def tearDown(self):
        del self.data
        del self.fofct_file
    
    def test_set_from_fofct(self):
        """Test the set_from_fofct method of CtFile.
        """
        ct = CtFile('test_ct.ct', 'w')
        ct.set_from_fofct(self.fofct_file)
        assert ct.genome.assembly == self.data['assembly']
        assert np.array_equal(ct.index.chromstr, self.data['chromstr'])
        assert np.array_equal(ct.index.start, self.data['start'])
        assert np.array_equal(ct.index.end, self.data['end'])
        assert ct.ncell == self.data['ncell']
        assert ct.ndomain == self.data['ndomain']
        assert ct.ncopy_max == self.data['ncopy_max']
        assert ct.nspot_max == self.data['nspot_max']
        assert ct.nspot_tot == self.data['nspot_tot']
        assert ct.ntrace_tot == self.data['ntrace_tot']
        assert np.array_equal(ct.ncopy, self.data['ncopy'])
        assert np.array_equal(ct.nspot, self.data['nspot'])
        assert np.allclose(ct.coordinates, self.data['coordinates'], equal_nan=True)
        ct.close()
        # delete the file
        os.remove('test_ct.ct')
    
    def test_merge(self):
        """Test the merge method of CtFile.
        """
        ct1 = CtFile('test_ct1.ct', 'w')
        ct1.set_from_fofct(self.fofct_file)
        ct2 = CtFile('test_ct2.ct', 'w')
        ct2.set_from_fofct(self.fofct_file)
        ct = ct1.merge(ct2, 'test_ct_merged.ct', tag1='1', tag2='2')
        ct1.close()
        ct2.close()
        os.remove('test_ct1.ct')
        os.remove('test_ct2.ct')
        assert ct.genome.assembly == self.data['assembly']
        assert np.array_equal(ct.index.chromstr, self.data['chromstr'])
        assert np.array_equal(ct.index.start, self.data['start'])
        assert np.array_equal(ct.index.end, self.data['end'])
        assert ct.ncell == self.data['ncell'] * 2
        assert ct.ndomain == self.data['ndomain']
        assert ct.ncopy_max == self.data['ncopy_max']
        assert ct.nspot_max == self.data['nspot_max']
        assert ct.nspot_tot == self.data['nspot_tot'] * 2
        assert ct.ntrace_tot == self.data['ntrace_tot'] * 2
        assert np.array_equal(ct.ncopy, np.concatenate((self.data['ncopy'],
                                                        self.data['ncopy']), axis=0))
        assert np.array_equal(ct.nspot, np.concatenate((self.data['nspot'],
                                                        self.data['nspot']), axis=0))
        assert np.allclose(ct.coordinates, np.concatenate((self.data['coordinates'],
                                                           self.data['coordinates']), axis=0),
                           equal_nan=True)
        ct.close()
        os.remove('test_ct_merged.ct')
    
    def test_set_manually(self):
        """Test the manual setting of CtFile.
        """
        ct = CtFile('test_ct.ct', 'w')
        genome = Genome(self.data['assembly'], usechr=np.unique(self.data['chromstr']))
        index = Index(chrom=self.data['chromstr'],
                      start=self.data['start'],
                      end=self.data['end'],
                      genome=genome)
        ct.set_manually(self.data['coordinates'], genome, index)
        assert ct.genome.assembly == self.data['assembly']
        assert np.array_equal(ct.index.chromstr, self.data['chromstr'])
        assert np.array_equal(ct.index.start, self.data['start'])
        assert np.array_equal(ct.index.end, self.data['end'])
        assert ct.ncell == self.data['ncell']
        assert ct.ndomain == self.data['ndomain']
        assert ct.ncopy_max == self.data['ncopy_max']
        assert ct.nspot_max == self.data['nspot_max']
        assert ct.nspot_tot == self.data['nspot_tot']
        assert ct.ntrace_tot == self.data['ntrace_tot']
        assert np.array_equal(ct.ncopy, self.data['ncopy'])
        assert np.array_equal(ct.nspot, self.data['nspot'])
        assert np.allclose(ct.coordinates, self.data['coordinates'], equal_nan=True)
        ct.close()
        os.remove('test_ct.ct')
    

if __name__ == '__main__':
    unittest.main()
