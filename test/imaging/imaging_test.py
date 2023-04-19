import unittest
import os
import numpy as np
import pickle as pkl
from alabtools.imaging import CtFile
from alabtools.utils import Genome, Index
from alabtools.imaging.phasing import WSPhaser

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
        self._assertCtFile(ct)
        
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
        self._assertCtFile(ct, merged=True)
        
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
        self._assertCtFile(ct)
        
        # close and delete the file
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
        self._assertCtFile(ct)
        
        # close and remove the CtFile
        ct.close()
        os.remove('test_ct.ct')
    
    def test_phasing(self):
        """Test the phasing method of WSPhaser.
        """
        
        # collapse the coordinates to a single copy
        coordinates_collapsed = np.reshape(self.data['coordinates'],
                                           (self.data['ncell'],
                                            self.data['ndomain'],
                                            1,
                                            self.data['ncopy_max'] * self.data['nspot_max'],
                                            3))
        # check that the coordinates are collapsed correctly
        for cellnum in range(self.data['ncell']):
            for domainnum in range(self.data['ndomain']):
                for i in range(3):
                    assert np.array_equal(self.data['coordinates'][cellnum, domainnum, :, :, i].flatten(),
                                          coordinates_collapsed[cellnum, domainnum, 0, :, i],
                                          equal_nan=True), 'Coordinates are not collapsed correctly.'
        
        # create CtFile object and set data manually
        ct = CtFile('test_ct.ct', 'w')
        genome = Genome(self.data['assembly'], usechr=np.unique(self.data['chromstr']))
        index = Index(chrom=self.data['chromstr'],
                      start=self.data['start'],
                      end=self.data['end'],
                      genome=genome)
        ct.set_manually(coordinates_collapsed, genome, index)  # use collapsed coordinates
        
        # configure the phaser and initialize it
        config = {'ct_name': 'test_ct.ct',
                  'parallel': {'controller': 'serial'},
                  'ncluster': {'#': 2, 'chrX': 1},
                  'additional_parameters': {'st': 1.2, 'ot': 2.5}}
        phaser = WSPhaser(config)
        
        # run the phasing
        ct_phsd = phaser.run()
        
        # check the results
        self._assertCtFile(ct_phsd)
        
        # clean up
        ct.close()
        ct_phsd.close()
        os.remove('test_ct.ct')
        os.remove('test_ct_phased.ct')
    
    
    def _assertCtFile(self, ct, merged=False):
        """Assert the CtFile object.

        Args:
            ct (CtFile)
        """
            
        self.assertEqual(ct.genome.assembly, self.data['assembly'])
        np.testing.assert_array_equal(ct.index.chromstr, self.data['chromstr'])
        np.testing.assert_array_equal(ct.index.start, self.data['start'])
        np.testing.assert_array_equal(ct.index.end, self.data['end'])
        self.assertEqual(ct.ndomain, self.data['ndomain'])
        self.assertEqual(ct.ncopy_max, self.data['ncopy_max'])
        self.assertEqual(ct.nspot_max, self.data['nspot_max'])
                
        if not merged:
            self.assertEqual(ct.ncell, self.data['ncell'])
            self.assertEqual(ct.nspot_tot, self.data['nspot_tot'])
            self.assertEqual(ct.ntrace_tot, self.data['ntrace_tot'])
            np.testing.assert_array_equal(ct.ncopy, self.data['ncopy'])
            np.testing.assert_array_equal(ct.nspot, self.data['nspot'])
            np.testing.assert_allclose(ct.coordinates, self.data['coordinates'], equal_nan=True)
        
        else:
            self.assertEqual(ct.ncell, 2 * self.data['ncell'])
            self.assertEqual(ct.nspot_tot, 2 * self.data['nspot_tot'])
            self.assertEqual(ct.ntrace_tot, 2 * self.data['ntrace_tot'])
            np.testing.assert_array_equal(ct.ncopy, np.concatenate((self.data['ncopy'],
                                                                    self.data['ncopy']),
                                                                   axis=0))
            np.testing.assert_array_equal(ct.nspot, np.concatenate((self.data['nspot'],
                                                                    self.data['nspot']),
                                                                   axis=0))
            np.testing.assert_allclose(ct.coordinates, np.concatenate((self.data['coordinates'],
                                                                       self.data['coordinates']),
                                                                      axis=0),
                                       equal_nan=True)  


if __name__ == '__main__':
    unittest.main()
