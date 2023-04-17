import unittest
import os
import numpy as np
import pickle as pkl
from alabtools.utils import Genome, Index
from alabtools.imaging import CtFile
from alabtools.imaging.phasing import WSPhaser

class TestWSPhaser(unittest.TestCase):
    """Test class for WSPhaser.
    
    Attention: This test might fail if the test data sample is small,
               due to spots potentially deemd outliers.
               Additionally, the test data sample should be nicely
               separated into clusters, so that there is no misclassification.

    Args:
        unittest (unittest.TestCase): Test class for unittest.
    """
    
    def setUp(self):
        super().setUp()
        # load the test data
        with open('test_data.pkl', 'rb') as f:
            self.data = pkl.load(f)
        self.fofct_file = 'test_fofct.csv'
    
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
        assert ct_phsd.genome.assembly == self.data['assembly']
        assert np.array_equal(ct_phsd.index.chromstr, self.data['chromstr'])
        assert np.array_equal(ct_phsd.index.start, self.data['start'])
        assert np.array_equal(ct_phsd.index.end, self.data['end'])
        assert ct_phsd.ncell == self.data['ncell']
        assert ct_phsd.ndomain == self.data['ndomain']
        assert ct_phsd.ncopy_max == self.data['ncopy_max']
        assert ct_phsd.nspot_max == self.data['nspot_max']
        assert ct_phsd.nspot_tot == self.data['nspot_tot']
        assert ct_phsd.ntrace_tot == self.data['ntrace_tot']
        assert np.array_equal(ct_phsd.ncopy, self.data['ncopy'])
        assert np.array_equal(ct_phsd.nspot, self.data['nspot'])
        assert np.allclose(ct_phsd.coordinates, self.data['coordinates'], equal_nan=True)
        
        # clean up
        ct.close()
        ct_phsd.close()
        os.remove('test_ct.ct')
        os.remove('test_ct_phased.ct')
        
    

if __name__ == '__main__':
    unittest.main()

