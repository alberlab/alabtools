import unittest
import os
import numpy as np
from alabtools.imaging import CtFile
from alabtools.utils import Genome, Index

class TestCtFile(unittest.TestCase):
    """Test class for CtFile.

    Args:
        unittest (unittest.TestCase): Test class for unittest.
    """
    
    def setUp(self):
        super().setUp()
        self.data = self.generate_data()
        self.fofct_file = self.write_fofct_file(self.data)
    
    def tearDown(self):
        del self.data
        del self.fofct_file
        os.remove('test_fofct.csv')
    
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
        genome = Genome(self.data['assembly'], usechr=np.unique(np.array(self.data['chromstr'])))
        index = Index(chrom=np.array(self.data['chromstr']),
                      start=np.array(self.data['start']),
                      end=np.array(self.data['end']),
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
        assert np.array_equal(ct.ncopy, self.data['ncopy'])
        assert np.array_equal(ct.nspot, self.data['nspot'])
        assert np.allclose(ct.coordinates, self.data['coordinates'], equal_nan=True)
        ct.close()
        os.remove('test_ct.ct')

    
    @staticmethod
    def write_fofct_file(data):
        """Write a fofct file.

        Args:
            data (dict): Dictionary with the data to write.

        Returns:
            str: Path to the fofct file.
        """
        
        # create the FoF-CT file
        fofct_file = open('test_fofct.csv', 'w')
        
        # write the header
        fofct_file.write('#FOF-CT_version=v0.1,\n')
        fofct_file.write('#genome_assembly=hg38,\n')
        fofct_file.write('##XYZ_unit=micron,\n')
        fofct_file.write('"#experimenter_name: Francesco Musella,,,\n')
        fofct_file.write('###lab_name: Frank Alber,\n')
        fofct_file.write('#""description: test FOFCT file for CtFile reading,\n')
        fofct_file.write('##columns=(Spot_ID,Trace_ID,X,Y,Z,' +
                         'Chrom,Chrom_Start,Chrom_End,Cell_ID,' +
                         'Extra_Cell_ROI_ID,Additional_Feature)\n')
        
        # write the data
        spotID = 1
        for cellnum in range(data['ncell']):
            for domainnum in range(data['ndomain']):
                chrom = data['chromstr'][domainnum]
                chrom_start = data['start'][domainnum]
                chrom_end = data['end'][domainnum]
                for copynum in range(data['ncopy'][cellnum, domainnum]):
                    for spotnum in range(data['nspot'][cellnum, domainnum, copynum]):
                        # I can update here to make x,y,z separated in space
                        # for different copies, to test phasing
                        x = data['coordinates'][cellnum, domainnum, copynum, spotnum, 0]
                        y = data['coordinates'][cellnum, domainnum, copynum, spotnum, 1]
                        z = data['coordinates'][cellnum, domainnum, copynum, spotnum, 2]
                        if np.isnan(x) or np.isnan(y) or np.isnan(z):
                            raise ValueError('Coordinates must not be NaN.')
                        fofct_file.write('{},'.format(spotID))
                        fofct_file.write('{}_{}_{},'.format(cellnum, chrom, copynum))
                        fofct_file.write('{},{},{},'.format(x, y, z))
                        fofct_file.write('{},{},{},'.format(chrom, chrom_start, chrom_end))
                        fofct_file.write('{},'.format(cellnum))
                        fofct_file.write('{},'.format(cellnum**2))  # useless
                        fofct_file.write('{}'.format(cellnum**3))  # useless
                        fofct_file.write('\n')
                        spotID += 1
        
        # close the file
        fofct_file.close()
        
        return fofct_file.name
        
    
    @staticmethod
    def generate_data():
        """Generate data for testing.

        Returns:
            str: Path to the fofct file.
        """
        
        # set seed for reproducibility
        np.random.seed(0)
        
        # set genome assembly
        assembly = 'hg38'
        chroms = ['chr1', 'chr2', 'chrX']
        
        # set the attributes
        ncell = 4
        ndomain = 15
        ncopy_max = 3
        nspot_max = 4
        
        # create the index
        sizes = np.random.rand(len(chroms))
        sizes = ndomain * sizes / np.sum(sizes)
        sizes = np.round(sizes).astype(int)        
        if np.sum(sizes) != ndomain:
            raise ValueError('The sum of sizes must be equal to ndomain.')
        chromstr = []
        start = []
        end = []
        for size, chrom in zip(sizes, chroms):
            for i in range(size):
                chromstr.append(chrom)
                # generate start[i] between end[-1]+1000 and end[i-1]+10000
                if i == 0:  # if i=0 there is end[-1], so start from 0
                    start.append(np.random.randint(0, 10000))
                else:
                    start.append(end[-1] + np.random.randint(1000, 10000))
                # generate end[i] between start[-1]+100 and start[i]+1000
                end.append(start[-1] + np.random.randint(100, 1000))
        
        # create ncopy
        # If I allows for ncopy to be 0, then there is the risk that,
        # for small data, a domain has no copy in any cell, and thus
        # that domain is missed in the test.
        # For now I am not allowing for ncopy to be 0, but this feature
        # can be added in the future.
        ncopy = np.random.randint(1, ncopy_max + 1, size=(ncell, ndomain))
        
        # create nspot
        nspot = np.zeros((ncell, ndomain, ncopy_max), dtype=int)
        for cellnum in range(ncell):
            for domainnum in range(ndomain):
                for copynum in range(ncopy[cellnum, domainnum]):
                    # Same as for ncopy, I am not allowing for nspot to be 0.
                    # If I allows for nspot to be 0, then there is the risk that,
                    # for small data, a copy has no spot in any cell, and thus
                    # that copy is missed in the test.
                    nspot[cellnum, domainnum, copynum] = np.random.randint(1, nspot_max + 1)

        # create coordinates
        coordinates = np.full((ncell, ndomain, ncopy_max, nspot_max, 3), np.nan)
        for cellnum in range(ncell):
            for domainnum in range(ndomain):
                for copynum in range(ncopy[cellnum, domainnum]):
                    for spotnum in range(nspot[cellnum, domainnum, copynum]):
                        coordinates[cellnum, domainnum, copynum, spotnum, :] = 300 * np.random.rand(3)
        
        # compute final attributes
        nspot_tot = np.sum(nspot)
        ntrace_tot = np.nan  # TODO
        
        # return everything as a dictionary
        data = {
            'assembly': assembly,
            'chromstr': chromstr,
            'start': start,
            'end': end,
            'ncell': ncell,
            'ndomain': ndomain,
            'ncopy_max': ncopy_max,
            'nspot_max': nspot_max,
            'nspot_tot': nspot_tot,
            'ntrace_tot': ntrace_tot,
            'ncopy': ncopy,
            'nspot': nspot,
            'coordinates': coordinates
        }
        
        return data
    

if __name__ == '__main__':
    unittest.main()
