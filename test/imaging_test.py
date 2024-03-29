import unittest
import os
import numpy as np
from alabtools import Genome, Index
from alabtools.imaging import CtFile
from alabtools.imaging.utils_imaging import flatten_coordinates
from alabtools.imaging.phasing import WSPhaser
from alabtools.imaging.ctenvelope import fit_alphashape


class TestCtFile(unittest.TestCase):
    """Test class for CtFile.
    Args:
        unittest (unittest.TestCase): Test class for unittest.
    """
    
    def setUp(self):
        super().setUp()
        # set file directory as the working directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        # create the test data
        self.data = createTestData()
        # write the fofct file
        self.fofct_file = 'test_fofct.csv'
        writeFofctFile(self.fofct_file, self.data)
    
    def tearDown(self):
        super().tearDown()
        # remove the fofct file
        os.remove(self.fofct_file)
    
    def test_set_from_fofct(self):
        """Test the set_from_fofct method of CtFile."""
        # create a CtFile object and set the data from fofct
        ct = CtFile('test_ct.ct', 'w')
        ct.set_from_fofct(self.fofct_file)
        # sort the cells by int-casted cell labels (with str labels, the order is not correct)
        ct.sort_cells(order=np.argsort(ct.cell_labels.astype(int)))
        # check the results
        self._assertCtFile(ct, self.data)
        # close and delete the file
        ct.close()
        os.remove('test_ct.ct')
    
    def test_merge(self):
        """Test the merge method of CtFile."""
        # modify the test data for merging
        mrgd_data = self.data.copy()
        mrgd_data['ncell'] = 2 * self.data['ncell']
        mrgd_data['nspot_tot'] = 2 * self.data['nspot_tot']
        mrgd_data['ntrace_tot'] = 2 * self.data['ntrace_tot']
        mrgd_data['ncopy'] = np.concatenate((self.data['ncopy'],
                                             self.data['ncopy']),
                                            axis=0)
        mrgd_data['nspot'] = np.concatenate((self.data['nspot'],
                                             self.data['nspot']),
                                            axis=0)
        mrgd_data['coordinates'] = np.concatenate((self.data['coordinates'],
                                                   self.data['coordinates']),
                                                  axis=0)
        mrgd_data['intensity'] = np.concatenate((self.data['intensity'],
                                                 self.data['intensity']),
                                                axis=0)
        # create two CtFile objects and merge them
        ct1 = CtFile('test_ct1.ct', 'w')
        ct1.set_from_fofct(self.fofct_file)
        ct1.sort_cells(np.argsort(ct1.cell_labels.astype(int)))
        ct2 = CtFile('test_ct2.ct', 'w')
        ct2.set_from_fofct(self.fofct_file)
        ct2.sort_cells(np.argsort(ct2.cell_labels.astype(int)))
        ct = ct1.merge(ct2, 'test_ct_merged.ct', tag1='1', tag2='2')
        # check the results
        self._assertCtFile(ct, mrgd_data)
        # close and delete the files
        ct1.close()
        ct2.close()
        ct.close()
        os.remove('test_ct1.ct')
        os.remove('test_ct2.ct')
        os.remove('test_ct_merged.ct')
    
    def test_set_manually(self):
        """Test the manual setting of CtFile."""    
        # open a CtFile object and set the data manually
        ct = CtFile('test_ct.ct', 'w')
        genome = Genome(self.data['assembly'], chroms=self.data['chroms'], origins=self.data['origins'], lengths=self.data['lengths'])
        index = Index(chrom=self.data['chromstr'],
                      start=self.data['start'],
                      end=self.data['end'],
                      genome=genome)
        ct.set_manually(self.data['coordinates'], genome, index, intensity=self.data['intensity'])
        # check the results
        self._assertCtFile(ct, self.data)
        # close and delete the file
        ct.close()
        os.remove('test_ct.ct')
    
    def test_trimming(self):
        """Test the trim method of CtFile for trimming.
        First we inclulde a number of NaN columns in copies and spots of the coordinates.
        Then we create a CtFile object and trim the spots."""
        ncopy_totrim = 2  # number of NaN columns to trim for copies
        nspot_totrim = 3  # number of NaN columns to trim for spots
        # include NaN columns in copies
        coordinates_totrim = np.concatenate((self.data['coordinates'],
                                            np.full((self.data['ncell'],
                                                    self.data['ndomain'],
                                                    ncopy_totrim,
                                                    self.data['nspot_max'], 3), np.nan)),
                                            axis=2)
        intensity_totrim = np.concatenate((self.data['intensity'],
                                           np.full((self.data['ncell'],
                                                    self.data['ndomain'],
                                                    ncopy_totrim,
                                                    self.data['nspot_max']), np.nan)),
                                           axis=2)
        # include NaN columns in spots
        coordinates_totrim = np.concatenate((coordinates_totrim,
                                            np.full((self.data['ncell'],
                                                    self.data['ndomain'],
                                                    self.data['ncopy_max']+ncopy_totrim,
                                                    nspot_totrim, 3), np.nan)),
                                            axis=3)
        intensity_totrim = np.concatenate((intensity_totrim,
                                           np.full((self.data['ncell'],
                                                    self.data['ndomain'],
                                                    self.data['ncopy_max']+ncopy_totrim,
                                                    nspot_totrim), np.nan)),
                                             axis=3)
        # create CtFile object and set data manually
        ct = CtFile('test_ct.ct', 'w')
        genome = Genome(self.data['assembly'], chroms=self.data['chroms'], origins=self.data['origins'], lengths=self.data['lengths'])
        index = Index(chrom=self.data['chromstr'],
                      start=self.data['start'],
                      end=self.data['end'],
                      genome=genome)
        ct.set_manually(coordinates_totrim, genome, index, intensity=intensity_totrim)  # use data with NaN columns
        # trim the spots
        ct.trim()
        # check the results
        self._assertCtFile(ct, self.data)
        # close and remove the CtFile
        ct.close()
        os.remove('test_ct.ct')
    
    def test_sort_copies(self):
        """Test the sort_copies method of CtFile."""
        n_nan_copy = 2  # number of NaN columns to add for copies
        # create data with NaN columns for copies
        coordinates_tosort = np.full((self.data['ncell'],
                                      self.data['ndomain'],
                                      self.data['ncopy_max']+n_nan_copy,
                                      self.data['nspot_max'],
                                      3),
                                     np.nan)
        intensity_tosort = np.full((self.data['ncell'],
                                    self.data['ndomain'],
                                    self.data['ncopy_max']+n_nan_copy,
                                    self.data['nspot_max']),
                                   np.nan)
        # fill in the data
        for cellnum in range(self.data['ncell']):
            for chrom in np.unique(self.data['chromstr']):
                # map the copy index without NaNs to the copy index with NaNs
                cp_map = np.sort(np.random.choice(np.arange(self.data['ncopy_max'] + n_nan_copy),
                                                  self.data['ncopy_max'],
                                                  replace=False))
                # fill in the coordinates and copies data for each copy
                for cp in range(self.data['ncopy_max']):
                    coordinates_tosort[cellnum, self.data['chromstr']==chrom, cp_map[cp], :, :] = \
                        self.data['coordinates'][cellnum, self.data['chromstr']==chrom, cp, :, :]
                    intensity_tosort[cellnum, self.data['chromstr']==chrom, cp_map[cp], :] = \
                        self.data['intensity'][cellnum, self.data['chromstr']==chrom, cp, :]
        # create the coordinates/nspot arrays to test
        coordinates_test = np.full((self.data['ncell'],
                                    self.data['ndomain'],
                                    self.data['ncopy_max']+n_nan_copy,
                                    self.data['nspot_max'],
                                    3),
                                   np.nan)
        coordinates_test[:, :, :self.data['ncopy_max'], :, :] = self.data['coordinates']
        intensity_test = np.full((self.data['ncell'],
                                  self.data['ndomain'],
                                  self.data['ncopy_max']+n_nan_copy,
                                  self.data['nspot_max']),
                                 np.nan)
        intensity_test[:, :, :self.data['ncopy_max'], :] = self.data['intensity']
        nspot_test = np.zeros((self.data['ncell'],
                                 self.data['ndomain'],
                                 self.data['ncopy_max']+n_nan_copy))
        nspot_test[:, :, :self.data['ncopy_max']] = self.data['nspot']
        # create a CtFile object and set the data manually
        ct = CtFile('test_ct.ct', 'w')
        genome = Genome(self.data['assembly'], chroms=self.data['chroms'], origins=self.data['origins'], lengths=self.data['lengths'])
        index = Index(chrom=self.data['chromstr'],
                      start=self.data['start'],
                      end=self.data['end'],
                      genome=genome)
        ct.set_manually(coordinates_tosort, genome, index, intensity=intensity_tosort)
        # sort the copies
        ct.sort_copies()
        # check the results
        np.testing.assert_array_almost_equal(ct.coordinates, coordinates_test, decimal=3)
        np.testing.assert_array_equal(ct.intensity, intensity_test)
        np.testing.assert_array_equal(ct.nspot, nspot_test)
        # close and remove the CtFile
        ct.close()
        os.remove('test_ct.ct')
    
    def test_sort_spots(self):
        """Test the sort_spots method of CtFile."""
        n_nan_spot = 3  # number of NaNs to add for spots
        # extend the data with NaN columns for spots
        coordinates_tosort = np.full((self.data['ncell'],
                                      self.data['ndomain'],
                                      self.data['ncopy_max'],
                                      self.data['nspot_max']+n_nan_spot,
                                      3),
                                     np.nan) 
        intensity_tosort = np.full((self.data['ncell'],
                                    self.data['ndomain'],
                                    self.data['ncopy_max'],
                                    self.data['nspot_max']+n_nan_spot),
                                   np.nan)   
        for cellnum in range(self.data['ncell']):
            for domainnum in range(self.data['ndomain']):
                for cp in range(self.data['ncopy_max']):
                    # map the spot index without NaNs to the spot index with NaNs
                    spt_map = np.sort(np.random.choice(np.arange(self.data['nspot_max'] + n_nan_spot),
                                                       self.data['nspot_max'],
                                                       replace=False))
                    for spt in range(self.data['nspot_max']):
                        coordinates_tosort[cellnum, domainnum, cp, spt_map[spt], :] = \
                            self.data['coordinates'][cellnum, domainnum, cp, spt, :]
                        intensity_tosort[cellnum, domainnum, cp, spt_map[spt]] = \
                            self.data['intensity'][cellnum, domainnum, cp, spt]
        # write the test data (with NaNs at the end of spots)
        coordinates_test = np.full((self.data['ncell'],
                                    self.data['ndomain'],
                                    self.data['ncopy_max'],
                                    self.data['nspot_max']+n_nan_spot,
                                    3),
                                    np.nan)
        coordinates_test[:, :, :, :self.data['nspot_max'], :] = self.data['coordinates']
        intensity_test = np.full((self.data['ncell'],
                                  self.data['ndomain'],
                                  self.data['ncopy_max'],
                                  self.data['nspot_max']+n_nan_spot),
                                 np.nan)
        intensity_test[:, :, :, :self.data['nspot_max']] = self.data['intensity']
        # create a CtFile object and set the data manually
        ct = CtFile('test_ct.ct', 'w')
        genome = Genome(self.data['assembly'], chroms=self.data['chroms'], origins=self.data['origins'], lengths=self.data['lengths'])
        index = Index(chrom=self.data['chromstr'],
                      start=self.data['start'],
                      end=self.data['end'],
                      genome=genome)
        ct.set_manually(coordinates_tosort, genome, index, intensity=intensity_tosort)
        # sort the spots
        ct.sort_spots()
        # check the results
        np.testing.assert_array_almost_equal(ct.coordinates, coordinates_test, decimal=3)
        np.testing.assert_array_equal(ct.intensity, intensity_test)
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
        intensity_collapsed = np.reshape(self.data['intensity'],
                                         (self.data['ncell'],
                                          self.data['ndomain'],
                                          1,
                                          self.data['ncopy_max'] * self.data['nspot_max']))
        # check data are collapsed correctly
        for cellnum in range(self.data['ncell']):
            for domainnum in range(self.data['ndomain']):
                assert np.array_equal(self.data['intensity'][cellnum, domainnum, :, :].flatten(),
                                      intensity_collapsed[cellnum, domainnum, 0, :],
                                      equal_nan=True), 'Intensity is not collapsed correctly.'
                for i in range(3):
                    assert np.array_equal(self.data['coordinates'][cellnum, domainnum, :, :, i].flatten(),
                                          coordinates_collapsed[cellnum, domainnum, 0, :, i],
                                          equal_nan=True), 'Coordinates are not collapsed correctly.'
        
        # create CtFile object and set data manually
        ct = CtFile('test_ct.ct', 'w')
        genome = Genome(self.data['assembly'], chroms=self.data['chroms'], origins=self.data['origins'], lengths=self.data['lengths'])
        index = Index(chrom=self.data['chromstr'],
                      start=self.data['start'],
                      end=self.data['end'],
                      genome=genome)
        ct.set_manually(coordinates_collapsed, genome, index, intensity=intensity_collapsed)
        
        # configure the phaser and initialize it
        config = {'ct_name': 'test_ct.ct',
                  'parallel': {'controller': 'serial'},
                  'ncluster': {'#': 2, 'chrX': 1},
                  'additional_parameters': {'st': 1.2, 'ot': 2.5}}
        phaser = WSPhaser(config)
        
        # run the phasing
        ct_phsd = phaser.run()

        # check the results
        self._assertCtFile(ct_phsd, self.data)
        
        # clean up
        ct.close()
        ct_phsd.close()
        os.remove('test_ct.ct')
        os.remove('test_ct_phased.ct')
    
    def test_flatten_coordinates(self):
        """Test the flatten_coordinates function in imaging utils."""
        # test with different number of dimensions
        n_tests = [(3, 2),  # n1=3, n2=2 (np.array((3, 2, 3)))
                   (3, 2, 4),  # n1=3, n2=2, n3=4
                   (3, 2, 4, 6),  # n1=3, n2=2, n3=4, n4=6
                   (3, 2, 4, 6, 1)]  # n1=3, n2=2, n3=4, n4=6, n5=1
        for n in n_tests:
            # create random coordinates (np.array(n1, n2, ..., nK, 3))
            crd = np.random.rand(*n, 3)
            # flatten the coordinates (np.array(n1*n2*...*nK, 3)
            # and the corresponding indices (np.array(n1*n2*...*nK, K))
            crd_flat, idx = flatten_coordinates(crd)
            # check that the flattened coordinates are the same as the original coordinates
            for w, ijk in enumerate(idx):
                np.testing.assert_allclose(crd[tuple(ijk)], crd_flat[w])
    
    def test_fit_alphashape(self):
        """Test the fit_alphashape function in imaging ctenvelope."""
        # create random coordinates within a unit sphere
        n_points = 5000  # enough points to fit a sphere
        theta = np.random.rand(n_points) * 2 * np.pi
        phi = np.arccos(2 * np.random.rand(n_points) - 1)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts = np.array([x, y, z]).T
        # fit an alpha shape
        alpha, mesh = fit_alphashape(0, pts, alpha=0.0005)
        # check that the alpha shape is a sphere
        # generate random points within a sphere of radius 2
        n_check = 1000
        for _ in range(n_check):
            r = 2 * np.random.rand()
            theta = np.random.rand() * 2 * np.pi
            phi = np.arccos(2 * np.random.rand() - 1)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            pt = np.array([[x, y, z]])
            if r < 0.9:  # points within the sphere should be contained in the alpha shape
                self.assertTrue(mesh.contains(pt))
            if r > 1.1:  # points outside the sphere should not be contained in the alpha shape
                self.assertFalse(mesh.contains(pt))
    
    
    def _assertCtFile(self, ct, test_data):
        """Assert the CtFile object with the test data."""
        self.assertEqual(ct.genome.assembly, test_data['assembly'])
        np.testing.assert_array_equal(ct.genome.chroms, test_data['chroms'])
        np.testing.assert_array_equal(ct.genome.origins, test_data['origins'])
        np.testing.assert_array_equal(ct.genome.lengths, test_data['lengths'])
        np.testing.assert_array_equal(ct.index.chromstr, test_data['chromstr'])
        np.testing.assert_array_equal(ct.index.start, test_data['start'])
        np.testing.assert_array_equal(ct.index.end, test_data['end'])
        self.assertEqual(ct.ndomain, test_data['ndomain'])
        self.assertEqual(ct.ncopy_max, test_data['ncopy_max'])
        self.assertEqual(ct.nspot_max, test_data['nspot_max'])
        self.assertEqual(ct.ncell, test_data['ncell'])
        self.assertEqual(ct.nspot_tot, test_data['nspot_tot'])
        self.assertEqual(ct.ntrace_tot, test_data['ntrace_tot'])
        np.testing.assert_array_equal(ct.ncopy, test_data['ncopy'])
        np.testing.assert_array_equal(ct.nspot, test_data['nspot'])
        np.testing.assert_allclose(ct.coordinates, test_data['coordinates'], equal_nan=True)
        np.testing.assert_array_equal(ct.intensity, test_data['intensity'])


if __name__ == '__main__':
    unittest.main()


def writeFofctFile(filename, data):
    """Write a FoF-CT file (csv format) from the data.
    
    Saves the file in the current directory with the input filename.

    Args:
        data (dict): Dictionary with the data to write.

    Returns:
        None
    """
    
    # create the FoF-CT file
    fofct_file = open(filename, 'w')
    
    # write the header
    fofct_file.write('#FOF-CT_version=v0.1,\n')
    fofct_file.write('#genome_assembly=hg38,\n')
    fofct_file.write('##XYZ_unit=micron,\n')
    fofct_file.write('"#experimenter_name: Francesco Musella,,,\n')
    fofct_file.write('###lab_name: Frank Alber,\n')
    fofct_file.write('#""description: test FOFCT file for CtFile reading,\n')
    fofct_file.write('##columns=(Spot_ID,Trace_ID,X,Y,Z,Intensity,' +
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
                    x = data['coordinates'][cellnum, domainnum, copynum, spotnum, 0]
                    y = data['coordinates'][cellnum, domainnum, copynum, spotnum, 1]
                    z = data['coordinates'][cellnum, domainnum, copynum, spotnum, 2]
                    lum = data['intensity'][cellnum, domainnum, copynum, spotnum]
                    if np.isnan(x) or np.isnan(y) or np.isnan(z):
                        raise ValueError('Coordinates must not be NaN.')
                    fofct_file.write('{},'.format(spotID))
                    fofct_file.write('{}_{}_{},'.format(cellnum, chrom, copynum))
                    fofct_file.write('{},{},{},'.format(x, y, z))
                    fofct_file.write('{},'.format(lum))
                    fofct_file.write('{},{},{},'.format(chrom, chrom_start, chrom_end))
                    fofct_file.write('{},'.format(cellnum))
                    fofct_file.write('{},'.format(cellnum**2))  # useless
                    fofct_file.write('{}'.format(cellnum**3))  # useless
                    fofct_file.write('\n')
                    spotID += 1
    
    # close the file
    fofct_file.close()
    
    return None

# Functions to generate random data for testing

def createTestData():
    """Generates random chromating tracing data for testing.
    
    For the coordinates generation, spots of different alleles are highly separated.
    In particular, they are generated in 3D boxes with distances much larger than the box size.
    
    The data consists of:
        - attributes (int): ncell, ndomain, ncopy_max, nspot_max, nspot_tot, ntrace_tot
        - genome (alabtools.Genome)
        - index (alabtools.Index)
        - ncopy (np.ndarray): ncell x ndomain
        - nspot (np.ndarray): ncell x ndomain x ncopy_max
        - coordinates (np.ndarray): ncell x ndomain x ncopy_max x nspot_max x 3
        - intensity (np.ndarray): ncell x ndomain x ncopy_max x nspot_max

    Returns:
        data (dict): Dictionary with the data.
    """
    
    np.random.seed(0)  # for reproducibility
    # set attributes
    assembly, chroms, origins, lengths, resolution, ncell, ncopy_max, nspot_max = set_attributes()
    # create the data
    chromstr, start, end = create_index(chroms, origins, lengths, resolution)
    ndomain = len(chromstr)
    cell_labels = create_cell_labels(ncell)
    ncopy = create_ncopy(ncell, ndomain, ncopy_max, chromstr)
    nspot = create_nspot(ncell, ndomain, ncopy_max, nspot_max, ncopy)
    coordinates = create_coordinates(ncell, ndomain, ncopy_max, nspot_max, ncopy, nspot)
    intensity = create_intensity(ncell, ndomain, ncopy_max, nspot_max, ncopy, nspot)
    nspot_tot = np.sum(nspot)
    ntrace_tot = compute_ntrace_tot(chroms, chromstr, ncopy)
    # return everything as a dictionary
    data = create_data_dicionary(assembly, chroms, origins, lengths,  # genome properties
                                 chromstr, start, end,  # index properties
                                 ncell, ndomain, ncopy_max, nspot_max, nspot_tot, ntrace_tot,  # attributes
                                 cell_labels, ncopy, nspot, coordinates, intensity)  # data
    return data


def set_attributes():
    """Set the attributes of the CtFile object for testing."""
    assembly = 'hg38'
    chroms = ['chr1', 'chr2', 'chrX']
    origins = [0, 4000, 1000]
    lengths = [100000, 76000, 59000]
    resolution = 1000
    ncell = 5
    ncopy_max = 2
    nspot_max = 7
    return assembly, chroms, origins, lengths, resolution, ncell, ncopy_max, nspot_max

def create_chromosome_index(chrom, origin, length, resolution):
    # generate the start array with fixed resolution
    start = np.arange(origin, origin + length, resolution).astype(int)
    chromstr = np.repeat(chrom, len(start)).astype('U10')
    # add random noise (positive or negative) to the start array
    sign = np.random.choice([-1, 1], len(start))
    noise = sign * np.random.randint(0, int(np.ceil(0.05 * resolution))+1, len(start))
    noise[0] = 0  # the first start must be the origin
    start += noise
    # Generate the end array
    end = []
    for i in range(len(start)-1):
        end.append(np.random.randint(start[i], start[i+1]))
    end.append(origin + length)
    end = np.array(end).astype(int)
    # Assert everything is correct
    assert start[0] >= origin, 'The first start is smaller than the origin.'
    assert end[-1] <= origin + length, 'The last end is larger than the origin + length.'
    for i in range(len(start)-1):
        assert start[i+1] > start[i], 'start[{}] is not > start[{}] for {}'.format(i+1, i, chrom)
        assert end[i+1] > end[i], 'end[{}] is not > end[{}] for {}'.format(i+1, i, chrom)
        assert end[i+1] > start[i], 'end[{}] is not > start[{}] for {}'.format(i+1, i, chrom)
        assert start[i+1] > end[i], 'start[{}] is not > end[{}] for {}'.format(i+1, i, chrom)
    return start, end, chromstr

def create_index(chroms, origins, lengths, resolution):
    """Create the index for testing, i.e. the chromosome, start and end arrays."""
    chromstr = []
    start = []
    end = []
    for chrom, origin, length in zip(chroms, origins, lengths):
        start_chrom, end_chrom, chromstr_chrom = create_chromosome_index(chrom, origin, length, resolution)
        chromstr = np.concatenate((chromstr, chromstr_chrom))
        start = np.concatenate((start, start_chrom))
        end = np.concatenate((end, end_chrom))
    chromstr = np.array(chromstr).astype('U10')
    start = np.array(start).astype(int)
    end = np.array(end).astype(int)
    return np.array(chromstr), np.array(start), np.array(end)

def create_cell_labels(ncell):
    """Create the cell labels for testing."""
    return np.arange(ncell).astype('S10')

def create_ncopy(ncell, ndomain, ncopy_max, chromstr):
    """Create the ncopy array for testing.
    We avoid 0 because - if data size is small - it may not be possible to
    reconstruct the original data from the CtFile object."""
    ncopy = np.random.randint(1, ncopy_max + 1, size=(ncell, ndomain))
    ncopy[:, chromstr == 'chrX'] = 1
    ncopy[:, chromstr == 'chrY'] = 1
    return ncopy

def create_nspot(ncell, ndomain, ncopy_max, nspot_max, ncopy):
    """Create the nspot array for testing.
    We avoid 0 because - if data size is small - it may not be possible to
    reconstruct the original data from the CtFile object."""
    nspot = np.zeros((ncell, ndomain, ncopy_max), dtype=int)
    for cellnum in range(ncell):
        for domainnum in range(ndomain):
            for copynum in range(ncopy[cellnum, domainnum]):
                nspot[cellnum, domainnum, copynum] = np.random.randint(1, nspot_max + 1)
    return nspot

def create_coordinates(ncell, ndomain, ncopy_max, nspot_max, ncopy, nspot):
    """Create the coordinates array for testing.
    The coordinates are generated such that the spots of different alleles are
    highly separated in 3D space, to avoid any ambiguity in the reconstruction."""
    boxsize = 500
    separation = 5000
    centers = np.linspace(0, separation * (ncopy_max - 1), ncopy_max)
    coordinates = np.full((ncell, ndomain, ncopy_max, nspot_max, 3), np.nan)
    for cellnum in range(ncell):
        for domainnum in range(ndomain):
            for copynum in range(ncopy[cellnum, domainnum]):
                for spotnum in range(nspot[cellnum, domainnum, copynum]):
                    x, y, z = centers[copynum] + boxsize * (np.random.rand(3) - 0.5)
                    coordinates[cellnum, domainnum, copynum, spotnum, :] = [x, y, z]
    return coordinates

def create_intensity(ncell, ndomain, ncopy_max, nspot_max, ncopy, nspot):
    """Create the intensity array for testing."""
    intensity = np.full((ncell, ndomain, ncopy_max, nspot_max), np.nan)
    for cellnum in range(ncell):
        for domainnum in range(ndomain):
            for copynum in range(ncopy[cellnum, domainnum]):
                for spotnum in range(nspot[cellnum, domainnum, copynum]):
                    intensity[cellnum, domainnum, copynum, spotnum] = np.random.randint(1, 100)
    return intensity

def create_data_dicionary(assembly, chroms, origins, lengths,
                          chromstr, start, end,
                          ncell, ndomain, ncopy_max, nspot_max, nspot_tot, ntrace_tot,
                          cell_labels, ncopy, nspot, coordinates, intensity):
    """Create the data dictionary for testing."""
    data = {
        'assembly': assembly,
        'chroms': chroms,
        'origins': origins,
        'lengths': lengths,
        'chromstr': chromstr,
        'start': start,
        'end': end,
        'ncell': ncell,
        'ndomain': ndomain,
        'ncopy_max': ncopy_max,
        'nspot_max': nspot_max,
        'nspot_tot': nspot_tot,
        'ntrace_tot': ntrace_tot,
        'cell_labels': cell_labels,
        'ncopy': ncopy,
        'nspot': nspot,
        'coordinates': coordinates,
        'intensity': intensity
    }
    return data

def compute_ntrace_tot(chroms, chromstr, ncopy):
    """Compute the total number of traces for testing."""
    ntrace_tot = 0
    # a trace is a chromosome copy in a cell
    # so we have to loop over chromosomes and check how many copies are in each cell
    for chrom in chroms:
        ncopy_chrom = ncopy[:, np.array(chromstr) == chrom]  # ncell x ndomain_chrom
        ncopy_max_chrom = np.max(ncopy_chrom, axis=1)  # ncell
        ntrace_tot += np.sum(ncopy_max_chrom)
    return ntrace_tot
