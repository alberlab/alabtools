# This file contains functions used to process and analyze chromatin tracing data.
# Author: Francesco Musella, University of California Los Angeles
# Date: April 3, 2023

import h5py
import warnings
import numpy as np
import os
import sys
from alabtools import Genome, Index
from .fofct import process as fofct_process
from .utils_imaging import sort_coord_by_boolmask

__author__ = "Francesco Musella"
__ct_version__ = 0.1
__email__ = "fmusella@g.ucla.edu"


class CtFile(h5py.File):
    """
    A class to store and analyze Chromatin Tracing data.
    Inherited from h5py.File.
    
    HDF5 can only store homogeneous arrays (i.e. hyperrectangulars).
    Imaging data are not homogeneous per nature, e.g. there can be multiple spots per domain in a cell.
    We overcome this issue by max-padding the arrays with NaNs.
    
    Attributes
    ---------
    version : str
    ncell : int
    ndomain : int
    nspot_tot: int
    ntrace_tot: int
    nspot_max: int
    ncopy_max: int
    
    Datasets
    ---------
    genome : Genome
    index : Index
    cell_labels = np.array(ncell, dtype='S10')
    coordinates: np.array(ncell, ndomain, ncopy_max, nspot_max, 3, dtype=float32)
    intensity: np.array(ncell, ndomain, ncopy_max, nspot_max, dtype=float32)
    nspot = np.array(ncell, ndomain, ncopy_max, dtype=int32)
    ncopy = np.array(ncell, ndomain, dtype=int32)
    """
    
    def __init__(self, *args, **kwargs):
        
        self._genome = None
        self._index = None
        self._data = None
        
        h5py.File.__init__(self, *args, **kwargs)  # inherits init from h5py.File
        
        try:
            self._version = self.attrs['version']
        except KeyError:
            self._version = __ct_version__
            if self.mode != 'r':
                self.attrs.create('version', self._version, dtype='float')
        
        try:  # Checks if ncell exists, otherwise create it
            self._ncell = self.attrs['ncell']
        except KeyError:
            self._ncell = 0
            if self.mode != 'r':
                self.attrs.create('ncell', self._ncell, dtype='int')
        
        try :
            self._ndomain = self.attrs['ndomain']
        except KeyError:
            self._ndomain = 0
            if self.mode != 'r':
                self.attrs.create('ndomain', self._ndomain, dtype='int')
        
        try:
            self._nspot_tot = self.attrs['nspot_tot']
        except KeyError:
            self._nspot_tot = 0
            if self.mode != 'r':
                self.attrs.create('nspot_tot', self._nspot_tot, dtype='int')
        
        try:
            self._ntrace_tot = self.attrs['ntrace_tot']
        except KeyError:
            self._ntrace_tot = 0
            if self.mode != 'r':
                self.attrs.create('ntrace_tot', self._ntrace_tot, dtype='int')
        
        try:
            self._nspot_max = self.attrs['nspot_max']
        except KeyError:
            self._nspot_max = 0
            if self.mode != 'r':
                self.attrs.create('nspot_max', self._nspot_max, dtype='int')
        
        try:
            self._ncopy_max = self.attrs['ncopy_max']
        except KeyError:
            self._ncopy_max = 0
            if self.mode != 'r':
                self.attrs.create('ncopy_max', self._ncopy_max, dtype='int')
                
        if self.mode == "r":
            if 'genome' not in self:
                warnings.warn('Read-only CtFile is missing genome')
            if 'index' not in self:
                warnings.warn('Read-only CtFile is missing index')
            if 'cell_labels' not in self:
                warnings.warn('Read-only CtFile is missing cell_labels')
            if 'coordinates' not in self:
                warnings.warn('Read-only CtFile is missing coordinates')
            if 'nspot' not in self:
                warnings.warn('Read-only CtFile is missing nspot')
            if 'ncopy' not in self:
                warnings.warn('Read-only CtFile is missing ncopy')
        
        self._check_consistency()
    
    def _check_consistency(self):
        if 'cell_labels' in self:
            assert self['cell_labels'].shape[0] == self._ncell, 'cell_labels.shape[0] != ncell'
        if 'coordinates' in self:
            assert self['coordinates'].shape[0] == self._ncell, 'coordinates.shape[0] != ncell'
            assert self['coordinates'].shape[1] == self._ndomain, 'coordinates.shape[1] != ndomain'
            assert self['coordinates'].shape[2] == self._ncopy_max, 'coordinates.shape[2] != ncopy_max'
            assert self['coordinates'].shape[3] == self._nspot_max, 'coordinates.shape[3] != nspot_max'
            assert self['coordinates'].shape[4] == 3, 'coordinates.shape[4] != 3'
        if 'intensity' in self:
            assert self['intensity'].shape[0] == self._ncell, 'intensity.shape[0] != ncell'
            assert self['intensity'].shape[1] == self._ndomain, 'intensity.shape[1] != ndomain'
            assert self['intensity'].shape[2] == self._ncopy_max, 'intensity.shape[2] != ncopy_max'
            assert self['intensity'].shape[3] == self._nspot_max, 'intensity.shape[3] != nspot_max'
        if 'nspot' in self:
            assert self['nspot'].shape[0] == self._ncell, 'nspot.shape[0] != ncell'
            assert self['nspot'].shape[1] == self._ndomain, 'nspot.shape[1] != ndomain'
            assert self['nspot'].shape[2] == self._ncopy_max, 'nspot.shape[2] != ncopy_max'
        if 'ncopy' in self:
            assert self['ncopy'].shape[0] == self._ncell, 'ncopy.shape[0] != ncell'
            assert self['ncopy'].shape[1] == self._ndomain, 'ncopy.shape[1] != ndomain'
        pass
    
    def __eq__(self, other):
        eq = self['ncell'] == other['ncell'] and self['ndomain'] == other['ndomain'] \
            and self['nspot_tot'] == other['nspot_tot'] and self['ntrace_tot'] == other['ntrace_tot'] \
                and self['nspot_max'] == other['nspot_max'] and self['ncopy_max'] == other['ncopy_max'] \
                    and np.array_equal(self['cell_labels'], other['cell_labels']) \
                        and np.array_equal(self['coordinates'], other['coordinates']) \
                            and np.array_equal(self['intensity'], other['intensity']) \
                                and np.array_equal(self['nspot'], other['nspot']) \
                                    and np.array_equal(self['ncopy'], other['ncopy'])
        return eq
    
    def get_ncell(self):
        return self._ncell
    
    def get_ndomain(self):
        return self._ndomain
    
    def get_nspot_tot(self):
        return self._nspot_tot
    
    def get_ntrace_tot(self):
        return self._ntrace_tot
    
    def get_nspot_max(self):
        return self._nspot_max
    
    def get_ncopy_max(self):
        return self._ncopy_max
    
    def get_genome(self):
        if self._genome is None:
            # If _genome is None, it initializes it
            # By passing self to the Genome class, Genome will read it from the HDF5 file
            # The info in the HDF5 file (chrstr, length, ...) have been saved in set_genome
            self._genome = Genome(self)
        return self._genome
    
    def get_index(self):
        if self._index is None:
            self._index = Index(self)
        return self._index
    
    def get_cell_labels(self):
        return self['cell_labels'][:].astype('U10')

    def get_coordinates(self):
        # These data are not huge, so we can load them all in memory
        return self['coordinates'][:]
    
    def get_intensity(self):
        return self['intensity'][:]

    def get_nspot(self):
        return self['nspot'][:]
    
    def get_ncopy(self):
        return self['ncopy'][:]
    
    def set_ncell(self, n):
        self.attrs['ncell'] = self._ncell = n
    
    def set_ndomain(self, n):
        self.attrs['ndomain'] = self._ndomain = n
    
    def set_nspot_tot(self, n):
        self.attrs['nspot_tot'] = self._nspot_tot = n
    
    def set_ntrace_tot(self, n):
        self.attrs['ntrace_tot'] = self._ntrace_tot = n
    
    def set_nspot_max(self, n):
        self.attrs['nspot_max'] = self._nspot_max = n
    
    def set_ncopy_max(self, n):
        self.attrs['ncopy_max'] = self._ncopy_max = n
    
    def set_genome(self, genome):
        assert isinstance(genome, Genome)
        self._genome = genome
        genome.save(self)

    def set_index(self, index):
        assert isinstance(index, Index)
        self._index = index
        index.save(self)
    
    def set_cell_labels(self, cell_labels):
        if 'cell_labels' in self:
            del self['cell_labels']
        self.create_dataset('cell_labels', data=cell_labels.astype('S10'), dtype=np.dtype('S10'))
    
    def set_coordinates(self, coordinates):
        if 'coordinates' in self:
            del self['coordinates']
        self.create_dataset('coordinates', data=coordinates)
    
    def set_intensity(self, intensity):
        if 'intensity' in self:
            del self['intensity']
        self.create_dataset('intensity', data=intensity)
    
    def set_nspot(self, nspot):
        if 'nspot' in self:
            del self['nspot']
        self.create_dataset('nspot', data=nspot)
    
    def set_ncopy(self, ncopy):
        if 'ncopy' in self:
            del self['ncopy']
        self.create_dataset('ncopy', data=ncopy)
    
    ncell = property(get_ncell, set_ncell, doc='number of cells (int)')
    ndomain = property(get_ndomain, set_ndomain, doc='number of domains (int)')
    nspot_tot = property(get_nspot_tot, set_nspot_tot, doc='total number of spots (int)')
    ntrace_tot = property(get_ntrace_tot, set_ntrace_tot, doc='total number of traces (int)')
    nspot_max = property(get_nspot_max, set_nspot_max, doc='maximum number of spots (across copies of domains) (int)')
    ncopy_max = property(get_ncopy_max, set_ncopy_max, doc='maximum number of copies (across domains) (int)')
    genome = property(get_genome, set_genome, doc='a alabtools.Genome instance')
    index = property(get_index, set_index, doc='a alabtools.Index instance')
    cell_labels = property(get_cell_labels, set_cell_labels, doc='labels of cell. np.array(ncell, dtype=object)')
    coordinates = property(get_coordinates, set_coordinates, doc='coordinates of spots. np.array(ncell, ndomain, ntrace_max, nspot_max, 3)')
    intensity = property(get_intensity, set_intensity, doc='intensity of spots. np.array(ncell, ndomain, ntrace_max, nspot_max)')
    nspot = property(get_nspot, set_nspot, doc='number of spots. np.array(ncell, ndomain, ntrace_max)')
    ncopy = property(get_ncopy, set_ncopy, doc='number of copies. np.array(ncell, ndomain)')
    
    def get_cellnum(self, cellID):
        findidx = np.flatnonzero(self.cell_labels == cellID)
        if len(findidx) == 0:
            return None
        else:
            return findidx[0]
    
    def get_cellID(self, cellnum):
        assert isinstance(cellnum, (int, np.int32, np.int32))
        return self.cell_labels[cellnum]
    
    def _compute_ntrace_tot(self):
        """Compute the total number of traces (ntrace_tot).
        Uses self.ncopy, self.index.chromstr, self.genome.chroms.
        """
        ntrace_tot = 0
        # I can't count it simply as np.sum(self.ncopy), because the copies of spots
        # of the same trace in the same chromosome count as 1
        # So I have to count the maximum number of copies of a trace in each chromosome,
        # and sum them over all cells and chromosomes
        for chrom in self.genome.chroms:
            # compute number of copies of each domain of the chromosome in each cell
            ncopy_chrom = self.ncopy[:, self.index.chromstr == chrom]  # np.array(ncell, ndomain_chrom)
            # in each cell, find the maximum number of copies across all domains of the chromosome
            ncopy_max_chrom = np.nanmax(ncopy_chrom, axis=1)  # np.array(ncell)
            # sum over all cells to get the total number of traces of the chromosome
            ntrace_tot += np.sum(ncopy_max_chrom)
        return ntrace_tot
    
    @staticmethod
    def _compute_nspot_ncopy(coordinates):
        """Compute the number of spots (nspot) and the number of copies (ncopy)
        from the coordinates of spots."""
        # nonan_map is a boolean array that is True if the coordinates are not nan
        nonan_map = ~np.isnan(coordinates)  # (ncell, ndomain, ncopy_max, nspot_max, 3)
        # remove the last axis by taking the x coordinate
        nonan_map = nonan_map[:, :, :, :, 0]  # (ncell, ndomain, ncopy_max, nspot_max) 
        # now sum over the last axis to get the number of spots in each copy
        nspot = np.sum(nonan_map, axis=3, dtype=np.int32)  # (ncell, ndomain, ncopy_max)
        # remove the last axis from nonan_map again by taking the first spot for each copy
        nonan_map = nonan_map[:, :, :, 0]  # (ncell, ndomain, ncopy_max)
        # now sum over the last axis to get the number of copies in each domain
        ncopy = np.sum(nonan_map, axis=2, dtype=np.int32)  # (ncell, ndomain)
        return nspot, ncopy
    
    def sort_cells(self, order):
        """Orders the CtFile according to the input order.

        Args:
            order (np.array(ncell, dtype=int)): New order of the cells.
                Must be a permutation of the numbers from 0 to ncell-1.
        """
        # assert the input order
        assert len(order) == self.ncell,\
            "The length of the input order must be equal to the number of cells."
        assert np.array_equal(np.sort(order), np.arange(self.ncell)),\
            "The input order must be a permutation of the numbers from 0 to ncell-1."
        # sort the datasets
        self.cell_labels = self.cell_labels[order]
        self.ncopy = self.ncopy[order]
        self.nspot = self.nspot[order]
        self.coordinates = self.coordinates[order]
        if 'intensity' in self:
            self.intensity = self.intensity[order]
    
    def sort_copies(self):
        """Sorts copies of each chromosome in each cell.
        
        It reorders the copies so that all-NaN traces are at the end.
        """
        
        # Initialize mask of NaN values (ndomain wide)
        nan_map_0 = np.isnan(self.coordinates[:, :, :, :, 0])  # ncell, ndomain, ncopy_max, nspot_max
        # LogicAND along the spot axis (True if all spots are NaN)
        nan_map_0 = np.all(nan_map_0, axis=-1)  # ncell, ndomain, ncopy_max
        
        # The previous map gives different True/False for each domain
        # We want to have the same True/False for all domains of the same chromosome in each cell
        nan_map = np.zeros((self.ncell, self.ndomain, self.ncopy_max), dtype=bool)  # ncell, ndomain, ncopy_max
        
        # Loop over chromosomes and fill the mask
        for chrom in self.genome.chroms:
            nan_chrom = np.all(nan_map_0[:, self.index.chromstr==chrom, :], axis=1)  # ncell, ncopy_max
            nan_map[:, self.index.chromstr==chrom, :] = np.repeat(nan_chrom[:, np.newaxis, :],
                                                                  np.sum(self.index.chromstr==chrom),
                                                                  axis=1)  # ncell, ndomain_chrom, ncopy_max
        
        # Sort nspot
        idx_srt = sort_coord_by_boolmask(nan_map, axis=2)
        self.nspot = self.nspot[idx_srt]
        
        # Expand the mask to include the spot axis
        nan_map = np.expand_dims(nan_map, axis=3)  # ncell, ndomain, ncopy_max, 1
        nan_map = np.repeat(nan_map, repeats=self.nspot_max, axis=3)  # ncell, ndomain, ncopy_max, nspot_max
        
        # Sort coordinates
        idx_srt = sort_coord_by_boolmask(nan_map, axis=2)
        coord_srt = np.copy(self.coordinates)
        for i in range(3):
            coord_srt[..., i] = self.coordinates[..., i][idx_srt]
        self.coordinates = coord_srt
        
        # Sort intensity
        if 'intensity' in self:
            self.intensity = self.intensity[idx_srt]
    
    def sort_spots(self):
        """Sorts the spots in each domain/cell.
        NaN spots are put at the end.
        """
        # Initialize NaN map
        nan_map = np.isnan(self.coordinates[..., 0])  # ncell, ndomain, ncopy_max, nspot_max
        # Sort coordinates
        idx = sort_coord_by_boolmask(nan_map, axis=3)
        coord_srt = np.copy(self.coordinates)
        for i in range(3):
            coord_srt[..., i] = self.coordinates[..., i][idx]
        self.coordinates = coord_srt
        # Sort intensity
        if 'intensity' in self:
            self.intensity = self.intensity[idx]
    
    def trim(self):
        """
        Trims the data to remove redundant copies and spots.
        This operation might be required if the data have an excessive max-padding.
        A copy label c is reduntant if:
            self.coordinates[:, :, c, :, :] is all NaN
        A spot label s is redundant if:
            self.coordinates[:, :, :, s, :] is all NaN
        ATTENTION: this method works if the data data are organized so that
        the reduntant copies and spots are at the end of the array.
        Use sort_copies() and sort_spots() to ensure this.
        """
        # Defines the map of NaN values from self.coordinates
        nan_map = np.isnan(self.coordinates)  # np.array(ncell, ndomain, ncopy_max, nspot_max, 3)
        # Trim copies
        while True:
            if not np.all(nan_map[:, :, self.ncopy_max-1, :, :]):
                # If the copy is not redundant, we can exit the loop
                break
            sys.stdout.write('Trimming copy {}...\n'.format(self.ncopy_max-1))
            self.nspot = self.nspot[:, :, :self.ncopy_max-1]
            self.coordinates = self.coordinates[:, :, :self.ncopy_max-1, :, :]
            if 'intensity' in self:
                self.intensity = self.intensity[:, :, :self.ncopy_max-1, :, :]
            self.ncopy_max = self.ncopy_max-1
            nan_map = np.isnan(self.coordinates)
        # Trim spots
        while True:
            if not np.all(nan_map[:, :, :, self.nspot_max-1, :]):
                break
            sys.stdout.write('Trimming spot {}...\n'.format(self.nspot_max-1))
            self.coordinates = self.coordinates[:, :, :, :self.nspot_max-1, :]
            if 'intensity' in self:
                self.intensity = self.intensity[:, :, :, :self.nspot_max-1]
            self.nspot_max = self.nspot_max-1
            nan_map = np.isnan(self.coordinates)
    
    def pop_cells(self, indices):
        """Removes the cells with the input indices.
        Args:
            indices (np.array(n, dtype=int)): Indices of the cells to remove.
                Must be a subset of the numbers from 0 to ncell-1.
        """
        # assert the input indices
        assert len(indices) <= self.ncell,\
            "The length of the input indices must be less than or equal to the number of cells."
        for i in indices:
            assert i in np.arange(self.ncell),\
                "The input indices must be a subset of the numbers from 0 to ncell-1."
        # remove the cells from the datasets
        self.cell_labels = np.delete(self.cell_labels, indices)
        self.ncopy = np.delete(self.ncopy, indices, axis=0)
        self.nspot = np.delete(self.nspot, indices, axis=0)
        self.coordinates = np.delete(self.coordinates, indices, axis=0)
        # update attributes
        self.ncell -= len(indices)
        self.nspot_tot = np.sum(self.nspot)
        self.ntrace_tot = self._compute_ntrace_tot()

    def merge(self, other, new_name, tag1=None, tag2=None):
        """
        Merge two CtFile instances.
        Creates a new CtFile instance with the data from both CtFile instances.
        
        This function is for merging datasets with no overlap, e.g. replicates.
        
        @param other: a CtFile instance
        @param new_name: name of the new CtFile instance
        @param tag1: tag to distinguish cell labels from self
        @param tag2: tag to distinguish cell labels from other
        @return new_ct: a CtFile instance
        """
        if self.genome != other.genome:
            raise ValueError('Cannot merge CtFile instances with different genomes.')
        if self.index != other.index:
            raise ValueError('Cannot merge CtFile instances with different indexes.')
        if len(np.intersect1d(self.cell_labels, other.cell_labels)) != 0 \
            and tag1 is None and tag2 is None:
            raise ValueError('There is an overlap in cell labels. Please provide tags to distinguish them.')
        
        new_ct = CtFile(new_name, mode='w')
        
        new_ct.set_genome(self.genome)
        new_ct.set_index(self.index)
        
        new_ct.set_ncell(self.ncell + other.ncell)
        new_ct.set_ndomain(self.ndomain)
        new_ct.set_nspot_tot(self.nspot_tot + other.nspot_tot)
        new_ct.set_ntrace_tot(self.ntrace_tot + other.ntrace_tot)
        new_ct.set_nspot_max(int(np.max([self.nspot_max, other.nspot_max])))
        new_ct.set_ncopy_max(int(np.max([self.ncopy_max, other.ncopy_max])))
        
        new_cell_labels = []
        if tag1 is None:
            tag1 = ''
        if tag2 is None:
            tag2 = ''
        new_cell_labels.extend([cellID + tag1 for cellID in self.cell_labels])
        new_cell_labels.extend([cellID + tag2 for cellID in other.cell_labels])
        new_cell_labels = np.array(new_cell_labels)
        new_ct.set_cell_labels(new_cell_labels)
        
        new_coord = np.nan * np.zeros((new_ct.ncell, new_ct.ndomain, new_ct.ncopy_max, new_ct.nspot_max, 3))
        new_coord[:self.ncell, :, :self.ncopy_max, :self.nspot_max, :] = self.coordinates
        new_coord[self.ncell:, :, :other.ncopy_max, :other.nspot_max, :] = other.coordinates
        new_ct.set_coordinates(new_coord)
        
        if 'intensity' in self and 'intensity' in other:
            new_intensity = np.nan * np.zeros((new_ct.ncell, new_ct.ndomain, new_ct.ncopy_max, new_ct.nspot_max))
            new_intensity[:self.ncell, :, :self.ncopy_max, :self.nspot_max] = self.intensity
            new_intensity[self.ncell:, :, :other.ncopy_max, :other.nspot_max] = other.intensity
            new_ct.set_intensity(new_intensity)
        
        new_nspot = np.nan * np.zeros((new_ct.ncell, new_ct.ndomain, new_ct.ncopy_max))
        new_nspot[:self.ncell, :, :self.ncopy_max] = self.nspot
        new_nspot[self.ncell:, :, :other.ncopy_max] = other.nspot
        new_ct.set_nspot(new_nspot)
        
        new_ncopy = np.concatenate((self.ncopy, other.ncopy), axis=0)
        new_ct.set_ncopy(new_ncopy)
        
        return new_ct
    
    def set_manually(self, coordinates, genome, index, cell_labels=None, intensity=None):
        """Set the CtFile attributes by manually inputting the data.
           ncopy and nspot are inferred from the coordinates array.

        Args:
            coordinates (np.array(ncell, ndomain, ncopy_max, nspot_max, 3), np.float32)
            genome (Genome)
            index (Index)
            cell_labels (np.array(ncell, dtype='U10'), optional):
                If None, set as np.arange(ncell).astype('U10'). Defaults to None.

        Returns:
            None
        """        
        # set the genome / index
        assert isinstance(genome, Genome), 'genome must be a Genome instance'
        self.set_genome(genome)
        assert isinstance(index, Index), 'index must be a Index instance'
        assert index.genome == self.genome, 'genome in index must match input genome'
        self.set_index(index)
        # set the coordinates
        assert len(coordinates.shape) == 5, 'coordinates must be a 5D array'
        assert coordinates.shape[1] == len(index),\
            'coordinates must have the same number of domains as the index'
        assert coordinates.shape[4] == 3, 'spatial coordinates must be 3D'
        try:
            self.set_coordinates(coordinates.astype(np.float32))
        except ValueError:
            "Coordinates must be numeric."
        # set the intensity
        if intensity is not None:
            assert len(intensity.shape) == 4, 'intensity must be a 4D array'
            assert intensity.shape == coordinates.shape[:-1],\
                'intensity must have the same shape as coordinates'
            try:
                self.set_intensity(intensity.astype(np.float32))
            except ValueError:
                "Intensity must be numeric."
        # set the cell labels
        if cell_labels is None:
            cell_labels = np.arange(coordinates.shape[0]).astype('U10')
        assert len(cell_labels) == coordinates.shape[0],\
            'cell_labels must have the same length as coordinates.shape[0]'
        assert cell_labels.dtype == 'U10', 'cell_labels must be of dtype U10'
        self.set_cell_labels(cell_labels)
        # set the attribues from coordinates.shape
        self.set_ncell(coordinates.shape[0])
        self.set_ndomain(coordinates.shape[1])
        self.set_ncopy_max(coordinates.shape[2])
        self.set_nspot_max(coordinates.shape[3])
        # set nspot and ncopy
        nspot, ncopy = self._compute_nspot_ncopy(coordinates)
        self.set_nspot(nspot)
        self.set_ncopy(ncopy)
        # set nspot_tot and ntrace_tot
        self.set_nspot_tot(np.sum(nspot))
        self.set_ntrace_tot(self._compute_ntrace_tot())
    
    
    def set_from_fofct(self, fofct_file, in_assembly=None):
        """
        Set the data from a FOF-CT file.
        
        @param fofct_file: path to the FOF-CT file
        @param in_assembly: assembly of the FOF-CT file. If None, it will be inferred from the file name.
        """
        
        if self.mode == 'r':
            raise IOError('Cannot set data from FOF-CT file. File is read-only.')
        
        genome, index, cell_labels, coordinates, intensity, ncell, ndomain, ncopy_max, nspot_max = fofct_process(fofct_file,
                                                                                                                 in_assembly=in_assembly)
        
        self.set_ndomain(ndomain)
        self.set_genome(genome)
        self.set_index(index)
        self.set_ncell(ncell)
        self.set_nspot_max(nspot_max)
        self.set_ncopy_max(ncopy_max)
        self.set_cell_labels(cell_labels)
        self.set_coordinates(coordinates)
        if not np.all(np.isnan(intensity)):
            self.set_intensity(intensity)
        nspot, ncopy = self._compute_nspot_ncopy(coordinates)
        self.set_nspot(nspot)
        self.set_ncopy(ncopy)
        self.set_nspot_tot(np.sum(self.nspot))
        self.set_ntrace_tot(self._compute_ntrace_tot())
