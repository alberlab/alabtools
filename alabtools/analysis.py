# Copyright (C) 2017 University of Southern California and
#                          Nan Hua
# 
# Authors: Nan Hua
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division, print_function

__author__  = "Guido Polles"

__license__ = "GPL"
__version__ = "0.0.3"
__email__   = "polles@usc.edu"

import numpy as np
import warnings
import h5py
from .utils import Genome, Index, COORD_DTYPE, RADII_DTYPE

__hss_version__ = 2.0
#COORD_CHUNKSIZE = (100, 100, 3)

class HssFile(h5py.File):

    '''
    h5py.File like object for .hss population files.
    Directly inherit from h5py.File, and keeps in memory only version, 
    number of beads and number of structures. Methods are provided to 
    get/set data.

    Attributes
    ----------
        version : int
            File version
        nstruct : int 
            Number of structures
        nbead : int
            Number of beads in each structure
    '''

    def __init__(self, *args, **kwargs):

        '''
        See h5py.File constructor.
        '''

        h5py.File.__init__(self, *args, **kwargs)
        try:
            self._version = self.attrs['version']
        except(KeyError):
            self._version = __hss_version__
            self.attrs.create('version', __hss_version__, dtype='int32')
        try:
            self._violation = self.attrs['violation']
        except(KeyError):
            self._violatoin = np.nan
            self.attrs.create('violation', np.nan, dtype='float')
        try:
            self._nbead = self.attrs['nbead']
        except(KeyError):
            self._nbead = 0
            self.attrs.create('nbead', 0, dtype='int32')
        try:
            self._nstruct = self.attrs['nstruct']
        except(KeyError):
            self._nstruct = 0
            self.attrs.create('nstruct', 0, dtype='int32')

        if self.mode == "r":
            if 'genome' not in self:
                warnings.warn('Read-only HssFile is missing genome')
            if 'index' not in self:
                warnings.warn('Read-only HssFile is missing index')
            if 'coordinates' not in self:
                warnings.warn('Read-only HssFile is missing coordinates')
            if 'radii' not in self:
                warnings.warn('Read-only HssFile is missing radii')

        self._check_consistency()
        
    def _assert_warn(self, expr, msg):
        if not expr:
            warnings.warn('Hss consistency warning: ' + msg, RuntimeWarning)

    def _check_consistency(self):
        n_bead  = self.attrs['nbead']
        
        if 'coordinates' in self:
            self._assert_warn(self.attrs['nstruct'] == self['coordinates'].shape[1],
                              'nstruct != coordinates length')
            self._assert_warn(n_bead == self['coordinates'].shape[0],
                              'nbead != coordinates second axis size')
        if 'index' in self:
            self._assert_warn(n_bead == self['index/chrom'].len(),
                              'nbead != index.chrom length')
            self._assert_warn(n_bead == self['index/copy'].len(),
                              'nbead != index.copy length')
            self._assert_warn(n_bead == self['index/start'].len(),
                              'nbead != index.start length')
            self._assert_warn(n_bead == self['index/end'].len(),
                              'nbead != index.end length')
            self._assert_warn(n_bead == self['index/label'].len(),
                              'nbead != index.label length')
        
        if 'radii' in self:
            self._assert_warn(n_bead == self['radii'].len(),
                              'nbead != radii length')

    def get_version(self):
        return self._version

    def get_nbead(self):
        return self._nbead

    def get_nstruct(self):
        return self._nstruct

    def get_genome(self):
        '''
        Returns
        -------
            alabtools.Genome
        '''
        return Genome(self)

    def get_index(self):
        '''
        Returns
        -------
            alabtools.Index
        '''
        return Index(self)

    def get_coordinates(self, read_to_memory=True):

        '''
        Parameters
        ----------
        read_to_memory (bool) :
            If True (default), the coordinates will be read and returned
            as a numpy.ndarray. If False, a h5py dataset object will be
            returned. In the latter case, note that the datased is valid 
            only while the file is open.
        '''
        
        if read_to_memory:
            return self['coordinates'][:]
        return self['coordinates']

    def get_bead_crd(self, key):
        return self['coordinates'][key][()]
    
    def get_violation(self):
        return self._violation
    
    def get_struct_crd(self, key):
        return self['coordinates'][:, key, :][()]

    def set_bead_crd(self, key, crd):
        self['coordinates'][key, :, :] = crd

    def set_struct_crd(self, key, crd):
        self['coordinates'][:, key, :] = crd

    def get_radii(self):
        return self['radii'][:]

    def set_nbead(self, n):
        self.attrs['nbead'] = self._nbead = n

    def set_nstruct(self, n):
        self.attrs['nstruct'] = self._nstruct = n
    
    def set_violation(self, v):
        self.attrs['violation'] = self._violation = v

    def set_genome(self, genome):
        assert isinstance(genome, Genome)
        genome.save(self)

    def set_index(self, index):
        assert isinstance(index, Index)
        index.save(self)

    def set_coordinates(self, coord):
        assert isinstance(coord, np.ndarray)
        if (len(coord.shape) != 3) or (coord.shape[2] != 3):
            raise ValueError('Coordinates should have dimensions ' 
                             '(nbeads x struct x 3), '
                             'got %s' % repr(coord.shape))
        if self._nstruct != 0 and self._nstruct != coord.shape[1]:
            raise ValueError('Coord first axis does not match number of '
                             'structures')
        if self._nbead != 0 and self._nbead != coord.shape[0]:
            raise ValueError('Coord second axis does not match number of '
                             'beads')

        if 'coordinates' in self:
            self['coordinates'][...] = coord
        else:
            #chunksz = list(COORD_CHUNKSIZE)
            #for i in range(2):
            #    if coord.shape[i] < COORD_CHUNKSIZE[i]:
            #        chunksz[i] = coord.shape[i]
            self.create_dataset('coordinates', data=coord, dtype=COORD_DTYPE)
        self.attrs['nstruct'] = self._nstruct = coord.shape[1]
        self.attrs['nbead'] = self._nbead = coord.shape[0]
        
    def set_radii(self, radii):
        assert isinstance(radii, np.ndarray)
        if len(radii.shape) != 1:
            raise ValueError('radii should be a one dimensional array')
        if self._nbead != 0 and self._nbead != len(radii):
            raise ValueError('Length of radii does not match number of beads')
        
        if 'radii' in self:
            self['radii'][...] = radii
        else:
            self.create_dataset('radii', data=radii, dtype=RADII_DTYPE, 
                                chunks=True, compression="gzip")
        self.attrs['nbead'] = self._nbead = radii.shape[0]

    coordinates = property(get_coordinates, set_coordinates)
    radii = property(get_radii, set_radii)
    index = property(get_index, set_index, doc='a alabtools.Index instance')
    genome = property(get_genome, set_genome, 
                      doc='a alabtools.Genome instance')
    nbead = property(get_nbead, set_nbead)
    nstruct = property(get_nstruct, set_nstruct)
    violation = property(get_violation, set_violation)
    
#================================================


