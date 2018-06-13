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
import matplotlib.pyplot as plt
import warnings
import os, errno
from .api import Contactmatrix
from .plots import plot_comparison, plotmatrix
import h5py
from .utils import Genome, Index, COORD_DTYPE, RADII_DTYPE
import matplotlib.colors as colors


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
            self._violation = np.nan
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

    def buildContactMap(self, contactRange = 2):
        from ._cmtools import BuildContactMap_func
        from .api import Contactmatrix
        from .matrix import sss_matrix
        mat = Contactmatrix(None, genome=None, resolution=None)
        mat.genome = self.genome
        mat.index = self.index
        DimB = len(self.radii)
        Bi = np.empty(int(DimB*(DimB+1)/2),dtype=np.int32)
        Bj = np.empty(int(DimB*(DimB+1)/2),dtype=np.int32)
        Bx = np.empty(int(DimB*(DimB+1)/2),dtype=np.float32)

        crd = self.coordinates
        if crd.flags['C_CONTIGUOUS'] is not True:
            crd = crd.copy(order='C')
        BuildContactMap_func(crd, self.radii, contactRange, Bi, Bj, Bx)

        mat.matrix = sss_matrix((Bx, (Bi, Bj)))
        mat.resolution = np.nan
        return mat

    coordinates = property(get_coordinates, set_coordinates)
    radii = property(get_radii, set_radii)
    index = property(get_index, set_index, doc='a alabtools.Index instance')
    genome = property(get_genome, set_genome,
                      doc='a alabtools.Genome instance')
    nbead = property(get_nbead, set_nbead)
    nstruct = property(get_nstruct, set_nstruct)
    violation = property(get_violation, set_violation)

#================================================

def get_simulated_hic(hssfname, contactRange=2):
    with HssFile(hssfname) as f:
        full_cmap = f.buildContactMap(contactRange)

    return full_cmap.sumCopies()

def compare_hic_maps(M, ref, fname='matrix'):
    dname = fname + '_hic_comparison'
    M = Contactmatrix(M)
    ref = Contactmatrix(ref)
    idx1 = M.index
    idx2 = ref.index
    if len(idx1) != len(idx2):
        warnings.warn(
            'Cannot compare maps. Their indices have different lengths'
            ' (%d, %d)' % (len(idx1) != len(idx2)) )
        return False
    try:
        os.makedirs(dname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    chroms = np.unique(idx1.chrom)
    for ci in chroms:
        cn  = idx1.genome.getchrom(ci)
        plot_comparison(M, ref,
                        chromosome=cn,
                        file=dname + '/{}.pdf'.format(cn))

    # plot the histogram of matrix differences

    dm1 = M.matrix.toarray()
    dm2 = ref.matrix.toarray()
    d = dm1 - dm2
    dmmax = np.maximum(dm1, dm2)
    dave, dstd, dmax = np.average(d), np.std(d), np.max(np.abs(d))
    print ('HiC maps difference\n'
           '-------------------\n'
           'max: %f ave: %f std: %f' % (dmax, dave, dstd) )

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    nnz = np.count_nonzero(d)
    nnzfrac = nnz / d.size


    plt.sca(ax[0])
    plt.hist(d[d != 0], bins=100)
    plt.xlabel('differences')
    plt.ylabel('count (nonzero fraction: %f)' % nnzfrac)

    plt.sca(ax[1])
    w = np.logical_and(
        dmmax != 0,
        dm2 != 0
    )
    plt.hist(d[w]/dmmax[w], bins=100, log=True)
    plt.xlabel('Normalized differences')
    plt.ylabel('count (nonzero fraction: %f)' % nnzfrac)

    plt.tight_layout()
    plt.savefig(dname + '/difference_histo.pdf')
    plt.close(fig)

    plotmatrix(dname + '/difference_matrix.pdf', d)

    # 2d hist to have a "scatter plot"
    fig = plt.figure()
    nrm = colors.LogNorm(vmin=1)
    plt.hist2d(dm1.ravel(), dm2.ravel(), bins=100, norm=nrm)
    plt.colorbar()
    plt.title('2D histogram of matrix probabilities')
    plt.xlabel('probability (M)')
    plt.ylabel('probability (ref)')
    plt.tight_layout()
    plt.savefig(dname + '/matrix_values_hist2d.pdf')
    plt.close()




def common_analysis(hssfname, **kwargs):

    if kwargs.get('contacts', True):
        with HssFile(hssfname) as f:
            cmap = f.buildContactMap(2)
        cmap.save(hssfname + 'full_cmap.hcs')
        cmap.plot(hssfname + 'full_cmap.png')
        cmap = cmap.sumCopies()
        cmap.save(hssfname + 'cmap.hcs')
        cmap.plot(hssfname + 'cmap.png')


    if kwargs.get('hic_compare', False):
        compare_hic_maps(cmap, kwargs.get('hic_compare'), hssfname)
