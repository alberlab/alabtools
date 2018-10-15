# Copyright (C) 2017 University of Southern California and
#                    Guido Polles, Nan Hua
#
# Authors: Guido Polles, Nan Hua
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

__author__  = "Guido Polles, Nan Hua"

__license__ = "GPL"
__version__ = "0.0.3"
__email__   = "polles@usc.edu"

import numpy as np
import warnings
import os, errno
from .api import Contactmatrix
from .plots import plot_comparison, plotmatrix
import h5py
from .utils import unicode, Genome, Index, COORD_DTYPE, RADII_DTYPE
import matplotlib.pyplot as plt
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
            if self.mode != 'r':
                self.attrs.create('version', __hss_version__, dtype='int32')
        try:
            self._violation = self.attrs['violation']
        except(KeyError):
            self._violation = np.nan
            if self.mode != 'r':
                self.attrs.create('violation', np.nan, dtype='float')
        try:
            self._nbead = self.attrs['nbead']
        except(KeyError):
            self._nbead = 0
            if self.mode != 'r':
                self.attrs.create('nbead', 0, dtype='int32')
        try:
            self._nstruct = self.attrs['nstruct']
        except(KeyError):
            self._nstruct = 0
            if self.mode != 'r':
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
        if 'structmajorcrd' in self:
            return self['structmajorcrd'][key][()]
        else:
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

    def has_struct_major(self):
        return 'structmajorcrd' in self

    def transpose_coords(self, max_items=int(1e8)):
        '''
        Transposes data for fast visualization retrieval
        '''
        from .utils import block_transpose
        if not self.has_struct_major():
            self.create_dataset('structmajorcrd', shape=(self.nstruct, self.nbead, 3), dtype=COORD_DTYPE)
        block_transpose(self['coordinates'], self['structmajorcrd'], max_items)

    coordinates = property(get_coordinates, set_coordinates)
    radii = property(get_radii, set_radii)
    index = property(get_index, set_index, doc='a alabtools.Index instance')
    genome = property(get_genome, set_genome,
                      doc='a alabtools.Genome instance')
    nbead = property(get_nbead, set_nbead)
    nstruct = property(get_nstruct, set_nstruct)
    violation = property(get_violation, set_violation)


    def getBeadRadialPositions(self, beads, nucleusRadius=5000.0):
        allrp = []

        if isinstance(nucleusRadius, tuple):
            if len(nucleusRadius) != 3:
                raise ValueError("Please provide 3 axis for radius")
        
        for i in np.array(beads):
            beadcrd = self.get_bead_crd(i) / nucleusRadius
            rp = np.linalg.norm(beadcrd, axis=1)
            
            allrp.append(rp)
    
        return np.array(allrp)

    def getChromBeadRadialPosition(self, chrnum, nucleusRadius=5000.0):
        rps = []
        for c in self.index.get_chrom_copies(chrnum):
            beads = self.index.get_chrom_pos(chrnum, copy=c)
            rps.append(self.getBeadRadialPositions(self, beads, nucleusRadius))
        rps = np.column_stack(rps)
        #print(rps.shape)
        return rps

    def savePDBx(self, output_file, max_struct=None, entry_id="Model", title="Model Population",
                 software=None,
                 citation=None,
                 citation_author=None):
        from .ihm import ihmWriter, DCDWriter

        #deal with file paths
        #will save the file as output file location/file_prefix/file
        #coordinates will be saved into subdirectory `data`
        numModel = self.nstruct
        if max_struct:
            numModel = int(min(numModel, max_struct))

        mainfile = os.path.abspath(output_file)
        path, filename = os.path.split(mainfile)
        prefix, ext = os.path.splitext(filename)

        if ext != ".ihm":
            filename += ".ihm"

        path = os.path.join(path, prefix)
        directory = os.path.dirname(os.path.join(path, "data/"))
        if not os.path.exists(directory):
            os.makedirs(directory)

        mainfile = os.path.join(path, filename)
        coordinate_filename = os.path.join("data", prefix+"_coordinates.dcd")

        ihm = ihmWriter(mainfile,[["_entry.id", entry_id],
                                  ["_struct.entry_id", entry_id],
                                  ["_struct.title", title]
                                 ])

        if software:
            ihm.write_software(software)
        if citation:
            ihm.write_citation(citation)
        if citation_author:
            ihm.write_citation_auther(citation_author)

        if 'genome' not in self:
            raise RuntimeError('HssFile is missing genome')
        if 'index' not in self:
            raise RuntimeError('HssFile is missing index')
        genome = self.genome
        index  = self.index

        #---write_entity
        #---write_struct_asym
        entity_data = []
        struct_asym_data = []

        for i in range(len(genome)):
            chrom = genome[i]
            detail = "{}_{}".format(unicode(genome.assembly), chrom)
            copynum = index.copy[np.flatnonzero(index.chrom == i)[-1]]+1
            entity_data.append([i+1, "polymer", "man", chrom, '?', copynum, detail])
            for j in range(copynum):
                struct_asym_data.append(["{}{}".format(chrom[3:], chr(65+j)), i+1, detail])
            #-
        #-
        ihm.write_entity(entity_data)
        ihm.write_struct_asym(struct_asym_data)



        #---write_chem_comp
        chem_comp_data = []

        for i in index.copy_index.keys():
            chrom = genome[index.chrom[i]]
            start = index.start[i]
            end = index.end[i]
            chem_comp_data.append(["Seg-{}".format(i+1), "{}:{}-{}".format(chrom, start, end), "other"])
        #-
        ihm.write_chem_comp(chem_comp_data)

        #---write_entity_poly
        #---write_entity_poly_seq

        entity_poly_data = []
        entity_poly_seq_data = []

        for i in range(len(genome)):
            entity_poly_data.append([i+1, "other", "no", "no", "no", "."])
            for j in range(index.offset[i+1]-index.offset[i]):
                entity_poly_seq_data.append([i+1, j+1, chem_comp_data[index.offset[i]+j][0], "."])
            #-
        #-
        ihm.write_entity_poly(entity_poly_data)
        ihm.write_entity_poly_seq(entity_poly_seq_data)


        #===write_ihm_struct_assembly
        #===write_ihm_model_representation
        ihm_struct_assembly_data = []
        ihm_model_representation_data = []
        k = 0
        for i in range(len(genome)):
            chrom = genome[i]
            copynum = index.copy[np.flatnonzero(index.chrom == i)[-1]]+1
            length = index.offset[i+1] - index.offset[i]
            for j in range(copynum):
                k += 1
                asym = "{}{}".format(chrom[3:], chr(65+j))
                ihm_struct_assembly_data.append([k, 1, chrom, i+1, asym, 1, length])
                ihm_model_representation_data.append([k,k,k, i+1, chrom, asym, 1, length, "sphere", ".", "flexible", "by-feature", 1 ])
        ihm.write_ihm_struct_assembly(ihm_struct_assembly_data)
        ihm.write_ihm_model_representation(ihm_model_representation_data)


        #===write_ihm_dataset_list
        ihm.write_ihm_dataset_list([[1, "in situ Hi-C", "no"]])

        #===write_ihm_dataset_group
        ihm.write_ihm_dataset_group([[1,1,1]])

        #===write_ihm_cross_link_list
        #===write_ihm_cross_link_restraint

        #+++write_ihm_modeling_protocol
        ihm.write_ihm_modeling_protocol([[1,1,1,1,1,".","PGS","A/M", "A/M", 10000, 10000, "no", "yes", "no"]])

        #---write_ihm_external_files
        ihm.write_ihm_external_files([[1,1,coordinate_filename,"DCD","Model ensemble coordinates",".", "DCD file"]])
        #---write_ihm_ensemble_info
        ihm.write_ihm_ensemble_info([[1, "Model_Population", ".", 1, ".", ".", numModel, numModel, ".", 1]])

        #===write_ihm_model_list
        #===write_ihm_sphere_obj_site
        radii = self.radii
        print("Loading coordinates into memory..")
        coords = self.coordinates
        numSample = 10
        ihm_model_list_data = []
        ihm_sphere_obj_site_data = []
        for s in range(numSample):
            struct_name = "Structure {}".format(s+1)
            ihm_model_list_data.append([s+1, s+1, 1, struct_name, "Sample_Population", 1, 1])
            xyz = coords[:, s, :]
            for i in range(len(index)):
                chromNum = index.chrom[i]
                chrom = genome[chromNum]
                asym = "{}{}".format(chrom[3:], chr(65+index.copy[i]))
                x, y, z = xyz[i]
                ihm_sphere_obj_site_data.append([len(index)*s+i+1, chromNum+1, i+1, i+1, asym, x,y,z, radii[i], ".", s+1])
            #-
        #-

        ihm.write_ihm_model_list(ihm_model_list_data)
        ihm.write_ihm_sphere_obj_site(ihm_sphere_obj_site_data)

        ihm.close()
        print("Writing DCD coordinates..")
        dcdfh = open(os.path.join(path, coordinate_filename), 'wb')
        dcdwriter = DCDWriter(dcdfh)

        for i in range(numModel):
            xyz = coords[:, i, :]
            dcdwriter._write_frame(xyz[:,0],xyz[:,1],xyz[:,2])

        dcdfh.close()

#================================================

def get_simulated_hic(hssfname, contactRange=2):
    '''
    Computes the contact map and sums the copies. Returns a Contactmap instance
    '''
    with HssFile(hssfname, 'r') as f:
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

    v0 = min(0.1, np.max(np.abs(d)))
    plotmatrix(dname + '/difference_matrix.pdf', d, cmap='coolwarm', vmin=-v0, vmax=v0)

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
