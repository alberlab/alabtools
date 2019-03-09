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
import numpy as np
import warnings
import os
import sys
import errno
from .api import Contactmatrix
from .plots import plot_comparison, plotmatrix
import h5py
from .utils import unicode, Genome, Index, COORD_DTYPE, RADII_DTYPE, CatmullRomSpline
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tempfile import mkdtemp, mktemp
import scipy.io
import scipy.sparse
import shutil
import colorsys
from subprocess import Popen, PIPE, STDOUT

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(a, *args, **kwargs):
        return a

__author__  = "Guido Polles, Nan Hua"

__license__ = "GPL"
__version__ = "0.0.3"
__email__   = "polles@usc.edu"

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

        self._index = None
        self._genome = None

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
        if self._genome is None:
            self._genome = Genome(self)
        return self._genome

    def get_index(self):
        '''
        Returns
        -------
            alabtools.Index
        '''
        if self._index is None:
            self._index = Index(self)
        return self._index

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
        self._genome = genome
        genome.save(self)

    def set_index(self, index):
        assert isinstance(index, Index)
        self._index = index
        index.save(self)

    def create_coordinates(self, shape=None, chunks=None, **kwargs):
        assert('coordinates' not in self)
        if shape is None:
            assert(self.nbead != 0 and self.nstruct != 0)
            shape = (self.nbead, self.nstruct, 3)
        return self.create_dataset('coordinates', shape=shape, dtype=COORD_DTYPE, chunks=chunks, **kwargs)

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

    @staticmethod
    def _get_number_of_contacts(ci, cj, cutoff):
        csq = cutoff*cutoff
        d2 = np.square(ci - cj).sum(axis=1)
        return np.count_nonzero(d2 <= csq)

    @staticmethod
    def _get_contact_submatrix(cis, cjs, cutoff, dtype=np.float32):
        return np.array([
            [HssFile._get_number_of_contacts(xi, xj, cutoff) for xj in cjs]
            for xi in cis
        ], dtype=dtype)

    def _ipyparallel_get_contact_probability_map(self, cutoff, absolute_cutoff=False, client=None, tmpdir=None):

        if client is None:
            from ipyparallel import Client
            client = Client()

        lbv = client.load_balanced_view()
        n_workers = len(client)
        if n_workers == 0:
            raise RuntimeError('No workers available')

        # ~4 items per worker
        n = self.nbead
        step = max(n // n_workers // 2, 1)

        # prepare directory and write current data, to be sure
        tdir = mkdtemp(prefix=tmpdir)
        try:
            for i in range(0, self.nbead, step):
                np.save(f'{tdir}/{i}.npy', [self.get_bead_crd(j) for j in range(i, min(i+step, self.nbead))])

            def task(tdir, i, j, cutoff):
                crdi = np.load(f'{tdir}/{i}.npy')
                crdj = np.load(f'{tdir}/{j}.npy')
                sm = HssFile._get_contact_submatrix(crdi, crdj, cutoff)
                sm = scipy.sparse.csr_matrix(sm)
                scipy.io.mmwrite(f'{tdir}/{i}_{j}.mtx', sm)

            ii = []
            jj = []
            cutoffs = []
            r = self.radii
            for i in range(0, self.nbead, step):
                for j in range(i, self.nbead, step):
                    ii.append(i)
                    jj.append(j)
                    if absolute_cutoff:
                        cutoffs.append(cutoff)
                    else:
                        cutoffs.append(cutoff * (r[i] + r[j]))

            n_a = len(ii)

            ar = lbv.map_async(task, [tdir] * n_a, ii, jj, cutoffs)

            n = len(list(range(0, self.nbead, step)))
            outs = [[None for _ in range(n)] for _ in range(n)]
            last = -1
            for i, j, r in tqdm(zip(ii, jj, ar)):
                if i != last:
                    outs[i // step][j // step] = scipy.io.mmread(f'{tdir}/{i}_{j}.mtx') / self.nstruct
                    if i == j:
                        outs[i // step][j // step] = scipy.sparse.triu(outs[i // step][j // step])

        finally:
            shutil.rmtree(tdir)

        idx_or_resolution = self.index.resolution()
        if idx_or_resolution is None:
            idx_or_resolution = self.index

        return Contactmatrix(scipy.sparse.bmat(outs),
                             genome=self.genome,
                             usechr=self.genome.chroms,
                             resolution=idx_or_resolution)

    def buildContactMap(self, contactRange = 2, use_ipyparallel=False, absolute_range=False, client=None):
        if absolute_range and not use_ipyparallel:
            raise NotImplementedError()
        if use_ipyparallel:
            return self._ipyparallel_get_contact_probability_map(contactRange,
                                                                 absolute_cutoff=absolute_range,
                                                                 client=client)
        from ._cmtools import BuildContactMap_func
        from .api import Contactmatrix
        from .matrix import sss_matrix
        idx_or_resolution = self.index.resolution()
        if idx_or_resolution is None:
            idx_or_resolution = self.index
        mat = Contactmatrix(None, genome=self.genome, resolution=idx_or_resolution)
        DimB = len(self.radii)
        Bi = np.empty(int(DimB*(DimB+1)/2), dtype=np.int32)
        Bj = np.empty(int(DimB*(DimB+1)/2), dtype=np.int32)
        Bx = np.empty(int(DimB*(DimB+1)/2), dtype=np.float32)

        crd = self.coordinates
        if crd.flags['C_CONTIGUOUS'] is not True:
            crd = crd.copy(order='C')
        BuildContactMap_func(crd, self.radii, contactRange, Bi, Bj, Bx)

        mat.matrix = sss_matrix((Bx, (Bi, Bj)))
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
            rps.append(self.getBeadRadialPositions(beads, nucleusRadius))
        rps = np.column_stack(rps)
        #print(rps.shape)
        return rps

    def resampleCoordinates(self, new_index, output_file):

        with HssFile(output_file, 'w') as hss:
            hss.index = new_index
            hss.nbead = len(new_index)
            hss.nstruct = self.nstruct
            hss.genome = self.genome
            hss.create_coordinates()
            for sid in tqdm(range(self.nstruct)):
                hss.set_struct_crd(
                    sid,
                    resample_coordinates(self.get_struct_crd(sid), self.index, new_index)
                )

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

    @staticmethod
    def _parse_range_string(s, vmax=-1):
        items = s.split(',')
        out = list()
        for it in items:
            if it.strip() == '':
                continue
            if ':' in it:  # slice style
                slice_items = it.split(':')
                if slice_items[0].strip() == '':
                    slice_items[0] = 0
                if slice_items[1].strip() == '':
                    slice_items[1] = vmax
                if len(slice_items) == 2:
                    slice_items.append(1)
                elif len(slice_items) == 3:
                    if slice_items[2].strip() == '':
                        slice_items[2] = 1
                else:
                    raise ValueError('Invalid slice %s' % it)
                slice_items = [int(x) for x in slice_items]
                out += list(range(*slice_items))
            elif '-' in it:
                rngs = it.split('-')
                if len(rngs) == 2:
                    out += list(range(int(rngs[0]), int(rngs[1]) + 1))
                else:
                    raise ValueError('invalid range %s' % it)
            else:
                out.append(int(it))

        return out

    def dump_pdb(self, confs, fname='structure_%d.pdb', render=False, high_quality=False,
                 fmt='png', wsize=(1024, 1024), show_bonds=True, **image_kwargs):
        if isinstance(confs, int):
            confs = [confs]
        elif isinstance(confs, str):
            confs = self._parse_range_string(confs, self.nstruct)

        radii = self.radii
        idx = self.index
        genome = self.genome
        chroms = self.index.get_chromosomes()
        n_chrom = max(chroms) + 1

        # set the tubewidth relative to the size of the beads
        tubewidth = min(radii/1000) * 0.6

        # Generate colors
        h = np.arange(float(n_chrom + 2)) / (n_chrom + 1)
        color = {}
        for i in range(n_chrom):
            color[i] = colorsys.hsv_to_rgb(h[i], 1, 1)

        if not fname.endswith('.pdb'):
            fname = fname + '.pdb'

        for conf in confs:
            crd = self.get_struct_crd(conf)

            if '%' in fname:
                ofname = fname % conf
            else:
                ofname = fname

            # Write PDB
            rn = ''
            rnum = 0
            bondlist = ''

            with open(ofname, 'w') as outf:
                for i, (x, y, z) in enumerate(crd):
                    cpy = idx.copy[i]
                    nrn = str(genome.getchrom(idx.chrom[i])).replace('chr', '')
                    if nrn != rn:
                        rnum += 1
                    else:
                        bondlist += ' {%d %d}' % (i - 1, i)
                    rn = nrn
                    vstr = '{:6s}{:5d} {:^4s}{:1s}{:3s}'.format('HETATM', i + 1, 'BDX', ' ', rn)
                    vstr += ' {:1s}{:4d}{:1s}   '.format(str(cpy), rnum, ' ')
                    vstr += '{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}'.format(x / 1000, y / 1000, z / 1000, radii[i] / 1000,
                                                                         0, 0, 0)
                    print(vstr, file=outf)

            # Write tcl file
            script = ''
            script += 'package require topotools\n'
            # Set colors

            for i in range(23):
                cn = i + 1
                script += 'color change rgb %d %f %f %f\n' % (cn, color[i][0], color[i][1], color[i][2])
            # Load coordinates
            script += 'mol new %s type pdb autobonds no\n' % ofname
            # Set radiuses
            script += 'mol delrep 0 top\n'
            script += 'set sel [atomselect top all]\n'
            script += '$sel set radius [$sel get occupancy]\n'
            # Create representation
            script += 'mol representation VDW 1.000000 12.000000\n'
            script += 'mol color ResName\n'
            script += 'mol selection {all}\n'
            script += 'mol material Diffuse\n'
            script += 'mol addrep top\n'

            script += 'topo setbondlist { %s }\n' % bondlist
            if show_bonds:
                script += 'mol representation Bonds %.6f 12.000000\n' % tubewidth
                script += 'mol selection {all}\n'
                script += 'mol material Diffuse\n'
                script += 'mol addrep top\n'

            script += f'''
                display projection Orthographic
                axes location Off
                color Display Background 300
                color change rgb 300 0.000000 0.000000 0.000000
                display resize {wsize[0]} {wsize[1]}
                display height 4
            '''

            if high_quality:
                script += 'display ambientocclusion on\n' \
                          'display shadows on\n'

            scriptname = ofname[:-4] + '.tcl'
            with open(scriptname, 'w') as outf:
                outf.write(script)

            if render:
                imfile = ofname[:-4] + '.tga'
                rendercmd = f'render TachyonInternal {imfile}\nquit\n'

                tmpfname = mktemp()
                with open(tmpfname, 'wb') as tmpf:
                    tmpf.write((script + rendercmd).encode('utf-8'))

                out, err = mktemp(), mktemp()
                r = os.system(f'vmd -e {tmpfname} -dispdev text > {out} 2> {err}')
                if r != 0:  # error occured
                    sys.stderr.write('Error executing vmd.\nDumped logs to hss.render.out and hss.render.err.\n')
                    shutil.copy(out, 'hss.render.out')
                    shutil.copy(err, 'hss.render.err')
                    continue
                else:
                    if fmt != 'tga':
                        try:
                            from PIL import Image
                            im = Image.open(imfile)
                            oifile = ofname[:-4] + '.' + fmt
                            im.save(oifile, **image_kwargs)
                            os.remove(imfile)
                        except ImportError:
                            pass

                os.remove(tmpfname)
                os.remove(out)
                os.remove(err)

    def dump_cmm(self, confs, fname='structure_%d.cmm'):
        markertemplate = '<marker id="%d" x="%.3f" y="%.3f" z="%.3f" r="%.3f" g="%.3f" b="%.3f" note="" ' \
                         'radius="%.3f" nr="%.3f" ng="%.3f" nb="%.3f" extra="%s"/>\n'
        linktemplate = '<link id1="%d" id2="%d" r="%.3f" g="%.3f" b="%.3f" radius="20" />\n'

        if isinstance(confs, int):
            confs = [confs]
        elif isinstance(confs, str):
            confs = self._parse_range_string(confs, self.nstruct)

        r = self.radii
        chroms = self.index.get_chromosomes()
        n_chrom = max(chroms) + 1
        idx = self.index.chrom

        # Generate colors
        h = np.arange(float(n_chrom + 2)) / (n_chrom + 1)
        color = {}
        for i in range(n_chrom):
            color[i] = colorsys.hsv_to_rgb(h[i], 1, 1)

        for conf in confs:
            crd = self.get_struct_crd(conf)

            if '%' in fname:
                ofname = fname % conf
            else:
                ofname = fname

            with open(ofname, 'w') as f:
                f.write('<marker_set name="bead">\n')
                for i in range(len(crd)):
                    f.write(markertemplate % (i + 1,
                                              crd[i, 0], crd[i, 1], crd[i, 2],
                                              color[idx[i]][0],
                                              color[idx[i]][1],
                                              color[idx[i]][2],
                                              r[i],
                                              color[idx[i]][0],
                                              color[idx[i]][1],
                                              color[idx[i]][2],
                                              ''))
                    if i > 0 and idx[i] == idx[i - 1]:
                        f.write(linktemplate % (i, i + 1,
                                                color[idx[i]][0],
                                                color[idx[i]][1],
                                                color[idx[i]][2]))
                f.write('</marker_set>\n')


# ================================================


def get_simulated_hic(hssfname, contactRange=2):
    '''
    Computes the contact map and sums the copies. Returns a Contactmap instance
    '''
    with HssFile(hssfname, 'r') as f:
        full_cmap = f.buildContactMap(contactRange)

    return full_cmap.sumCopies(norm='min')


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


def resample_coordinates(old_crd, old_index, new_index):
    new_crd = np.empty((len(new_index), 3))
    for chrom in old_index.get_chromosomes():
        for ccopy in old_index.get_chrom_copies(chrom):
            old_chrom_idx = old_index.get_chrom_pos(chrom, ccopy)
            old_chrom_crd = old_crd[old_chrom_idx]
            old_chrom_midpts = (old_index.end[old_chrom_idx] + old_index.start[old_chrom_idx]) / 2
            spline = CatmullRomSpline(old_chrom_crd, old_chrom_midpts)

            new_chrom_idx = new_index.get_chrom_pos(chrom, ccopy)
            new_chrom_midpts = (new_index.end[new_chrom_idx] + new_index.start[new_chrom_idx]) / 2
            new_chrom_crd = np.array([spline(x) for x in new_chrom_midpts])
            new_crd[new_chrom_idx] = new_chrom_crd
    return new_crd