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

__author__ = "Nan Hua"

__license__ = "GPL"
__version__ = "0.0.4"
__email__ = "nhua@usc.edu"

import numpy as np

import warnings

warnings.simplefilter('ignore', FutureWarning)

import os.path
import h5py
import scipy.sparse
import cooler

try:
    import cPickle as pickle
except:
    import pickle
from six import string_types

from . import utils
from . import matrix


class Contactmatrix(object):
    """
    A flexible matrix instant that supports various methods for processing HiC
    contacts. It provides a syntax to easily generate submatrices.

    Parameters
    ----------
    mat : str or matrix or None
        read matrix from filename, or initialize the matrix from a scipy
        sparse matrix, or a numpy dense matrix, a cooler file
        or with an empty matrix
    genome : alabtools.Genome or str
        Name of the genome assembly e.g.'hg19','mm9'. It is ignored if
        loading from hcs, hdf5, or cooler
    resolution : int or utils.Index
        the resolution for the hic matrix e.g. 100000, or a alabtools index
        describing the binning
    usechr : list of str
        chromosomes used for generating the matrix, es: ['#', 'X', 'Y']

    Attributes
    ----------
    matrix : sparse skyline matrix sss_matrix
    index : utils.Index
    genome : utils.Genome, for the genome
    resolution : resolution for the contact matrix

    Examples
    --------

    # load a matrix
    cm = Contactmatrix('mm9_10kb.cool')

    # remove 2% of low coverage bins
    cm.maskLowCoverage(2)

    # krnormalize it
    cm.krnorm()

    # make a 100kb matrix probability from 10kb matrix, to have average
    # rowsum equal to 25
    cm_100kb = cm.iterativeScaling(10, 25)

    # get the chr1-chr1 submatrix
    cm_chr1 = cm_100kb['chr1']

    # plot it to a file, clipping to 0.2
    com_chr1.plot('chr1.pdf', clip_max=0.2)


    """

    def __init__(self, mat=None, genome=None, resolution=None, usechr=('#', 'X'), tri='upper', lazy=False):

        self.resolution = None
        self.bias = None
        self.genome = None
        self.index = None
        self.h5 = None
        # matrix from file
        if isinstance(mat, string_types):
            if not os.path.isfile(mat):
                raise IOError("File %s doesn't exist!\n" % (mat))
            if os.path.splitext(mat)[1] == '.hdf5' or os.path.splitext(mat)[1] == '.hmat':
                self._load_hmat(mat)
            elif os.path.splitext(mat)[1] == '.hcs':
                self._load_hcs(mat, lazy=lazy)
            elif os.path.splitext(mat)[1] == '.cool':
                # h5 = h5py.File(mat)
                # self._load_cool(h5, assembly=genome, usechr=usechr)
                # h5.close()
                cool = cooler.Cooler(mat)
                self._load_cool_new(cool, assembly=genome, usechr=usechr)
                cool.close()
            elif os.path.splitext(mat)[1] == '.mcool':
                h5 = h5py.File(mat)
                print("Loading matrix from mcool, resolution={}".format(resolution))
                self._load_cool(h5['resolutions']['{}'.format(resolution)], assembly=genome, usechr=usechr)
                h5.close()
            elif os.path.splitext(mat)[1] == '.hic':
                self._load_hic(mat, resolution=resolution, usechr=usechr)
            elif os.path.splitext(mat)[1] == '.gz' and os.path.splitext(os.path.splitext(mat)[0])[1] == '.pairs':
                self._load_pairs_bgzip(mat, resolution, usechr=['#', 'X'])
            else:
                raise ValueError("unrecognized constructor constructor usage")
        else:
            # build genome/index
            if genome is not None:
                if isinstance(genome, utils.Genome):
                    self.genome = genome
                else:
                    self._build_genome(assembly=genome, usechr=usechr)

                if resolution is not None:
                    if isinstance(resolution, utils.Index):
                        self.index = resolution
                        self.resolution = 0
                    else:
                        self._build_index(resolution=resolution)

            # matrix from scipy sparse matrix
            if isinstance(mat, scipy.sparse.base.spmatrix):
                mm = mat.tocsr()
                diag = mm.diagonal()
                mm.setdiag(0)
                mm.eliminate_zeros()
                self.matrix = matrix.sss_matrix((mm.data, mm.indices,
                                                 mm.indptr, diag))
            # matrix from numpy dense matrix
            elif isinstance(mat, np.ndarray):
                if tri == 'upper':
                    mat = np.triu(mat)
                elif tri == 'lower':
                    mat = np.tril(mat)
                else:
                    raise ValueError('tri should be either `upper` or `lower`, got `%s`' % repr(tri))
                mm = scipy.sparse.csr_matrix(mat)
                diag = mm.diagonal()
                mm.setdiag(0)
                mm.eliminate_zeros()
                self.matrix = matrix.sss_matrix((mm.data, mm.indices,
                                                 mm.indptr, diag))
            # empty matrix
            elif mat is None:
                if self.index is not None:
                    self.matrix = matrix.sss_matrix(([], [],
                                                     [0] * (len(self.index) + 1),
                                                     [0] * len(self.index)))
            else:
                raise (ValueError, 'If mat argument is none, you must specify genome and resolution')
            # assert(self.matrix.shape[0] == self.matrix.shape[1] == len(self.index))
            # -

    def _build_genome(self, assembly, usechr=('#', 'X', 'Y'), chroms=None,
                      origins=None, lengths=None):
        self.genome = utils.Genome(assembly, chroms=chroms, origins=origins, lengths=lengths, usechr=usechr)

    def _build_index(self, resolution):
        self.index = self.genome.bininfo(resolution)
        self.resolution = resolution

    def _set_index(self, chrom, start, end, label=[], copy=[], chrom_sizes=[]):
        self.index = utils.Index(chrom=chrom, start=start, end=end, label=label, copy=copy, chrom_sizes=chrom_sizes)

    def _load_hcs(self, filename, lazy=False):
        self.h5 = h5py.File(filename, 'r')
        self.resolution = self.h5.attrs["resolution"]
        self.genome = utils.Genome(self.h5)
        self.index = utils.Index(self.h5)

        if 'bias' in self.h5:
            self.bias = self.h5['bias'][()]

        self.matrix = matrix.sss_matrix(self.h5["matrix"], lazy=lazy)
        if not lazy:
            self.h5.close()
            del self.h5

    def _load_cool(self, h5, assembly=None, usechr=('#', 'X')):
        if assembly is None:
            assembly = h5.attrs['genome-assembly']
        self.resolution = h5.attrs['bin-size']
        self._build_genome(assembly, usechr=usechr, chroms=h5['chroms']['name'][:], lengths=h5['chroms']['length'][:])
        self._build_index(self.resolution)

        # this doesn't work right now because origenome and self.genome have different order
        # maybe we can use
        # origenome = utils.Genome(assembly, h5['chroms']['name'][:],
        #                          lengths=h5['chroms']['length'][:], usechr=["#", "X", "Y"])?
        
        # origenome = utils.Genome(assembly, usechr=['#', 'X', 'Y'], silence=True)
        origenome = utils.Genome(assembly, h5['chroms']['name'][:],
                                 lengths=h5['chroms']['length'][:], usechr=["#", "X", "Y"])
        allChrId = [origenome.getchrnum(self.genome[x]) for x in range(len(self.genome))]
        chrIdRange = [[allChrId[0], allChrId[0] + 1]]
        for i in allChrId[1:]:
            if i == chrIdRange[-1][1]:
                chrIdRange[-1][1] = i + 1
            else:
                chrIdRange.append([i, i + 1])

        indptr = np.empty(len(self.index) + 1, np.int32)
        indptr[0] = 0
        indices = []
        data = []

        # loop through all used chromosomes
        for cil, cih in chrIdRange:
            i0, i1 = (h5['indexes']['chrom_offset'][cil], h5['indexes']['chrom_offset'][cih])
            for r in range(i0, i1):
                edge = h5['indexes']['bin1_offset'][r:r + 2]
                # print(edge)
                rowind = h5['pixels']['bin2_id'][edge[0]:edge[1]]
                rowdata = h5['pixels']['count'][edge[0]:edge[1]]
                newind = []
                newdata = []
                # get data within range
                for cjl, cjh in chrIdRange:
                    if cjl < cil:
                        continue
                    j0, j1 = (h5['indexes']['chrom_offset'][cjl], h5['indexes']['chrom_offset'][cjh])
                    chroffset = self.index.offset[self.genome.getchrnum(origenome[cjl])]
                    mask = (rowind >= j0) & (rowind < j1)
                    newind.append(rowind[mask] - j0 + chroffset)
                    newdata.append(rowdata[mask])
                # -
                rowind = np.concatenate(newind, axis=0)
                rowdata = np.concatenate(newdata, axis=0)
                indices.append(rowind)
                data.append(rowdata)
            # --
        # --
        for r in range(len(indices)):
            indptr[r + 1] = indptr[r] + len(indices[r])

        indices = np.concatenate(indices)
        data = np.concatenate(data)

        self.matrix = matrix.sss_matrix((data, indices, indptr), shape=(len(self.index), len(self.index)))
    
    @staticmethod
    def _read_cool_attribute(cool, attr_keys):
        """Try to read an attribute from a cool file.
        Since the attribute can be in either cool._info or cool.info, we try both.

        Args:
            cool (cooler.Cooler): Cooler file.
            attr_name (list of str): List of possible attribute names.

        Raises:
            ValueError: If attribute is not found (i.e. is None).
            
        Returns:
            str: Attribute value.
        """
        
        # Initialize the attribute to None
        attr = None
        
        # Try to find the attribute in cool._info or cool.info or cool.__dict__
        for key in attr_keys:
            try:
                attr = cool.info[key]
                break
            except AttributeError:
                pass
            try:
                attr = cool._info[key]
                break
            except AttributeError:
                pass
            try:
                attr = cool.__dict__[key]
                break
            except AttributeError:
                pass
        
        # If attribute is still None, raise error
        if attr is None:
            raise ValueError('Attribute not found.')

        return attr

    def _load_cool_new(self, cool, assembly=None, usechr=('#', 'X', 'Y')):
        
        # Read the assembly
        if assembly is None:
            assembly = self._read_cool_attribute(cool, ['assembly', 'genome-assembly'])
        
        # Read the resolution
        self.resolution = self._read_cool_attribute(cool, ['bin-size', 'resolution'])

        # Build the genome
        self._build_genome(assembly, usechr=usechr,
                           chroms=list(cool.chromnames),
                           lengths=list(cool.chromsizes))
        
        # Build the index
        self._build_index(self.resolution)
        
        # Get the complete domain BED of the cooler file
        chroms, start, end  = cool.bins()[:]
        
        # Get the complete contact matrix of the cooler file
        mat = cool.matrix(balance=False)[:]

        # remove 'chrM' from chroms and mat
        chroms = chroms[chroms != 'chrM']
        mat = mat[chroms != 'chrM', :][:, chroms != 'chrM']

        return None

    def _load_pairs_bgzip(self, filename, resolution, usechr=('#', 'X')):
        import pypairix
        tb = pypairix.open(filename)
        self.resolution = resolution

        header = tb.get_header()

        chroms = []
        length = []
        assembly = ''
        for line in header:
            s = line.split()
            if s[0] == "#chromsize:":
                chroms.append(s[1])
                length.append(int(s[2]))
            if s[0] == "#genome_assembly:":
                assembly = s[1]

        self.genome = utils.Genome(assembly, chroms=chroms, lengths=length, usechr=usechr)
        self.index = self.genome.bininfo(resolution)

        indptr = np.zeros(len(self.index) + 1, dtype=np.int32)
        indices = []
        data = []
        for i in range(len(self.index)):
            chrom1 = self.genome.getchrom(self.index.chrom[i])

            start1 = self.index.start[i]
            end1 = self.index.end[i]
            print(chrom1, start1, end1)
            for j in range(i, len(self.index)):
                chrom2 = self.genome.getchrom(self.index.chrom[j])
                start2 = self.index.start[j]
                end2 = self.index.end[j]
                # querystr='{}:{}-{}|{}:{}-{}'.format(chrom1, start1, end1, chrom2, start2, end2)
                # it = tb.querys2D(querystr)
                it = tb.query2D(chrom1, start1, end1, chrom2, start2, end2, 1)
                n = len([x for x in it])

                if n > 0:
                    indices.append(j)
                    data.append(n)
            # -
            indptr[i + 1] = len(indices)
        # -
        self.matrix = matrix.sss_matrix((data, indices, indptr), shape=(len(self.index), len(self.index)))

    # -------------------
     
    def sort(self):
        """Sorts the HCS file by chromosome.
        Modifies Genome, Index, and Matrix in place.
        """
        
        # Get the original chromint from the index
        chromint_old = self.index.get_chromint()
        
        # Sort the genome
        self.genome.sort()
        
        # Get the sorted index
        self.index = self.genome.bininfo(self.resolution)
        
        # Sort the matrix
        order = np.argsort(chromint_old)  # get the order of the chromosomes before sorting
        mat = self.matrix.toarray()  # convert to dense matrix
        mat = mat[order, :]
        mat = mat[:, order]
        self.matrix = matrix.sss_matrix(mat)
    
    def rowsum(self):
        return self.matrix.sum(axis=1)

    def icp(self, lowp=0, exclude_diagonal=True):
        '''
        Returns the interchromosomal probability for each row, defined as
        the sum of inter-chromosomal pixels divided by rowsum.
        '''
        N = len(self)
        cis = np.zeros(N)
        trans = np.zeros(N)

        nchrom = len(self.index.offset) - 1
        for i in range(nchrom):
            s, e = self.index.offset[i], self.index.offset[i + 1]
            xm = self.matrix.csr[s:e]
            if lowp > 0:
                xm = xm.multiply(xm >= lowp)

            cis[s:e] = np.array(
                xm[:, s:e].sum(axis=0)
                + xm[:, s:e].sum(axis=1).T
            )[0]

            if e < N:
                trans[s:e] += np.array(
                    xm[:, e:].sum(axis=1).T
                )[0]

                trans[e:] += np.array(xm[:, e:].sum(axis=0))[0]

            if not exclude_diagonal:
                d = self.matrix.diagonal[s:e]
                if lowp > 0:
                    d = np.multiply(d, d >= lowp)
                cis[s:e] += d

        # we are ok when returning nan for 0 rowsum rows, turn off warnings for 0 division
        old_settings = np.seterr(divide='ignore', invalid='ignore')
        icp = trans / (trans + cis)
        np.seterr(**old_settings)

        return icp

    def columnsum(self):
        return self.matrix.sum(axis=0)

    def _getZeroEntry(self):
        self.mask = np.flatnonzero(self.rowsum() == 0)

    def _getMask(self, mask=None):
        if mask is None:
            self._getZeroEntry()
            return 0
        else:
            if isinstance(mask, np.ndarray):
                self.mask = mask
                return 1
            else:
                raise TypeError("Invalid argument type, numpy.ndarray is required")

    def getSubMatrix(self, c0, c1, sum_copies=True, copies=None):
        '''
        Select a chromosome-chromosome submatrix

        Parameters
        ----------
        c0 : int or str
            first chromosome
        c1 : int or str
            second chromosome
        sum_copies: bool
            if a chromosome has multiple copies, sum the values
        copies: int or iterable
            if specified, considers only some copy ids
        '''
        if isinstance(c0, string_types):
            c0 = self.genome.getchrnum(c0)

        if isinstance(c1, string_types):
            c1 = self.genome.getchrnum(c1)

        idx = [
            np.where(self.index.chrom == c0)[0],
            np.where(self.index.chrom == c1)[0]
        ]

        if copies is None:
            copies = [
                np.unique(self.index.copy[idx[0]]),
                np.unique(self.index.copy[idx[1]])
            ]

        cidx = [list(), list()]
        for i in [0, 1]:
            try:
                copies[i] = list(copies[i])
            except TypeError:
                copies[i] = [copies[i]]

            for cpy in copies[i]:
                kk = np.where(self.index.copy[idx[i]] == cpy)[0]
                cidx[i].append(idx[i][kk])

        l = len(cidx[0][0]), len(cidx[1][0])
        if sum_copies:
            n, m = l[0], l[1]
        else:
            n, m = l[0] * len(copies[0]), l[1] * len(copies[1])

        sm = np.zeros((n, m))

        for icpy in range(len(copies[0])):
            for jcpy in range(len(copies[1])):

                ii = cidx[0][icpy]
                jj = cidx[1][jcpy]
                transpose = False

                if jj[0] < ii[0]:
                    ii, jj = jj, ii
                    transpose = True

                chunk = self.matrix.get_triu_row(ii)[:, jj]
                if jj[0] == ii[0]:
                    chunk += chunk.T
                    chunk[np.diag_indices(len(chunk))] /= 2
                elif transpose:
                    chunk = chunk.T
                # chunk = self.matrix[cidx[0][icpy], cidx[1][jcpy]]

                if sum_copies:
                    sm += chunk
                else:
                    sm[l[0] * icpy:l[0] * (icpy + 1)][:, l[1] * jcpy:l[1] * (jcpy + 1)] = chunk

        return sm

    def sumCopies(self, norm='none'):
        """
        Parameters
        ----------
        norm : str
            if specified, normalizes each chrom/chrom submatrix according
            to the number of copies. Possible values are
            - min:
                use the minimum of the number of copies. For example, in
                a male cell, chr1-chr2 will be divided by 2, while
                chr1 - chrY will be divided by 1. I believe this is a good approximation
                of the probability of an HiC contact to be in one structure in a population.
                It may yield values larger than one, but it should be quite rare. It should, in spirit,
                compare to balanced Hi-C matrices.
            - max:
                use the maximum of the number of copies. For example, in
                a male cell, everything will be divided by 2, except for chrX-chrX, chrX-chrY, chrY-chrY.
            - prod:
                use the product of the number of copies, returning effectively an average
                over the number of instances. Each output pixel is guarantee to be smaller or equal the
                maximum value in the matrix.

        """
        ii = np.array([i for i in self.index.copy_index.keys()])
        new_index = self.index.get_sub_index(ii)
        n = len(new_index)
        chroms = new_index.get_chromosomes()
        n_chrom = len(chroms)
        chrom_idxs = [
            [
                self.index.get_chrom_pos(c, z) for z in self.index.get_chrom_copies(c)
            ] for c in chroms
        ]

        out_matrix = np.empty((n, n), dtype='object')
        for i in range(n_chrom):
            for j in range(i, n_chrom):
                oi = new_index.get_chrom_index(chroms[i])
                oj = new_index.get_chrom_index(chroms[j])
                x = np.zeros((len(oi), len(oj)))
                for ci in chrom_idxs[i]:
                    for cj in chrom_idxs[j]:
                        if ci[0] <= cj[0]:
                            # we are working on a triangular matrix, so we have to take care to
                            # not select the lower triangle.
                            x += self.matrix.get_triu_row((ci, cj))
                        elif i == j:
                            # if we are working on intra-chromosomal contacts, ignore lower off diagonal: they are
                            # duplicates. We transpose the lower triangle later.
                            continue
                        else:
                            # if we are on the lower triangular part,
                            # select submatrix from upper triangle and transpose
                            x += self.matrix.get_triu_row((cj, ci)).T

                if i == j:
                    # if we are dealing with intra chromosomal contacts,
                    # the lower triangular part comes from different copies
                    # we have to flip it and add it to the upper triangle
                    x += np.tril(x, -1).T
                    x = np.triu(x)

                if norm:
                    nc1, nc2 = len(chrom_idxs[i]), len(chrom_idxs[j])
                    if norm == 'min':
                        x /= min(nc1, nc2)
                    elif norm == 'max':
                        x /= max(nc1, nc2)
                    elif norm == 'prod':
                        x /= nc1 * nc2

                out_matrix[i, j] = scipy.sparse.csr_matrix(x)

        full_matrix = scipy.sparse.bmat(out_matrix, format='csr')
        return Contactmatrix(full_matrix, genome=self.genome, resolution=new_index)

    def __len__(self):
        return self.index.__len__()

    def __repr__(self):
        if self.genome:
            assembly = str(self.genome.assembly).strip()
        else:
            assembly = 'Assembly not specified'
        return '<alabtools.Contactmatrix: {:d} x {:d} | {:s} | {:s}>'.format(
            self.matrix.shape[0],
            self.matrix.shape[1],
            assembly,
            str(self.resolution)
        )

    def __getIntra(self, start, stop):
        uniqueChroms = np.unique(self.index.chrom[start:stop])
        usechr = []
        for c in uniqueChroms:
            chrom = self.genome.getchrom(c)
            usechr.append(chrom[3:])

        newMatrix = Contactmatrix(None, genome=None, resolution=self.resolution)
        newMatrix._build_genome(self.genome.assembly,
                                usechr=usechr,
                                chroms=self.genome.chroms,
                                origins=self.genome.origins,
                                lengths=self.genome.lengths)

        newchrom = np.copy(self.index.chrom[start:stop])
        for i in range(len(newchrom)):
            chrom = self.genome.getchrom(newchrom[i])
            newchrom[i] = newMatrix.genome.getchrnum(chrom)

        newMatrix._set_index(newchrom,
                             self.index.start[start:stop],
                             self.index.end[start:stop],
                             self.index.label[start:stop],
                             self.index.copy[start:stop],
                             [])

        submat = self.matrix.csr[start:stop, start:stop]

        newMatrix.matrix = matrix.sss_matrix((submat.data,
                                              submat.indices,
                                              submat.indptr,
                                              self.matrix.diagonal[start:stop]
                                              ))
        return newMatrix

    def __getitem__(self, key):
        if isinstance(key, (string_types, bytes)):
            if isinstance(key, str):
                chrnum = self.genome.getchrnum(key.encode())
            else:
                chrnum = self.genome.getchrnum(key)
            chrstart = np.flatnonzero(self.index.chrom == chrnum)[0]
            chrend = np.flatnonzero(self.index.chrom == chrnum)[-1]
            return self.__getIntra(chrstart, chrend + 1)

        elif isinstance(key, slice):
            if key.step is None:
                step = 1
            else:
                step = key.step
            if key.start is None:
                start = 0
            else:
                start = key.start
            if key.stop is None:
                stop = len(self)
            else:
                stop = key.stop

            if start < 0: start += len(self)
            if stop < 0: stop += len(self)
            if start > len(self) or stop > len(self):  raise IndexError("The index out of range")
            return self.__getIntra(start, stop)
        else:
            raise TypeError("Invalid argument type")

    # -------------------

    def maskLowCoverage(self, cutoff=2):
        """
        Removes "cutoff" percent of bins with least counts

        Parameters
        ----------
        cutoff : int, 0<cutoff<100
            Percent of lowest-counts bins to be removed
        """
        rowsum = self.rowsum()
        self.mask = np.flatnonzero(rowsum <= np.percentile(rowsum[rowsum > 0], cutoff))
        print("{} bins are masked.".format(len(self.mask)))

    def normalize(self, bias):
        """
        norm matrix by bias vector

        Parameters
        ----------
        bias : np.array column vector

        """
        # stores the bias
        self.bias = bias
        self.matrix.normalize(np.array(bias))

    # -

    def getKRnormBias(self, mask=None, **kwargs):
        """
        using KR balancing algorithm to calculate bias vector

        Parameters
        ----------
        mask : list/array
            mask is a 1-D vector with the same length as the matrix where 1s specify the row/column to be ignored\
            or a 1-D vector specifing the indexes of row/column to be ignored\
            if no mask is given, row/column with rowsum==0 will be automatically detected and ignored

        Returns
        -------
        out : numpy column array
            vector of bias
        """
        from .norm import bnewt
        if not hasattr(self, "mask"):
            self._getMask(mask)
        x = bnewt(self.matrix, mask=self.mask, check=0, **kwargs) * 100
        return x

    def krnorm(self, mask=None, force=False, **kwargs):
        """
        using krnorm balacing the matrix (overwriting the matrix!)

        Parameters
        ----------
        mask : list/array
            mask is a 1-D vector with the same length as the matrix where 1s specify the row/column to be ignored\
            or a 1-D vector specifing the indexes of row/column to be ignored\
            if no mask is given, row/column with rowsum==0 will be automatically detected and ignored
        force : bool
            force to normalize the matrix
        """
        x = self.getKRnormBias(mask, **kwargs)
        self.normalize(x)

    def computeConfidenceMatrix(self, M=None):
        from ._cmtools import CalculatePixelConfidence
        if M is None:
            M = self.matrix.toarray()
        cc = np.empty(self.matrix.shape, dtype=np.float32)
        ee = np.empty(self.matrix.shape, dtype=np.float32)
        # do confidence calc for each chr-chr block otherwise makes no sense
        for chr1 in range(len(self.genome.chroms)):
            id1 = np.flatnonzero(self.index.chrom == chr1)
            print(self.genome.chroms[chr1])
            for chr2 in range(len(self.genome.chroms)):
                id2 = np.flatnonzero(self.index.chrom == chr2)
                m = M[id1[0]:id1[-1] + 1, id2[0]:id2[-1] + 1].copy()

                c = np.zeros(m.shape, dtype=np.float32)
                e = np.zeros(m.shape, dtype=np.float32)
                CalculatePixelConfidence(m, c, e)
                cc[id1[0]:id1[-1] + 1, id2[0]:id2[-1] + 1] = c
                ee[id1[0]:id1[-1] + 1, id2[0]:id2[-1] + 1] = e
        cc[np.isnan(cc)] = 0
        return cc, ee

    def filterByConfidence(self, cut=0.25, passes=2):
        from ._cmtools import CalculatePixelConfidence
        newM = self.matrix.toarray()
        for _ in range(passes):
            for chr1 in range(len(self.genome.chroms)):
                id1 = np.flatnonzero(self.index.chrom == chr1)
                print(self.genome.chroms[chr1])
                for chr2 in range(len(self.genome.chroms)):
                    id2 = np.flatnonzero(self.index.chrom == chr2)
                    m = newM[id1[0]:id1[-1] + 1, id2[0]:id2[-1] + 1].copy()

                    c = np.zeros(m.shape, dtype=np.float32)
                    e = np.zeros(m.shape, dtype=np.float32)

                    CalculatePixelConfidence(m, c, e)
                    c[np.isnan(c)] = 0

                    ii = np.where(c < cut)
                    m[ii] = e[ii]
                    newM[id1[0]:id1[-1] + 1, id2[0]:id2[-1] + 1] = m

        return Contactmatrix(newM, genome=self.genome, resolution=self.index)

    def expectedRestraints(self, cut=0.01, which='both'):
        '''
        Returns the number of expected restraints per structure per bead when theta == cut

        Parameters
        ----------
        cut : float
            the theta cutoff
        which : ['intra', 'inter', 'both']
        '''
        if which in ['intra', 'cis']:
            ii = (self.matrix.data >= cut) & self.getIntraSparseMask()
        elif which in ['inter', 'trans']:
            ii = (self.matrix.data >= cut) & self.getInterSparseMask()
        elif which in ['both', 'all']:
            ii = self.matrix.data >= cut
        else:
            raise ValueError(
                '`which` argument not understood: %s.\nAllowed values are "intra", "inter", "both" (default)' % which)

        return np.sum(self.matrix.data[ii]) / self.matrix.shape[0]

    def getInterSparseMask(self):
        jj = self.matrix.indices
        # create row index vector
        ip = self.matrix.indptr
        ii = np.empty(len(jj), dtype=int)
        for i in range(len(ip) - 1):
            ii[ip[i]: ip[i + 1]] = i
        return self.index.chrom[ii] != self.index.chrom[jj]

    def getIntraSparseMask(self):
        jj = self.matrix.indices
        ip = self.matrix.indptr
        ii = np.empty(len(jj), dtype=int)
        for i in range(len(ip) - 1):
            ii[ip[i]: ip[i + 1]] = i
        return self.index.chrom[ii] == self.index.chrom[jj]

    def getInterValues(self):
        return self.matrix.data[self.getInterSparseMask()]

    def getIntraValues(self):
        return self.matrix.data[self.getIntraSparseMask()]

    def getInterIJV(self):
        jj = self.matrix.indices
        ip = self.matrix.indptr
        ii = np.empty(len(jj), dtype=int)
        for i in range(len(ip) - 1):
            ii[ip[i]: ip[i + 1]] = i
        mask = self.index.chrom[ii] != self.index.chrom[jj]
        return ii[mask], jj[mask], self.matrix.data[mask]

    def getIntraIJV(self):
        jj = self.matrix.indices
        ip = self.matrix.indptr
        ii = np.empty(len(jj), dtype=int)
        for i in range(len(ip) - 1):
            ii[ip[i]: ip[i + 1]] = i
        mask = self.index.chrom[ii] == self.index.chrom[jj]
        return ii[mask], jj[mask], self.matrix.data[mask]

    def getInterMask(self):
        return self.index.chrom[:, None] != self.index.chrom[None, :]

    def getIntraMask(self):
        return self.index.chrom[:, None] == self.index.chrom[None, :]

    def toarray(self):
        return self.matrix.toarray()

    def makeSummaryMatrix(self, step=10):
        """
        Filter the matrix to get a lower resolution matrix. New resolution will be resolution*step

        Parameters
        ----------
        step : int/string
            int : The length of submatrix to be checked.
            string : Bed file defining the TAD (fields required:[chrom, start, end, label])

        Returns
        -------
        out : contactmatrix instance
            A contactmatrix instance of the lower resolution matrix.

        """

        if isinstance(step, int) and step == 1:
            newMatrix = self.copy()
            return newMatrix

        from ._cmtools import TopmeanSummaryMatrix_func
        DimA = len(self.index)

        newMatrix = Contactmatrix(None, genome=None, resolution=None)
        newMatrix._build_genome(self.genome.assembly,
                                usechr=['#', 'X', 'Y'],
                                chroms=self.genome.chroms,
                                origins=self.genome.origins,
                                lengths=self.genome.lengths)

        if isinstance(step, string_types):
            tadDef = np.genfromtxt(step, dtype=None, encoding=None)

            chrom = tadDef['f0']
            start = tadDef['f1']
            end = tadDef['f2']
            if 'f3' in tadDef:
                label = tadDef['f3']
            else:
                label = [''] * len(chrom)
            nchrom = np.array([newMatrix.genome.getchrnum(x) for x in chrom])

            f_tadDef = np.sort(
                np.rec.fromarrays([nchrom, start, end, label]),
                order=['f0', 'f1']
            )
            chrom = f_tadDef['f0']
            start = f_tadDef['f1']
            end = f_tadDef['f2']
            if 'f3' in f_tadDef:
                label = f_tadDef['f3']
            else:
                label = [''] * len(chrom)
            tad_sizes = end - start
            if hasattr(self, 'resolution'):
                if np.count_nonzero(tad_sizes < self.resolution):
                    # it is ok if they are the last beads of a chromosome tho.
                    warnings.warn('%d TAD(s) are too small for this matrix resolution (%d)' %
                                  (np.count_nonzero(tad_sizes < self.resolution),
                                   self.resolution)
                                  )
            newMatrix._set_index(chrom, start, end, label)
        else:
            newMatrix._build_index(self.resolution * step)

        DimB = len(newMatrix.index)

        _, fwmap, _ = utils.get_index_mappings(newMatrix.index, self.index)

        mapping = np.empty(DimB + 1, dtype=np.int32)
        for i in range(len(fwmap)):
            mapping[i] = fwmap[i][0]
        mapping[-1] = fwmap[-1][-1] + 1

        # row = 0
        # for i in range(DimB):
        #     incStep = int((newMatrix.index.end[i] - newMatrix.index.start[i]) / self.resolution)
        #     row += incStep

        #     if (row >= DimA) or (newMatrix.index.chrom[i] != self.index.chrom[row]):
        #         #row = 1 + np.flatnonzero(self.index.chrom == newMatrix.index.chrom[i])[-1]
        #         row = self.index.offset[self.index.chrom[row-incStep]+1]
        #     mapping[i+1] = row

        Bi = np.empty(int(DimB * (DimB + 1) / 2), dtype=np.int32)
        Bj = np.empty(int(DimB * (DimB + 1) / 2), dtype=np.int32)
        Bx = np.empty(int(DimB * (DimB + 1) / 2), dtype=np.float32)
        TopmeanSummaryMatrix_func(self.matrix.indptr,
                                  self.matrix.indices,
                                  self.matrix.data,
                                  DimA, DimB,
                                  mapping,
                                  Bi, Bj, Bx)
        newMatrix.matrix = matrix.sss_matrix((Bx, (Bi, Bj)))
        newMatrix.resolution = -1
        return newMatrix

    def fmaxScaling(self, fmax, force=False):
        """
        use fmax to generate probability matrix

        """
        if isinstance(fmax, float) or isinstance(fmax, np.float32) or isinstance(fmax, int):
            self.matrix.csr.data /= fmax
            self.matrix.diagonal /= fmax
            self.matrix.csr.data = self.matrix.csr.data.clip(max=1)
            self.matrix.diagonal = self.matrix.diagonal.clip(max=1)
            self.matrix.data = self.matrix.csr.data

    def copy(self):
        newMatrix = Contactmatrix(None, genome=None, resolution=None)
        newMatrix.index = self.index
        newMatrix.genome = self.genome
        newMatrix.matrix = self.matrix.copy()
        newMatrix.resolution = self.resolution
        return newMatrix

    def iterativeScaling(self, domain=10, averageContact=24, theta=0.001, tol=0.01):
        """
        Iterative Scale the matrix to probability matrix at lower resolution, such that the average contact for each bin matches the Parameters.

        Parameters
        ----------
        domain : int/string
            int : The length of submatrix to be checked.
            string : Bed file defining the TAD (fields required:[chrom, start, end, label])

        averageContact : int
            Target average contact for each bin.

        theta : float
            Target theta value expected in modeling step.

        tol : float
            average contact tolarance in order to converge.

        Returns
        -------

        out : Contactmatrix instance
            A probability matrix at given resolution

        """
        average = 0
        originalData = self.matrix.csr.data.copy()
        originalDiag = self.matrix.diagonal.copy()

        fmax = self.rowsum().mean() / (averageContact + 0.2)
        left = 0
        right = fmax * 2

        self.fmaxScaling(fmax, force=True)
        newMat = self.makeSummaryMatrix(domain)
        newMat.matrix.data[newMat.matrix.data < theta] = 0

        rowsums = newMat.rowsum()
        rowsums = rowsums[rowsums > 0]
        average = rowsums.mean()

        print("({}, {}) => {} : {}".format(left, right, fmax, average))

        if average < averageContact:
            right = fmax
            fmax = (left + fmax) / 2

        else:
            left = fmax
            fmax = (right + fmax) / 2

        while abs(average - averageContact) / averageContact > tol:

            self.matrix.csr.data[:] = originalData.copy()
            self.matrix.diagonal[:] = originalDiag.copy()

            self.fmaxScaling(fmax, force=True)

            newMat = self.makeSummaryMatrix(domain)

            newMat.matrix.data[newMat.matrix.data < theta] = 0

            rowsums = newMat.rowsum()
            rowsums = rowsums[rowsums > 0]
            average = rowsums.mean()

            print("({}, {}) => {} : {}".format(left, right, fmax, average))
            if average < averageContact:
                right = fmax
                fmax = (left + fmax) / 2

            else:
                left = fmax
                fmax = (right + fmax) / 2
            # fmax = fmax/averageContact*average

        # ==

        # Reset to our original status
        self.matrix.csr.data[:] = originalData.copy()
        self.matrix.diagonal[:] = originalDiag.copy()

        self.fmaxScaling(fmax, force=True)

        newMat = self.makeSummaryMatrix(domain)
        rowsums = newMat.rowsum()
        rowsums = rowsums[rowsums > 0]
        average = rowsums.mean()

        # Reset to our original status
        self.matrix.csr.data[:] = originalData
        self.matrix.diagonal[:] = originalDiag

        print("Fmax = {} Rowmean = {}".format(fmax, average))
        newMat.matrix.csr.eliminate_zeros()
        return newMat

    # =============plotting method

    def plot(self, filename, log=False, bin_size=None, **kwargs):
        '''
        Plots the current contact matrix to a file

        Parameters
        ----------
        filename: str
            Output filename
        log: bool, optional
            Use a log-scaled color map. Default is False
        bin_size: int, optional
            If set, plots the matrix by separating each bin
            in the index in equally spaced bins of size
            bin_size bases. Useful to plot TAD level matrices on
            a fixed kilobase scale.

        Additional keyword arguments
        ----------------------------

        cmap : matplotlib color map
            Color map used in matrix, e.g cm.Reds, cm.bwr, default is red
        clip_min : float, optional
            The lower clipping value. If an element of a matrix is <clip_min, it is
            plotted as clip_min.
        clip_max : float, optional
            The upper clipping value.
        label : str, optional
            Colorbar label
        ticklabels1 : list, optional
            Custom tick labels for the first dimension of the matrix.
        ticklabels2 : list, optional
            Custom tick labels for the second dimension of the matrix.
        max_resolution : int, optional
            Set a maximum resolution for the output file in pixels.
        '''

        from .plots import plotmatrix, red

        mat = self.matrix.toarray()
        if log:
            mat = np.log(mat)

        if bin_size is not None:
            sizes = self.index.end - self.index.start
            nidx = []
            for i, s in enumerate(sizes):
                if s % bin_size == 0:
                    n = s // bin_size
                else:
                    n = (s // bin_size) + 1
                nidx += [i] * n
            if len(nidx) > 4096 and 'max_resolution' not in kwargs:
                warnings.warn('Very large matrix (%d x %d) to be plotted' %
                              (len(nidx), len(nidx)))
            mat = mat[nidx]
            mat = mat[:, nidx]

        cmap = kwargs.pop('cmap', red)

        plotmatrix(filename, mat, cmap=cmap, **kwargs)

    def plotComparison(self, m2, chromosome=None, file=None, dpi=300, **kwargs):
        from .plots import plot_comparison, red
        cmap = kwargs.pop('cmap', red)
        plot_comparison(self, m2, chromosome, file, dpi, cmap=cmap, **kwargs)

    # =============saveing method
    def save(self, filename, compression='gzip', compression_opts=6):
        """
        save into file
        """

        if (filename[-4:] != '.hcs'):
            filename += '.hcs'

        with h5py.File(filename, 'w') as h5f:
            h5f.attrs["resolution"] = self.resolution
            h5f.attrs["version"] = __version__
            h5f.attrs["nbin"] = len(self.index)
            self.genome.save(h5f, compression=compression, compression_opts=compression_opts)

            self.index.save(h5f, compression=compression, compression_opts=compression_opts)

            if self.bias is not None:
                h5f.create_dataset('bias', data=self.bias, compression=compression, compression_opts=compression_opts)

            mgrp = h5f.create_group("matrix")
            mgrp.create_dataset("data", data=self.matrix.data, compression=compression,
                                compression_opts=compression_opts)
            mgrp.create_dataset("indices", data=self.matrix.indices, compression=compression,
                                compression_opts=compression_opts)
            mgrp.create_dataset("indptr", data=self.matrix.indptr, compression=compression,
                                compression_opts=compression_opts)
            mgrp.create_dataset("diagonal", data=self.matrix.diagonal, compression=compression,
                                compression_opts=compression_opts)

    #
    def __del__(self):
        try:
            self.h5.close()
        except:
            pass
