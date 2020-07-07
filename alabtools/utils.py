# Copyright (C) 2017 University of Southern California and
#                          Nan Hua
#
# Authors: Nan Hua, Guido Polles
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

__author__ = "Nan Hua and Guido Polles"

__license__ = "GPL"
__version__ = "0.0.3"
__email__ = "nhua@usc.edu"

import os
import re
import math
import numpy as np
import warnings
import subprocess
import itertools
import h5py
import json
import scipy
import sys
import collections
import pandas as pd
import scipy.sparse.linalg
from six import string_types

if sys.version_info > (3, 0):
    # python 3.x
    def unicode(s):
        if isinstance(s, bytes):
            return s.decode()
        return s
    from io import StringIO
else:
    # python 2.x
    from cStringIO import StringIO
    unicode = unicode

CHROMS_DTYPE = np.dtype('U10')
ORIGINS_DTYPE = np.int32
LENGTHS_DTYPE = np.int32

CHROM_DTYPE = np.int32
START_DTYPE = np.int32
END_DTYPE = np.int32
COPY_DTYPE = np.int32
LABEL_DTYPE = np.dtype('U10')
CHROM_SIZES_DTYPE = np.int32

COORD_DTYPE = np.float32
RADII_DTYPE = np.float32

# size of buckets (in basepairs) for fast search in index (see index.loc function)
BUCKET_SIZE = 1000000


class Genome(object):
    """
    And instance which holds genome information

    Parameters
    ----------
    assembly : str
        The name of the genome. e.g. "hg19" or "mm9"

    chroms : array[str] like, optional
        List of chromosome names. e.g ['chr1','chr2' ... 'chrX']

    origins : array[int] like, optional
        List of chr region start, default will be arry of zeros

    lengths : array[int] like, optional
        List of chr region length. e.g [249250621, 243199373, ...]

    usechr : array[str] like, optional
        Specified chromsome to use. e.g. ['#','X'] or ['1','2','3'..'10']\
        '#' indicates all autosome chromosomes.

    Notes
    -----

    If ``chroms`` or ``lengths`` is(are) not specified, genome info will be read from genomes/*.info file

    Attributes
    ----------

    assembly : str
        Name of genome
    chroms : np.array[string10]
        chromosome name
    origins : np.array[int32]
        chromosome region start
    lengths : np.array[int32]
        chromosome region length
    """

    def __init__(self, assembly, chroms=None, origins=None, lengths=None,
                 usechr=None, silence=False):

        # If the first argument is a string, and no other info is specified,
        # check if we can read from info or hdf5.
        # Genome assembly name has precedence over hdf5 file names.
        # Will raise a IOError if cannot read any.
        if isinstance(assembly, string_types) and (
            (chroms is None) or (lengths is None)
            ):
            datafile = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "genomes/" + assembly + ".info"
            )
            if os.path.isfile(datafile):
                if not silence:
                    sys.stderr.write("chroms or lengths not given, reading from genomes info file.\n")
                info = np.genfromtxt(datafile, dtype=[("chroms", CHROMS_DTYPE), ("lengths", LENGTHS_DTYPE)])
            else:
                assembly = h5py.File(assembly, 'r')

        if isinstance(assembly, Genome):
            genome = assembly
            assembly = genome.assembly
            if chroms is None:
                chroms = genome.chroms
            if origins is None:
                origins = genome.origins
            if lengths is None:
                lengths = genome.lengths

        if isinstance(assembly, h5py.File):
            chroms = np.array(assembly["genome"]["chroms"][:], CHROMS_DTYPE)
            origins = assembly["genome"]["origins"]
            lengths = assembly["genome"]["lengths"]
            assembly = unicode(assembly["genome"]["assembly"][()])

        if (chroms is None) or (lengths is None):
            chroms = info["chroms"].astype(CHROMS_DTYPE)
            lengths = info["lengths"].astype(LENGTHS_DTYPE)
            origins = np.zeros(len(lengths), dtype=ORIGINS_DTYPE)
        else:
            if origins is None:
                origins = np.zeros(len(lengths), dtype=ORIGINS_DTYPE)
            if len(chroms) != len(lengths) or len(chroms) != len(origins):
                raise RuntimeError("Dimension of chroms and lengths do not match.")

            chroms = np.array(chroms, dtype=CHROMS_DTYPE)
            lengths = np.array(lengths, dtype=LENGTHS_DTYPE)
            origins = np.array(origins, dtype=ORIGINS_DTYPE)

        choices = np.zeros(len(chroms), dtype=bool)

        if usechr is None:
            # if not specified, use all of them
            usechr = np.unique(chroms)

        for chrnum in usechr:
            if chrnum == '#':
                choices = np.logical_or([re.search(r"chr[\d]+$",c) != None for c in chroms], choices)
            else:
                # if specified with full name, remove chr
                chrnum = chrnum.replace('chr', '')
                choices = np.logical_or(chroms == ("chr%s" % chrnum), choices)
        self.chroms = chroms[choices]  # convert to unicode for python2/3 compatibility
        self.origins = origins[choices]
        self.lengths = lengths[choices]
        self.assembly = unicode(assembly)

    # -

    def __eq__(self, other):
        try:
            return (
                    np.all(self.chroms == other.chroms) and
                    np.all(self.origins == other.origins) and
                    np.all(self.lengths == other.lengths) and
                    np.all(self.assembly == other.assembly)
            )
        except:
            return False

    def bininfo(self, resolution):

        """
        Bin the genome by resolution

        Parameters
        ----------
        resolution : int
            resolution of the matrix

        Return
        ------
        utils.Index instance

        """

        binSize = [int(math.ceil(float(x) / resolution)) for x in self.lengths]

        chromList = []
        binLabel = []
        for i in range(len(self.chroms)):
            chromList += [i for j in range(binSize[i])]
            binLabel += [j + int(self.origins[i] / resolution) for j in range(binSize[i])]

        startList = [binLabel[j] * resolution for j in range(sum(binSize))]
        endList = [binLabel[j] * resolution + resolution for j in range(sum(binSize))]

        binInfo = Index(chromList, startList, endList, chrom_sizes=binSize, genome=self)
        return binInfo

    def getchrnum(self, chrom):

        findidx = np.flatnonzero(self.chroms == unicode(chrom))

        if len(findidx) == 0:
            return None
        else:
            return findidx[0]

    def getchrom(self, chromNum):
        assert isinstance(chromNum, (int, np.int32, np.int64))
        return self.chroms[chromNum]

    def __getitem__(self, key):
        if isinstance(key, (int, np.int32, np.int64)):
            return self.getchrom(key)

    def __len__(self):
        return len(self.chroms)

    def __repr__(self):
        represent = "Genome Assembly: " + self.assembly + '\n'
        for i in range(len(self.chroms)):
            represent += (self.chroms[i].astype(str) + '\t' + str(self.origins[i]) + '-' + str(
                self.origins[i] + self.lengths[i]) + '\n')
        return represent

    def save(self, h5f, compression="gzip", compression_opts=6):

        """
        Save genome information into a hd5f file handle. The information will be saved as a group of datasets:
        genome/
            |- assembly
            |- chroms
            |- origins
            |- lengths

        Parameters
        ----------
        h5f : h5py.File object

        compression : string
            "gzip" as default

        compression_opts : int
            compression level, higher the better
        """

        assert isinstance(h5f, h5py.File)
        if 'genome' in h5f:
            ggrp = h5f["genome"]
        else:
            ggrp = h5f.create_group("genome")

        if 'assembly' in ggrp:
            ggrp['assembly'][...] = self.assembly
        else:
            ggrp.create_dataset("assembly", data=self.assembly)

        if 'chroms' in ggrp:
            ggrp['chroms'][...] = np.array(self.chroms, dtype='S10')
        else:
            ggrp.create_dataset("chroms", data=np.array(self.chroms, dtype='S10'),  # hdf5 does not like unicode
                                compression=compression,
                                compression_opts=compression_opts,
                                )

        if 'origins' in ggrp:
            ggrp['origins'][...] = self.origins
        else:
            ggrp.create_dataset("origins", data=self.origins,
                                compression=compression,
                                compression_opts=compression_opts)

        if 'lengths' in ggrp:
            ggrp['lengths'][...] = self.lengths
        else:
            ggrp.create_dataset("lengths", data=self.lengths,
                                compression=compression,
                                compression_opts=compression_opts)


# --------------------


class Index(object):

    """
    Matrix/System indexes. Maps matrix bins or model beads to
    genomic regions.

    Parameters
    ----------
    chrom : list[int32] or np.ndarray[int]
        numeric chromosome id (starting from 0) for each bin/bead.
        Es.: 0 -> chr1, 1 -> chr2, ..., 22 -> chrX
    start : list[int32] or np.ndarray[int]
        genomic starting positions of each bin (in bp, with respect to the
        chromosome start)
    end : list[int32] or np.ndarray[int]
        genomic ending positions of each bin (in bp, with respect to the
        chromosome start)
    label : list[string10] or np.ndarray[str]
        label for each bin (usually, 'CEN', 'gap', 'domain', although it
        can be any string of less than 10 characters)
    copy : list[int32] or np.ndarray[int], optional
        In systems of beads, there may be multiple indistinguishable
        copies of the same chromosome in the system. The copy vector specifies
        which copy of the chromosome each bead maps to. If not specified,
        is computed assuming non-contiguous groups of beads with the same
        `chrom` value belong to different copies.
        Each bead mapping to the same (chrom, start, end) tuple should,
        in general, have a different copy value.
    chrom_sizes : list[int32] or np.ndarray[int], optional
        number of bins/beads in each chromosome. It is useful to specify it if
        two copies of the same chromosome appear as contiguous in the index.
        If not specified, it is automatically computed assuming non-contiguous
        groups of beads with the same `chrom` value belong to different copies.
    genome: alabtools.utils.Genome or str, optional
        genome info for the index.
    usecols: list of ints
        if reading from a text file, use only a subset of columns
    ignore_headers: boolean
        if True and reading from a text file, ignore any column name specification.
        Defaults to False.

    """

    def __init__(self, chrom=[], start=[], end=[], label=[], copy=[], chrom_sizes=[], genome=None, usecols=None,
                 ignore_headers=False, **kwargs):

        self.custom_tracks = []
        self.chrom_sizes = chrom_sizes
        self.genome = genome
        self.chromstr = None
        if isinstance(chrom, string_types):
            try:
                # tries to load a hdf5 file
                with h5py.File(chrom, 'r') as f:
                    self.load_h5f(f)

            except IOError:
                # if it is not a hdf5 file, try to read as text BED

                colnames, n_header = self._look_for_bed_header(chrom)
                if ignore_headers:
                    colnames = None

                data = np.genfromtxt(chrom, usecols=usecols, dtype=None, encoding=None, skip_header=n_header)
                n_fields = len(data.dtype.names)

                if n_fields < 3:
                    raise ValueError('Provided text file does not seem to have at least 3 columns (chrom, start, end).')

                if colnames:
                    assert len(colnames) == n_fields
                    if usecols:
                        colnames = np.array(colnames)[usecols].tolist()
                        assert (colnames[:3] == ['chrom', 'start', 'end'])
                else:
                    colnames = ['chrom', 'start', 'end'] + ['track%d' % i for i in range(n_fields - 3)]

                data.dtype.names = colnames

                # transform chrom names to integers ids
                self.chromstr = data['chrom']
                if self.genome is None:
                    cmap = {s: i for i, s in enumerate(natural_sort(np.unique(data['chrom'])))}
                    self.chrom = [cmap[c] for c in data['chrom']]
                else:
                    if not isinstance(self.genome, Genome):
                        self.genome = Genome(self.genome)
                    self.chrom = np.array([self.genome.getchrnum(c) for c in data['chrom']])

                # set the variables for further processing
                self.start = data['start']
                self.end = data['end']
                if 'label' in data.dtype.names:
                    self.label = data['label']
                else:
                    self.label = []

                if 'copy' in data.dtype.names:
                    self.copy = data['copy']
                else:
                    self.copy = []

                # add custom fields as keyword arguments, to be added
                custom_field_names = set(data.dtype.names) - {'chrom', 'start', 'end', 'copy', 'label'}
                custom_fields = {n: data[n] for n in custom_field_names}
                kwargs.update(custom_fields)

        elif isinstance(chrom, h5py.File):
            self.load_h5f(chrom)

        elif isinstance(chrom, Index):
            index = chrom
            self.chrom = index.chrom
            self.start = index.start
            self.end = index.end
            self.copy = index.copy
            self.genome = index.genome

        else:
            self.chrom = chrom
            self.start = start
            self.end = end
            self.label = label
            self.copy = copy

        # fix datatypes and eventual problems
        if not (len(self.chrom) == len(self.start) == len(self.end)):
            raise RuntimeError("Dimensions do not match.")

        if len(self.chrom):
            try:
                int(self.chrom[0])
            except:
                if isinstance(self.chrom[0], string_types):
                    self.chromstr = self.chrom
                    if self.genome is not None:
                        self.chrom = [self.genome.getchrnum(x) for x in self.chromstr]
                    else:
                        cmap = {s: i for i, s in enumerate(natural_sort(np.unique(self.chromstr)))}
                        self.chrom = [cmap[c] for c in self.chromstr]
                else:
                    raise RuntimeError("chrom should be a list of integers or strings.")

        if len(self.start) and not isinstance(self.start[0], (int, np.int32, np.int64, np.uint32, np.uint64)):
            raise RuntimeError("start should be a list of integers.")
        if len(self.end) and not isinstance(self.end[0], (int, np.int32, np.int64, np.uint32, np.uint64)):
            raise RuntimeError("end should be list of integers.")
        self.chrom = np.array(self.chrom, dtype=CHROM_DTYPE)
        self.start = np.array(self.start, dtype=START_DTYPE)
        self.end = np.array(self.end, dtype=END_DTYPE)

        if len(self.copy) != len(self.chrom):
            self.copy = np.zeros(len(self.chrom), dtype=COPY_DTYPE)
            self._compute_copy_vec()
        else:
            self.copy = np.array(self.copy, dtype=COPY_DTYPE)

        if len(self.chrom_sizes) == 0 and len(self.chrom) != 0:  # chrom sizes have not been computed yet
            self.chrom_sizes = [
                len(list(g))
                for _, g in itertools.groupby(
                    zip(self.chrom, self.copy)
                )
            ]

        self.chrom_sizes = np.array(self.chrom_sizes, dtype=CHROM_SIZES_DTYPE)

        self.num_chrom = len(self.chrom_sizes)

        if len(self.label) != len(self.chrom):
            self.label = np.array(["-"] * len(self.chrom), dtype=LABEL_DTYPE)
        else:
            self.label = np.array(self.label, dtype=LABEL_DTYPE)

        self.offset = np.array([sum(self.chrom_sizes[:i])
                                for i in range(len(self.chrom_sizes) + 1)])

        if not hasattr(self, 'copy_index'):
            self._compute_copy_index()

        self._compute_ref_vec()

        self.loctree = None

        # add additional attributes
        for k, v in kwargs.items():
            self.add_custom_track(k, v)

        if self.chromstr is None:
            if self.genome is not None:
                self.chromstr = self.genome.chroms[self.chrom]
            else:
                self.chromstr = np.array(['chr%d' % (i + 1) for i in self.chrom])

        self._map_chrom_id = {}
        self._map_id_chrom = {}
        for i, s in zip(self.chrom, self.chromstr):
            self._map_chrom_id[s] = i
            self._map_id_chrom[i] = s
        # -

    @staticmethod
    def _look_for_bed_header(file_descriptor):
        '''
        Bed is a somewhat inconsistent format. Apparently can have a header or not,
        and it starts with `track` or `browser`. But I've seen people using the first line
        as column names, or using comments (#) to put column names.
        Here I try to find out if there is a header with column names or not.
        Far from failproof, but should cover the cases I found.
        '''
        colnames = None
        n_head = 0

        if isinstance(file_descriptor, string_types):
            file_descriptor = open(file_descriptor, 'r')

        for line in file_descriptor:
            line = line.strip()
            if line:
                if line.startswith('#'):
                    line = line.replace('#', '')
                    colnames = line.split()
                else:
                    # if has a bed header, I don't know exactly how to parse it,
                    # so check only the other cases
                    if not (line.startswith('track') or line.startswith('browser')):
                        ss = StringIO(line)
                        x = np.genfromtxt(ss, dtype=None, encoding=None)
                        if x.dtype.kind == 'U':
                            # only unicode string, no integers, maybe a header
                            colnames = x.tolist()
                        else:
                            # in this case, we should be reading the chrom start end version
                            if colnames:
                                # if we found column names, unset them if they are not consistent
                                if len(colnames) != len(line.split()):
                                    colnames = None
                                # also, if we have column names, they should start with chrom, start, end
                                elif colnames[:3] != ['chrom', 'start', 'end']:
                                    colnames = None
                            break

            # increase the number of header lines
            n_head += 1

        return colnames, n_head

    def __eq__(self, other):
        '''
        equality check
        '''
        return (
            np.all(self.chrom == other.chrom) and
            np.all(self.start == other.start) and
            np.all(self.end == other.end) and
            np.all(self.label == other.label) and
            np.all(self.copy == other.copy)
        )

    def __add__(self, other):
        '''
        concatenate indices
        '''
        return Index(
            np.concatenate([self.chrom, other.chrom]),
            np.concatenate([self.start, other.start]),
            np.concatenate([self.end, other.end]),
            np.concatenate([self.label, other.label]),
            genome=self.genome
        )

    def resolution(self):
        '''
        Checks if all the region sizes are equal (except for the last region of each chromosome).
        If they are, returns the only size as a resolution. If a resolution cannot be determined
        returns None.
        :return: int or None
        '''
        sizes = self.end - self.start
        last_beads = np.array([self.offset[i] - 1 for i in range(1, len(self.offset))])
        mask = np.ones(len(self), np.bool)
        mask[last_beads] = False
        us = np.unique(sizes[mask])
        if len(us) == 1:
            return us[0]
        return None

    def chrom_to_id(self, c):
        return self._map_chrom_id[c]

    def id_to_chrom(self, c):
        return self._map_id_chrom[c]

    def get_chrom_mask(self, c, copy=None):
        '''
        Return the mask relative to a chromosome

        Parameters
        ----------
        c : str or int
            the chromosome. To access by string, the Index needs to have the
            genome member variable set.
        copy : int or list, optional
            If specified, select only the specified copy/copies.

        Returns
        -------
        idx : np.ndarray

        '''
        if isinstance(c, string_types):
            if self.genome is None:
                c = int(c.replace('chr', '')) - 1
            else:
                c = self.genome.getchrnum(c)
        idx = self.chrom == c
        if copy is not None:
            idx = np.logical_and(idx, np.isin(self.copy, copy))
        return idx

    def get_chrom_pos(self, c, copy=None):
        return np.flatnonzero(self.get_chrom_mask(c, copy))

    def get_chrom_index(self, c, copy=None):
        ii = self.get_chrom_pos(c, copy)
        return Index(
            self.chrom[ii],
            self.start[ii],
            self.end[ii],
            self.label[ii],
            self.copy[ii],
            genome=self.genome
        )

    def get_chrom_copies(self, c):
        return np.unique(self.copy[self.get_chrom_mask(c)])

    def get_chromosomes(self):
        return pd.unique(self.chrom)

    def get_chrom_names(self):
        '''
        Returns the unique names of chromosomes in this index
        '''
        return pd.unique(self.chromstr)

    def __getitem__(self, key):
        return np.rec.fromarrays((self.chrom[key],
                                  self.start[key],
                                  self.end[key],
                                  self.label[key]),
                                 dtype=[("chrom", CHROM_DTYPE),
                                        ("start", START_DTYPE),
                                        ("end", END_DTYPE),
                                        ("label", LABEL_DTYPE)])

    def __len__(self):
        return len(self.chrom)

    def __repr__(self):
        return '<alabtools.Index: %d chroms, %d segments>' % (
            self.num_chrom,
            len(self.chrom)
        )

    def add_custom_track(self, k, v, force=False):
        a = np.array(v)
        assert (len(a) == len(self))

        if k in self.custom_tracks:
            if not force:
                raise KeyError('track %s already present' % k)
            else:
                self.remove_custom_track(k)
        self.custom_tracks.append(k)
        self.__setattr__(k, a)

    def remove_custom_track(self, k):
        if (not hasattr(self, k)) or (k not in self.custom_tracks):
            raise KeyError('track %s not present' % k)
        self.custom_tracks.remove(k)
        t = getattr(self, k)
        delattr(self, k)
        return t

    def rename_custom_track(self, k, nk):
        t = self.remove_custom_track(k)
        self.add_custom_track(nk, t)

    def get_custom_track(self, k):
        return getattr(self, k)

    def get_sub_index(self, keys):
        dt = self[keys]
        custom_tracks = {
            k: self.__getattribute__(k)[keys] for k in self.custom_tracks
        }
        return Index(
            dt['chrom'],
            dt['start'],
            dt['end'],
            dt['label'],
            self.copy[keys],
            genome=self.genome,
            **custom_tracks
        )

    def get_haploid(self):
        return self.get_sub_index(self.copy == 0)

    def _compute_copy_vec(self):
        '''
        Tries to guess the copy vector. It assumes every copy of a chromosome
        to be sequential and sorted by start/end. When bins have the same
        chromosome id but are separated in sequence, or the latter starts
        before the end of the previous one, the two bins are assumed to
        refer to different copies.
        '''
        if len(self.copy) == 0:
            return
        chrom_ids = set(self.chrom)
        copy_no = {c: -1 for c in chrom_ids}
        copy_no[self.chrom[0]] = 0
        self.copy[0] = 0
        for i in range(1, len(self.chrom)):
            cc = self.chrom[i]
            if self.chrom[i - 1] != cc or self.start[i] < self.end[i - 1]:
                copy_no[cc] += 1
            self.copy[i] = copy_no[cc]

    def _compute_copy_index(self):
        '''
        Returns a index of the copies of the same genomic region in form of a
        dictionary. The key of each unique region is the id of the first
        bin/bead mapping to it. The values are lists of all the bead id's
        (including the key) which map to that particular region.
        In the case of an haploid system, or a contact map, this dictionary
        is completely trivial, i.e. {i : i for i in range(len(index))}.
        '''
        tmp_index = {}
        for i, v in enumerate(self):
            locus = (int(v.chrom), int(v.start), int(v.end))
            if locus not in tmp_index:
                tmp_index[locus] = [i]
            else:
                tmp_index[locus] += [i]
        # use first index as a key
        self.copy_index = {ii[0]: ii for locus, ii in tmp_index.items()}

    def _compute_ref_vec(self):
        '''
        Sets a vector of references, the inverse of the copy index
        '''
        self.refs = [None] * len(self)
        for i, ii in self.copy_index.items():
            for j in ii:
                self.refs[j] = i

    def load_h5f(self, h5f):
        self.chrom = h5f["index"]["chrom"][()]
        self.start = h5f["index"]["start"][()]
        self.end = h5f["index"]["end"][()]
        self.copy = h5f["index"]["copy"][()]
        self.label = np.array(h5f["index"]["label"][()], LABEL_DTYPE)
        self.chrom_sizes = h5f["index"]["chrom_sizes"][()]
        if 'genome' in h5f:
            try:
                self.genome = Genome(h5f)
            except:
                pass
        try:
            # try to load the copy index
            tmp = json.loads(h5f["index"]["copy_index"][()])
            # it may happen that in json dump/loading keys are considered
            # as strings.
            self.copy_index = {int(k): v for k, v in tmp.items()}
        except (KeyError, ValueError):
            pass

        try:
            self.chromstr = np.array(h5f["index"]["chromstr"][()], CHROMS_DTYPE)
        except:
            pass
        # tries to load additional data tracks
        if 'custom_tracks' in h5f["index"]:
            self.custom_tracks = json.loads(h5f["index"]["custom_tracks"][()])
            for k in self.custom_tracks:
                v = h5f["index"][k][()]
                if not isinstance(v, np.ndarray):
                    v = np.array(json.loads(v))
                setattr(self, k, v)

    def load_txt(self, f):
        pass

    def save(self, h5f, compression="gzip", compression_opts=6):

        """
        Save index information into a hd5f file handle. The information will
        be saved as a group of datasets:
        index/
            |- chrom
            |- start
            |- end
            |- label
            |- copy
            |- chrom_sizes

        Parameters
        ----------
        h5f : h5py.File object

        compression : string
            'gzip' as default

        compression_opts : int
            compression level, higher the better
        """

        assert isinstance(h5f, h5py.File)

        if 'index' in h5f:
            igrp = h5f['index']
        else:
            igrp = h5f.create_group("index")

        if 'chrom' in igrp:
            igrp['chrom'][...] = self.chrom
        else:
            igrp.create_dataset("chrom", data=self.chrom,
                                compression=compression,
                                compression_opts=compression_opts)

        if 'start' in igrp:
            igrp['start'][...] = self.start
        else:
            igrp.create_dataset("start", data=self.start,
                                compression=compression,
                                compression_opts=compression_opts)

        if 'end' in igrp:
            igrp['end'][...] = self.end
        else:
            igrp.create_dataset("end", data=self.end,
                                compression=compression,
                                compression_opts=compression_opts)

        if 'label' in igrp:
            igrp['label'][...] = np.array(self.label, dtype=np.dtype('S10'))
        else:
            igrp.create_dataset("label", data=np.array(self.label, dtype=np.dtype('S10')),
                                compression=compression,
                                compression_opts=compression_opts)

        if 'copy' in igrp:
            igrp['copy'][...] = self.copy
        else:
            igrp.create_dataset("copy", data=self.copy,
                                compression=compression,
                                compression_opts=compression_opts)

        if 'chrom_sizes' in igrp:
            igrp['chrom_sizes'][...] = self.chrom_sizes
        else:
            igrp.create_dataset("chrom_sizes", data=self.chrom_sizes,
                                compression=compression,
                                compression_opts=compression_opts)

        if 'copy_index' in igrp:
            igrp['copy_index'][...] = json.dumps(self.copy_index)
        else:
            # scalar datasets don't support compression
            igrp.create_dataset("copy_index", data=json.dumps(self.copy_index))

        if 'chromstr' in igrp:
            igrp['chromstr'][...] = np.array(self.chromstr, dtype=np.dtype('S10'))
        else:
            # scalar datasets don't support compression
            igrp.create_dataset("chromstr", data=np.array(self.chromstr, dtype=np.dtype('S10')))

        for k in self.custom_tracks:
            if k in igrp:
                try:
                    igrp[k][...] = self.__getattribute__(k)
                except:
                    igrp[k][...] = json.dumps(self.__getattribute__(k).tolist())
            else:
                # scalar datasets don't support compression
                try:
                    igrp.create_dataset(k, data=self.__getattribute__(k))
                except:
                    igrp.create_dataset(k, data=json.dumps(self.__getattribute__(k).tolist()))

        if 'custom_tracks' in igrp:
            igrp['custom_tracks'][...] = json.dumps(self.custom_tracks)
        else:
            # scalar datasets don't support compression
            igrp.create_dataset("custom_tracks", data=json.dumps(self.custom_tracks))

        if self.genome and "genome" not in h5f:
            self.genome.save(h5f)

        h5f.flush()

    def dump_csv(self, file=sys.stdout, include=None, exclude=None, header=True, header_style='comment', sep=";",
                 cut_chrom_ends=False):

        if isinstance(file, str):
            file = open(file, 'w')

        if include is None:
            include = ['chrom', 'start', 'end', 'label', 'copy'] + self.custom_tracks
        else:
            include = ['chrom', 'start', 'end'] + include

        if exclude is not None:
            for e in exclude:
                if e in include:
                    include.remove(e)

        if header:
            if header_style == 'comment':
                file.write('# ')
            file.write(sep.join(include) + '\n')

        for i in range(len(self)):
            if cut_chrom_ends:
                save_end = self.end[i]
                if self.end[i] > self.genome.lengths[self.chrom[i]]:
                    self.end[i] = self.genome.lengths[self.chrom[i]]
            file.write(
                sep.join([
                    str(self.__getattribute__(col)[i]) if col != 'chrom' else self.chromstr[i] for col in include
                ]) + '\n'
            )
            if cut_chrom_ends:
                self.end[i] = save_end

    def dump_bed(self, file=sys.stdout, include=None, exclude=None, header=True, cut_chrom_ends=False):
        self.dump_csv(file, include, exclude, header, sep='\t', cut_chrom_ends=cut_chrom_ends)

    def loc(self, chrom, start, end=None, copy=None):
        '''
        Get the indexes corresponding to the specified genomic location or
        segment.

        Parameters
        ----------
        chrom (str or int) : chromosome name or 0 based number
        start (int) : position or starting position of a region (basepairs)
        end (int, optional) : ending position of the region
        copy (int or list, optional) : consider only the specified copies
            of the chromosome

        Returns
        -------
        np.ndarray[int] : an array containing the positions of index's
            regions which overlap with the input region.
        '''

        if self.loctree is None:
            self.loctree = LocStruct(self)

        if end is None:
            buckets = self.loctree[chrom][start // BUCKET_SIZE: start // BUCKET_SIZE + 1]
        else:
            buckets = self.loctree[chrom][start // BUCKET_SIZE: (end - 1) // BUCKET_SIZE + 1]
        locs = [np.array([], dtype=int)]
        for bucket in buckets:
            locs.append(bucket.get_intersections(start, end))
        locs = np.concatenate(locs)
        if copy is not None:
            if not isinstance(copy, collections.Iterable):
                copy = [copy]
            locs = [i for i in locs if self.copy[i] in copy]
        return np.sort(np.unique(locs))


# --------------------


def loadstream(filename):
    """
    Convert a file location, return a file handle
    zipped file are automaticaly unzipped using stream
    """

    if not os.path.isfile(filename):
        raise IOError("File %s doesn't exist!\n" % (filename))
    if os.path.splitext(filename)[1] == '.gz':
        p = subprocess.Popen(["zcat", filename], stdout=subprocess.PIPE)
        f = StringIO(p.communicate()[0])
    elif os.path.splitext(filename)[1] == '.bz2':
        p = subprocess.Popen(["bzip2 -d", filename], stdout=subprocess.PIPE)
        f = StringIO(p.communicate()[0])
    else:
        f = open(filename, 'r')
    return f


def make_diploid(index):
    didx = {}
    for k in ['chrom', 'start', 'end', 'label']:
        didx[k] = np.concatenate([index.__dict__[k], index.__dict__[k]])
    didx['copy'] = np.concatenate([index.__dict__['copy'],
                                   index.__dict__['copy'] + 1])
    return Index(didx['chrom'], didx['start'], didx['end'], label=didx['label'], copy=didx['copy'])


def make_multiploid(index, chroms, copies):
    '''
    Returns a multiploid index mapping based on the input index.
    The index is ordered by the copy id and then by the input chrom order.
    For example, using chroms=[0, 1, 2] and copies=[3, 2, 1], the output
    index chromosomes will be in this order: [0, 1, 2, 0, 1, 0].

    Parameters
    ----------
    index (alabtools.utils.Index): Haploid input index
    chroms (array[int] like): the chromosomes ids to be included
        in the output index
    copy_number (array[int] like): the number of copies for each
        of the chromosomes in `chrom`

    Returns
    -------
    alabtools.utils.Index: a index with `copies[i]` copies of the
        `chroms[i]` chromosome, for each element in `chroms`.

    Examples
    --------
    # human genome assembly
    g = Genome('hg19', usechr=['#', 'X', 'Y'])

    # get the index at 1MB resolution
    idx = g.bininfo(1000000)

    # obtain the index for a 1MB resolution male diploid cell
    # i.e. 2 copies for chromosomes 1-22 and 1 each for chromosomes X and Y
    num_copies = [2]*22 + [1, 1]
    idx = make_multiploid(idx, range(24), num_copies)
    '''
    nchrom = []
    nstart = []
    nend = []
    ncopy = []
    nlabel = []
    csizes = []
    add_tracks = {k: [] for k in index.custom_tracks}
    for z in range(max(copies)):
        for cid, cnum in zip(chroms, copies):
            if z < cnum:
                idxs = np.where(index.chrom == cid)[0]
                nchrom.append(index.chrom[idxs])
                nstart.append(index.start[idxs])
                nend.append(index.end[idxs])
                nlabel.append(index.label[idxs])
                ncopy.append(np.array([z] * len(idxs)))
                csizes.append(len(idxs))
                for k in index.custom_tracks:
                    add_tracks[k].append(index.get_custom_track(k)[idxs])

    nchrom = np.concatenate(nchrom)
    nstart = np.concatenate(nstart)
    nend = np.concatenate(nend)
    ncopy = np.concatenate(ncopy)
    nlabel = np.concatenate(nlabel)
    for k in index.custom_tracks:
        add_tracks[k] = np.concatenate(add_tracks[k])

    return Index(nchrom, nstart, nend, copy=ncopy, label=nlabel, chrom_sizes=csizes, **add_tracks)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_index_from_bed(
        file,
        genome=None,
        usecols=None,
    ):

    return Index(file, genome, usecols)


_ftpi = 4. / 3. * np.pi
DEFAULT_NUCLEAR_VOLUME = _ftpi * (5000 ** 3)


def compute_radii(index, occupancy=0.2, volume=DEFAULT_NUCLEAR_VOLUME):
    sizes = [b.end - b.start for b in index]
    totsize = sum(sizes)
    prefactor = volume * occupancy / (_ftpi * totsize)
    rr = [(prefactor * sz) ** (1. / 3.) for sz in sizes]
    return np.array(rr, dtype=RADII_DTYPE)


# if we are dealing with h5py datasets, preloads data for performance
class H5Batcher():
    def __init__(self, ds, bsize):
        self.ds = ds
        self.n = ds.shape[0]
        self.b = 0
        self.bsize = bsize
        self.batch = ds[0:min(bsize, self.n)]

    def __getitem__(self, key):
        if key > self.n:
            raise KeyError()
        z = key // self.bsize
        if z != self.b:
            self.b = z
            self.batch = self.ds[
                         self.bsize * z:
                         min(self.n, self.bsize * (z + 1))
                         ][()]
        return self.batch[key % self.bsize]

    def __len__(self):
        return self.ds.__len__()


def remap(s0, s1):
    '''
    Creates forward and backward maps between two segmentations, assuming
    no gaps in the mapping: each segment will have at least one corresponding
    partner in the other segmentation. Assumes the two segmentations have
    the same starting and ending points. Ugly stuff can happen if this is not
    the case.

    Parameters
    ----------

    s0, s1 : list of ints
        the boundaries of two segmentations, including start and end positions.
        A segmentation in N segments will have N+1 boundary positions.

    Returns
    -------

    dmap : list of lists of ints
        the map from s0 to s1. Each element dmap[ i ], 0 <= i < N holds
        the indexes of the segments on s1 corresponding to the i-th segment
        on s0. Each segment in s0 is guarantee to have at least one
        corresponding segment in s1.
    imap : list of lists of ints
        the inverse mapping from s1 to s0.
    '''

    dmap = [list() for _ in range(len(s0) - 1)]
    imap = [list() for _ in range(len(s1) - 1)]

    # Define an overlap score to generalize the choice of corresponding
    # elements when boundaries do not match. Simply the size of the
    # overlapping section
    def overlap(i, j):
        b = max(s0[i], s1[j])
        e = min(s0[i + 1], s1[j + 1])
        return max(0, e - b)

    N = len(s0) - 1
    M = len(s1) - 1

    # the first elements are necessarily mapped one to the other
    i = 0
    j = 0

    while True:
        dmap[i].append(j)
        imap[j].append(i)

        if i == N - 1 and j == M - 1:
            break

        # if we get to the end of a segmentation, assign all the rest of
        # the other one to this last element
        if j == M - 1:
            i += 1
        elif i == N - 1:
            j += 1
        else:
            # decide on which sequence(s) to advance based on overlap
            v = [overlap(i + 1, j), overlap(i, j + 1), overlap(i + 1, j + 1)]

            k = v.index(max(v))

            if k == 0 or k == 2:
                i += 1
            if k == 1 or k == 2:
                j += 1

    return dmap, imap


def get_index_mappings(idx0, idx1):
    # make sure that the two subdivisions map the same chromosomes
    assert idx0.num_chrom == idx1.num_chrom
    n_chrom = idx0.num_chrom

    # get the starting index for each chromosome
    cc0 = idx0.offset
    cc1 = idx1.offset

    # get the mapping for each chromosome
    cmap, fwmap, bwmap = [], [], []
    for i in range(n_chrom):
        # create the arrays of boundaries
        v0 = np.concatenate([idx0.start[cc0[i]:cc0[i + 1]], [idx0.end[cc0[i + 1] - 1]]])
        v1 = np.concatenate([idx1.start[cc1[i]:cc1[i + 1]], [idx1.end[cc1[i + 1] - 1]]])

        # get the maps chromosome by chromosome
        dm, im = remap(v0, v1)
        cmap.append([dm, im])

        # the full mapping needs to add the chromosome offset
        # to every element (note, elements in direct map refer to
        # the second segmentation, and viceversa)
        fwmap += [
            [
                x + idx1.offset[i] for x in row
            ] for row in dm
        ]
        bwmap += [
            [
                x + idx0.offset[i] for x in row
            ] for row in im
        ]

    return cmap, fwmap, bwmap


def region_intersect(b, e, x, y):
    a = x < e
    b = y > b
    return (a and b)


def region_intersect_one(b, e, x, y):
    return (x >= b) and (x < e)


class Node:
    """
    Class Node
    """
    def __init__(self, value=None):
        self.left = None
        self.data = value
        self.right = None


class BucketTree:
    def __init__(self, ids, starts, ends):
        self.root = Node()
        # sort by start, end, id
        self.fill(self.root, ids, starts, ends)

    @staticmethod
    def fill(node, ids, starts, ends):
        n = len(starts)
        if n == 0:
            return
        elif n == 1:
            node.data = (n, ids[0], starts[0], ends[0])
        else:
            i = n // 2
            node.data = (n, None, starts[0], ends[-1])
            node.left = Node()
            node.right = Node()
            BucketTree.fill(node.left, ids[:i], starts[:i], ends[:i])
            BucketTree.fill(node.right, ids[i:], starts[i:], ends[i:])

    @staticmethod
    def _traverse_tree(node, ids, x, y, f=region_intersect):
        if node.data is None:
            return
        n, i, s, e = node.data
        if f(s, e, x, y):
            if n == 1:
                ids.append(i)
            else:
                BucketTree._traverse_tree(node.left, ids, x, y, f)
                BucketTree._traverse_tree(node.right, ids, x, y, f)

    def get_intersections(self, x, y=None):
        ids = []
        if y is None:
            self._traverse_tree(self.root, ids, x, y, region_intersect_one)
        else:
            self._traverse_tree(self.root, ids, x, y)
        return ids


class BucketLinear:
    def __init__(self, ids, starts, ends):
        self.starts = np.array(starts, dtype=int)
        self.ends = np.array(ends, dtype=int)
        self.ids = np.array(ids, dtype=int)

    def get_intersections(self, x, y=None):
        if y is None:
            ii = (self.starts <= x) & (self.ends > x)
        else:
            ii = (self.ends > x) & (self.starts < y)
        return self.ids[ii]


class LocStruct:
    def __init__(self, index):
        self.chroms = {}
        chromids = index.get_chromosomes()
        chroms = index.get_chrom_names()
        for cid, c in zip(chromids, chroms):
            ii = index.get_chrom_pos(cid)
            ind = np.lexsort((index.start[ii], index.end[ii], ii))
            starts, ends, ids = index.start[ii][ind], index.end[ii][ind], ii[ind]
            nbuckets = ends[-1] // BUCKET_SIZE + 1
            bS = [list() for _ in range(nbuckets)]
            bE = [list() for _ in range(nbuckets)]
            bI = [list() for _ in range(nbuckets)]
            for s, e, i in zip(starts, ends, ids):
                i0, i1 = s // BUCKET_SIZE, (e - 1) // BUCKET_SIZE
                for k in range(i0, i1 + 1):
                    bS[k].append(s)
                    bE[k].append(e)
                    bI[k].append(i)
            self.chroms[c] = list()
            for i in range(nbuckets):
                self.chroms[c].append(BucketLinear(bI[i], bS[i], bE[i]))

            self.chroms[cid] = self.chroms[c]

    def __getitem__(self, c):
        return self.chroms[c]


def underline(*args, **kwargs):
    '''
    Underlines a string.Takes a variable number of unnamed arguments, and the
        final string is assembled like in the print function. The width of
        the under-line matches the longest line in the output (or the terminal
        width, if `terminal` is set to True)

    Keyword arguments
    -----------------
        char : a character or a string to use to underline
        terminal : if set to True, limits the line width to the terminal

    '''
    import re
    char = kwargs.pop('char', '-')
    terminal = kwargs.pop('terminal', False)
    s = ' '.join([str(a) if hasattr(a, '__str__') else repr(a) for a in args])
    ss = re.split('[\\r\\n]+', s)
    l = max([len(x) for x in ss])
    if terminal:
        import shutil
        c, r = shutil.get_terminal_size()
        l = min(l, c)
    u = (char * (l // len(char) + 1))[:l]
    return s + '\n' + u


def block_transpose(x1, x2, max_items=int(1e8)):
    '''
    Transposes a matrix in blocks smaller than max_items.

    Parameters
    ----------
        x1: ndarray-like
            input matrix
        x2: indarray-like
            output matrix
        max_items: int
            maximum number of matrix elements to be transposed at once
    '''
    n = x1.shape[0]
    k = max(max_items // n, 1)
    for i in range(0, n, k):
        s = min(k, n - i)
        block = x1[i:i + s].swapaxes(0, 1)  # get a subset and transpose in memory
        x2[:, i:i + s] = block


def isSymmetric(x):
    return np.all(x.T == x)


# See details in Imakaev et al. (2012)
def PCA(A, numPCs=6, verbose=False):
    """performs PCA analysis, and returns 6 best principal components
    result[0] is the first PC, etc"""
    # A = np.array(A, float)
    if np.sum(np.sum(A, axis=0) == 0) > 0:
        warnings.warn("Columns with zero sum detected. Use zeroPCA instead")
    M = (A - np.mean(A.T, axis=1)).T
    covM = np.dot(M, M.T)
    [latent, coeff] = scipy.sparse.linalg.eigsh(covM, numPCs)
    if verbose:
        print("Eigenvalues are:", latent)
    return (np.transpose(coeff[:, ::-1]), latent[::-1])


def EIG(A, numPCs=3):
    """Performs mean-centered engenvector expansion
    result[0] is the first EV, etc.;
    by default returns 3 EV
    """
    # A = np.array(A, float)
    if np.sum(np.sum(A, axis=0) == 0) > 0:
        warnings.warn("Columns with zero sum detected. Use zeroEIG instead")
    M = (A - np.mean(A))  # subtract the mean (along columns)
    if isSymmetric(A):
        [latent, coeff] = scipy.sparse.linalg.eigsh(M, numPCs)
    else:
        [latent, coeff] = scipy.sparse.linalg.eigs(M, numPCs)
    alatent = np.argsort(np.abs(latent))
    print("eigenvalues are:", latent[alatent])
    coeff = coeff[:, alatent]
    return (np.transpose(coeff[:, ::-1]), latent[alatent][::-1])


def zeroPCA(data, numPCs=3, verbose=False):
    """
    PCA which takes into account bins with zero counts
    """
    nonzeroMask = np.sum(data, axis=0) > 0
    data = data[nonzeroMask]
    data = data[:, nonzeroMask]
    PCs = PCA(data, numPCs, verbose)
    PCNew = [np.zeros(len(nonzeroMask), dtype=float) for _ in PCs[0]]
    for i in range(len(PCs[0])):
        PCNew[i][nonzeroMask] = PCs[0][i]
    return PCNew, PCs[1]


def zeroEIG(data, numPCs=3):
    """
    Eigenvector expansion which takes into account bins with zero counts
    """
    nonzeroMask = np.sum(data, axis=0) > 0
    data = data[nonzeroMask]
    data = data[:, nonzeroMask]
    PCs = EIG(data, numPCs)
    PCNew = [np.zeros(len(nonzeroMask), dtype=float) for _ in PCs[0]]
    for i in range(len(PCs[0])):
        PCNew[i][nonzeroMask] = PCs[0][i]
    return PCNew, PCs[1]


def isiterable(a):
    try:
        for _ in a:
            break
    except:
        return False
    return True


def spline_4p(t, p):
    """ Catmull-Rom """
    # wikipedia Catmull-Rom -> Cubic_Hermite_spline
    # 0 -> p0,  1 -> p1,  1/2 -> (- p_1 + 9 p0 + 9 p1 - p2) / 16
    # assert 0 <= t <= 1
    return (
           t * ((2 - t) * t - 1) * p[0]
           + (t * t * (3 * t - 5) + 2) * p[1]
           + t * ((4 - 3 * t) * t + 1) * p[2]
           + (t - 1) * t * t * p[3]
    ) / 2


class CatmullRomSpline:
    """
    Computes a Catmull-Rom spline curve on n-dimensional points.
    Optionally, one can specify the value of the curve parameter of each point,
    usually (but not necessarily) in the range [0, 1]. This is useful for curves
    which are not equally sampled, for example.
    When a value of curve parameters outside the extremes of chain_positions
    (0 and 1 by default) is provided, a linear interpolation using the two
    last points is used.

    Parameters
    ----------
    points: ndarray
        list of points defining the curve
    chain_positions: array, optional
        list of curve parameters for each point. If not specified, points are
        assumed to be equally spaced

    Notes
    -----
    Since spacing between points can be uneven, a search must be done for each call,
    hurting performance.

    Example
    -------
    points = np.random.random((10, 3))
    cs = CatmullRomSpline(points)
    resampled_points = np.array([ cs(x) for x in np.linspace(0, 1, 100)])
    """
    def __init__(self, points, chain_positions=None):
        points = np.array(points)
        # create extension points at beginning and end
        vb = (points[0] - points[1]) * 0.1
        ve = (points[-1] - points[-2]) * 0.1
        if chain_positions is None:
            chain_positions = np.linspace(0, 1, len(points))

        points = np.concatenate([
            [points[0] + vb],
            points,
            [points[-1] + ve]
        ])

        self.points = points
        self.pos = chain_positions
        self.steps = np.diff(self.pos, 1)

        # try to make average case scenario search faster by creating buckets
        mean_size = self.steps.mean()
        n_buckets = int((self.pos[-1] - self.pos[0]) / mean_size) + 1
        self._buckets = [list() for _ in range(n_buckets)]
        for z in range(1, len(self.pos)):
            start = int((self.pos[z - 1] - self.pos[0]) / mean_size)
            end = int((self.pos[z] - self.pos[0]) / mean_size)
            for i in range(start, end + 1):
                self._buckets[i].append(z - 1)
        self._bucket_step = mean_size

    def __call__(self, t):
        if t < self.pos[0]:
            q = (self.pos[0] - t) / self.steps[0]
            return self.points[1] + (self.points[1] - self.points[2]) * q
        elif t >= self.pos[-1]:
            q = (t - self.pos[-1]) / self.steps[-1]
            return self.points[-2] + (self.points[-2] - self.points[-3]) * q
        else:
            k = int((t - self.pos[0]) / self._bucket_step)
            a = self.pos[self._buckets[k]]
            ib = np.searchsorted(a, t, side='right') - 1  # left interval boundary
            i = self._buckets[k][ib]
            q = (t - self.pos[i]) / self.steps[i]
            return spline_4p(q, self.points[i:i + 4])
