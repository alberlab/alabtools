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

__author__  = "Nan Hua and Guido Polles"

__license__ = "GPL"
__version__ = "0.0.3"
__email__   = "nhua@usc.edu"


import os
import re
import math
import numpy as np
import subprocess
import itertools
import h5py
import json
import sys
from six import string_types

if (sys.version_info > (3, 0)):
    # python 3.x
    def unicode(s):
        if isinstance(s, bytes):
            return s.decode()
        return s
    from io import StringIO
else:
    # python 2.x 
    from cStringIO import StringIO

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
    
    def __init__(self,assembly,chroms=None,origins=None,lengths=None,usechr=['#','X'],silence=False):
        if isinstance(assembly,h5py.File):
            chroms  = np.array(assembly["genome"]["chroms"][:], CHROMS_DTYPE)
            origins = assembly["genome"]["origins"]
            lengths = assembly["genome"]["lengths"]
            assembly = assembly["genome"]["assembly"].value
            usechr = ['#','X','Y']
            
        if (chroms is None) or (lengths is None) :
            if not silence:
                print("chroms or lengths not given, reading from genomes info file.")
            datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)),"genomes/" + assembly + ".info")
            info = np.genfromtxt(datafile,dtype=[("chroms",CHROMS_DTYPE),("lengths",LENGTHS_DTYPE)])
            chroms = info["chroms"].astype(CHROMS_DTYPE)
            lengths = info["lengths"].astype(LENGTHS_DTYPE)
            origins = np.zeros(len(lengths),dtype=ORIGINS_DTYPE)
        else :
            if origins is None:
                origins = np.zeros(len(lengths),dtype=ORIGINS_DTYPE)
            if len(chroms) != len(lengths) or len(chroms) != len(origins):
                raise RuntimeError("Dimension of chroms and lengths do not match.")
            
            chroms = np.array(chroms,dtype=CHROMS_DTYPE)
            lengths = np.array(lengths,dtype=LENGTHS_DTYPE)
            origins = np.array(origins,dtype=ORIGINS_DTYPE)
            
        choices = np.zeros(len(chroms),dtype=bool)
        for chrnum in usechr:
            if chrnum == '#':
                choices = np.logical_or([re.search("chr[\d]+$",c) != None for c in chroms], choices)
            else:
                choices = np.logical_or(chroms == ("chr%s" % chrnum), choices)
        self.chroms = chroms[choices] # convert to unicode for python2/3 compatibility
        self.origins = origins[choices]
        self.lengths = lengths[choices]
        self.assembly = str(assembly)
    #-

    def __eq__(self, other):
        return ( 
            np.all(self.chroms == other.chroms) and
            np.all(self.origins == other.origins) and
            np.all(self.lengths == other.lengths) and
            np.all(self.assembly == other.assembly) 
        )
    
    def bininfo(self,resolution):
        
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
        
        binSize    = [int(math.ceil(float(x)/resolution)) for x in self.lengths]
        
        chromList  = []
        binLabel   = []
        for i in range(len(self.chroms)):
            chromList += [i for j in range(binSize[i])]
            binLabel  += [j+int(self.origins[i]/resolution) for j in range(binSize[i])]
   
        startList  = [binLabel[j]*resolution for j in range(sum(binSize))]
        endList    = [binLabel[j]*resolution + resolution for j in range(sum(binSize))]
        
        binInfo    = Index(chromList,startList,endList,chrom_sizes=binSize)
        return binInfo
    
    def getchrnum(self, chrom):
        
        findidx = np.flatnonzero(self.chroms==unicode(chrom))
    
        if len(findidx) == 0:
            return None
        else:
            return findidx[0]
  
    def getchrom(self, chromNum):
        assert isinstance(chromNum,(int,np.int32,np.int64))
        return self.chroms[chromNum]
    
    def __getitem__(self, key):
        if isinstance(key, (int, np.int32, np.int64)):
            return self.getchrom(key)
    def __len__(self):
        return len(self.chroms)
    def __repr__(self):
        represent = "Genome Assembly: " + self.assembly + '\n'
        for i in range(len(self.chroms)):
            represent += (self.chroms[i].astype(str) + '\t' + str(self.origins[i]) + '-' + str(self.origins[i]+self.lengths[i]) + '\n')
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
            ggrp.create_dataset("chroms", data=np.array(self.chroms, dtype='S10'), # hdf5 does not like unicode 
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
#--------------------


class Index(object):
    
    """
    Matrix/System indexes. Maps matrix bins or model beads to
    genomic regions.
    
    Parameters
    ----------
    chrom : list[int32]
        numeric chromosome id (starting from 0) for each bin/bead.
        Es.: 0 -> chr1, 1 -> chr2, ..., 22 -> chrX  
    start : list[int32]
        genomic starting positions of each bin (in bp, with respect to the 
        chromosome start)
    end : list[int32]
        genomic ending positions of each bin (in bp, with respect to the 
        chromosome start) 
    label : list[string10]
        label for each bin (usually, 'CEN', 'gap', 'domain', although it 
        can be any string of less than 10 characters)
    copy : list[int32], optional
        In systems of beads, there may be multiple indistinguishable 
        copies of the same chromosome in the system. The copy vector specifies
        which copy of the chromosome each bead maps to. If not specified,
        is computed assuming non-contiguous groups of beads with the same
        `chrom` value belong to different copies.
        Each bead mapping to the same (chrom, start, end) tuple should,
        in general, have a different copy value. 
    chrom_sizes : list[int32], optional
        number of bins/beads in each chromosome. It is useful to specify it if 
        two copies of the same chromosome appear as contiguous in the index.
        If not specified, it is automatically computed assuming non-contiguous 
        groups of beads with the same `chrom` value belong to different copies. 
    """

    def __init__(self, chrom=[], start=[], end=[], label=[], copy=[], chrom_sizes=[], **kwargs):
        
        if isinstance(chrom, string_types):
            try:
                # tries to open a hdf5 file
                chrom = h5py.File(chrom, 'r')
            except IOError:
                # if it is not a hdf5 file, try to read as text BED

                cols=[
                    ('chrom', CHROMS_DTYPE), # note that this is a string type
                    ('start', START_DTYPE), 
                    ('end', END_DTYPE), 
                    ('label', LABEL_DTYPE), 
                    ('copy', COPY_DTYPE)
                ]

                usecols = kwargs.get('usecols', None)
                if usecols is None:
                    # try to determine if it is a 3, 4 or 5 columns bed
                    with open(chrom) as f:
                        line = ''
                        while line == '' or line[0] == '#':
                            line = f.readline().strip()
                        ncols = len(line.split())
                        if ncols < 3:
                            raise ValueError('bed file appears to have less than'
                                             ' 3 columns')
                        usecols = range(ncols)
                ncols = len(usecols)
                
                # define columns fields
                
                data = np.genfromtxt(chrom, usecols=usecols, dtype=cols[:ncols])

                genome = kwargs.get('genome', None)
                if genome is None:
                    # transform chrom names to integers ids
                    i, nmap = 0, {}
                    for c in data['chrom']:
                        if c not in nmap:
                            nmap[c] = i
                            i += 1
                    chrom = [ nmap[c] for c in data['chrom'] ] 
                else:
                    if not isinstance(genome, Genome):
                        genome = Genome(genome)
                    chrom = np.array([genome.getchrnum(c) for c in data['chrom']])

                # set the variables for further processing
                
                start = data['start'] 
                end   = data['end']
                if 'label' in data.dtype.names:
                    label = data['label']
                    
                if 'copy' in data.dtype.names:
                    copy = data['copy']
                
        if isinstance(chrom, h5py.File):
            h5f = chrom
            chrom = h5f["index"]["chrom"]
            start = h5f["index"]["start"]
            end   = h5f["index"]["end"]
            copy  = h5f["index"]["copy"]
            label = np.array(h5f["index"]["label"][:], LABEL_DTYPE)
            chrom_sizes = h5f["index"]["chrom_sizes"]
            try:
                # try to load the copy index
                tmp = json.loads(h5f["index"]["copy_index"][()])
                # it may happen that in json dump/loading keys are considered
                # as strings.
                self.copy_index = { int(k): v for k, v in tmp.items() }
            except (KeyError, ValueError):
                pass
        
        if not(len(chrom) == len(start) == len(end)):
            raise RuntimeError("Dimensions do not match.")
        if len(chrom) and not isinstance(chrom[0], (int, np.int32, np.int64, np.uint32, np.uint64)):
            raise RuntimeError("chrom should be a list of integers.")
        if len(start) and not isinstance(start[0], (int, np.int32, np.int64, np.uint32, np.uint64)):
            raise RuntimeError("start should be a list of integers.")
        if len(end) and not isinstance(end[0], (int, np.int32, np.int64, np.uint32, np.uint64)):
            raise RuntimeError("end should be list of integers.")
        self.chrom = np.array(chrom, dtype=CHROM_DTYPE)
        self.start = np.array(start, dtype=START_DTYPE)
        self.end   = np.array(end, dtype=END_DTYPE)
        
        if len(chrom_sizes) == 0 and len(self.chrom) != 0: # chrom sizes have not been computed yet
            chrom_sizes = [len(list(g)) for _, g in itertools.groupby(self.chrom)]
        
        self.chrom_sizes = np.array(chrom_sizes, dtype=CHROM_SIZES_DTYPE)

        self.num_chrom = len(self.chrom_sizes)
        
        if len(label) != len(self.chrom):
            self.label = np.array([""] * len(self.chrom), dtype=LABEL_DTYPE)
        else:
            self.label = np.array(label, dtype=LABEL_DTYPE)
        
        if len(copy) != len(self.chrom):
            self.copy = np.zeros(len(self.chrom), dtype=COPY_DTYPE)
            self._compute_copy_vec()
        else:
            self.copy= np.array(copy, dtype=COPY_DTYPE)
            
        self.offset = np.array([sum(self.chrom_sizes[:i]) 
                                for i in range(len(self.chrom_sizes) + 1)])

        if not hasattr(self, 'copy_index'):
            self._compute_copy_index()

    #-


    def __eq__(self, other):
        return ( 
            np.all(self.chrom == other.chrom) and
            np.all(self.start == other.start) and
            np.all(self.end == other.end) and
            np.all(self.label == other.label) and
            np.all(self.copy == other.copy) 
        )
    
    def __getitem__(self,key):
        return np.rec.fromarrays((self.chrom[key], 
                                  self.start[key],
                                  self.end[key],
                                  self.label[key]),
                                 dtype=[("chrom",CHROM_DTYPE),
                                        ("start",START_DTYPE),
                                        ("end",END_DTYPE),
                                        ("label",LABEL_DTYPE)])
    
    def __len__(self):
        return len(self.chrom)

    def __repr__(self):
        return '<alabtools.Index: %d chroms, %d segments>' % (
            self.num_chrom, 
            len(self.chrom) 
        )

    def _compute_copy_vec(self):
        if len(self.copy) == 0:
            return
        chrom_ids = set(self.chrom)
        copy_no = { c: -1 for c in chrom_ids}
        copy_no[self.chrom[0]] = 0
        self.copy[0] = 0
        for i in range(1, len(self.chrom)):
            cc = self.chrom[i]
            if self.chrom[i - 1] != cc:
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
        
    def save(self,h5f,compression="gzip", compression_opts=6):

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
#--------------------


def loadstream(filename):

    """
    Convert a file location, return a file handle
    zipped file are automaticaly unzipped using stream
    """
    
    if not os.path.isfile(filename):
        raise IOError("File %s doesn't exist!\n" % (filename))
    if os.path.splitext(filename)[1] == '.gz':
        p = subprocess.Popen(["zcat", filename], stdout = subprocess.PIPE)
        f = StringIO(p.communicate()[0])
    elif os.path.splitext(filename)[1] == '.bz2':
        p = subprocess.Popen(["bzip2 -d", filename], stdout = subprocess.PIPE)
        f = StringIO(p.communicate()[0])
    else:
        f = open(filename,'r')
    return f


def make_diploid(index):
    didx = {}
    for k in ['chrom', 'start', 'end', 'label']:
        didx[k] = np.concatenate([index.__dict__[k], index.__dict__[k]])
    didx['copy'] = np.concatenate([index.__dict__['copy'], 
                                   index.__dict__['copy'] + 1 ])
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
    nend   = []
    ncopy  = []
    nlabel = []
    csizes = []
    for z in range(max(copies)):
        for cid, cnum in zip(chroms, copies):
            if z < cnum:
                idxs = np.where(index.chrom == cid)[0]
                nchrom.append(index.chrom[idxs])
                nstart.append(index.start[idxs])
                nend.append(index.end[idxs])
                nlabel.append(index.label[idxs])
                ncopy.append(np.array([z]*len(idxs)))
                csizes.append(len(idxs))

    nchrom = np.concatenate(nchrom)
    nstart = np.concatenate(nstart)
    nend = np.concatenate(nend)
    ncopy = np.concatenate(ncopy)
    nlabel = np.concatenate(nlabel)
    
    return Index(nchrom, nstart, nend, copy=ncopy, label=nlabel, chrom_sizes=csizes)


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


_ftpi = 4./3. * np.pi
DEFAULT_NUCLEAR_VOLUME = _ftpi * (5000**3)
def compute_radii(index, occupancy=0.2, volume=DEFAULT_NUCLEAR_VOLUME):
    sizes = [b.end - b.start for b in index]
    totsize = sum(sizes)
    prefactor = volume * occupancy / (_ftpi * totsize)
    rr = [(prefactor*sz)**(1./3.) for sz in sizes]
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
                self.bsize*z : 
                min(self.n, self.bsize * (z + 1))
            ][()]
        return self.batch[key % self.bsize]

    def __len__(self):
        return self.ds.__len__()
