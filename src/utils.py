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

__author__  = "Nan Hua"

__license__ = "GPL"
__version__ = "0.0.3"
__email__   = "nhua@usc.edu"


import os
import re
import math
import numpy as np
import subprocess
import warnings
import h5py
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

CHROMS_DTYPE = np.dtype('S10')
ORIGINS_DTYPE = np.int32
LENGTHS_DTYPE = np.int32

CHROM_DTYPE = np.int32
START_DTYPE = np.int32
END_DTYPE = np.int32
COPY_DTYPE = np.int32
LABEL_DTYPE = np.dtype('S10')
CHROM_SIZES_DTYPE = np.int32

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
            chroms  = assembly["genome"]["chroms"]
            origins = assembly["genome"]["origins"]
            lengths = assembly["genome"]["lengths"]
            assembly = assembly["genome"]["assembly"].value
            usechr = ['#','X','Y']
            
        if (chroms is None) or (lengths is None) :
            if not silence:
                print("chroms or lengths not given, reading from genomes info file.")
            datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)),"genomes/" + assembly + ".info")
            
            f = loadstream(datafile)
            info = np.genfromtxt(f,dtype=[("chroms",CHROMS_DTYPE),("lengths",LENGTHS_DTYPE)])
            f.close()
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
                choices = np.logical_or([re.search("chr[0-9]",c) != None for c in chroms],choices)
            else:
                choices = np.logical_or(chroms == ("chr"+str(chrnum)), choices)
        self.chroms = chroms[choices]
        self.origins = origins[choices]
        self.lengths = lengths[choices]
        self.assembly = str(assembly)
    #-
    
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
    
    def getchrnum(self,chrom):
        findidx = np.flatnonzero(self.chroms==chrom)
    
        if len(findidx) == 0:
            return None
        else:
            return findidx[0]
  
    def getchrom(self,chromNum):
        assert isinstance(chromNum,(int,np.int32,np.int64))
        return self.chroms[chromNum]
    
    def __getitem__(self,key):
        if isinstance(key,(int,np.int32,np.int64)):
            return self.getchrom(key)
    def __len__(self):
        return len(self.chroms)
    def __repr__(self):
        represent = "Genome Assembly: " + self.assembly + '\n'
        for i in range(len(self.chroms)):
            represent += self.chroms[i] + '\t' + str(self.origins[i]) + '-' + str(self.origins[i]+self.lengths[i]) + '\n'
        return represent
        
    def save(self,h5f,compression="gzip", compression_opts=6):
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
        assert(h5f,h5py.File)
        try:
            ggrp = h5f.create_group("genome")
        except:
            ggrp = h5f["genome"]
            
        ggrp.create_dataset("assembly",data=self.assembly)
        ggrp.create_dataset("chroms",data=self.chroms, compression=compression,compression_opts=compression_opts)
        ggrp.create_dataset("origins",data=self.origins, compression=compression,compression_opts=compression_opts)
        ggrp.create_dataset("lengths",data=self.lengths, compression=compression,compression_opts=compression_opts)
        
#--------------------
class Index(object):
    """
    Matrix indexes
    
    Parameters
    ----------
    chrom : list[int32]
        chromosome index starting from 0 (which is chr1)
    start : list[int32]
        bin start
    end : list[int32]
        bin end
    copy : list[int32]
        ploidy of each bin
    label : list[string10]
        label for each bin
    chrom_sizes : list[int32]
        number of bins of each chromosome
    """
    def __init__(self,chrom,start,end,**kwargs):
        if isinstance(chrom,h5py.File):
            start = chrom["index"]["start"]
            end   = chrom["index"]["end"]
            label = chrom["index"]["label"]
            copy  = chrom["index"]["copy"]
            chrom_sizes = chrom["index"]["chrom_sizes"]
            chrom = chrom["index"]["chrom"]
        else:
            label = []
            copy = []
            chrom_sizes = []
        
        if not(len(chrom) == len(start) and len(start) == len(end)):
            raise RuntimeError("Dimensions do not match.")
        if not isinstance(chrom[0],(int,np.int32,np.int64)):
            raise RuntimeError("chrom should be a list of integers.")
        if not isinstance(start[0],(int,np.int32,np.int64)):
            raise RuntimeError("start should be a list of integers.")
        if not isinstance(end[0],(int,np.int32,np.int64)):
            raise RuntimeError("end should be list of integers.")
        self.chrom = np.array(chrom,dtype=CHROM_DTYPE)
        self.start = np.array(start,dtype=START_DTYPE)
        self.end   = np.array(end,dtype=END_DTYPE)
        
        chrom_sizes = kwargs.pop("chrom_sizes",chrom_sizes)
        if len(chrom_sizes) != len(self.chrom):
            chromList = np.unique(self.chrom)
            self.chrom_sizes = np.zeros(len(chromList),dtype=CHROM_SIZES_DTYPE)
            for i in chromList:
                self.chrom_sizes[i] = sum(self.chrom == i)
        else:
            self.chrom_sizes = np.array(chrom_sizes,dtype=CHROM_SIZES_DTYPE)
        
        label = kwargs.pop("label",label)
        if len(label) != len(self.chrom):
            self.label = np.array([""]*len(self.chrom),dtype=LABEL_DTYPE)
        else:
            self.label = np.array(label,dtype=LABEL_DTYPE)
        
        copy = kwargs.pop("copy",copy)
        if len(copy) != len(self.chrom):
            self.copy = np.zeros(len(self.chrom),dtype=COPY_DTYPE)
        else:
            self.copy= np.array(copy,dtype=COPY_DTYPE)
            
        self.offset = np.array([sum(self.chrom_sizes[:i]) for i in range(len(self.chrom_sizes)+1)])
    #-
    
    def __getitem__(self,key):
        return np.rec.fromarrays((self.chrom[key],self.start[key],self.end[key],self.label[key]),
                                 dtype=[("chrom",CHROM_DTYPE),("start",START_DTYPE),("end",END_DTYPE),("label",LABEL_DTYPE)])
    
    def __len__(self):
        return len(self.chrom)
    def __repr__(self):
        return self.chrom_sizes.__repr__()
    
    def save(self,h5f,compression="gzip", compression_opts=6):
        """
        Save index information into a hd5f file handle. The information will be saved as a group of datasets:
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
        assert(h5f,h5py.File)
        try:
            igrp = h5f.create_group("index")
        except:
            igrp = h5f["index"]
            
        igrp.create_dataset("chrom",data=self.chrom, compression=compression,compression_opts=compression_opts)
        igrp.create_dataset("start",data=self.start, compression=compression,compression_opts=compression_opts)
        igrp.create_dataset("end",  data=self.end,   compression=compression,compression_opts=compression_opts)
        igrp.create_dataset("label",data=self.label, compression=compression,compression_opts=compression_opts)
        igrp.create_dataset("copy", data=self.copy,  compression=compression,compression_opts=compression_opts)
        igrp.create_dataset("chrom_sizes", data=self.chrom_sizes,  compression=compression,compression_opts=compression_opts)
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
