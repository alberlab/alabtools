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
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

class genome(object):
    """
    And instance which holds genome information
    
    Parameters
    ----------
    assembly : str
        The name of the genome. e.g. "hg19" or "mm9"
        
    chroms : array[str] like, optional
        List of chromosome names. e.g ['chr1','chr2' ... 'chrX']
        
    origin : array[int] like, optional
        List of chr region start, default will be arry of zeros
        
    length : array[int] like, optional
        List of chr region length. e.g [249250621, 243199373, ...]
        
    usechr : array[str] like, optional
        Specified chromsome to use. e.g. ['#','X'] or ['1','2','3'..'10']\
        '#' indicates all autosome chromosomes.
        
    Notes
    -----
    
    If ``chroms`` or ``length`` is(are) not specified, genome info will be read from genomes/*.info file
    
    Attributes
    ----------
    
    assembly : str
        Name of genome
    chroms : np.array[string10]
        chromosome name
    origin : np.array[int64]
        chromosome region start
    length : np.array[int64]
        chromosome region length
        
    """
    def __init__(self,assembly,chroms=None,origin=None,length=None,usechr=['#','X'],silence=False):
        if (chroms is None) or (length is None) :
            if not silence:
                print("chroms or length not given, reading from genomes info file.")
            datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'genomes/' + assembly + '.info')
            
            f = loadstream(datafile)
            info = np.genfromtxt(f,dtype=[('chroms','S10'),('length',int)])
            f.close()
            chroms = info['chroms'].astype('S10')
            length = info['length'].astype(int)
            origin = np.zeros(len(length),dtype=int)
        else :
            if origin is None:
                origin = np.zeros(len(length),dtype=int)
            if len(chroms) != len(length) or len(chroms) != len(origin):
                raise RuntimeError("Dimension of chroms and length do not match.")
            chroms = np.array(chroms).astype('S32')
            length = np.array(length).astype(int)
            origin = np.array(origin).astype(int)
            
        choices = np.zeros(len(chroms),dtype=bool)
        for chrnum in usechr:
            if chrnum == '#':
                choices = np.logical_or([re.search('chr[0-9]',c) != None for c in chroms],choices)
            else:
                choices = np.logical_or(chroms == ('chr'+str(chrnum)), choices)
        self.chroms = chroms[choices]
        self.origin = origin[choices]
        self.length = length[choices]
        self.assembly = assembly
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
        utils.index instance
        
        """
        binSize    = [int(math.ceil(float(x)/resolution)) for x in self.length]
        
        chromList  = []
        binLabel   = []
        for i in range(len(self.chroms)):
            chromList += [i for j in range(binSize[i])]
            binLabel  += [j+int(self.origin[i]/resolution) for j in range(binSize[i])]
   
        startList  = [binLabel[j]*resolution for j in range(sum(binSize))]
        endList    = [binLabel[j]*resolution + resolution for j in range(sum(binSize))]
        
        binInfo    = index(chromList,startList,endList,size=binSize)
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
    
#--------------------
class index(object):
    """
    Matrix indexes
    
    Parameters
    ----------
    chrom : list[int32]
        chromosome index starting from 0 (which is chr1)
    start : list[int64]
        bin start
    end : list[int64]
        bin end
    label : list[string10]
        label for each bin
    chrom_sizes : list[int32]
        number of bins of each chromosome
    """
    def __init__(self,chrom,start,end,**kwargs):
        if not(len(chrom) == len(start) and len(start) == len(end)):
            raise RuntimeError("Dimensions do not match.")
        if not isinstance(chrom[0],(int,np.int32,np.int64)):
            raise RuntimeError("chrom should be a list of integers.")
        if not isinstance(start[0],(int,np.int32,np.int64)):
            raise RuntimeError("start should be a list of integers.")
        if not isinstance(end[0],(int,np.int32,np.int64)):
            raise RuntimeError("end should be list of integers.")
        self.chrom = np.array(chrom,dtype=np.int32)
        self.start = np.array(start,dtype=int)
        self.end   = np.array(end,dtype=int)
        
        chrom_sizes = kwargs.pop('chrom_sizes',[])
        if len(chrom_sizes) != len(self.chrom):
            chromList = np.unique(self.chrom)
            self.chrom_sizes = np.zeros(len(chromList),dtype=np.int32)
            for i in chromList:
                self.chrom_sizes[i] = sum(self.chrom == i)
        else:
            self.chrom_sizes = np.array(chrom_sizes,dtype=np.int32)
        
        label = kwargs.pop('label',[])
        if len(label) != len(self.chrom):
            self.label = np.array(['']*len(self.chrom),dtype='S10')
        else:
            self.label = np.array(label,dtype='S10')
        
        copy = kwargs.pop('copy',[])
        if len(copy) != len(self.chrom):
            self.copy = np.zeros(len(self.chrom),dtype=np.int32)
        else:
            self.copy= np.array(copy,dtype=np.int32)
            
        self.offset = np.array([sum(self.chrom_sizes[:i]) for i in range(len(self.chrom_sizes)+1)])
    #-
    
    def __getitem__(self,key):
        return np.array([self.chrom[key],self.start[key],self.end[key],self.label[key]])
    def __len__(self):
        return len(self.chrom)
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
