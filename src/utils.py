# Copyright (C) 2015 University of Southern California and
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
from cStringIO import StringIO


class genome(object):
    """
    And instance which holds genome information
    
    Parameters
    ----------
    genomeName : str
        The name of the genome. e.g. "hg19" or "mm9"
    chroms : array[str] like, optional
        List of chromosome names. e.g ['chr1','chr2' ... 'chrX']
    length : array[int] like, optional
        List of chromosome length. e.g [249250621, 243199373, ...]
    usechr : array[str] like, optional
        Specified chromsome to use. e.g. ['#','X'] or ['1','2','3'..'10']
        '#' indicates all autosome chromosomes.
        
    Notes
    -----
    
    If ``chroms`` or ``length`` is(are) not specified, genome info will be read from genomes/*.info file
    
    Attributes
    ----------
    
    genome : str
        Name of genome
    chroms : np.array[str]
        chromosome name
    length : np.array[int]
        chromosome length
        
    """
    def __init__(self,genomeName,chroms=None,length=None,usechr=['#','X']):
        if (chroms is None) or (length is None) :
            print("chroms or length not given, reading from genomes info file.")
            datafile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'genomes/' + genomeName + '.info')
            
            f = loadstream(datafile)
            info = np.genfromtxt(f,dtype=[('chroms','S32'),('length',int)])
            f.close()
            chroms = info['chroms'].astype('S32')
            length = info['length'].astype(int)
        else :
            if len(chroms) != len(length):
                raise RuntimeError, "Dimention of chroms and length not met."
            chroms = np.array(chroms).astype('S32')
            length = np.array(length).astype(int)
        choices = np.zeros(len(chroms),dtype=bool)
        for chrnum in usechr:
            if chrnum == '#':
                choices = np.logical_or([re.search('chr[0-9]',c) != None for c in chroms],choices)
            else:
                choices = np.logical_or(chroms == ('chr'+str(chrnum)), choices)
        self.chroms = chroms[choices]
        self.length = length[choices]
        self.genome = genomeName
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
            binLabel  += [j for j in range(binSize[i])]
   
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
        assert isinstance(chromNum,int)
        return self.chroms[chromNum]
    
    def __getitem__(self,key):
        if isinstance(key,int):
            return self.getchrom(key)
    def __len__(self):
        return len(self.chroms)
    
#--------------------
class index(object):
    """
    Matrix indexes
    
    Parameters
    ----------
    chrom : list[int]
        chromosome index starting from 0 (which is chr1)
    start : list[int]
        bin start
    end : list[int]
        bin end
    """
    def __init__(self,chrom,start,end,**kwargs):
        if not(len(chrom) == len(start) and len(start) == len(end)):
            raise RuntimeError, "Dimention not met."
        if not isinstance(chrom[0],int):
            raise RuntimeError, "chrom should be list of integers."
        if not isinstance(start[0],int):
            raise RuntimeError, "start should be list of integers."
        if not isinstance(end[0],int):
            raise RuntimeError, "end should be list of integers."
        self.chrom = np.array(chrom,dtype=int)
        self.start = np.array(start,dtype=int)
        self.end   = np.array(end,dtype=int)
        
        size = kwargs.pop('size',[])
        if len(size) != len(self.chrom):
            chromList = np.unique(self.chrom)
            self.size = np.zeros(len(chromList),dtype=int)
            for i in chromList:
                self.size[i] = sum(self.chrom == i)
        else:
            self.size = np.array(size,dtype=int)
        
        self.offset = np.array([sum(self.size[:i]) for i in range(len(self.size))])
    #-
    
    def __getitem__(self,key):
        return np.array([self.chrom[key],self.start[key],self.end[key]])
    
#--------------------
def loadstream(filename):
    """
    Convert a file location, return a file handle
    zipped file are automaticaly unzipped using stream
    """
    if not os.path.isfile(filename):
        raise IOError,"File %s doesn't exist!\n" % (filename)
    if os.path.splitext(filename)[1] == '.gz':
        p = subprocess.Popen(["zcat", filename], stdout = subprocess.PIPE)
        f = StringIO(p.communicate()[0])
    elif os.path.splitext(filename)[1] == '.bz2':
        p = subprocess.Popen(["bzip2 -d", filename], stdout = subprocess.PIPE)
        f = StringIO(p.communicate()[0])
    else:
        f = open(filename,'r')
    return f
