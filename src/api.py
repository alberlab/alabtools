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

import numpy as np
import os.path
import re
import h5py
try:
   import cPickle as pickle
except:
   import pickle
import warnings
import utils
import matrix

class contactmatrix(object):
    """
    A flexible matrix instant that supports various methods for processing HiC contacts
    
    Parameters
    ----------
    filename : 
    genome : string for a genome e.g.'hg19','mm9'
    resolution : int, the resolution for the hic matrix e.g. 100000
    usechr : list, containing the chromosomes used for generating the matrix
    
    Attributes
    ----------
    matrix : sparse skyline matrix sss_matrix
    idx : utils.genome
    genome : string, for the genome
    resolution : resolution for the contact matrix
    
    """
    
    def __init__(self,filename=None,genome='hg19',resolution=100000,usechr=['#','X']):
        if isinstance(filename,str):
            if not os.path.isfile(filename):
                raise IOError("File %s doesn't exist!\n" % (filename))
            if os.path.splitext(filename)[1] == '.hdf5' or os.path.splitext(filename)[1] == '.hmat':
                self._load_hmat(filename)
            elif os.path.splitext(filename)[1] == '.hcs':
                self._load_hcs(filename)
            elif os.path.splitext(filename)[1] == '.cool':
                self._load_cool(filename,usechr=usechr)
            elif os.path.splitext(filename)[1] == '.hic':
                self._load_hic(filename,resolution=resolution,usechr=usechr)
            else:
                raise ValueError("unrecognized constructor constructor usage")
        else:
            self._build_genome(genomeName=genome,usechr=usechr)
            self._build_index(resolution=resolution)
        
    def _build_genome(self,genomeName,usechr=['#','X'],chroms=None,origin=None,length=None):
        self.genome = utils.genome(genomeName,chroms=chroms,origin=origin,length=length,usechr=usechr)
    
    def _build_index(self,resolution):
        self.idx = self.genome.bininfo(resolution)
        
    def _set_index(self,chrom,start,end,size):
        self.idx = utils.index(chrom=chrom,start=start,end=end,size=size)
        
    def _load_hcs(self,filename):
        h5 = h5py.File(filename)
        self.resolution = h5.attrs["resolution"]
        self._build_genome(h5.attrs["genome"],usechr=['#','X','Y'],
                           chroms = h5["genome"]["chroms"],
                           origin = h5["genome"]["origin"],
                           length = h5["genome"]["length"])
        
        self._set_index(h5["idx"]["chrom"],
                        h5["idx"]["start"],
                        h5["idx"]["end"],
                        h5["idx"]["size"])
        
        self.matrix = matrix.sss_matrix((h5["matrix"]["data"],
                                         h5["matrix"]["indices"],
                                         h5["matrix"]["indptr"],
                                         h5["matrix"]["diagonal"]))
        h5.close()
        
    def _load_cool(self,filename,usechr=['#','X']):        
        h5 = h5py.File(filename)
        genome = h5.attrs['genome-assembly']
        nbins = h5.attrs['nbins']
        self.resolution = h5.attrs['bin-size']
        self._build_genome(genome,usechr=usechr,chroms=h5['chroms']['name'][:],length=h5['chroms']['length'][:])
        self._build_index(self.resolution)
        
        origenome = utils.genome(genome,usechr=['#','X','Y'],silence=True)
        allChrId = [origenome.getchrnum(self.genome[x]) for x in range(len(self.genome))]
        chrIdRange = [[allChrId[0],allChrId[0]+1]]
        for i in allChrId[1:]:
            if i == chrIdRange[-1][1]:
                chrIdRange[-1][1] = i+1
            else:
                chrIdRange.append([i,i+1])
        
        indptr = np.empty(len(self.idx)+1,np.int32)
        indptr[0] = 0
        indices = []
        data = []
        
        #loop through all used chromosomes
        for cil,cih in chrIdRange:
            i0,i1 = (h5['indexes']['chrom_offset'][cil],h5['indexes']['chrom_offset'][cih])
            for r in xrange(i0,i1):
                edge    = h5['indexes']['bin1_offset'][r:r+2]
                #print(edge)
                rowind  = h5['pixels']['bin2_id'][edge[0]:edge[1]]
                rowdata = h5['pixels']['count'][edge[0]:edge[1]]
                newind = []
                newdata = []
                #get data within range
                for cjl,cjh in chrIdRange:
                    if cjl < cil:
                        continue
                    j0,j1 = (h5['indexes']['chrom_offset'][cjl],h5['indexes']['chrom_offset'][cjh])
                    chroffset = self.idx.offset[self.genome.getchrnum(origenome[cjl])]
                    mask = (rowind >= j0) & (rowind < j1)
                    newind.append(rowind[mask] - j0 + chroffset)
                    newdata.append(rowdata[mask])
                #-
                rowind  = np.concatenate(newind,axis=0)
                rowdata = np.concatenate(newdata,axis=0)
                indices.append(rowind)
                data.append(rowdata)
            #--
        #--
        for r in xrange(len(indices)):
            indptr[r+1] = indptr[r] + len(indices[r])
        
        indices = np.concatenate(indices)
        data    = np.concatenate(data)
        
        self.matrix = matrix.sss_matrix((data,indices,indptr),shape=(len(self.idx),len(self.idx)))
        h5.close()
    #-------------------
    def rowsum(self):
        return self.matrix.sum(axis=1)
        
    def columnsum(self):
        return self.matrix.sum(axis=0)
    
    def _getZeroEntry(self):
        self.mask   = np.flatnonzero(self.rowsum() == 0)
    
    def _getMask(self,mask = None):
        if mask is None:
            self._getZeroEntry()
            return 0
        else:
            if isinstance(mask,np.ndarray):
                self.mask = mask
                return 1
            else:
                raise TypeError, "Invalid argument type, numpy.ndarray is required"
    #-------------------
    def maskLowCoverage(self,cutoff = 2):
        """
        Removes "cutoff" percent of bins with least counts

        Parameters
        ----------
        cutoff : int, 0<cutoff<100
            Percent of lowest-counts bins to be removed
        """
        rowsum   = self.rowsum()
        self.mask= np.flatnonzero(rowsum < np.percentile(rowsum[rowsum > 0],cutoff))
        print("{} bins are masked.".format(len(self.mask)))
        
    def krnorm(self,mask = None, force = False, **kwargs):
        """
        using krnorm balacing the matrix (overwriting the matrix!)
        Parameters
        ----------
        mask: list/array 
            mask is a 1-D vector with the same length as the matrix where 1s specify the row/column to be ignored\
            or a 1-D vector specifing the indexes of row/column to be ignored\
            if no mask is given, row/column with rowsum==0 will be automatically detected and ignored
        force: bool
            force to normalize the matrix  
        """
        from norm import bnewt
        if not hasattr(self,"mask"):
            self._getMask(mask)
        x = bnewt(self.matrix,mask=self.mask,check=0,**kwargs)*100
        self.matrix.norm(x) 
    
    def makeSummaryMatrix(self,step=10):
        """
        Filter the matrix to get a lower resolution matrix. New resolution will be resolution/step
        Parameters
        ----------
        step : int
            The length of submatrix to be checked.
        
        Returns
        -------
        A contactmatrix instance of the lower resolution matrix.
        """
        from ._cmtools import TopmeanSummaryMatrix_func
        DimA = len(self.idx)
        
        newMatrix = contactmatrix(filename=None,genome=self.genome.genome,resolution=self.resolution*step)
        DimB = len(newMatrix.idx)
        mapping = np.empty(DimB+1,dtype=np.int32)
        mapping[0] = 0
        
        row = 0
        for i in range(DimB):
            row += step
            if (row > DimA) or (newMatrix.idx.chrom[i] != self.idx.chrom[row]):
                row = 1 + np.flatnonzero(self.idx.chrom == newMatrix.idx.chrom[i])[-1]
            mapping[i+1] = row
        
        Bi = np.empty(int(DimB*(DimB+1)/2),dtype=np.int32)
        Bj = np.empty(int(DimB*(DimB+1)/2),dtype=np.int32)
        Bx = np.empty(int(DimB*(DimB+1)/2),dtype=np.float32)
        TopmeanSummaryMatrix_func(self.matrix.indptr,
                                  self.matrix.indices,
                                  self.matrix.data,
                                  DimA,DimB,
                                  mapping,
                                  Bi,Bj,Bx)
        newMatrix.matrix = matrix.sss_matrix((Bx,(Bi,Bj)))
        return newMatrix
    #=============saveing method
    def save(self,filename,compression='gzip', compression_opts=6):
        """
        save into file
        """
        
        if (filename[-4:] != '.hcs'):
            filename += '.hcs'
            
        h5f = h5py.File(filename, 'w')
        h5f.attrs["genome"] = self.genome.genome
        h5f.attrs["resolution"] = self.resolution
        ggrp = h5f.create_group("genome")
        ggrp.create_dataset("chroms",data=self.genome.chroms, compression=compression,compression_opts=compression_opts)
        ggrp.create_dataset("origin",data=self.genome.origin, compression=compression,compression_opts=compression_opts)
        ggrp.create_dataset("length",data=self.genome.length, compression=compression,compression_opts=compression_opts)
        
        igrp = h5f.create_group("idx")
        igrp.create_dataset("chrom",data=self.idx.chrom, compression=compression,compression_opts=compression_opts)
        igrp.create_dataset("start",data=self.idx.start, compression=compression,compression_opts=compression_opts)
        igrp.create_dataset("end",  data=self.idx.end,   compression=compression,compression_opts=compression_opts)
        igrp.create_dataset("size", data=self.idx.size,  compression=compression,compression_opts=compression_opts)
        
        mgrp = h5f.create_group("matrix")
        mgrp.create_dataset("data",   data=self.matrix.data,    compression=compression,compression_opts=compression_opts)
        mgrp.create_dataset("indices",data=self.matrix.indices, compression=compression,compression_opts=compression_opts)
        mgrp.create_dataset("indptr", data=self.matrix.indptr,  compression=compression,compression_opts=compression_opts)
        mgrp.create_dataset("diagonal",data=self.matrix.diagonal,  compression=compression,compression_opts=compression_opts)
        
        h5f.close()
    #
    
        
