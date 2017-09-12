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
__version__ = "0.0.4"
__email__   = "nhua@usc.edu"

import numpy as np
import os.path
import re
import h5py
import scipy.sparse
try:
   import cPickle as pickle
except:
   import pickle
import warnings
from six import string_types

from . import utils
from . import matrix

class Contactmatrix(object):
    """
    A flexible matrix instant that supports various methods for processing HiC contacts
    
    Parameters
    ----------
    mat : str or matrix or None
        read matrix from filename, or initialize the matrix from a scipy
        sparse matrix, or a numpy dense matrix, or with an empty matrix
    genome : string for a genome e.g.'hg19','mm9'
    resolution : int, the resolution for the hic matrix e.g. 100000
    usechr : list, containing the chromosomes used for generating the matrix
    
    Attributes
    ----------
    matrix : sparse skyline matrix sss_matrix
    index : utils.Index
    genome : utils.Genome, for the genome
    resolution : resolution for the contact matrix
    
    """
    def __init__(self, mat=None, genome='hg19', resolution=100000, usechr=['#','X']):
        # matrix from file
        if isinstance(mat, string_types):
            if not os.path.isfile(mat):
                raise IOError("File %s doesn't exist!\n" % (mat))
            if os.path.splitext(mat)[1] == '.hdf5' or os.path.splitext(mat)[1] == '.hmat':
                self._load_hmat(mat)
            elif os.path.splitext(mat)[1] == '.hcs':
                self._load_hcs(mat)
            elif os.path.splitext(mat)[1] == '.cool':
                self._load_cool(mat,usechr=usechr)
            elif os.path.splitext(mat)[1] == '.hic':
                self._load_hic(mat,resolution=resolution,usechr=usechr)
            elif os.path.splitext(mat)[1] == '.gz' and os.path.splitext(os.path.splitext(mat)[0])[1] == '.pairs':
                self._load_pairs_bgzip(mat,resolution,usechr=['#','X'])
            else:
                raise ValueError("unrecognized constructor constructor usage")
        else:
            # build genome/index
            if genome is not None:
                self._build_genome(assembly=genome, usechr=usechr)
                if resolution is not None:
                    self._build_index(resolution=resolution)
                elif isinstance(resolution, utils.Index):
                    self.index = resolution

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
                mm = scipy.sparse.csr_matrix(mat)
                diag = mm.diagonal()
                mm.setdiag(0)
                mm.eliminate_zeros()
                self.matrix = matrix.sss_matrix((mm.data, mm.indices, 
                                                 mm.indptr, diag))
            # empty matrix
            elif mat is None:
                if hasattr(self,"index"):
                    self.matrix = matrix.sss_matrix(([], [],
                                                    [0] * (len(self.index)+1),
                                                    [0] * len(self.index)))
            else:
                raise(ValueError, 'Invalid mat argument')
            #assert(self.matrix.shape[0] == self.matrix.shape[1] == len(self.index)) 
            #-
        
    def _build_genome(self, assembly, usechr=['#','X'], chroms=None, 
                      origins=None, lengths=None):
        self.genome = utils.Genome(assembly,chroms=chroms,origins=origins,lengths=lengths,usechr=usechr)
    
    def _build_index(self,resolution):
        self.index = self.genome.bininfo(resolution)
        self.resolution = resolution
        
    def _set_index(self,chrom,start,end,label=[],copy=[],chrom_sizes=[]):
        self.index = utils.Index(chrom=chrom,start=start,end=end,label=label,copy=copy,chrom_sizes=chrom_sizes)
        
    def _load_hcs(self,filename):
        h5 = h5py.File(filename)
        self.resolution = h5.attrs["resolution"]
        self.genome = utils.Genome(h5)
        self.index = utils.Index(h5)
        
        self.matrix = matrix.sss_matrix((h5["matrix"]["data"],
                                         h5["matrix"]["indices"],
                                         h5["matrix"]["indptr"],
                                         h5["matrix"]["diagonal"]))
        h5.close()
        
    def _load_cool(self,filename,usechr=['#','X']):        
        h5 = h5py.File(filename)
        assembly = h5.attrs['genome-assembly']
        nbins = h5.attrs['nbins']
        self.resolution = h5.attrs['bin-size']
        self._build_genome(assembly,usechr=usechr,chroms=h5['chroms']['name'][:],lengths=h5['chroms']['length'][:])
        self._build_index(self.resolution)
        
        origenome = utils.Genome(assembly,usechr=['#','X','Y'],silence=True)
        allChrId = [origenome.getchrnum(self.genome[x]) for x in range(len(self.genome))]
        chrIdRange = [[allChrId[0],allChrId[0]+1]]
        for i in allChrId[1:]:
            if i == chrIdRange[-1][1]:
                chrIdRange[-1][1] = i+1
            else:
                chrIdRange.append([i,i+1])
        
        indptr = np.empty(len(self.index)+1,np.int32)
        indptr[0] = 0
        indices = []
        data = []
        
        #loop through all used chromosomes
        for cil,cih in chrIdRange:
            i0,i1 = (h5['indexes']['chrom_offset'][cil],h5['indexes']['chrom_offset'][cih])
            for r in range(i0,i1):
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
                    chroffset = self.index.offset[self.genome.getchrnum(origenome[cjl])]
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
        for r in range(len(indices)):
            indptr[r+1] = indptr[r] + len(indices[r])
        
        indices = np.concatenate(indices)
        data    = np.concatenate(data)
        
        self.matrix = matrix.sss_matrix((data,indices,indptr),shape=(len(self.index),len(self.index)))
        h5.close()
        
    def _load_pairs_bgzip(self,filename,resolution,usechr=['#','X']):
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

        self.genome = utils.Genome(assembly,chroms=chroms,lengths=length,usechr=usechr)
        self.index = self.genome.bininfo(resolution)

        indptr = np.zeros(len(self.index)+1,dtype=np.int32)
        indices = []
        data = []
        for i in range(len(self.index)):
            chrom1 = self.genome.getchrom(self.index.chrom[i])
            
            start1 = self.index.start[i]
            end1   = self.index.end[i]
            print(chrom1,start1,end1)
            for j in range(i,len(self.index)):
                chrom2 = self.genome.getchrom(self.index.chrom[j])
                start2 = self.index.start[j]
                end2   = self.index.end[j]
                #querystr='{}:{}-{}|{}:{}-{}'.format(chrom1, start1, end1, chrom2, start2, end2)
                #it = tb.querys2D(querystr)
                it = tb.query2D(chrom1, start1, end1, chrom2, start2, end2, 1)
                n = len([x for x in it])
                
                if n > 0:
                    indices.append(j)
                    data.append(n)
            #-
            indptr[i+1] = len(indices)
        #-
        self.matrix = matrix.sss_matrix((data,indices,indptr),shape=(len(self.index),len(self.index)))
      
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
                raise TypeError("Invalid argument type, numpy.ndarray is required")
    
    def __len__(self):
        return self.index.__len__()
    
    def __getIntra(self,start,stop):
        uniqueChroms = np.unique(self.index.chrom[start:stop])
        usechr = []
        for c in uniqueChroms:
            chrom = self.genome.getchrom(c)
            usechr.append(chrom[3:].decode())
        
        newMatrix = Contactmatrix(None,genome=None,resolution=None)
        newMatrix._build_genome(self.genome.assembly,
                                usechr=usechr,
                                chroms = self.genome.chroms,
                                origins = self.genome.origins,
                                lengths = self.genome.lengths)
        
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
        
        submat = self.matrix.csr[start:stop,start:stop]
        
        newMatrix.matrix = matrix.sss_matrix((submat.data,
                                              submat.indices,
                                              submat.indptr,
                                              self.matrix.diagonal[start:stop]
                                              ))
        return newMatrix
        
    def __getitem__(self,key):
        if isinstance(key,(string_types,bytes)):
            if isinstance(key,str):
                chrnum = self.genome.getchrnum(key.encode())
            else:
                chrnum = self.genome.getchrnum(key)
            chrstart = np.flatnonzero(self.index.chrom == chrnum)[0]
            chrend   = np.flatnonzero(self.index.chrom == chrnum)[-1]
            return self.__getIntra(chrstart,chrend+1)
        
        elif isinstance(key,slice):
            if key.step is None: step = 1
            else: step = key.step
            if key.start is None: start = 0
            else: start = key.start
            if key.stop is None: stop = len(self)
            else: stop = key.stop

            if start < 0: start += len(self)
            if stop < 0: stop += len(self)
            if start > len(self) or stop > len(self) :  raise IndexError("The index out of range")
            return self.__getIntra(start,stop)
        else:
            raise TypeError("Invalid argument type")
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
        self.mask= np.flatnonzero(rowsum <= np.percentile(rowsum[rowsum > 0],cutoff))
        print("{} bins are masked.".format(len(self.mask)))
    
    def normalize(self,bias):
        """
        norm matrix by bias vector
        
        Parameters
        ----------
        bias : np.array column vector
        
        """
        self.matrix.normalize(np.array(bias))
        
    #-
    
    def getKRnormBias(self, mask = None, **kwargs):
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
        if not hasattr(self,"mask"):
            self._getMask(mask)
        x = bnewt(self.matrix, mask=self.mask, check=0, **kwargs) * 100
        return x
        
    def krnorm(self, mask = None, force = False, **kwargs):
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
    
    def makeSummaryMatrix(self,step=10):
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
        from ._cmtools import TopmeanSummaryMatrix_func
        DimA = len(self.index)
        
        newMatrix = Contactmatrix(None,genome=None,resolution=None)
        newMatrix._build_genome(self.genome.assembly,
                                usechr=['#','X','Y'],
                                chroms = self.genome.chroms,
                                origins = self.genome.origins,
                                lengths = self.genome.lengths)
        
        if isinstance(step, string_types):
            tadDef = np.genfromtxt(step,dtype = None)
            chrom  = tadDef['f0']
            start  = tadDef['f1']
            end    = tadDef['f2']
            label  = tadDef['f3']
            nchrom = np.array([newMatrix.genome.getchrnum(x) for x in chrom])
            
            f_tadDef = np.sort(np.rec.fromarrays([nchrom,start,end,label]),order=['f0','f1'])
            chrom  = f_tadDef['f0']
            start  = f_tadDef['f1']
            end    = f_tadDef['f2']
            label  = f_tadDef['f3']
            newMatrix._set_index(chrom,start,end,label)
        else:
            newMatrix._build_index(self.resolution*step)
            
        DimB = len(newMatrix.index)
        mapping = np.empty(DimB+1,dtype=np.int32)
        mapping[0] = 0
        
        row = 0
        for i in range(DimB):
            incStep = int((newMatrix.index.end[i] - newMatrix.index.start[i]) / self.resolution)
            row += incStep
            
            if (row > DimA) or (newMatrix.index.chrom[i] != self.index.chrom[row]):
                #row = 1 + np.flatnonzero(self.index.chrom == newMatrix.index.chrom[i])[-1]
                row = self.index.offset[self.index.chrom[row-incStep]+1]
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
        newMatrix.resolution = -1
        return newMatrix
    
    def fmaxScaling(self,fmax,force=False):
        """
        use fmax to generate probability matrix
        
        """
        if isinstance(fmax,float) or isinstance(fmax,np.float32) or isinstance(fmax,int):
            self.matrix.csr.data /= fmax
            self.matrix.diagonal /= fmax
            self.matrix.csr.data = self.matrix.csr.data.clip(max=1)
            self.matrix.diagonal = self.matrix.diagonal.clip(max=1)
            self.matrix.data = self.matrix.csr.data
            
        
    def iterativeScaling(self,domain=10,averageContact=24,theta=0.001,tol=0.01):
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
        originalData = np.copy(self.matrix.csr.data)
        originalDiag = np.copy(self.matrix.diagonal)
        fmax = self.rowsum().mean()/(averageContact+0.2)
        while abs(average - averageContact)/averageContact > tol :
            print(fmax,average)
            self.matrix.csr.data = np.copy(originalData)
            self.matrix.diagonal = np.copy(originalDiag)
            self.fmaxScaling(fmax,force=True)
            
            newMat = self.makeSummaryMatrix(domain)
            
            newMat.matrix.data[newMat.matrix.data < theta] = 0
            
            rowsums = newMat.rowsum()
            rowsums = rowsums[rowsums > 0]
            average = rowsums.mean()
            fmax = fmax/averageContact*average
        #==
        
        newMat = self.makeSummaryMatrix(domain)
        
        print("{} {}".format(fmax,average))
        return newMat
        
    #=============plotting method
    
    def plot(self,filename,log=False,**kwargs):
        from .plots import plotmatrix,red
        
        mat = self.matrix.toarray()
        if log:
            mat = np.log(mat)
        
        cmap = kwargs.pop('cmap',red)
        
        plotmatrix(filename,mat,cmap=cmap,**kwargs)
        
    #=============saveing method
    def save(self,filename,compression='gzip', compression_opts=6):
        """
        save into file
        """
        
        if (filename[-4:] != '.hcs'):
            filename += '.hcs'
            
        h5f = h5py.File(filename, 'w')
        h5f.attrs["resolution"] = self.resolution
        h5f.attrs["version"] = __version__
        h5f.attrs["nbin"] = len(self.index)
        self.genome.save(h5f,compression=compression,compression_opts=compression_opts)
        
        self.index.save(h5f,compression=compression,compression_opts=compression_opts)
        
        mgrp = h5f.create_group("matrix")
        mgrp.create_dataset("data",   data=self.matrix.data,    compression=compression,compression_opts=compression_opts)
        mgrp.create_dataset("indices",data=self.matrix.indices, compression=compression,compression_opts=compression_opts)
        mgrp.create_dataset("indptr", data=self.matrix.indptr,  compression=compression,compression_opts=compression_opts)
        mgrp.create_dataset("diagonal",data=self.matrix.diagonal,  compression=compression,compression_opts=compression_opts)
        
        h5f.close()
    #
    
        
