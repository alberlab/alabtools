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

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, dia_matrix, isspmatrix_csr
from scipy.sparse.sputils import isshape
from scipy.sparse._sparsetools import coo_tocsr
class sss_matrix(object):
    """
    Sparse Skyline Format
    
    Inherited from CSR. SSS format is used for storing sparse symmetric matrices. 
    The diagonal is stored in a separate (full) vector and the strict upper triangle is stored in CSR format.
    
    Attributes
    ----------
    
    csr : Scipy.sparse Compressed Sparse Row matrix
    data : csr_matrix.data
    indptr : csr_matrix.indptr
    indices : csr_matrix.indices
    
    
    
    """
    def __init__(self,arg1,shape=None,dtype=np.float32,copy=False,mem=True):
        if isinstance(arg1,str):
            pass
        elif isinstance(arg1, tuple):
            if isshape(arg1):
                # It's a tuple of matrix dimensions (M, N)
                # create empty matrix
                self.csr = csr_matrix(arg1, shape, dtype, copy)
            else:
                if len(arg1) == 2:
                    # (data, ij) format
                    try:
                        obj, (row, col) = arg1
                    except (TypeError, ValueError):
                        raise TypeError('invalid input format')
                    
                    if shape is None:
                        if len(row) == 0 or len(col) == 0:
                            raise ValueError('cannot infer dimensions from zero '
                                             'sized index arrays')
                        M = np.max(row) + 1
                        N = np.max(col) + 1
                        shape = (M, N)
                    else:
                        # Use 2 steps to ensure shape has length 2.
                        M, N = shape
                        shape = (M, N)
                        
                    row = np.array(row,dtype=np.int32)
                    col = np.array(col,dtype=np.int32)
                    obj = np.array(obj,dtype=dtype)
                    nnz = len(obj)
                    
                    indptr = np.empty(M+1,dtype=np.int32)
                    indices = np.empty_like(col,dtype=np.int32)
                    data = np.empty_like(obj,dtype=dtype)
                    
                    coo_tocsr(M, N, nnz, row, col, obj,
                              indptr, indices, data)
                    self.csr = csr_matrix((data,indices,indptr), shape, dtype, copy)
                    self._pop_diag()
                    
                elif len(arg1) == 3:
                    (data, indices, indptr) = arg1
                    data    = np.array(data,dtype=np.float32)
                    indices = np.array(indices,dtype=np.int32)
                    indptr  = np.array(indptr,dtype=np.int32)
                    
                    self.csr = csr_matrix((data,indices,indptr), shape, dtype, copy)
                    self._pop_diag()
                    
                elif len(arg1) == 4:
                    (data, indices, indptr, diag) = arg1
                    data    = np.array(data,dtype=np.float32)
                    indices = np.array(indices,dtype=np.int32)
                    indptr  = np.array(indptr,dtype=np.int32)
                    diag    = np.array(diag,dtype=np.float32)
                    if shape is None:
                        shape = (len(diag),len(diag))
                    else:
                        shape = (max(shape[0],len(diag)),max(shape[1],len(diag)))
                        
                    self.csr = csr_matrix((data,indices,indptr), shape, dtype, copy)
                    self.diagonal = diag
                else:
                    raise ValueError("unrecognized sss_matrix constructor usage")
                #-
            #--
        #if instance
        self.data    = self.csr.data
        self.indices = self.csr.indices
        self.indptr  = self.csr.indptr
        self.shape   = self.csr.shape
        
    #==
    def _pop_diag(self):
        if not hasattr(self,"diagonal"):
            self.diagonal = np.zeros(self.csr.shape[0],dtype=self.csr.dtype)
        td = self.csr.diagonal()
        self.diagonal += td
        nonzero = np.flatnonzero(td)
        self.csr[nonzero,nonzero] = 0
        self.csr.eliminate_zeros()

    def toarray(self,order=None,out=None):
        """See the docstring for `spmatrix.toarray`."""
        mt = self.csr.toarray(order=order, out=out)
        mt = mt + mt.T
        np.fill_diagonal(mt,self.diagonal)
        return mt

    def sum(self,axis=None):
        """
            Sum of the symmetric matrix
            
            Parameters
            ----------
            axis : {0,1,None}
            
            Returns
            -------
            A matrix with the same shape as self, with the specified axis removed.
            
        """
        if axis is None:
            return self.csr.sum() * 2 + self.diagonal.sum()
        elif (axis == 0) or (axis == 1):
            return np.array(self.csr.sum(axis = 0) + 
                            self.csr.sum(axis = 1).T + 
                            self.diagonal)[0]
        else : 
            raise ValueError("unrecognized axis usage")
    #-
    
    def dot(self,other):
        """
            Ordinary dot product
            
            Parameters
            ----------
            other : np.array 
        """
        d = dia_matrix((self.diagonal,[0]),shape=self.shape)
            
        return self.csr.dot(other) + d.dot(other) + self.csr.T.dot(other)
    #-
    
    def dotv(self,other):
        """
            Ordinary dot product using MKL with vector
            
            Parameters
            ----------
            other : np.array 
        """
        return SpMV_SM_viaMKL(self.csr, other) + np.array([self.diagonal]).T*other
    
    def norm(self,bias):
        """
        norm matrix by bias vector
        
        Parameters
        ----------
        bias : np.array column vector
        
        """
        from numutils import NormCSR_ByBiasVector
        bias = bias.flatten()
        
        if len(bias) != self.shape[0] :
            raise ValueError("unrecognized input shape, should be array of length %s" % (self.shape[0]))
        
        self.diagonal *= bias*bias
        NormCSR_ByBiasVector(data,indices,indptr,bias)
        #for i in xrange(len(self.indptr)-1):
            #for j in xrange(self.indptr[i],self.indptr[i+1]):
                #self.data[j] = self.data[j] * bias[i] * bias[self.indices[j]]
        
    #-
    
    
#=================================================

def SpMV_viaMKL( A, x ,Atranspose=False):
    """
    Wrapper to Intel's SpMV
    (Sparse Matrix-Vector multiply)
    For medium-sized matrices, this is 4x faster
    than scipy's default implementation
    Stephen Becker, April 24 2014
    stephen.beckr@gmail.com
    """
    from ctypes import POINTER,c_void_p,c_int,c_char,c_float,byref,cdll
    mkl = cdll.LoadLibrary("libmkl_rt.so")
    if Atranspose:
        tras = 'N'
    else :
        tras = 'T'
    SpMV = mkl.mkl_cspblas_scsrgemv
    # Dissecting the "cspblas_dcsrgemv" name:
    # "c" - for "c-blas" like interface (as opposed to fortran)
    #    Also means expects sparse arrays to use 0-based indexing, which python does
    # "sp"  for sparse
    # "s"   for single-precision
    # "csr" for compressed row format
    # "ge"  for "general", e.g., the matrix has no special structure such as symmetry
    # "mv"  for "matrix-vector" multiply

    if not isspmatrix_csr(A):
        raise Exception("Matrix must be in csr format")
    (m,n) = A.shape

    # The data of the matrix
    data    = A.data.ctypes.data_as(POINTER(c_float))
    indptr  = A.indptr.ctypes.data_as(POINTER(c_int))
    indices = A.indices.ctypes.data_as(POINTER(c_int))

    # Allocate output, using same conventions as input
    nVectors = 1
    if x.ndim is 1:
        y = np.empty(m,dtype=np.float32,order='F')
        if x.size != n:
            raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
    elif x.shape[1] is 1:
        y = np.empty((m,1),dtype=np.float32,order='F')
        if x.shape[0] != n:
            raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
    else:
        nVectors = x.shape[1]
        y = np.empty((m,nVectors),dtype=np.float32,order='F')
        if x.shape[0] != n:
            raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))

    # Check input
    if x.dtype.type is not np.float32:
        x = x.astype(np.float32,copy=True)
    # Put it in column-major order, otherwise for nVectors > 1 this FAILS completely
    if x.flags['F_CONTIGUOUS'] is not True:
        x = x.copy(order='F')

    if nVectors == 1:
        np_x = x.ctypes.data_as(POINTER(c_float))
        np_y = y.ctypes.data_as(POINTER(c_float))
        # now call MKL. This returns the answer in np_y, which links to y
        SpMV(byref(c_char(tras)), byref(c_int(m)),data ,indptr, indices, np_x, np_y ) 
    else:
        for columns in xrange(nVectors):
            xx = x[:,columns]
            yy = y[:,columns]
            np_x = xx.ctypes.data_as(POINTER(c_float))
            np_y = yy.ctypes.data_as(POINTER(c_float))
            SpMV(byref(c_char(tras)), byref(c_int(m)),data,indptr, indices, np_x, np_y ) 

    return y

def SpMV_SM_viaMKL( A, x ):

    from ctypes import POINTER,c_void_p,c_int,c_char,c_float,byref,cdll
    mkl = cdll.LoadLibrary("libmkl_rt.so")

    SpMV = mkl.mkl_cspblas_scsrsymv
    # Dissecting the "cspblas_dcsrgemv" name:
    # "c" - for "c-blas" like interface (as opposed to fortran)
    #    Also means expects sparse arrays to use 0-based indexing, which python does
    # "sp"  for sparse
    # "s"   for double-precision
    # "csr" for compressed row format
    # "sy"  for "symmetry"
    # "mv"  for "matrix-vector" multiply

    if not isspmatrix_csr(A):
        raise Exception("Matrix must be in csr format")
    (m,n) = A.shape

    # The data of the matrix
    data    = A.data.ctypes.data_as(POINTER(c_float))
    indptr  = A.indptr.ctypes.data_as(POINTER(c_int))
    indices = A.indices.ctypes.data_as(POINTER(c_int))

    # Allocate output, using same conventions as input
    nVectors = 1
    if x.ndim is 1:
        y = np.empty(m,dtype=np.float32,order='F')
        if x.size != n:
            raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
    elif x.shape[1] is 1:
        y = np.empty((m,1),dtype=np.float32,order='F')
        if x.shape[0] != n:
            raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
    else:
        nVectors = x.shape[1]
        y = np.empty((m,nVectors),dtype=np.float32,order='F')
        if x.shape[0] != n:
            raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))

    # Check input
    if x.dtype.type is not np.float32:
        x = x.astype(np.float32,copy=True)
    # Put it in column-major order, otherwise for nVectors > 1 this FAILS completely
    if x.flags['F_CONTIGUOUS'] is not True:
        x = x.copy(order='F')

    if nVectors == 1:
        np_x = x.ctypes.data_as(POINTER(c_float))
        np_y = y.ctypes.data_as(POINTER(c_float))
        # now call MKL. This returns the answer in np_y, which links to y
        SpMV(byref(c_char("U")), byref(c_int(m)),data ,indptr, indices, np_x, np_y ) 
    else:
        for columns in xrange(nVectors):
            xx = x[:,columns]
            yy = y[:,columns]
            np_x = xx.ctypes.data_as(POINTER(c_float))
            np_y = yy.ctypes.data_as(POINTER(c_float))
            SpMV(byref(c_char("U")), byref(c_int(m)),data,indptr, indices, np_x, np_y ) 

    return y
