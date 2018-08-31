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
import h5py
from scipy.sparse import coo_matrix, csr_matrix, dia_matrix, isspmatrix_csr, SparseEfficiencyWarning
from scipy.sparse.sputils import isshape
from scipy.sparse._sparsetools import coo_tocsr
from .utils import H5Batcher
import warnings
warnings.simplefilter('ignore', SparseEfficiencyWarning)

DATA_DTYPE = np.float32
INDPTR_DTYPE = np.int32
INDICES_DTYPE = np.int32


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

    def __init__(self,arg1,shape=None,dtype=DATA_DTYPE,copy=False,mem=True, lazy=False, h5group='.'):

        if isinstance(arg1, np.ndarray):
            self.csr = csr_matrix(np.triu(arg1), shape, dtype, copy)
            self._pop_diag()

        if isinstance(arg1,str):
            h5 = h5py.File(arg1, 'r')
            self._load_from_h5(h5[h5group], lazy)

        elif isinstance(arg1, h5py.Group):
            self._load_from_h5(arg1[h5group], lazy)

        elif isinstance(arg1, tuple):
            if isshape(arg1):
                # It's a tuple of matrix dimensions (M, N)
                # create empty matrix
                self.csr = csr_matrix(arg1, shape, dtype, copy)
                self._pop_diag()
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

                    row = np.array(row,dtype=INDICES_DTYPE)
                    col = np.array(col,dtype=INDICES_DTYPE)
                    obj = np.array(obj,dtype=dtype)
                    nnz = len(obj)

                    indptr = np.empty(M+1,dtype=INDPTR_DTYPE)
                    indices = np.empty_like(col,dtype=INDICES_DTYPE)
                    data = np.empty_like(obj,dtype=dtype)

                    coo_tocsr(M, N, nnz, row, col, obj,
                              indptr, indices, data)
                    self.csr = csr_matrix((data,indices,indptr), shape, dtype, copy)
                    self._pop_diag()

                elif len(arg1) == 3:
                    (data, indices, indptr) = arg1
                    data    = np.array(data,dtype=dtype)
                    indices = np.array(indices,dtype=INDICES_DTYPE)
                    indptr  = np.array(indptr,dtype=INDPTR_DTYPE)

                    self.csr = csr_matrix((data,indices,indptr), shape, dtype, copy)
                    self._pop_diag()

                elif len(arg1) == 4:
                    (data, indices, indptr, diag) = arg1
                    data    = np.array(data,dtype=dtype)
                    indices = np.array(indices,dtype=INDICES_DTYPE)
                    indptr  = np.array(indptr,dtype=INDPTR_DTYPE)
                    diag    = np.array(diag,dtype=dtype)
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

    def get_column(self, key):
        ix = np.flatnonzero(self.indices[:self.indptr[key]] == key)
        js = np.searchsorted(self.indptr, ix, side='right') - 1
        col = np.zeros(self.shape[1])
        col[js] = self.data[ix]
        return col

    def get_triu_row(self, key):
        n, m = self.shape

        if np.issubdtype(type(key), np.integer):
            i = key
            row = np.zeros(m)
            row[ self.indices[self.indptr[i]:self.indptr[i+1]] ] = self.data[self.indptr[i]:self.indptr[i+1]]
            row[i] = self.diagonal[i]
            return row

        elif isinstance(key, list) or isinstance(key, np.ndarray):
            rows = np.zeros((len(key), m))
            for k, i in enumerate(key):
                rows[k] = self.get_triu_row(i)
            return rows

        elif isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if start == None:
                start = 0
            if stop == None or stop > n:
                stop = n
            if step == None:
                step = 1

            rng = range(start, stop, step)
            return self.get_triu_row(list(rng))

        elif isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('get_triu_row should receive maximum 2 dimensions')
            rows, cols = key
            return self.get_triu_row(rows)[:, cols]

        else:
            raise RuntimeError('Invalid index type')

    def __getitem__(self, key):
        n, m = self.shape

        if np.issubdtype(type(key), np.integer):
            i = key
            row = np.zeros(m)
            row[ self.indices[self.indptr[i]:self.indptr[i+1]] ] = self.data[self.indptr[i]:self.indptr[i+1]]
            row[i] = self.diagonal[i]
            #lower diagonal
            col = self.get_column(i)
            row[:i] = col[:i]
            return row

        elif isinstance(key, list) or isinstance(key, np.ndarray):
            rows = np.zeros((len(key), m))
            for k, i in enumerate(key):
                rows[k] = self.__getitem__(i)
            return rows

        elif isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if start == None:
                start = 0
            if stop == None or stop > n:
                stop = n
            if step == None:
                step = 1

            rng = range(start, stop, step)
            return self.__getitem__(list(rng))

        elif isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('__getitem__ should receive maximum 2 dimensions')
            rows, cols = key
            return self.__getitem__(rows)[:, cols]

        else:
            raise RuntimeError('Invalid index type')

    def toarray(self,order=None,out=None):
        """See the docstring for `spmatrix.toarray`."""
        mt = self.csr.toarray(order=order, out=out)
        mt = mt + mt.T
        np.fill_diagonal(mt,self.diagonal)
        return mt

    def tocsr(self):
        mt = self.csr.copy()
        mt.setdiag(self.diagonal)
        return mt

    def tocoo(self):
        mt = self.csr.tocoo()
        mt.setdiag(self.diagonal)
        return mt

    def copy(self):
        mt = sss_matrix(self.shape)
        mt.csr = self.csr.copy()
        mt.data = mt.csr.data
        mt.indices = mt.csr.indices
        mt.indptr = mt.csr.indptr
        mt.diagonal = self.diagonal.copy()
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

    def normalize(self,bias):
        """
        normalize matrix by bias vector

        Parameters
        ----------
        bias : np.array column vector

        """
        from .numutils import NormCSR_ByBiasVector
        bias = bias.flatten()

        if len(bias) != self.shape[0] :
            raise ValueError("unrecognized input shape, should be array of length %s" % (self.shape[0]))

        self.diagonal *= bias*bias
        NormCSR_ByBiasVector(self.data,self.indices,self.indptr,bias)
        #for i in xrange(len(self.indptr)-1):
            #for j in xrange(self.indptr[i],self.indptr[i+1]):
                #self.data[j] = self.data[j] * bias[i] * bias[self.indices[j]]

    def coo_generator(self, batch_size=100000):

        # read stuff in batches if lazy evaluating for significant
        # performance improvements.

        if isinstance(self.indptr, h5py.Dataset):
            indptr = H5Batcher(self.indptr, batch_size)
        else:
            indptr = self.indptr

        if isinstance(self.indices, h5py.Dataset):
            indices = H5Batcher(self.indices, batch_size)
        else:
            indices = self.indices

        if isinstance(self.data, h5py.Dataset):
            data = H5Batcher(self.data, batch_size)
        else:
            data = self.data

        if isinstance(self.diagonal, h5py.Dataset):
            diagonal = H5Batcher(self.diagonal, batch_size)
        else:
            diagonal = self.diagonal

        vp, curr_row, next_row = 0, 0, 1

        while vp < len(self.data):
            while vp < indptr[next_row] and indices[vp] < curr_row:
                yield curr_row, indices[vp], data[vp]
                vp += 1

            if curr_row < min(self.shape) and diagonal[curr_row] != 0:
                yield curr_row, curr_row, diagonal[curr_row]

            while vp < indptr[next_row]:
                yield next_row - 1, indices[vp], data[vp]
                vp += 1

            curr_row += 1
            next_row += 1

        while curr_row < len(self.diagonal):
            yield curr_row, curr_row, diagonal[curr_row]
            curr_row += 1

    def nnz(self):
        return len(self.data) + np.count_nonzero(self.diagonal != 0)

    def _load_from_h5(self, grp, lazy=False):
        data = grp['data']
        indices = grp['indices']
        indptr = grp['indptr']
        diagonal = grp['diagonal']
        self.shape = (len(diagonal), len(diagonal))

        if lazy:
            self.csr = csr_matrix(self.shape)
            self.csr.data    = data
            self.csr.indices = indices
            self.csr.indptr  = indptr
            self.diagonal = diagonal
        else:
            self.csr = csr_matrix((data, indices, indptr), shape=self.shape)
            self.diagonal = diagonal[:]
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
        tras = b'N'
    else :
        tras = b'T'
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
        SpMV(byref(c_char(b"U")), byref(c_int(m)),data ,indptr, indices, np_x, np_y )
    else:
        for columns in xrange(nVectors):
            xx = x[:,columns]
            yy = y[:,columns]
            np_x = xx.ctypes.data_as(POINTER(c_float))
            np_y = yy.ctypes.data_as(POINTER(c_float))
            SpMV(byref(c_char(b"U")), byref(c_int(m)),data,indptr, indices, np_x, np_y )

    return y
