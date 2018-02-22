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

__author__  = "Nan Hua"

__license__ = "GPL"
__version__ = "0.0.3"
__email__   = "nhua@usc.edu"

import numpy as np
cimport numpy as np 
cimport cython   


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def NormCSR_ByBiasVector(data,indices,indptr,bias):
    cdef int i,j,N,left,right
    
    cdef int[::1] Aj = indices
    cdef int[::1] Ap = indptr
    cdef float[::1] D = data
    cdef float[::1] B = bias
    N = len(indptr)-1
    for i in range(N):
        for j in range(Ap[i],Ap[i+1]):
            D[j] *= B[i] * B[Aj[j]]
