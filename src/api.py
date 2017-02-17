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
import os.path
import re
try:
   import cPickle as pickle
except:
   import pickle

import warnings


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
    matrix : numpy 2d array storing all infor for the hic contact matrix
    idx : numpy structure array for matrix index
    genome : string, for the genome
    resolution : resolution for the contact matrix
    
    """
