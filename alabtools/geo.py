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

def PolarizationIndex(PointsA, PointsB):
    """
        Calculate Polarization Index defined by \sqrt{(1-V_s/V_a)(1-V_s/V_b)}
        
        Parameters
        ----------
        PointsA, PointsB : numpy 2D array (float)
        
        Returns
        -------
        Polarization index of 2 ConvexHulls defined by 2 point sets
        
    """
    from scipy.spatial import ConvexHull
    from ._geotools import ConvexHullIntersection
    
    Ach = ConvexHull(PointsA)
    Bch = ConvexHull(PointsB)
    
    PointsI = ConvexHullIntersection(PointsA[Ach.vertices].reshape(len(Ach.vertices)*3),
                                     PointsB[Bch.vertices].reshape(len(Bch.vertices)*3))
    PointsI = np.array(PointsI)
    PointsI = PointsI.reshape((int(len(PointsI)/3), 3))
    
    if len(PointsI) < 4:
        return 1
    
    Ich = ConvexHull(PointsI)
    
    return ((1-Ich.volume/Ach.volume) * (1-Ich.volume/Bch.volume)) ** 0.5

    
def CenterOfMass(xyz,mass):
    """
        Calculate center of mass of a list of particles, given xyz coordinates and radius
        
        Parameters
        ----------
        xyz : numpy 2D array, N*3 2D array of coordinates
        mass : 1D array, mass of each beads
        
        Returns
        -------
        numpy array, (x,y,z) coordinate of center of mass
        
    """
    try:
        np.array(mass).shape[1]
    except:
        mass = np.array(mass)[:,None]
        
    if len(xyz) != len(mass):
        raise(RuntimeError,"Dimension not agree")

    return np.sum(xyz*mass,axis=0)/sum(mass)

def RadiusOfGyration(xyz,r):
    """
        Calculate radius of gyration of particles
        rg^2 = sum(massi*(ri - rcom)^2)/sum(massi)
        
        Parameters
        ----------
        xyz : numpy 2D array, N*3 2D array of coordinates
        r : 1D array, radii of each beads
        
        Returns
        -------
        float, radius of gyration
        
    """
    try:
        np.array(r).shape[1]
    except:
        r = np.array(r)[:,None]
        
    if len(xyz) != len(r):
        raise(RuntimeError,"Dimension not agree")
    
    mass = r**3
    
    r0 = CenterOfMass(xyz,mass)
    
    return np.sqrt(np.sum(((xyz - r0)**2) * mass)/np.sum(mass))

    
def GenerateTomogramFromStructure(size, xyz, r, rexpansion=1.0, sratio=1.0):
    """
    Calculate MRC grids for xyz
    """
    
    if isinstance(size, int):
        dim1 = dim2 = dim3 = size
    elif isinstance(size, tuple):
        assert(len(size) == 3)
        dim1, dim2, dim3 = size
    else:
        raise(RuntimeError, "Size should be 1 dim or 3 dim")
    
    tomo = np.zeros((dim1, dim2, dim3), dtype= np.float32, order='C')
    from ._cmtools import CalculateTomogramsFromStructure
    if xyz.dtype.type is not np.float32:
        xyz = xyz.astype(np.float32, copy=True)
    if r.dtype.type is not np.float32:
        r = r.astype(np.float32, copy=True)
        
    CalculateTomogramsFromStructure(xyz, r, rexpansion, sratio, tomo)
    
    return tomo
