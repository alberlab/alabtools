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

    
    
    