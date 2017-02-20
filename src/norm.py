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
__version__ = "0.0.1"
__email__   = "nhua@usc.edu"

import numpy as np
import math
import time

def bnewt(A, mask=[], tol = 1e-5, delta_lower = 0.1, delta_upper = 3, fl = 0, check = 1, largemem = 0, chunk_size = 10000):
    """
    BNEWT A balancing algorithm for symmetric matrices
    X = BNEWT(A) attempts to find a vector X such that
    diag(X)*A*diag(X) is close to doubly stochastic. A must
    be symmetric and nonnegative.
    
    X0: initial guess. TOL: error tolerance.
    delta/Delta: how close/far balancing vectors can get
    to/from the edge of the positive cone.
    We use a relative measure on the size of elements.
    FL: intermediate convergence statistics on/off.
    RES: residual error, measured by norm(diag(x)*A*x - e).
    """
    # see details in Knight and Ruiz (2012)
    (n,m) = A.shape
    #np.seterr(divide='ignore')
  
    e        = np.ones((n,1),dtype=np.float32)
    e[mask]  = 0
    #res      = np.empty((n,1))
    
    g        = 0.9
    etamax   = 0.1
    eta      = etamax
    stop_tol = tol*0.5
    x        = e #initial guess
    rt       = tol*tol

    v      = x*A.dotv(x)

    rk       = 1 - v
    rk[mask] = 0
    rho_km1  = np.dot(np.transpose(rk),rk)
    rout     = rho_km1
    rold     = rout
    
    MVP = 0 #matrix vector products
    i = 0
  
    while rout > rt:
        i = i+1
        k=0
        y=e
        innertol = max(eta*eta*rout,rt)
    
        while rho_km1 > innertol: #inner iteration by CG
            k = k+1
            if k==1:
                with np.errstate(invalid='ignore'):
                    Z       = rk/v
                Z[mask] = 0
                p       = Z
                rho_km1 = np.dot(np.transpose(rk),Z)
            else:
                beta = rho_km1/rho_km2
                p    =  Z + beta*p
      
            #update search direction 

            w   = x*A.dotv(x*p) + v*p
      
            alpha = rho_km1/np.dot(np.transpose(p),w)
            ap = alpha*p
      
            #test distance to boundary of cone
            ynew = y + ap
            if min(np.delete(ynew,mask)) <= delta_lower:
                if delta_lower == 0:
                    break
                else:
                    ind = np.nonzero(ap < 0)
                    gamma = min((delta_lower - y[ind])/ap[ind])
                    y = y + gamma*ap
                    break
                if max(ynew) >= delta_upper:
                    ind = np.nonzero(ynew > delta_upper)
                    gamma = min((delta_upper-y[ind])/ap[ind])
                    y = y + gamma*ap
                    break
      
            y       = ynew
            rk      = rk - alpha*w
            rho_km2 = rho_km1
            with np.errstate(invalid='ignore'):
                Z       = rk/v
            Z[mask] = 0
            rho_km1 = np.dot(np.transpose(rk),Z)
        #end inner iteration
    
        x        = x*y

        v      = x*A.dotv(x)
        
        rk       = 1-v
        rk[mask] = 0
        rho_km1  = np.dot(np.transpose(rk),rk)
        rout     = rho_km1
        MVP      = MVP + k + 1
        #print MVP,res
        #update inner iteration stopping criterion
        rat      = rout/rold
        rold     = rout
        res_norm = math.sqrt(rout)
        eta_o    = eta
        eta      = g*rat
    
        if g*eta_o*eta_o > 0.1:
            eta = max(eta,g*eta_o*eta_o)
        eta = max(min(eta,etamax),stop_tol/res_norm)
    
        if fl == 1:
            print("{:4d} {:6d} {:.3f}".format(i,k,MVP))
      
        if MVP > 500:
            break
    #end outer
  
    print("Matrix vector products = {:6d}".format(MVP))

    return x
