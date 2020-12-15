#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:22:28 2020

@author: jordi
"""
import numpy as np

import gala.potential as gp
import gala.dynamics as gd

from def_potenciales import POT_multi_SFDM, POT_multi_SFDM2
from scipy.misc import derivative
di = 'baja_dens/'

class SFDM_Potential(gp.PotentialBase):
    def __init__(self,A,nsol, units=None):
#    def __init__(self,A, nsol, units=galactic):
        self.C2 = np.sqrt(5)/2.
        self.C = 3.*np.sqrt(5)        
        pars = dict(A = A, nsol = nsol)      
        super(SFDM_Potential, self).__init__(units = units, 
             parameters = pars, ndim = 3)

    def _energy(self, xyz, t):
        A = self.parameters['A'].value
        nsol = self.parameters['nsol'].value
        x,y,z = xyz.T
        r = np.sqrt(x**2 + y**2 + z**2)
        costh = z/r 
        V00, r2V20 = POT_multi_SFDM(r, nsol)
        en = A*(V00 + self.C2*(3.*costh**2 - 1.)*r2V20)
        return en
   
    def _gradient(self, xyz, t):
        A = self.parameters['A'].value
        nsol = self.parameters['nsol'].value
        x,y,z = xyz.T
        rho = np.sqrt(x**2 + y**2)
        r = np.sqrt(x**2 + y**2 + z**2)
        costh = z/r
        _, r2V20 = POT_multi_SFDM(r, nsol)
        v000, r2v200 = POT_multi_SFDM2(r, nsol)
        dV0dr = derivative(v000, r)
        dr2V2dr = derivative(r2v200, r)  
        dVdr = dV0dr + self.C2*(3.*costh**2 - 1.)*dr2V2dr
        grad = np.zeros_like(xyz)
        grad[:,0] = A*(x*dVdr/r - self.C*x*z**2*r2V20/r**4)
        grad[:,1] = A*(y*dVdr/r - self.C*y*z**2*r2V20/r**4)
        grad[:,2] = A*(z*dVdr/r + self.C*z*rho**2*r2V20/r**4)
        return grad 