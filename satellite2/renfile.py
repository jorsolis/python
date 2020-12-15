#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:44:33 2020

@author: jordi
"""

import os

for n in range(1, 104,1):
    os.rename('/home/jordi/satellite/schrodinger_poisson/potpaco/baja_dens/potcore/orbitas_random_vel_new/18inc/cords_%d.npy'%n,
              '/home/jordi/satellite/schrodinger_poisson/potpaco/baja_dens/potcore/orbitas_random_vel_new/18inc/cords_%d.npy'%(n+477))