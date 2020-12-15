#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:53:24 2019

@author: jordi
"""
from matplotlib.animation import FFMpegWriter, PillowWriter
from numpy import (linspace, sin, cos, pi, outer, ones, size, meshgrid,
                   array, load, shape)

from matplotlib import pyplot as plt

def anim2d2(x1, y, Rho, Z, fi, filename,fps=50):   
    writer = FFMpegWriter(fps=fps)    
    fig, ax1 = plt.subplots(1, 1, sharey=True, figsize=(10,6))  
    ax1.pcolormesh(Rho, Z, fi)
    l, = ax1.plot([], [], '*-r')
    x0, z0 = 0, 0
    dpi= 100.

    with writer.saving(fig, filename, dpi):
        for i in range(shape(x1)[0]):
            x0 = x1[i]
            z0 = y[i]
            l.set_data(x0, z0)
            writer.grab_frame()  
    plt.show()
import numpy as np           

rho= np.linspace(0,1,100)
z= np.linspace(0,1,100)

Rho, Z = np.meshgrid(rho,z)
def f(rho,z):
    return np.sin(rho)*np.cos(z)

t=np.linspace(0,1,100)
x = np.sin(t)

anim2d2(t, x, Rho, Z, f(Rho,Z), 'prueba.mpg',fps=50)
