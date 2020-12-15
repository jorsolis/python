#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:26:19 2018

@author: jordis
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import csv

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def main():
    # Generate Data...
    numdata = 100
    x = np.random.random(numdata)
    y = np.random.random(numdata)
    z = x**2 + y**2 + 3*x**3 + y + np.random.random(numdata)

    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)

    # Evaluate it on a grid...
    nx, ny = 20, 20
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), 
                         np.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)

    # Plot
    plt.imshow(zz, extent=(x.min(), y.max(), x.max(), y.min()))
    plt.scatter(x, y, c=z)
    plt.show()
     
def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

x = np.array([])
y = np.array([])
z = np.array([])
print type(z)

with open("coords2.csv") as tsvfile:
    lines = csv.reader(tsvfile, delimiter='\t')
    for row in lines:
        x= np.append(x,float(row[0]))
        y =np.append(y,float(row[1]))
with open("val2.csv") as tsvfile:
    lines = csv.reader(tsvfile, delimiter='\t')
    for row in lines:
        z=np.append(z,float(row[0]))

print type(x), x.size, type(y), y.size, type(z), z.size
# Fit a 3rd order, 2d polynomial
m = polyfit2d(x,y,z)
#print m
zz = polyval2d(x, y, m)

fig = plt.figure(figsize=(13.3,10))
ax = plt.axes(projection='3d')
ax.scatter(x, y, zz, marker='_')
ax.scatter(x, y, z, marker=',')
ax.set_xlabel(r'$r$',fontsize=16)
ax.set_ylabel(r'$theta$',fontsize=16)
ax.set_zlabel('D',fontsize=16)
plt.show