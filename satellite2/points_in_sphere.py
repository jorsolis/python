#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:49:34 2020

@author: jordi
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect("equal")

# top: elev=90, side: elev=0
ax.view_init(elev=0, azim=0)

u = np.linspace(0, 2 * np.pi, 120)
v = np.linspace(0, np.pi, 60)

x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

#ax.plot_surface(x, y, z,  rstride=2, cstride=2, color='b', #alpha = 0.3, linewidth = 0, cmap=cm.jet)
ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha = 0.3, linewidth = 0)

data = np.loadtxt(r'C:\TEST2\CDF\first.csv', delimiter=',',dtype=None)

xx, yy, zz = [], [], []
for d in data:
	xx.append(d[0])
	yy.append(d[1])
	zz.append(d[2])

ax.scatter(xx,yy,zz,color="k",s=1)

plt.title('Correctly distributed - Side View')
plt.axis('off')
plt.show()