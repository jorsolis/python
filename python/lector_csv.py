#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
...
"""

import numpy as np
import csv

r = np.array([])
z = np.array([])
theta = np.array([])
coord = np.array([[0,0]])

with open("coords_j1.csv") as tsvfile:
    lines = csv.reader(tsvfile, delimiter='\t')
    for row in lines:
        r= np.append(r,float(row[0]))
        theta =np.append(theta,float(row[1]))

with open("coords_j1.csv") as tsvfile:
    lines = csv.reader(tsvfile, delimiter='\t')
    for row in lines:
        coord= np.append(coord,[[float(row[0]),float(row[1])]],axis=0)
with open("val_j1.csv") as tsvfile:
    lines = csv.reader(tsvfile, delimiter='\t')
    for row in lines:
        z=np.append(z,float(row[0]))
print z.size, 'theta shape=', theta.shape, 'r shape=', r.shape, 'z shape=', z.shape
print 'coord shape =', coord.shape
coorde = np.delete(coord, 0, 0)
print 'new coord shape =', coorde.shape


#escribir archivo
with open('r.csv', mode='w') as r_coord_file:
    r_coord_writer = csv.writer(r_coord_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    i = 0
    while i < r.size-1:
#        print i
        r_coord_writer.writerow([r[i]])
        i+=13
with open('theta.csv', mode='w') as theta_coord_file:
    theta_coord_writer = csv.writer(theta_coord_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    i = 0
    while i < 13:
#        print i
        theta_coord_writer.writerow([theta[i]])
        i+=1
with open('z.csv', mode='w') as z_coord_file:
    z_coord_writer = csv.writer(z_coord_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    i = 0
    while i < z.size-1:
#        print i
        z_coord_writer.writerow([z[i], z[i+1], z[i+2], z[i+3], z[i+4],
                                 z[i+5], z[i+6], z[i+7], z[i+8], z[i+9], z[i+10],
                                 z[i+11], z[i+12]])
        i+=13