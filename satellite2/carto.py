#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:51:25 2020

@author: jordi
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

proy = ccrs.Mollweide(central_longitude=0, globe=None, 
                             false_easting=None, false_northing=None)

proy = ccrs.PlateCarree()

fig = plt.figure(figsize=(20,6))
ax = fig.add_subplot(111, projection=proy)
ax.stock_img()

ny_lon, ny_lat = -75, 43
delhi_lon, delhi_lat = 77.23, 28.61

plt.plot([ny_lon, delhi_lon], [ny_lat, delhi_lat],
         color='blue', linewidth=2, marker='o',
         transform=ccrs.Geodetic(),
         )

plt.plot([ny_lon, delhi_lon], [ny_lat, delhi_lat],
         color='gray', linestyle='--',
         transform=ccrs.PlateCarree(),
         )

plt.text(ny_lon - 3, ny_lat - 12, 'New York',
         horizontalalignment='right',
         transform=ccrs.Geodetic())

plt.text(delhi_lon + 3, delhi_lat - 12, 'Delhi',
         horizontalalignment='left',
         transform=ccrs.Geodetic())


#ax.coastlines()

ax.gridlines(draw_labels=True, dms=True, x_inline=True, y_inline=False,
             linewidth=2, color='gray', alpha=0.5, linestyle='--')
sc = ax.scatter([ny_lon, delhi_lon], [ny_lat, delhi_lat], s=10, c='r')#, c=r0) 
#plt.colorbar(sc).ax.set_ylabel(r'$r$(kpc)')   
plt.show()

