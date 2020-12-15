#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:51:58 2019

@author: jordi
"""
import numpy as np
import plots_jordi as pts
from numpy import (linspace, sin, cos, pi, outer, ones, size, meshgrid,
                   array, load, shape)
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as amp
from pandas import DataFrame
from matplotlib.animation import FFMpegWriter, PillowWriter
from scipy.special import sph_harm
from matplotlib import rcParams

rcParams['axes.titlesize'] = 22

def animacion3dim(x,y,z,t, realt, directory, corr, galaxy = True, DM=True, R=60):
    df = DataFrame({"time": t ,"realt" : realt, "x" : x, "y" : y, "z" : z})
    def update_graph(num):
        data=df[df['time']==num]
        graph.set_data (data.x, data.y)
        graph.set_3d_properties(data.z)
        title.set_text('$t=${} $\hat\mu c t$'.format(df['realt'][num]))#.format(num))
        return title, graph, 
#    R = 60
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
#    ax.axis('off')
    if DM == True:
        th, phi = meshgrid(linspace(0, pi, 51),linspace(0, 2*pi, 181))
        xyz = array([sin(th)*sin(phi),sin(th)*cos(phi),cos(th)]) 
        Y_lm = sph_harm(0,1, phi, th)
        ro = 150
        r = abs(Y_lm.real)*ro*xyz
        ax.scatter(r[0], r[1], r[2], marker='_')
    if galaxy==True:
        rad = R/2 #/5
        rad2 = R/500 
        u = linspace(0, 2 * pi, 100)
        v = linspace(0, pi, 100)
        xb = rad * outer(cos(u), sin(v))
        yb = rad * outer(sin(u), sin(v))
        zb = rad2 * outer(ones(size(u)), cos(v))
        ax.plot_surface(xb, yb, zb, cmap='viridis')
    title = ax.set_title('Orbita', fontsize=22)
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)    
    ax.set_xlabel("$\hat\mu x$")
    ax.set_ylabel("$\hat\mu y$")
    ax.set_zlabel("$\hat\mu z$",rotation=45)   
    if R!=0:
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_zlim(-R, R)  
    ax.view_init(20,30)
    data=df[df['time']==0]
    graph, = ax.plot(data.x, data.y, data.z, "*r")    
#    ani = amp.FuncAnimation(fig, update_graph, 100)
#    writer = FFMpegWriter(fps = 5)
#    ani.save("%s/orbita_%d.mp4"%(directory,corr), writer=writer)   
    
    writer = FFMpegWriter(fps = 50)
    with writer.saving(fig, "%s/orbita_%d.mp4"%(directory,corr), 200):
        for i in df['time']:
            data=df[df['time']==i]
            x0 = data.x
            z0 = data.z
            y0 = data.y
            graph.set_data(x0, y0)
            graph.set_3d_properties(z0)
            writer.grab_frame()
            
    plt.show()

def anim2d2(x1, x2, y,t, Rho, Z, fi, x1label, x2label, ylabel, filename,
            fps=50,Rmax = 100, titulo1= 'a', titulo2= 'b'):   
    metadata = dict(title='Movie', artist='Jordi', comment='May the force be with you!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,6))
    
#    fig = plt.figure(figsize=(10,6))   
#    ax3 = fig.add_subplot(1, 2, 1,projection="3d")
#    ax1 = fig.add_subplot(2, 2, 2)
#    ax2 = fig.add_subplot(2, 2, 4)  
#    R = 100
#    ax3.set_xlim(-R, R)
#    ax3.set_ylim(-R, R)
#    ax3.set_zlim(-R, R)  
#    ax3.grid(False)
#    ax3.axis('off')   
#    ax3.set_xlabel("$\hat\mu x$")
#    ax3.set_ylabel("$\hat\mu y$")
#    ax3.set_zlabel("$\hat\mu z$",rotation=45)  
#    th, phi = meshgrid(linspace(0, pi, 51),linspace(0, 2*pi, 181))
#    xyz = array([sin(th)*sin(phi),sin(th)*cos(phi),cos(th)]) 
#    Y_lm = sph_harm(0,1, phi, th)
#    ro = 150
#    r = abs(Y_lm.real)*ro*xyz
#    ax3.scatter(r[0], r[1], r[2], marker='_')
#    graph, = ax3.plot([], [], [], "*r") 
   
    ax1.pcolormesh(Rho, Z, fi)
    ax2.pcolormesh(Rho, Z, fi)
    l, = ax1.plot([], [], '*-r')
    m, = ax2.plot([], [], 'D-r')
    x0, y0, z0 = 0, 0, 0
    dpi= 100.
    ax1.set_xlabel(x1label)
    ax1.set_ylabel(ylabel)
    ax1.set_title(titulo1)
    ax1.set_xlim([-Rmax, Rmax])
    ax1.set_ylim([-Rmax, Rmax])
    ax2.set_xlabel(x2label)
    title = ax2.set_title(titulo2)
    ax2.set_xlim([-Rmax, Rmax])
    ax2.set_ylim([-Rmax, Rmax])
    with writer.saving(fig, filename, dpi):
        for i in range(shape(x1)[0]):
            x0 = x1[i]
            y0 = x2[i]
            z0 = y[i]
#            t0= t[i]
            l.set_data(x0, z0)
            m.set_data(y0, z0)
#            title.set_text('{}, $t=${} $\hat\mu c t$'.format(titulo2, t0))
#            graph.set_data(x0, y0)
#            graph.set_3d_properties(z0)
            writer.grab_frame()
 
            
def surface_animation(X,Y,Z):    
    # Create a figure and a 3D Axes
    fig = plt.figure()
    ax = Axes3D(fig)
    from matplotlib import cm
    def init():
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        return fig,  
    def animate(i):
        # azimuth angle : 0 deg to 360 deg
        # elev angle : 0 deg to 90 deg
        ax.view_init(elev=i/2, azim=i*4)
        return fig,
    ani = amp.FuncAnimation(fig, animate, init_func=init,
                                   frames=90, interval=50, blit=True)
    writer = FFMpegWriter(fps = 10)
    ani.save("prueba.mp4", writer=writer) 

#X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
#Z = np.sin(np.sqrt(X**2 + Y**2))
#surface_animation(X,Y,Z)   
    
def scatter3d_animation(X,Y,Z,xlab,ylab,zlab,title,nom_archivo,galaxy= False,
                        R = 0, elevado=True):    
    # Create a figure and a 3D Axes
    fig = plt.figure()
    ax = Axes3D(fig)
    if galaxy==True:
        if R!=0:
            rad = R/5
            rad2 = R/500 
        else:
            rad = 50.
            rad2= 0.6
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xb = rad * np.outer(np.cos(u), np.sin(v))
        yb = rad * np.outer(np.sin(u), np.sin(v))
        zb = rad2 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xb, yb, zb, cmap='viridis')
    def init():
        ax.scatter(X, Y, Z, marker='*', color= 'Red')
        plt.title(title,fontsize=20)
        plt.xticks(fontsize=18, rotation=0)
        plt.yticks(fontsize=18, rotation=0)
        ax.set_xlabel(xlab)#,fontsize=16)
        ax.set_ylabel(ylab)
        ax.set_zlabel(zlab,rotation=45)  
        if R!=0:
            ax.set_xlim(-R, R)
            ax.set_ylim(-R, R)
            ax.set_zlim(-R, R)

        return fig,  
    def animate(i):
        if elevado == True:
            ax.view_init(elev=i, azim=i*4)# elev angle : 0 deg to 90 deg
        else:
            ax.view_init(0, azim=i*4)# azimuth angle : 0 deg to 360 deg
        return fig,
    ani = amp.FuncAnimation(fig, animate, init_func=init,
                                   frames=90, interval=50, blit=True)
    writer = FFMpegWriter(fps = 10)
    ani.save(nom_archivo, writer=writer) 
   
def los2plots3dcyl(ro, z, phi, t, direct, ncor, title, R, galaxia= False, 
                   MO=False, save = True):
#    pts.plot3d(ro*np.cos(phi), ro*np.sin(phi), z, t, r'$\hat{\mu} x$', r'$\hat{\mu} y$', 
#               r'$\hat{\mu} z$', 10, 30, title, "%s/cyl_3d_%d" % (direct, ncor),
#               galaxy= galaxia, R= R, DM=MO, save = save)
    pts.plot3d(ro*np.cos(phi), ro*np.sin(phi), z, t, r'$\hat{\mu} x$', r'$\hat{\mu} y$',
               r'$\hat{\mu} z$', 0, 15, title, "%s/cyl_3d_%d_2" % (direct, ncor),
               galaxy= galaxia, R=R, DM=MO)
#    pts.plot3d(ro*np.cos(phi), ro*np.sin(phi), z, t, r'$\hat{\mu} x$', r'$\hat{\mu} y$',
#               r'$\hat{\mu} z$', 0, 105, title, "%s/cyl_3d_%d_3" % (direct, ncor),
#               galaxy= galaxia, R=R, DM=MO)
    
def plotscoordscyl(t, ro, z, phi, vro, vz, uf, direct, ncor, title):
    a = (direct, ncor)
#    pts.plotmultiple([t, t, t], [ro, z, np.sqrt(ro**2 + z**2)],
#                     [r'$\hat{\mu}\rho$', r'$\hat{\mu}z$', r'$\hat{\mu}r$'],
#                     r'$\hat\mu ct$', '', title, "%s/rho(t)_%d" % a,
#                     save = True, loc_leg='best')
#    pts.coordsplotang(t, np.arctan2(ro,z), r'$\hat\mu ct$', r'$\theta$', title,
#                      "%s/theta(t)_%d" % a)
#    pts.coordsplotang(t, abs(phi), r'$\hat\mu ct$', r'$\phi$', title,
#                      "%s/phi(t)_%d" % a, ylim=(0,2*np.pi))
#    pts.plotmultiple([t, t], [vro, vz], [r'$v_\rho/c$', r'$v_z/c$'],r'$\hat\mu ct$',
#                     '', title, "%s/vels_%d" % a, save =True, loc_leg='best')
#    pts.parametricplot(t, np.sqrt(vro**2 + vz**2), r'$\hat\mu ct$', r'$\|v\|/c$',
#                       title, "%s/v_%d" % a, save= False)
    pts.plotmultiple([ro*np.cos(phi)], [ro*np.sin(phi)],
                      [r'$xy$ plane'], r'$\hat\mu x$',
                      r'$\hat\mu y$', title, "%s/xy_xz_%d" % a,
                      save = True, loc_leg='best')
    pts.plotmultiple([ro*np.sin(phi)], [z],
                      [r'$yz$ plane'], r'$\hat\mu y$',
                      r'$\hat\mu z$', title, "%s/xy_xz_%d" % a,
                      save = True, loc_leg='best')
##    pts.plotmultiple([ro*np.cos(phi), ro*np.sin(phi)], [ro*np.sin(phi), z],
##                      [r'$xy$ plane', r'$yz$ plane'], r'$\hat\mu x$ or $\hat\mu y$',
##                      r'$\hat\mu y$ or $\hat\mu z$', title, "%s/xy_xz_%d" % a,
##                      save = True, loc_leg='best')
    pts.plotmultiple([ro*np.cos(phi)], [z],
                      [r'$xz$ plane'], r'$\hat\mu x$',
                      r'$\hat\mu z$', title, "%s/xz_%d" % a, save = True, loc_leg='best')

def anim2d(x,y,xlabel,ylabel,title,filename, fps=15,Rmax = 60):
    metadata = dict(title='Movie Test', artist='Jordi',
                    comment='May the force be with you!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)    
    fig = plt.figure()
    l, = plt.plot([], [], 'k-o')
    plt.xlim(-Rmax, Rmax)
    plt.ylim(-Rmax, Rmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    x0, y0 = 0, 0
    dpi= 100.
    with writer.saving(fig, filename, dpi):
        for i in range(shape(x)[0]):
            x0 = x[i]
            y0 = y[i]
            l.set_data(x0, y0)
            writer.grab_frame()
            
#def animacion(x,y,xlabel,ylabel,name,titulo, xlim=[-100, 100], ylim=[-100., 100]):
#    import matplotlib.pyplot as plt
#    import animatplot as amp
#    X, Y = amp.util.parametric_line(x, y)
#    timeline = amp.Timeline(x, 's', 12)
#    ax = plt.axes(xlim=xlim, ylim=ylim)
#    block1 = amp.blocks.Line(X, Y, ax=ax)
#    anim = amp.Animation([block1], timeline)
#    # Your standard matplotlib stuff
#    plt.figure(figsize=(10,10))
#    plt.title(titulo)
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    # Create Interactive Elements
##    anim.toggle()
##    anim.timeline_slider()
#    # Save
#    anim.save(name, writer=PillowWriter(fps=5))
#    plt.show()