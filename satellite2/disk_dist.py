#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:04:10 2020

Para generar distribucion de disco con anchura exponencial

@author: jordi
"""
from pot_paco import DE
from galpy.df import dehnendf
import plots_jordi as pts
import matplotlib.pyplot as plt
import numpy as np
from plots_orbitas_schrodinger_poisson import scatter3d_animation
#from pot_paco_orbitas2 import fuerza_interpolada
###############################################################################
##########               MATRICES DE ROTACION      #############################
###############################################################################
def Rx(theta):
    R=[[1.,0,0],
       [0., np.cos(theta), -np.sin(theta)],
       [0.,np.sin(theta), np.cos(theta)]]
    return(np.array(R))
def Ry(theta):
    R=[[np.cos(theta), 0., np.sin(theta)],
        [0,1., 0.],
       [-np.sin(theta), 0., np.cos(theta)]]
    return(np.array(R))

def Rz(theta):
    R=[[np.cos(theta), -np.sin(theta), 0],
       [np.sin(theta), np.cos(theta), 0],
       [0,0,1]]
    return(np.array(R)) 

if __name__ == '__main__':    
    rut = "/home/jordi/satellite/schrodinger_poisson/potpaco/baja_dens"
    nsol = 6
#    carp = 'orbitas_disc_dist'
#    carp = 'orbitas_disc_dist_new'    
    carp = 'orbitas_disc_dist_new2'    #otra corrida
############################################################################
#########                CREACION DEL DISCO      #############################
###############################################################################
#    np.random.seed(12345)# para 'orbitas_disc_dist_new'
#    np.random.seed(12) # para 'orbitas_disc_dist_new2'
#    dfc= dehnendf(beta=0.)
#    o= dfc.sample(n=10000,returnOrbit=True,nphi=1)
#    
#    xs = [e.x() for e in o]
#    ys = [e.y() for e in o]
#    xs = np.array(xs)
#    ys = np.array(ys)
#    zs = np.random.exponential(scale = 0.01, size = 10000)
#    sig = np.random.choice([-1,1], size = 10000)
#    zs = sig*zs
#    xs = xs/3.
#    ys = ys/3.
#    zs = zs/3.
#    rho = np.sqrt(xs**2 + ys**2)
#    np.save("%s/pot_%d/%s/RHO_0.npy"%(rut,nsol, carp), rho)
#    np.save("%s/pot_%d/%s/X_0.npy"%(rut,nsol, carp), xs)
#    np.save("%s/pot_%d/%s/Y_0.npy"%(rut,nsol, carp), ys)
#    np.save("%s/pot_%d/%s/Z_0.npy"%(rut,nsol, carp), zs)
#############################################################################
########                PLOT DEL DISCO      #############################
############################################################################    
    R=1.
#    mue = 25
#    mu = DE[nsol][mue]['mu']
#    rlam = DE[nsol][mue]['rlam']
#    lanp = 100./7.5
# 
#
    tt = 0
    xs = np.load("%s/pot_%d/%s/X_%d.npy"%(rut,nsol, carp, tt))
    ys = np.load("%s/pot_%d/%s/Y_%d.npy"%(rut,nsol, carp, tt))
    zs = np.load("%s/pot_%d/%s/Z_%d.npy"%(rut,nsol, carp, tt))
#    t =  np.load("%s/pot_%d/%s/1/tiemp_1.npy"%(rut,nsol, carp ))
    rs = np.sqrt(xs**2 + ys**2 +zs**2)
    ths = np.arccos(zs/rs)

#    R = lanp*R/(rlam*mu) 
#    xs = lanp*xs/(rlam*mu) 
#    ys = lanp*ys/(rlam*mu) 
#    zs = lanp*zs/(rlam*mu) 
       
#    tt= 3120
#    xs = np.load("%s/pot_%d/%s/X_%d.npy"%(rut,nsol, carp, tt))
#    ys = np.load("%s/pot_%d/%s/Y_%d.npy"%(rut,nsol, carp, tt))
#    zs = np.load("%s/pot_%d/%s/Z_%d.npy"%(rut,nsol, carp, tt))
#    R = 2
#
#    R = 10*lanp*R/(rlam*mu) 
#    xs = 10*lanp*xs/(rlam*mu) 
#    ys =  10*lanp*ys/(rlam*mu) 
#    zs =  10*lanp*zs/(rlam*mu) 

    pts.scater(xs,ys,'x(kpc)','y(kpc)','', z3D=True, z=zs, zlab='z', 
               initialview=[0,45],
              name = "%s/pot_%d/%s/particles_%d"%(rut,nsol, carp,tt), 
              R = R, s = 1)
    pts.scater(xs,ys,'x(kpc)','y(kpc)','', z3D=True, z=zs, zlab='z',
               initialview=[45,45],
              name = "%s/pot_%d/%s/particles_%d_1"%(rut,nsol, carp,tt),
              R = R, s = 1)
    pts.scater(xs,ys,'x(kpc)','y(kpc)','', z3D=True, z=zs, zlab='z', 
               initialview=[90,45],
              name = "%s/pot_%d/%s/particles_%d_2"%(rut,nsol, carp,tt),
              R = R, s = 1)
#    scatter3d_animation(xs,ys,zs,'$x$','$y$','$z$','$t=$%.2f'%tt,
#                        "%s/pot_%d/%s/haz_%d.mp4"%(rut,nsol, carp,tt),
#                        galaxy=False, R = R)
#
    pts.histo(rs, r'$r$', bins = 80)  
    pts.histo(ths, r'$\theta$', bins = 80, xangular =True)  
################################################################################
###########           CREACION DEL DISCO INCLINADO     #############################
################################################################################
#    xs = np.load("%s/pot_%d/orbitas_disc_dist/X_0.npy"%(rut,nsol))
#    ys = np.load("%s/pot_%d/orbitas_disc_dist/Y_0.npy"%(rut,nsol))
#    zs = np.load("%s/pot_%d/orbitas_disc_dist/Z_0.npy"%(rut,nsol))
#    rho0 = np.sqrt(xs**2 + ys**2)
#
#    print(np.shape(xs),np.shape(ys),np.shape(zs),np.shape(rhos) )
#    pos = np.array([xs,ys,zs])
#      
#    rot = Ry(np.pi/4)  ######## matriz de rotacion
#    posn = []
#    for i in range(0, 10000):
#        posn.append(rot.dot(pos[:,i]))    
#    posn = np.array(posn)
#    posn = posn.T
#    xsn = posn[0,:]
#    ysn = posn[1,:]
#    zsn = posn[2,:]
#    np.save("%s/pot_%d/orbitas_disc_rot/X_0.npy"%(rut,nsol), xsn)
#    np.save("%s/pot_%d/orbitas_disc_rot/Y_0.npy"%(rut,nsol), ysn)
#    np.save("%s/pot_%d/orbitas_disc_rot/Z_0.npy"%(rut,nsol), zsn)
#    np.save("%s/pot_%d/orbitas_disc_rot/RHO_0.npy"%(rut,nsol),rho0)
    
#    di = 'baja_dens/'
#    derrho, derz = fuerza_interpolada(nsol, refina = 3, di = di)
#    vrho0 = np.zeros(np.shape(xs))#estas son las velocidades del disco
#    vz0 = np.zeros(np.shape(xs))#horizontal
#    vphi0 = np.sqrt(-rho0*derrho(rho0,0))
#    #  convertidas a cartesianas, giradas y luego convertidas
#    # a cilindricas de nuevo 
#    VRHO = vphi0*(1. - np.cos(np.pi/4))*xs*ys/rho0**2 
#    VPHI = vphi0*(np.cos(np.pi/4)*ys**2 + xs**2)/rho0**2
#    VZ = vphi0*np.cos(np.pi/4)*ys/rho0 
#    carp = 'orbitas_disc_rot'
#    print(np.shape(VRHO),np.shape(VZ),np.shape(VPHI))
#    np.save("%s/pot_%d/%s/VRHO_0.npy"%(rut,nsol,carp), VRHO)
#    np.save("%s/pot_%d/%s/VZ_0.npy"%(rut,nsol,carp), VZ)
#    np.save("%s/pot_%d/%s/VPHI_0.npy"%(rut,nsol,carp),VPHI)
#    
###############################################################################
###########            PLOT DEL DISCO INCLINADO      ###########################
################################################################################
#    tt= 3120
#    tt = 0
#    carp = 'orbitas_disc_rot'
#    xs = np.load("%s/pot_%d/%s/X_%d.npy"%(rut,nsol, carp, tt))
#    ys = np.load("%s/pot_%d/%s/Y_%d.npy"%(rut,nsol, carp, tt))
#    zs = np.load("%s/pot_%d/%s/Z_%d.npy"%(rut,nsol, carp, tt))
#    
#    pts.scater(xs,ys,'x(kpc)','y(kpc)','', z3D=True, z=zs, zlab='z', initialview=[0,-60],
#              name = "%s/pot_%d/%s/particles_%d"%(rut,nsol, carp,tt), 
#              R = R, s = 1)
#    pts.scater(xs,ys,'x(kpc)','y(kpc)','', z3D=True, z=zs, zlab='z', initialview=[45,00],
#              name = "%s/pot_%d/%s/particles_%d_1"%(rut,nsol, carp,tt),
#              R = R, s = 1)
#    pts.scater(xs,ys,'x(kpc)','y(kpc)','', z3D=True, z=zs, zlab='z', initialview=[90,00],
#              name = "%s/pot_%d/%s/particles_%d_2"%(rut,nsol, carp,tt),
#              R = R, s = 1)
#    scatter3d_animation(xs,ys,zs,'$x$','$y$','$z$','$t=$%.2f'%tt,
#                        "%s/pot_%d/%s/haz_%d.mp4"%(rut,nsol, carp,tt),
#                        galaxy=False, R = R)