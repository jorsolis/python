#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:50:57 2020

Calcula torsión y momento angular especifico

@author: jordi
"""
import numpy as np
import matplotlib.pyplot as plt
import plots_jordi as pts
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
###############################################################################
######################     TORSION          ###################################
###############################################################################
def calc_acc_tau(nsol, nump = 1000, 
                       rut = "/home/jordi/satellite/schrodinger_poisson/potpaco",
                       carp = 'orbitas_random_vel_new', de = 'baja_dens/'):
    t =  np.load("%s/%spot_%d/%s/1/tiemp_1.npy"%(rut,de,nsol,carp))
    dt = t[1]-t[0]
    dt2 = dt*dt
    for k in range(1,101,1): ##orbitas random_new
        print(k)
        for ncor in range(1, nump + 1):
            w =np.load("%s/%spot_%d/%s/%d/cords_cyl_%d.npy"%(rut,de,nsol,carp,k, ncor))
            rho, z, fi, vrho,vzi,vph = w
            vxi = vrho*np.cos(fi) - vph*np.sin(fi)
            vyi = vrho*np.sin(fi) - vph*np.cos(fi)
            v = np.array([vxi[:-2], vyi[:-2], vzi[:-2]])
            vxdoti = []
            vydoti = []
            vzdoti = []
            for i in range(0, np.shape(vxi)[0] - 1):
                vxdoti.append((vxi[i + 1] - vxi[i])/dt)
                vydoti.append((vyi[i + 1] - vyi[i])/dt)
                vzdoti.append((vzi[i + 1] - vzi[i])/dt)
            vxdoti = np.array(vxdoti[:-1])
            vydoti = np.array(vydoti[:-1])
            vzdoti = np.array(vzdoti[:-1])
          
            vxdot2i = []
            vydot2i = []
            vzdot2i = []
        
            for i in range(0, np.shape(vxi)[0] - 2):
                vxdot2i.append((vxi[i + 2] - 2.*vxi[i + 1] + vxi[i])/dt2)
                vydot2i.append((vyi[i + 2] - 2.*vyi[i + 1] + vyi[i])/dt2)
                vzdot2i.append((vzi[i + 2] - 2.*vzi[i + 1] + vzi[i])/dt2)
            vxdot2i = np.array(vxdot2i)
            vydot2i = np.array(vydot2i)
            vzdot2i = np.array(vzdot2i)   
            
            vx, vy , vz = v
            alpha = vy*vzdoti - vz*vydoti
            beta = vz*vxdoti - vx*vzdoti
            gama = vx*vydoti - vy*vxdoti
            tau = (alpha*vxdot2i + beta*vydot2i + gama*vzdot2i)/(alpha**2 + beta**2 + gama**2)
            np.save("%s/%spot_%d/%s/%d/tau_%d.npy"%(rut,de,nsol,carp,k,ncor), tau)
       
def calc_acc_tau2(nsol, nump = 1000, 
                       rut = "/home/jordi/satellite/schrodinger_poisson/potpaco",
                       carp = 'orbitas_random_vel_new', de = 'baja_dens/'):   
    t =  np.load("%s/%spot_%d/%s/1/tiemp_1.npy"%(rut,de,nsol,carp))
    dt = t[1]-t[0]
    dt2 = dt*dt
    for k in range(1,9,1):
        print(k)
        for ncor in range(1, nump):
            w =np.load("%s/%spot_%d/%s/%d/cords_%d.npy"%(rut,de,nsol,carp,k, ncor))
            xi,yi,zi,vxi,vyi,vzi = w    
            v = np.array([vxi[:-2], vyi[:-2], vzi[:-2]])
            vxdoti = []
            vydoti = []
            vzdoti = []
            for i in range(0, np.shape(vxi)[0] - 1):
                vxdoti.append((vxi[i + 1] - vxi[i])/dt)
                vydoti.append((vyi[i + 1] - vyi[i])/dt)
                vzdoti.append((vzi[i + 1] - vzi[i])/dt)
            vxdoti = np.array(vxdoti[:-1])
            vydoti = np.array(vydoti[:-1])
            vzdoti = np.array(vzdoti[:-1])
          
            vxdot2i = []
            vydot2i = []
            vzdot2i = []
        
            for i in range(0, np.shape(vxi)[0] - 2):
                vxdot2i.append((vxi[i + 2] - 2.*vxi[i + 1] + vxi[i])/dt2)
                vydot2i.append((vyi[i + 2] - 2.*vyi[i + 1] + vyi[i])/dt2)
                vzdot2i.append((vzi[i + 2] - 2.*vzi[i + 1] + vzi[i])/dt2)
            vxdot2i = np.array(vxdot2i)
            vydot2i = np.array(vydot2i)
            vzdot2i = np.array(vzdot2i)   
            
            vx, vy , vz = v
            alpha = vy*vzdoti - vz*vydoti
            beta = vz*vxdoti - vx*vzdoti
            gama = vx*vydoti - vy*vxdoti
            tau = (alpha*vxdot2i + beta*vydot2i + gama*vzdot2i)/(alpha**2 + beta**2 + gama**2)
            np.save("%s/%spot_%d/%s/%d/tau_%d.npy"%(rut,de,nsol,carp,k, ncor), tau)

def taus_t(nsol, te= 0, carp = '', rut = '', de = '', kmax = 10, nump = 1000):
    T =[]
    for k in range(1,kmax + 1, 1):
        for ncor in range(1, nump + 1):
            tau =np.load("%s/%spot_%d/%s/%d/tau_%d.npy"%(rut,de,nsol,carp,k,
                                                         ncor))
            T.append(tau[te])
    T = np.array(T)
    return T

##carp = 'orbitas_random_vel_new2'
##
#for tt in range(975,1000,25000):
##for tt in range(0,2998,250):
#    T =   taus_t(nsol, te= tt, carp = carp, rut = rut, de = de,
#                 kmax = 100, nump = 1000)      
#    pts.histo(T, r'$\tau$', 
#              rang=(-0.04,0.04), 
#              bins = 200,
#              title = r'$t = 20\tau_s$',
##              title = r'$t = 0$',
##              title = r'$t = %f $'%t[tt],
#              normalized = False,
#              nom_archivo = "%s/%spot_%d/%s/hist_tau_%d_t%d"%(rut,de,nsol,carp,nsol,tt))
    
###############################################################################
################           ORBITAL POLES          #############################
###############################################################################
##orbitas random_new        kmax = 101
##orbitas_disc_dist_new     kmax = 2
def calc_orbital_poles(nsol, kmax = 100, nump = 1000, 
                       rut = "/home/jordi/satellite/schrodinger_poisson/potpaco",
                       carp = 'orbitas_random_vel_new', de = 'baja_dens/'):
    for k in range(1,kmax + 1,1):
        print(k)
        for ncor in range(1, nump + 1):
#            if nsol ==6:####           Pot 6 
#                w =np.load("%s/%spot_%d/%s/%d/cords_cyl_%d.npy"%(rut,de,nsol,carp,k, ncor))
#                w2 =np.load("%s/%spot_%d/%s/%d/cords_%d.npy"%(rut,de,nsol,carp,k, ncor))
#                _, _, fi, vrho,vz,vph = w
#                x,y,z = w2
#                vx = vrho*np.cos(fi) - vph*np.sin(fi)
#                vy = vrho*np.sin(fi) - vph*np.cos(fi)
#            elif nsol==2:###       Pot 2 
#                w2 =np.load("%s/%spot_%d/%s/%d/cords_%d.npy"%(rut,de,nsol,carp,k, ncor))
#                x,y,z, vx, vy, vz = w2

            w2 =np.load("%s/%spot_%d/%s/%d/cords_%d.npy"%(rut,de,nsol,carp,k, ncor))
            x,y,z, vx, vy, vz = w2
                
            lx = y*vz -z*vy
            ly = z*vx -x*vz
            lz = x*vy - y*vx
            norm = np.sqrt(lx**2 + ly**2 + lz**2)
            lxn = lx/norm
            lyn = ly/norm
            lzn = lz/norm
            l = np.array([lxn, lyn, lzn])
            lr = np.sqrt(lxn**2 + lyn**2 + lzn**2)
            lth = np.arccos(lzn/lr)
            lphi = np.arctan2(lyn,lxn)
            np.save("%s/%spot_%d/%s/%d/ang_mom_%d.npy"%(rut,de,nsol,carp,k,ncor),
                    l)
            np.save("%s/%spot_%d/%s/%d/ang_mom_th_%d.npy"%(rut,de,nsol,carp,k,ncor),
                    lth)
            np.save("%s/%spot_%d/%s/%d/ang_mom_ph_%d.npy"%(rut,de,nsol,carp,k,ncor),
                    lphi)
            
if __name__ == '__main__':
    carp = 'orbitas_random_vel_new'
    
    #carp = 'orbitas_disc_dist_new2'
    rut = "/home/jordi/satellite/schrodinger_poisson/potpaco"
    #nsol = 6
    nsol = 2
    
    de = 'baja_dens/'
    nump = 1000
##############################################################################
    #pts.plotmultiple([t,t,t], [l[0],l[1],l[2]], [], 't','l', 'title','', ylim=(0,0),
    #                 xlim=(0,0), logy= False, save = True, loc_leg='best',
    #                 angular=False, xangular=False, logx = False, show = True,
    #                 data=False, xd=[], yd=[], err=False, yerr=[], markersize = 20,
    #                 fill_between=False, fbx=[], fby1=1, fby2=0, text= '', xv=[])  