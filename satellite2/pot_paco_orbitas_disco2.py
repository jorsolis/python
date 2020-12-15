#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:01:22 2019

@author: jordi
"""
import numpy as np
from scipy.integrate import solve_ivp
import plots_jordi as pts
from plots_orbitas_schrodinger_poisson import (plotscoordscyl, los2plots3dcyl)
from pot_paco import DE
from pot_paco_orbitas_analisis import (mptodos, filtro,
                               mptodos_plot)
from disk_dist import Ry
import gala.potential as gp
import gala.dynamics as gd
from gala.units import galactic
from numpy import load
from scipy.interpolate import interp2d

rmin = .01
def filtro2(rho, z, phi, vrho, vz ,L,  value = rmin):
    vphi = L/rho
    n = 0
    for i in range(0, np.shape(rho)[0]):
        if (rho[i]**2 + z[i]**2) < value**2:
            rho[i] = 0.
            z[i] = 0.
            n = i
            print(n)
            break
    if n ==0:
        pass
    else:
        print('r^2 menores que %d'%value)
        for j in range(n, np.shape(rho)[0]):
            rho[j] = 0.
            z[j] = 0.
            phi[j]= 0.
            vrho[j]= 0.
            vz[j]=0.
            vphi[j] = 0.
    return rho,z,phi, vrho, vz, vphi

class SFDM_Potential(gp.PotentialBase):
#    def __init__(self,A, units=None):
    def __init__(self,A, nsol, units=galactic):
        pars = dict(A=A, nsol=nsol)
        super(SFDM_Potential, self).__init__(units=units,parameters =pars,ndim=3)

    def _energy(self, xyz, t):
        A = self.parameters['A'].value
        x,y,z = xyz.T
        rho = np.sqrt(x**2 + y**2)
        pot_cyl = potencial_interpolado(nsol, refina = 3, di = di)
        return A*pot_cyl(rho, z)
   
    def _gradient(self, xyz, t):
        A = self.parameters['A'].value
        x,y,z = xyz.T
        rho = np.sqrt(x**2 + y**2)
        grad = np.zeros_like(xyz)
        derrho, derz = fuerza_interpolada(nsol, refina = 3, di = di) 
        grad[:,0] = A*derrho(rho,z)*x/rho
        grad[:,1] = A*derrho(rho,z)*y/rho
        grad[:,2] = A*derz(rho, z)
        return grad    
def fuerza_interpolada(nsol, refina = 3, di = ''):
    dVdrho = load("%sdVdrho_%d_%d.npy"%(di,nsol,refina))
    dVdz = load("%sdVdz_%d_%d.npy"%(di,nsol,refina))
    rho = load("%scoordrho_%d_%d.npy"%(di,nsol,refina))
    z = load("%scoordz_%d_%d.npy"%(di,nsol,refina))
    rhor = rho[:50]
    derrho = interp2d(rhor, z, dVdrho, kind='linear', copy=True, bounds_error=False)
    zr = z[:100]
    derz = interp2d(rho, zr, dVdz, kind='linear', copy=True, bounds_error=False)    
    return derrho, derz

def potencial_interpolado(nsol, refina = 3, di = ''):
    rho = load("%scoordrho_%d_%d.npy"%(di,nsol,refina))
    z = load("%scoordz_%d_%d.npy"%(di,nsol,refina))
    potxz = load("%spotxz_%d_%d.npy"%(di,nsol,refina))    
    Vu = interp2d(rho, z, potxz, kind='linear', copy=True, bounds_error=False)
    return Vu  
def principal(derrho, derz, nsol,  Rf, tiem, ncor, y0a=[], plotting=True,
              direct = "/home/jordi/satellite/schrodinger_poisson/potpaco",
              de = '', k = 1):
  
    def resolvedor(u0,uf,y0,method,teval,L,ncor, conds):

        def fun(t,y):
            vrho, rho, vz, z, phi = y
            return [L**2/rho**3  + derrho(rho,z),
                    vrho,
                    derz(rho,z),
                    vz,
                    L/rho**2]   
        sol = solve_ivp(fun, [u0,uf], y0, method=method, t_eval=teval,
                        dense_output=True)#, events=event)
        if sol.status==0:
            rho, z, phi, vrho, vz = sol.y[1], sol.y[3],sol.y[4], sol.y[0], sol.y[2]
#            
            R2 = rho**2 + z**2
            if np.any(R2<rmin**2)==True:
                rho, z, phi, vrho, vz, vphi = filtro2(rho, z, phi, vrho, vz,
                                                     L, value = rmin)
            else:
                vphi = L/rho    
            if plotting ==True:
                plotscoordscyl(sol.t,  rho,z,phi, vrho, vz,
                               uf, "%s/%spot_%d/orbitas_disc_dist/%d"%(direct,de,nsol,k), ncor,
                               '')
#                los2plots3dcyl(rho, z, phi, sol.t,
#                               "%s/%spot_%d/orbitas_disc_dist/%d"%(direct,de,nsol,k), ncor,
#                               '', Rf, galaxia=False, MO=False)

            return [sol.t, rho, z, phi, vrho, vz, vphi]
        else:
            print("status", sol.status)
    def resolvedor2(y0a, du, tiem):    
        potdm = SFDM_Potential(A=-1., nsol = nsol,units=galactic)   
        vrho0, rho0, vz0, z0, phi0, vphi0 = y0a   
        x0 = rho0*np.cos(phi0)
        y0 = rho0*np.sin(phi0)
        vx0 = vrho0*np.cos(phi0) - vphi0*np.sin(phi0)
        vy0 = vrho0*np.sin(phi0) + vphi0*np.cos(phi0)
        
        w0 = gd.PhaseSpacePosition(pos=[x0,y0,z0], vel=[vx0,vy0,vz0])
        orbit = gp.Hamiltonian(potdm).integrate_orbit(w0, dt=du, n_steps=1000)  
    #    figs = orbit.plot(marker=',', linestyle='none')
    #    sixplots_orbits(orbit[0:1000])    
        w =  orbit.w(units=galactic)     
        t = uev
        x, y, z, vx,vy,vz = w
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)
        vrho = x*vx/rho + y*vy/rho
        vphi = x*vy/rho - y*vx/rho
        R2 =  np.sqrt(x**2 + y**2 + z**2)
        if np.any(R2<rmin**2)==True:
            print(ncor,'r min !!!!!')
        if np.any(R2>1.**2)==True:
            print(ncor,'r max !!!!!')
            print('rmax=', np.amax(R2))
            print('v2(0)=', np.sqrt(vrho[0]**2 + vz[0]**2+ vphi[0]**2))        
#        if plotting ==True:
            plotscoordscyl(t, rho, z, phi, vrho, vz, tiem, 
                           "%s/pot_%d/orbitas"%(direct,nsol),
                           ncor, 'title')
#            los2plots3dcyl(rho, z, phi, t,
#                           "%s/%spot_%d/orbitas/%d"%(direct,de,nsol,k), ncor,
#                           '', Rf, galaxia=False, MO=False)

        return t,rho,z,phi, vrho,vz,vphi    
    du = tiem/1000
    uev = np.arange(0., tiem, du)
    vphi0= y0a[5]
    rho0 = y0a[1]
    L = rho0*vphi0
    conds = (L, y0a[0], rho0, y0a[2], y0a[3])
    y0aa = [y0a[0], rho0, y0a[2], y0a[3], y0a[4]]
#    if (rho0**2 +y0a[3]**2) < rmin**2:
#        t = uev
#        rho = np.zeros(np.shape(uev))
#        z = np.zeros(np.shape(uev))
#        phi = np.zeros(np.shape(uev))
#        vrho = np.zeros(np.shape(uev))
#        vz = np.zeros(np.shape(uev))
#        vphi = np.zeros(np.shape(uev))
#        print('pos ini origen')
#    else:
#    t,rho,z,phi , vrho, vz, vphi = resolvedor(0,tiem,y0aa,'RK45',uev,L,
#                                              ncor, conds)
    t,rho,z,phi , vrho, vz, vphi = resolvedor2(y0a, du, tiem)       
    return t,rho,z,phi, vrho, vz, vphi

rut = "/home/jordi/satellite/schrodinger_poisson/potpaco"
###############################################################################
###############        Random positions & velocities        ###################
###############################################################################
di = 'baja_dens/'
nsol = 6

rf = 2.
tiemp = 3200 
###############################################################################
#carp = 'orbitas_disc_rot'
carp = 'orbitas_disc_dist_new'

#nump = 10000
nump = 6528
##########      posiciones iniciales  disco   
derrho, derz = fuerza_interpolada(nsol, refina = 3, di = di)

x0 = np.load("%s/%spot_%d/%s/X_0.npy"%(rut,di,nsol,carp))
y0 = np.load("%s/%spot_%d/%s/Y_0.npy"%(rut,di,nsol,carp))
phi0 =np.arctan2(y0,x0)
z0 = np.load("%s/%spot_%d/%s/Z_0.npy"%(rut,di,nsol,carp))
rho0 = np.load("%s/%spot_%d/%s/RHO_0.npy"%(rut,di,nsol,carp))
##############################################################################
############           cuando no hay rotacion        ########################
##############################################################################
vrho0 = np.zeros(np.shape(x0))
vz0 = np.zeros(np.shape(x0))
vphi0 = np.sqrt(-rho0*derrho(rho0,0))
##############################################################################
############           cuando si hay rotacion        ########################
############################################################################
#vrho0 = np.load("%s/%spot_%d/%s/VRHO_0.npy"%(rut,di,nsol,carp))
#vz0 = np.load("%s/%spot_%d/%s/VZ_0.npy"%(rut,di,nsol,carp))
#vphi0 = np.load("%s/%spot_%d/%s/VPHI_0.npy"%(rut,di,nsol,carp))
##############################################################################   
for ncor in range(0,nump,1):
    print(ncor)
    #######      condiciones  iniciales
    y0= [vrho0[ncor], rho0[ncor], vz0[ncor], z0[ncor], phi0[ncor], vphi0[ncor]] 
    t, rho, z, phi, vrho, vz, vphi = principal(derrho,derz, nsol, rf, tiemp,
                                               ncor, y0a=y0, plotting = True,
                                               de=di, k = 1)
    x= rho*np.cos(phi)
    y= rho*np.sin(phi)
    cord= np.array([x,y,z])
    cord_cyl= np.array([rho,z,phi,vrho,vz,vphi]) # disco inclinado
    np.save("%s/%spot_%d/%s/%d/cords_%d.npy"%(rut,di,nsol,carp,1,ncor), cord)
    np.save("%s/%spot_%d/%s/%d/cords_cyl_%d.npy"%(rut,di,nsol,carp,1,ncor), cord_cyl)
    np.save("%s/%spot_%d/%s/%d/tiemp_%d.npy"%(rut,di,nsol,carp,1,ncor), t)
###############################################################################
##############     plots fase space       ##################################
###############################################################################
#for i in range(975,999,2500):
##    print(nsol, i)
##    mptodos(nsol,te= i,carp = carp, de=di, kmax=1,ncormin = 0,
##            ncormax= nump, rut = rut)
#    mptodos_plot(nsol,te= i, Rf = rf, carp = carp, de=di,
#                 numpa = nump -150, rut = rut,
##                 histo = True, 
#                 tresD = True,
##                 Fase = True
#                 )
################################################################################
##############     quitar r<rmin       ##################################
###############################################################################
#for ncor in range(150, nump-1):
#    a = np.load("%s/%spot_%d/%s/%d/cords_cyl_%d.npy"%(rut,di,nsol,carp,1,ncor))
##    print(np.shape(a))
#    rho,z,phi,vrho,vz = a 
#    rho,z,phi,vrho,vz,_ = filtro2(rho, z, phi, vrho, vz ,np.zeros(np.shape(rho)),  value = rmin)
#    x= rho*np.cos(phi)
#    y= rho*np.sin(phi)
#    np.save("%s/%spot_%d/%s/%d/cords_cyl_%d.npy"%(rut,di,nsol,carp,1,ncor), np.array([rho,z,phi,vrho,vz]))
#    np.save("%s/%spot_%d/%s/%d/cords_%d.npy"%(rut,di,nsol,carp,1,ncor), np.array([x,y,z]))
    
