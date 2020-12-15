#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
órbitas con el potencial del sistema Schrodinger-Poisson

"""
import scipy.integrate as spi
import numpy as np
from schrodinger_poisson_fenics_cyl import read_saved_file
from dolfin import grad, project, VectorFunctionSpace, plot
from time import time
import plots_jordi as pj
from plots_orbitas_schrodinger_poisson import (plotscoordscyl, los2plots3dcyl,
                                               anim2d2, animacion3dim, anim2d, scatter3d_animation)
from plots_jordi import densityplot, plotfunc3d

def principal(edobase, Rf, u0, r0, th0, L, ncor, vsat = 0.86*1e-4, adel=False,
              direct = "/home/jordi/satellite/schrodinger_poisson/Positive/orbitas",
              plotting = True, phi0= 0.):
    tol = 0.01
    ub = read_saved_file(edobase, 
                         direct = '/home/jordi/satellite/schrodinger_poisson/Positive')
    ub.set_allow_extrapolation(True)
    
    psi_b, _ = ub.split()
    
    Vb = psi_b.function_space()
    meshb = Vb.mesh()
    degreeb = Vb.ufl_element().degree()
    Wb = VectorFunctionSpace(meshb, 'P', degreeb)    
    dVb = project(grad(psi_b), Wb)
    dVb.set_allow_extrapolation(True)
    dVb_x, dVb_y = dVb.split(deepcopy=True) # extract components
    dVb_x.set_allow_extrapolation(True)
    dVb_y.set_allow_extrapolation(True)
    psi_b.set_allow_extrapolation(True)

#    def psiplot(rho,z):    
#        Phi = []
#        for j in range(0,np.shape(rho)[0]):
#            Phii=[]
#            for i in range(0,np.shape(rho)[1]):
#                point = (rho[j][i], abs(z[j][i]))                   
#                if rho[j][i]**2 + z[j][i]**2 > Rf**2:
#                    the = np.arctan(abs(z[j][i])/rho[j][i])
#                    point = ((Rf - tol)*np.cos(the), (Rf - tol)*np.sin(the))
#                    Phii.append(psi_b(point))                   
#                else:
#                     Phii.append(psi_b(point))
#            Phi.append(Phii)
#        Phi = np.array(Phi)
#        return Phi 
#    
#    def derrhoplot(rho,z):    
#        Phi = []
#        for j in range(0,np.shape(rho)[0]):
#            Phii=[]
#            for i in range(0,np.shape(rho)[1]):
#                point = (rho[j][i], abs(z[j][i]))                   
#                if rho[j][i]**2 + z[j][i]**2 > Rf**2:
#                    the = np.arctan(abs(z[j][i])/rho[j][i])
#                    point = ((Rf - tol)*np.cos(the), (Rf - tol)*np.sin(the))
#                    Phii.append(dVb_x(point))                  
#                else:
#                     Phii.append(dVb_x(point))
#            Phi.append(Phii)
#        Phi = np.array(Phi)
#        return Phi    
    def derrho(rho,z):    
        z = abs(z)
        point = (rho,z)
        if rho**2 + z**2 > Rf**2:
            the = np.arctan(z/rho)
            point = ((Rf - tol)*np.cos(the), (Rf - tol)*np.sin(the))
            return dVb_x(point)
        else:
            return dVb_x(point)
#        
#    def derzplot(rho,z):
#        Phi = []
#        for j in range(0,np.shape(rho)[0]):
#            Phii=[]
#            for i in range(0,np.shape(rho)[1]):
#                point = (rho[j][i], z[j][i]) 
#                if z[j][i] < 0:
#                    point = (rho[j][i], -z[j][i])
#                    if rho[j][i]**2 + z[j][i]**2 > Rf**2:
#                        the = np.arctan(z[j][i]/rho[j][i])
#                        point = ((Rf - tol)*np.cos(the), -(Rf - tol)*np.sin(the))
#                        Phii.append(-dVb_y(point))
#                    else:
#                        Phii.append(-dVb_y(point))                
#                else:
#                    if rho[j][i]**2 + z[j][i]**2 > Rf**2:
#                        the = np.arctan(z[j][i]/rho[j][i])
#                        point = ((Rf - tol)*np.cos(the), (Rf - tol)*np.sin(the))
#                        Phii.append(dVb_y(point))
#                    else:
#                        Phii.append(dVb_y(point))
#            Phi.append(Phii)
#        Phi = np.array(Phi)
#        return np.array(Phi)
#        
    def derz(rho,z):
        if z>0:
            point = (rho,z)
            if rho**2 + z**2 > Rf**2:
                the = np.arctan(z/rho)
                point = ((Rf - tol)*np.cos(the), (Rf - tol)*np.sin(the))
                return dVb_y(point)      
            else:
                return dVb_y(point)
#    
        else:
            point = (rho,-z)
            if rho**2 + z**2 > Rf**2:
                the = np.arctan(z/rho)
                point = ((Rf - tol)*np.cos(the), -(Rf - tol)*np.sin(the))
                return -dVb_y(point)             
            else:
                return -dVb_y(point)
#    
#    rf = Rf
#    x1, x2 = np.meshgrid(np.linspace(0,rf, 100), np.linspace(-rf,rf, 100))
#
#    densityplot(x1,x2,psiplot(x1,x2),r'$\rho$',r'$z$',r'$\psi$','title',name=None)
#
#    densityplot(x1,x2,derrhoplot(x1,x2),r'$\rho$',r'$z$',r'$\frac{d\psi}{d\rho}$','title',name=None)
#    densityplot(x1,x2,derzplot(x1,x2),r'$\rho$',r'$z$',r'$\frac{d\psi}{dz}$','title',name=None)

    def resolvedor(u0,uf,y0,method,teval,L,ncor):
        #method= 'RK45', 'Radau' or 'LSODA'
        def event(t,y):
            vrho, rho, vz, z, phi = y
            return z
        
        def fun(t,y):
            vrho, rho, vz, z, phi = y
            return [L**2/rho**3 - derrho(rho,z),#rho/np.sqrt(rho**2 + z**2)**3,
                    vrho,
                    - derz(rho,z),# z/np.sqrt(rho**2 + z**2)**3, 
                    vz,
                    L/rho**2]   
        sol = spi.solve_ivp(fun, [u0,uf], y0, method=method, t_eval=teval,
                            events=event, dense_output=True)#, jac = jac)
        if sol.status==0:
            if plotting==True:
                plotscoordscyl(sol.t, sol.y[1], sol.y[3], sol.y[4], sol.y[0], sol.y[2],
                               uf, direct, ncor, labcond % conds)
                los2plots3dcyl(sol.y[1], sol.y[3], sol.y[4], sol.t, direct, ncor,
                               labcond % conds, Rf, galaxia=False, MO=False)
#            tc = sol.t_events
#            print("tc=",tc)
#            print("ic=",sol.sol(sol.t_events[0]))
#            print( "tiempo final=", u0-tc[0])
            return [sol.t, sol.y[1], sol.y[3],sol.y[4]]
        else:
            print("status", sol.status)

    labcond = r"$l\hat{\mu}/c=$ %.5f, $v_\rho(0)=$ %.5f $c$, $\hat{\mu}\rho(0)=$ %.1f, $v_z(0)=$ %.5f $c$, $\hat{\mu} z(0)=$ %.2f"
    rho0 = r0*np.sin(th0)
    z0 = r0*np.cos(th0)
    if vsat==0.:
        vrho0=0.
        vz0=0.
    else:
        vrho0 = 0.00008
        vz0 = np.sqrt( vsat**2 - vrho0**2) 
    y0 = [vrho0, rho0, vz0, z0, phi0] 
    conds = (L,vrho0, rho0, vz0, z0) 
    
    if adel == False:
        uf = 0.
        du = -u0/1000
        uev = np.arange(u0, uf, du)                           
        return resolvedor(u0,uf,y0,'RK45',uev,L,ncor)

    elif adel == True:
        du = u0/1000
        uev = np.arange(0., u0, du)    
        t,rho,z,phi = resolvedor(0.,u0,y0,'RK45',uev,L,ncor)
        return t,rho,z,phi

direct = '/home/jordi/satellite/schrodinger_poisson/Positive/orbitas'
rut = '/home/jordi/satellite/schrodinger_poisson/Positive/orbitas'

tiemp=1000.
ncor = 50
nsol = 1


for i in range(1,2,1):
    r0 = np.random.uniform(0.5, 8)
    th0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
    phi0 = np.random.uniform(0.,2.*np.pi)
    print(r0,th0)
    L = 0.08

    t, rho, z, phi  = principal(nsol, 10., tiemp, r0, th0, L, ncor, vsat=0., adel=True,
                                plotting = True, phi0= phi0)
    x= rho*np.cos(phi)
    y= rho*np.sin(phi)
    cord= np.array([x,y,z])
    np.save("%s/cords_%d.npy"%(direct,ncor), cord)
    np.save("%s/tiemp_%d.npy"%(direct,ncor), t)
    ncor += 1

