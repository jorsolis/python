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
from plots_orbitas_schrodinger_poisson import (plotscoordscyl, los2plots3dcyl,
                                               anim2d2, animacion3dim, anim2d)
from plots_jordi import densityplot, plotfunc3d

def principal(edo, Rf, u0, r0, th0, L, ncor, edobase=3, Rf_base=10, vsat = 0.86*1e-4, adel=False, uf=0,
              y0a=[],direct = "/home/jordi/satellite/schrodinger_poisson/mix/orbitas"):
    tol = 0.01
    u = read_saved_file(edo,direct = "/home/jordi/satellite/schrodinger_poisson")
    u.set_allow_extrapolation(True)
    
    psi_1, phi_1 = u.split()
    
    V2 = psi_1.function_space()
    mesh2 = V2.mesh()
    degree = V2.ufl_element().degree()
    W = VectorFunctionSpace(mesh2, 'P', degree)    
    dV = project(grad(psi_1), W)
    dV.set_allow_extrapolation(True)
    dV_x, dV_y = dV.split(deepcopy=True) # extract components
    dV_x.set_allow_extrapolation(True)
    dV_y.set_allow_extrapolation(True)
    psi_1.set_allow_extrapolation(True)

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

    def psiplot(rho,z):    
        Phi = []
        for j in range(0,np.shape(rho)[0]):
            Phii=[]
            for i in range(0,np.shape(rho)[1]):
                point = (rho[j][i], abs(z[j][i]))                   
                if rho[j][i]**2 + z[j][i]**2 > Rf**2:
                    the = np.arctan(abs(z[j][i])/rho[j][i])
                    point = ((Rf - tol)*np.cos(the), (Rf - tol)*np.sin(the))
                    Phii.append(psi_1(point))
                elif rho[j][i]**2 + z[j][i]**2 < Rf_base**2:
                    Phii.append(psi_1(point) + psi_b(point))                    
                else:
                     Phii.append(psi_1(point))
            Phi.append(Phii)
        Phi = np.array(Phi)
        return Phi 
    
    def derrhoplot(rho,z):    
        Phi = []
        for j in range(0,np.shape(rho)[0]):
            Phii=[]
            for i in range(0,np.shape(rho)[1]):
                point = (rho[j][i], abs(z[j][i]))                   
                if rho[j][i]**2 + z[j][i]**2 > Rf**2:
                    the = np.arctan(abs(z[j][i])/rho[j][i])
                    point = ((Rf - tol)*np.cos(the), (Rf - tol)*np.sin(the))
                    Phii.append(dV_x(point))
                elif rho[j][i]**2 + z[j][i]**2 < Rf_base**2:
                    Phii.append(dV_x(point) + dVb_x(point))                    
                else:
                     Phii.append(dV_x(point))
            Phi.append(Phii)
        Phi = np.array(Phi)
        return Phi    
    def derrho(rho,z):    
        z = abs(z)
        point = (rho,z)
        if rho**2 + z**2 > Rf**2:
            the = np.arctan(z/rho)
            point = ((Rf - tol)*np.cos(the), (Rf - tol)*np.sin(the))
            return dV_x(point)
        elif rho**2 + z**2 < Rf_base**2:
            return (dV_x(point) + dVb_x(point)) 
        else:
            return dV_x(point)
        
    def derzplot(rho,z):
        Phi = []
        for j in range(0,np.shape(rho)[0]):
            Phii=[]
            for i in range(0,np.shape(rho)[1]):
                point = (rho[j][i], z[j][i]) 
                if z[j][i] < 0:
                    point = (rho[j][i], -z[j][i])
                    if rho[j][i]**2 + z[j][i]**2 > Rf**2:
                        the = np.arctan(z[j][i]/rho[j][i])
                        point = ((Rf - tol)*np.cos(the), -(Rf - tol)*np.sin(the))
                        Phii.append(-dV_y(point))
                    elif rho[j][i]**2 + z[j][i]**2 < Rf_base**2:
                        Phii.append(-dV_y(point) - dVb_y(point))
                    else:
                        Phii.append(-dV_y(point))                
                else:
                    if rho[j][i]**2 + z[j][i]**2 > Rf**2:
                        the = np.arctan(z[j][i]/rho[j][i])
                        point = ((Rf - tol)*np.cos(the), (Rf - tol)*np.sin(the))
                        Phii.append(dV_y(point))
                    elif rho[j][i]**2 + z[j][i]**2 < Rf_base**2:
                        Phii.append(dV_y(point) + dVb_y(point))
                    else:
                        Phii.append(dV_y(point))
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.array(Phi)
        
    def derz(rho,z):
        if z>0:
            point = (rho,z)
            if rho**2 + z**2 > Rf**2:
                the = np.arctan(z/rho)
                point = ((Rf - tol)*np.cos(the), (Rf - tol)*np.sin(the))
                return dV_y(point)
            elif rho**2 + z**2 < Rf_base**2:
                return (dV_y(point) + dVb_y(point))        
            else:
                return dV_y(point)
    
        else:
            point = (rho,-z)
            if rho**2 + z**2 > Rf**2:
                the = np.arctan(z/rho)
                point = ((Rf - tol)*np.cos(the), -(Rf - tol)*np.sin(the))
                return -dV_y(point)
            elif rho**2 + z**2 < Rf_base**2:
                return -(dV_y(point) + dVb_y(point))              
            else:
                return -dV_y(point)
    
#    rf = Rf
#    x1, x2 = np.meshgrid(np.linspace(0,rf, 100), np.linspace(-rf,rf, 100))

#    densityplot(x1,x2,psiplot(x1,x2),r'$\rho$',r'$z$',r'$\psi$','title',name=None)
#    plotfunc3d(x1,x2,psiplot(x1,x2),r'$\rho$',r'$z$',r'$\psi$','title',name=None)

#    densityplot(x1,x2,derrhoplot(x1,x2),r'$\rho$',r'$z$',r'$\frac{d\psi}{d\rho}$','title',name=None)
#    plotfunc3d(x1,x2,derrhoplot(x1,x2),r'$\rho$',r'$z$',r'$\frac{d\psi}{d\rho}$','title',name=None)
#    densityplot(x1,x2,derzplot(x1,x2),r'$\rho$',r'$z$',r'$\frac{d\psi}{dz}$','title',name=None)
#    plotfunc3d(x1,x2,derzplot(x1,x2),r'$\rho$',r'$z$',r'$\frac{d\psi}{dz}$','title',name=None)    
    def resolvedor(u0,uf,y0,method,teval,L,ncor):
        #method= 'RK45', 'Radau' or 'LSODA'
        def event(t,y):
            vrho, rho, vz, z, phi = y
            return z
        
        def jac(t,y):
            vrho, rho, vz, z, phi = y
            return [[0,-3.*L**2/rho**4 - (z**2 - 2.*rho**2)/np.sqrt(rho**2 + z**2)**5 , 0., 3.*z*rho/np.sqrt(rho**2 + z**2)**5, 0. ],
                    [1.,0.,0.,0.,0.],
                    [0.,3.*z*rho/np.sqrt(rho**2 + z**2)**5,0.,- (rho**2 - 2.*z**2)/np.sqrt(rho**2 + z**2)**5,0.],
                    [0.,0.,1.,0.,0.],
                    [0.,-2.*L/rho**3,0.,0.,0.]]
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
            plotscoordscyl(sol.t, sol.y[1], sol.y[3], sol.y[4], sol.y[0], sol.y[2],
                           uf, direct, ncor, labcond % conds)
            los2plots3dcyl(sol.y[1], sol.y[3], sol.y[4], sol.t, direct, ncor,
                           labcond % conds, 2*Rf/3, galaxia=False, MO=False)
#            tc = sol.t_events
#            print("tc=",tc)
#            print("ic=",sol.sol(sol.t_events[0]))
#            print( "tiempo final=", u0-tc[0])
            return [sol.t, sol.y[1], sol.y[3],sol.y[4]]
        else:
            print("status", sol.status)

    labcond = r"$l\hat{\mu}/c=$ %.5f, $v_\rho(0)=$ %.5f $c$, $\hat{\mu}\rho(0)=$ %.1f, $v_z(0)=$ %.5f $c$, $\hat{\mu} z(0)=$ %.2f"
    if adel == False:
        uf = 0.
        du = -u0/1000
        uev = np.arange(u0, uf, du)
        rho0 = r0*np.sin(th0)
        z0 = r0*np.cos(th0) 
        if vsat==0.:
            vrho0=0.
            vz0=0.
        else:
            vrho0 = 0.00008
            vz0 = np.sqrt( vsat**2 - vrho0**2)   
#        print('vrho0=',vrho0, 'vz0=',vz0)
        conds = (L,vrho0, rho0, vz0, z0)
        y0 = [vrho0, rho0, vz0, z0, 0.]    
        return resolvedor(u0,uf,y0,'RK45',uev,L,ncor)
    elif adel == True:
        u0 = 0.
        du = uf/1000
        uev = np.arange(u0, uf, du)
        conds = (L,y0a[0], y0a[1], y0a[2], y0a[3])    
        t,rho,z,phi = resolvedor(u0,uf,y0a,'RK45',uev,L,ncor)
        return t,rho,z,phi

principal(83, 100., 20000., 90., np.pi/8, 0.05, 32)
##

