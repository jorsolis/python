#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
órbitas con el potencial del sistema Schrodinger-Poisson

"""
import scipy.integrate as spi
import numpy as np
from schrodinger_poisson_fenics_cyl import read_saved_file
from dolfin import grad, project, VectorFunctionSpace
from time import time
from plots_orbitas_schrodinger_poisson import (plotscoordscyl, los2plots3dcyl,
                                               anim2d2, animacion3dim, anim2d)
from plots_jordi import densityplot, plotfunc3d

def principal(edo, Rf, u0, r0, th0, L, ncor, vsat = 0.86*1e-4, adel=False, uf=0,
              y0a=[],direct = "/home/jordi/satellite/schrodinger_poisson/orbitas"):
    tol = 0.01
    u = read_saved_file(edo)
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
                    else:
                        Phii.append(-dV_y(point))                
                else:
                    if rho[j][i]**2 + z[j][i]**2 > Rf**2:
                        the = np.arctan(z[j][i]/rho[j][i])
                        point = ((Rf - tol)*np.cos(the), (Rf - tol)*np.sin(the))
                        Phii.append(dV_y(point))
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
            else:
                return dV_y(point)
    
        else:
            point = (rho,-z)
            if rho**2 + z**2 > Rf**2:
                the = np.arctan(z/rho)
                point = ((Rf - tol)*np.cos(the), -(Rf - tol)*np.sin(the))
                return -dV_y(point)
            else:
                return -dV_y(point)

    def derrho2(rho,z):
        return -rho/np.sqrt(rho**2 + z**2)**3
    def derz2(rho,z):
        return -z/np.sqrt(rho**2 + z**2)**3

#    x1, x2 = np.meshgrid(np.linspace(0.5,2, 100), np.linspace(-2,2, 100))
#    densityplot(x1,x2,derrhoplot(x1,x2),r'$\rho$',r'$z$',r'$\frac{d\psi}{d\rho}$','title',name=None)
#    densityplot(x1,x2,derrho2(x1,x2),r'$\rho$',r'$z$',r'$-\frac{d\psi}{d\rho}$','title',name=None)
#    densityplot(x1,x2,derzplot(x1,x2),r'$\rho$',r'$z$',r'$\frac{d\psi}{dz}$','title',name=None)
#    densityplot(x1,x2,derz2(x1,x2),r'$\rho$',r'$z$',r'$-\frac{d\psi}{dz}$','title',name=None)
        
    def resolvedor(u0,uf,y0,method,teval,L,ncor):
        #method= 'RK45', 'Radau' or 'LSODA'
        def event(t,y):
            vrho, rho, vz, z, phi = y
            return z
        
#        def jac(t,y):
#            vrho, rho, vz, z, phi = y
#            return [[0,-3.*L**2/rho**4 - (z**2 - 2.*rho**2)/np.sqrt(rho**2 + z**2)**5 , 0., 3.*z*rho/np.sqrt(rho**2 + z**2)**5, 0. ],
#                    [1.,0.,0.,0.,0.],
#                    [0.,3.*z*rho/np.sqrt(rho**2 + z**2)**5,0.,- (rho**2 - 2.*z**2)/np.sqrt(rho**2 + z**2)**5,0.],
#                    [0.,0.,1.,0.,0.],
#                    [0.,-2.*L/rho**3,0.,0.,0.]]
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

#principal(11, 10., 1300., 7., np.pi/8, 0.002, 4)  
#principal(11, 10., 1300., 7., np.pi/8, 0.01, 5)
#principal(11, 10., 1300., 9., np.pi/8, 0.01, 6)     
#principal(11, 10., 1300., 9., np.pi/8, 0.0009, 11)
    
#principal(11, 10., 1300., 9., np.pi/8, 0.001, 12)

#t,rho,z,phi=principal(9, 100., 100000., 90., np.pi/8, 0.05, 13)

#cor = 26
#ad = "/home/jordi/satellite/schrodinger_poisson/orbitas/adelante"
#t,rho,z,phi=principal(83, 100., 90000., 60., np.pi/2, 0.05, cor, vsat=0., 
#                      adel =True, uf=90000.,y0a=[0.,60.,0.,0.,0.], direct =ad)

#cor = 21
#t,rho,z,phi=principal(83, 100., 90000., 40., np.pi/2, 0.05, cor, vsat=0., 
#                      adel =True, uf=90000.,y0a=[0.,10.,0.,0.,0.], direct =ad)
#x= rho*np.cos(phi)
#y= rho*np.sin(phi)
#cord= np.array([x,y,z])
#np.save("%s/cords_%d.npy"%(ad,cor), cord)
#np.save("%s/tiemp_%d.npy"%(ad,cor), t)


#cor = 26
#ad = "/home/jordi/satellite/schrodinger_poisson/orbitas/adelante"
#x,y,z = np.load("%s/cords_%d.npy"%(ad,cor))
#realt = np.load("%s/tiemp_%d.npy"%(ad,cor))
#t = np.array([int(np.ones(1))*i for i in range(1000)]).flatten()
#t = np.array([np.ones(10)*i for i in range(100)]).flatten()
#
#animacion3dim(x,y,z,t, realt, ad, cor)

#Rho, Z, fi = np.load("%s/rhozfi.npy"%ad)           
#anim2d2(x, y, z, t, Rho, Z, fi, r'$\hat\mu x$', r'$\hat\mu y$', r'$\hat\mu z$',
#        "%s/orbita_2d_%d.mp4"%(ad,cor), titulo1=r'$xz$-plane projection',
#        titulo2=r'$yz$-plane projection')


#anim2d(x,z,r'$\hat\mu x$', r'$\hat\mu z$', "Orbitas","%s/orbita_xz_%d_.mp4"%(ad,cor), fps=50, Rmax = 100)
#anim2d(y,z,r'$\hat\mu y$', r'$\hat\mu z$', "Orbitas","%s/orbita_yz_%d_.mp4"%(ad,cor), fps=50, Rmax = 100)

#ad = "/home/jordi/satellite/schrodinger_poisson/orbitas/adelante"

#L = 0.05
#
#a=principal(83, 100., 90000., 40., np.pi/2, L, 24, vsat=0., adel =True,
#            uf=70000.,y0a=[0.,40.,0.,0.,0.], direct =ad)
#yes = []
#for i in range(1,7,1):
#    cor = 20 + i
#    a=principal(83, 100., 90000., i*10., np.pi/2, L, cor, vsat=0., adel =True,
#                uf=90000.,y0a=[0.,i*10.,0.,0.,0.], direct =ad)
#    t, rho, z, phi = a
#    x= rho*np.cos(phi)
#    y= rho*np.sin(phi)
#    cord= np.array([x,y,z])
#    cord_cyl = [t, rho, z, phi]
#    np.save("%s/cords_%d.npy"%(ad,cor), cord)
#    np.save("%s/cords_cyl_%d.npy"%(ad,cor), cord_cyl)
#    yes.append(a)
#
#yes= np.array(yes)

#import plots_jordi as pts
##
#def mp(ncor_i,ncor_f):
#    x = []
#    y = []
#    z = []
#    th = []
#    leg = []
#    T = []
#    for ncor in range(ncor_i,ncor_f,1):
#        xi,yi,zi =  np.load("%s/cords_%d.npy"%(ad,ncor))
#        ti,rhoi,_,phii =  np.load("%s/cords_cyl_%d.npy"%(ad,ncor))
#        x.append(xi[:400])
#        y.append(yi[:400])
#        z.append(zi[:400])
#        T.append(ti[:400])
#        th.append(np.arctan2(rhoi[:400],zi[:400]))
#        r0 = ((1.*ncor)/10. - 2)*100
#        leg.append(r'$r(0)= %.1f$'%r0)
#    Rf = 80    
#    pts.multiplot3d(x,y,z,leg,r'$x$',r'$y$',r'$z$',0,45,
#                    r'$\theta(0)=\pi/2$, $v_z=v_\rho = 0$',
#                    "%s/haz%d"%(ad,ncor_i), R = Rf) 
#    pts.multiplot3d(x,y,z,leg,r'$x$',r'$y$',r'$z$',20,45,
#                    r'$\theta(0)=\pi/2$, $v_z=v_\rho = 0$',
#                    "%s/haz%d_2"%(ad,ncor_i), R = Rf)
#    pts.plotmultiple(x,y,leg,r'$x$',r'$y$',
#                    r'$\theta(0)=\pi/2$, $v_z=v_\rho = 0$',
#                    "%s/haz%d_3"%(ad,ncor_i))
#    pts.plotmultiple(x,z,leg,r'$x$',r'$z$',
#                    r'$\theta(0)=\pi/2$, $v_z=v_\rho = 0$',
#                    "%s/haz%d_4"%(ad,ncor_i))
#    pts.plotmultiple(y,z,leg,r'$y$',r'$z$',
#                    r'$\theta(0)=\pi/2$, $v_z=v_\rho = 0$',
#                    "%s/haz%d_5"%(ad,ncor_i))
##    pts.plotmultiple(T,th,leg,r'$y$',r'$z$',
##                    r'$\theta(0)=\pi/2$, $v_z=v_\rho = 0$',
##                    "%s/haz%d_6"%(ad,ncor_i))
#
#
#mp(21,24)
#mp(24,27)
#mp(21,27)

#from plots_jordi import plotmultiple
#plotmultiple([yes[0,0,:],yes[1,0,:],yes[2,0,:]],
#                 [yes[0,2,:],yes[1,2,:],yes[2,2,:]],
#                 (r'$\rho(0)=10$',r'$\rho(0)=20$',r'$\rho(0)=30$'),
#                 r'$\mu c t$',r'$z$',r'$z(0)=0, v_\rho(0)=0, v_z(0)=0, l\hat{\mu}/c=%.4f$'%L,
#                 '%s/zetas_0_2'%ad)
#plotmultiple([yes[3,0,:],yes[4,0,:]],
#                 [yes[3,2,:],yes[4,2,:]],
#                 (r'$\rho(0)=40$',r'$\rho(0)=50$'),
#                 r'$\mu c t$',r'$z$',r'$z(0)=0, v_\rho(0)=0, v_z(0)=0, l\hat{\mu}/c=%.4f$'%L,
#                 '%s/zetas_1_2'%ad)
#plotmultiple([yes[0,0,:],yes[1,0,:],yes[2,0,:]],
#                 [np.arctan2(yes[0,1,:],yes[0,2,:]),np.arctan2(yes[1,1,:],yes[1,2,:]),np.arctan2(yes[2,1,:],yes[2,2,:])],
#                 (r'$\rho(0)=10$',r'$\rho(0)=20$',r'$\rho(0)=30$'),
#                 r'$\mu c t$',r'$\theta$',r'$z(0)=0, v_\rho(0)=0, v_z(0)=0, l\hat{\mu}/c=%.4f$'%L,
#                 '%s/thetas_0_2'%ad, angular=True)
#plotmultiple([yes[3,0,:],yes[4,0,:]],
#                 [np.arctan2(yes[3,1,:],yes[3,2,:]),np.arctan2(yes[4,1,:],yes[4,2,:])],
#                 (r'$\rho(0)=40$',r'$\rho(0)=50$'),
#                 r'$\mu c t$',r'$\theta$',r'$z(0)=0, v_\rho(0)=0, v_z(0)=0, l\hat{\mu}/c=%.4f$'%L,
#                 '%s/thetas_1_2'%ad, angular=True)
        
#z = 80
#rho = 40
#principal(83, 100., 90000., np.sqrt(rho**2 + z**2), np.arctan(rho/z),
#          0.05, 30)
#print(np.sqrt(rho**2 + z**2),np.arctan(rho/z),np.arctan2(rho,z)/np.pi, .1475*np.pi )

#z = 90
#rho = 25
#principal(83, 100., 90000., np.sqrt(rho**2 + z**2), np.arctan(rho/z), 0.05, 
#          31, vsat=0.)
#principal(83, 100., 20000., 90., np.pi/8, 0.05, 32)
##

#L=0.05
#for i in range(13,15,1):
##    principal(9, 100., 30000., 90., np.pi/8, L, i)
#    principal(9, 100., 45000., 90., np.pi/8, L, i)
#    L = L/10
#    print(i, L)

#L=0.08
#for i in range(1,4,1):
#    principal(11, 10., 1300., 9., np.pi/8, L, i)
#    L += 0.01
#    print(i, L)

#principal(11, 10., 2600., 7., np.pi/8, 0.002, 20) # orbita en una de las bolitas

#############################################################################
#uf = 993.89 
#y0 = [-1.40876840e-02, 4.17306484, -2.35500298e-02, 5.55111512e-16,-6.30073902]

#uf = 3625.72 
#y0 = [-1.93517016e-02, 1.43559426, 2.66982688e-02, 6.66133815e-16,-3.12038596]
#principal(11, 10., 1300., 9., np.pi/8, 0.001, 12, adel =True,uf = uf, y0a=y0,
#          direct = ad)

#uf =300922.5
#y0 = [-4.57711293e-03,  1.41672107e+01, 1.10828170e-02, -8.88178420e-15, -3.20669789]
#principal(9, 100., 30000., 90., np.pi/8, 0.05, 13, adel =True, uf=uf, y0a=y0,
#          direct = ad)
# 
#y0 = [0.,  5.41672107e+01, 0., -8.88178420e-15, -3.20669789]    
#principal(9, 100.,30000.,90.,np.pi/8,0.05,14 , adel = True, uf=uf, y0a=y0,
#          direct = ad)
#
#y0 = [0.,  8.41672107e+01, 0., -8.88178420e-15, -3.20669789]    
#principal(9, 100.,30000.,90.,np.pi/8,0.05,15 , adel = True, uf=uf, y0a=y0,
#          direct = ad)
