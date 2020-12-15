#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:55:42 2019

@author: jordis
"""
import scipy.integrate as spi
import numpy as np
import plots_jordi as pts

def solver(u0,uf,y0,ncor,metodo,teval,L,M,a,b,d,f,b_bar,titles,nomarchvs,ruta):
    #metodo= 'RK45', 'Radau' o 'LSODA'
    def fun(t,y):
        return [y[4]**2/(y[1]**3*np.sin(y[3])**2)
                - (b_bar/y[1]**2 + b/y[1]**2 + (M*y[1]*(1.+((d*np.cos(y[3])**2)/np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))))/(np.sqrt(d**2 + f**2 + y[1]**2 + 2*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdx(M,a,b,d,f,y),
                y[0],
                np.cos(y[3])*y[4]**2/(y[1]**2*np.sin(y[3])**3) 
                - (a*(np.pi -2.*y[3]) - (y[1]**2*d*M*np.cos(y[3])*np.sin(y[3]))/(np.sqrt(f**2+y[1]**2*np.cos(y[3])**2) * np.sqrt(d**2 + f**2 + y[1]**2 + 2.*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdtheta(M,a,b,d,f,y),
                y[2]/y[1]**2,
                0,
                y[4]/(y[1]**2*np.sin(y[3])**2)]    
    sol = spi.solve_ivp(fun, [u0,uf], y0, method=metodo, t_eval=teval)
    if sol.status==0:
        pts.los3plotscoords(sol.t,sol.y[1],sol.y[3],sol.y[4],
                        "%s/r(t)_%s_%d" % (ruta,nomarchvs,ncor),
                        "%s/theta(t)_%s_%d" % (ruta,nomarchvs,ncor),
                        "%s/phi(t)_%s_%d" % (ruta,nomarchvs,ncor), titles)
        
        pts.los2plotsvels(sol.t,sol.y[0],sol.y[2]/sol.y[1],#estoy graficando r\dot{theta}/c
                        "%s/vr(t)_%s_%d" % (ruta,nomarchvs,ncor),
                        "%s/vtheta(t)_%s_%d" % (ruta,nomarchvs,ncor), titles)

#        pts.parametricplot(sol.t,sol.y[4],sol.t,r'$\hat{\mu}ct$',
#                           r'$\hat{\mu}l/c$', titles, "%s/l(t)_%s_%d" % (ruta,nomarchvs,ncor))

        pts.parametricplot(sol.y[1]*np.cos(sol.y[4]),sol.y[1]*np.sin(sol.y[4]),
                           sol.t,r"$x\mu$",r"$y\mu$",
                           titles,"%s/xy_paramt_%s_%d" % (ruta,nomarchvs,ncor))

#        pts.parametricplot(sol.y[1]*np.sin(sol.y[3]),sol.y[1]*np.cos(sol.y[3]),
#                       r"$x\mu$",r"$y\mu$",
#                       titles,"%s/xz_paramt_%s_%d" % (ruta,nomarchvs,ncor))
 
        pts.parametricplot(sol.t, sol.y[0]**2 + sol.y[1]**2*sol.y[2]**2 + L**2/(sol.y[1]**2*np.sin(sol.y[3])**2) - 2*a*(sol.y[3]-np.pi/2)**2 - 2*b/sol.y[1] - 2*M/np.sqrt(d**2 + f**2 + sol.y[1]**2 + 2*d*np.sqrt(f**2+sol.y[1]**2*np.cos(sol.y[3])**2)),
                           sol.t,r"$\mu c t$",r"$2 E$",
                           titles,"%s/energia_%s_%d" % (ruta,nomarchvs,ncor))

#        pts.parametricplot(sol.t, sol.y[0]**2 + sol.y[1]**2*sol.y[2]**2,
#                           sol.t,r"$\mu c t$",r"$K$",
#                           titles,"%s/energiacin_%s_%d" % (ruta,nomarchvs,ncor))
#
#        pts.parametricplot(sol.t, L**2/(sol.y[1]**2*np.sin(sol.y[3])**2) - a*(sol.y[3]-np.pi/2)**2 - b/sol.y[1] - M/np.sqrt(d**2 + f**2 + sol.y[1]**2 + 2*d*np.sqrt(f**2+sol.y[1]**2*np.cos(sol.y[3])**2)),
#                           sol.t,r"$\mu c t$",r"$\psi_{eff}$",
#                           titles,"%s/poteff_%s_%d" % (ruta,nomarchvs,ncor))

        pts.los2plots3d(sol.y[1],sol.y[3],sol.y[4],sol.t,
                    "%s/orbita3d_%s_1_%d" % (ruta,nomarchvs,ncor),
                    "%s/orbita3d_%s_2_%d" % (ruta,nomarchvs,ncor), titles)  
        
        print "xmax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.cos(sol.y[4]))/15637.8, "Kpc"
        print "ymax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.sin(sol.y[4]))/15637.8, "Kpc"
        print "zmax=", np.amax(sol.y[1]*np.cos(sol.y[3]))/15637.8, "Kpc"
        print "rmax=", np.amax(sol.y[1])/15637.8, "Kpc"

    else:
        print "status", sol.status
       

def solver2(u0, uf, y0,ncor, method,max_steps, L, M, a, b, d, f, b_bar,titles,nomarchvs,ruta):
    def func(t,y):
        return [y[4]**2/(y[1]**3*np.sin(y[3])**2)
                - (b_bar/y[1]**2 + b/y[1]**2 + (M*y[1]*(1.+((d*np.cos(y[3])**2)/np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))))/(np.sqrt(d**2 + f**2 + y[1]**2 + 2*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdx(M,a,b,d,f,y),
                y[0],
                np.cos(y[3])*y[4]**2/(y[1]**2*np.sin(y[3])**3) 
                - (a*(np.pi -2.*y[3]) - (y[1]**2*d*M*np.cos(y[3])*np.sin(y[3]))/(np.sqrt(f**2+y[1]**2*np.cos(y[3])**2) * np.sqrt(d**2 + f**2 + y[1]**2 + 2.*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdtheta(M,a,b,d,f,y),
                y[2]/y[1]**2,
                0,
                y[4]/(y[1]**2*np.sin(y[3])**2)] 
    coord = np.array([y0])
    t = np.array([u0])
    if method=='LSODA':
        sols = spi.LSODA(func, u0, y0, uf, first_step=None)
    elif method=='BDF':
        sols = spi.BDF(func, u0, y0, uf, first_step=None)
    else:
        sols = spi.Radau(func, u0, y0, uf, first_step=None)
    
    sols.y 
    for i in range(max_steps):
        sols.step()
        t= np.append(t,[sols.t],axis=0)
        coord= np.append(coord,[sols.y],axis=0)
    
    print coord.shape, t.shape 
        
    pts.los3plotscoords(t,coord[:,1],coord[:,3],coord[:,4],
                    "%s/r(t)_LSODA_%s_%d" % (ruta,nomarchvs,ncor),
                    "%s/theta(t)_LSODA_%s_%d" % (ruta,nomarchvs,ncor),
                    "%s/phi(t)_LSODA_%s_%d" % (ruta,nomarchvs,ncor), titles)
    pts.los2plotsvels(t,coord[:,0],coord[:,2],
                "%s/vr(t)_LSODA_%s_%d" % (ruta,nomarchvs,ncor),
                "%s/vtheta(t)_LSODA_%s_%d" % (ruta,nomarchvs,ncor), titles)

    pts.parametricplot(coord[:,1]*np.cos(coord[:,4]),coord[:,1]*np.sin(coord[:,4]),
                       t,r"$x\mu$",r"$y\mu$",titles,
                       "%s/xy_paramt_LSODA_%s_%d" % (ruta,nomarchvs,ncor))
    
    pts.los2plots3d(coord[:,1],coord[:,3],coord[:,4],t,
                "%s/orbita3d_%s_1_LSODA_%d" % (ruta,nomarchvs,ncor),
                "%s/orbita3d_%s_2_LSODA_%d" % (ruta,nomarchvs,ncor),
                titles)

def solver3(u0,uf,du,y0,ncor, L, M, a, b, d, f, b_bar,titles,nomarchvs,ruta):
    def func(t,y,M,a,b,d,f,L,b_bar):
        return [y[4]**2/(y[1]**3*np.sin(y[3])**2)
                - (b_bar/y[1]**2 + b/y[1]**2 + (M*y[1]*(1.+((d*np.cos(y[3])**2)/np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))))/(np.sqrt(d**2 + f**2 + y[1]**2 + 2*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdx(M,a,b,d,f,y),
                y[0],
                np.cos(y[3])*y[4]**2/(y[1]**2*np.sin(y[3])**3) 
                - (a*(np.pi -2.*y[3]) - (y[1]**2*d*M*np.cos(y[3])*np.sin(y[3]))/(np.sqrt(f**2+y[1]**2*np.cos(y[3])**2) * np.sqrt(d**2 + f**2 + y[1]**2 + 2.*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdtheta(M,a,b,d,f,y),
                y[2]/y[1]**2,
                0,
                y[4]/(y[1]**2*np.sin(y[3])**2)] 
    u = np.arange(u0, uf, du)
    y = spi.odeint(func, y0, u, args=(M,a,b,d,f,L,b_bar), tfirst = True)

    pts.los3plotscoords(u,y[:, 1],y[:, 3],y[:, 4],
                        "%s/r(t)_ode_%s_%d" % (ruta,nomarchvs,ncor),
                        "%s/theta(t)_ode_%s_%d" % (ruta,nomarchvs,ncor),
                        "%s/phi(t)_ode_%s_%d" % (ruta,nomarchvs,ncor), titles)
    
    pts.los2plotsvels(u,y[:, 0],y[:, 2],
                    "%s/vr(t)_ode_%s_%d" % (ruta,nomarchvs,ncor),
                    "%s/vtheta(t)_ode_%s_%d" % (ruta,nomarchvs,ncor),
                    titles)
    
    pts.parametricplot(y[:, 1]*np.cos(y[:, 4]),y[:, 1]*np.sin(y[:, 4]),u,
                   r"$x\mu$",r"$y\mu$",
                   titles,"%s/xy_paramt_ode_%s_%d" % (ruta,nomarchvs,ncor))
#    
    pts.los2plots3d(y[:, 1],y[:, 3],y[:, 4], u,
                "%s/orbita3D_%s_1_ode_%d" % (ruta,nomarchvs,ncor),
                "%s/orbita3D_%s_2_ode_%d" % (ruta,nomarchvs,ncor), titles)