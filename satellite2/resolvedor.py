#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:16:47 2019

@author: jordis
"""
import numpy as np
import scipy.integrate as spi
import scipy.special as spe
from integral import dIx, dIth
import plots_jordi as pts
#POTENCIAL
#V(x,y)=- b/x - M/(d^2 + f^2 + x^2 + 2*d*(f^2+x^2*cos(y)^2)^(1/2))^(1/2) 
# x es \hat{\mu} r 
# y es \theta  
      
def solver(u0,uf,y0,ncor,metodo,teval,L,M,b,d,f,b_bar,titles,nomarchvs,ruta):
#    metodo= 'RK45', 'Radau' o 'LSODA' 
    def fun(t,y):
        return [L**2/(y[1]**3*np.sin(y[3])**2) + y[1]*y[2]**2 - (b_bar/y[1]**2 + b/y[1]**2 + (M*y[1]*(1.+((d*np.cos(y[3])**2)/np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))))/(np.sqrt(d**2 + f**2 + y[1]**2 + 2*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdx(M,a,b,d,f,y),
                y[0],
                -2.*y[0]*y[2]/y[1] + (L**2*np.cos(y[3]))/(y[1]**4*np.sin(y[3])**3) 
                - (- (d*M*np.cos(y[3])*np.sin(y[3]))/(np.sqrt(f**2+y[1]**2*np.cos(y[3])**2) * np.sqrt(d**2 + f**2 + y[1]**2 + 2.*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdtheta(M,a,b,d,f,y),
                y[2],
                L/(y[1]**2*np.sin(y[3])**2)]
    sol = spi.solve_ivp(fun, [u0,uf], y0, method=metodo, t_eval=teval)
    if sol.status==0:
        pts.plotsint_esf(sol.t,sol.y[1],sol.y[3],sol.y[4],
                         sol.y[0],sol.y[2],L,M,b,d,f,b_bar,
                         titles,ruta,nomarchvs,ncor) 
        print("xmax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.cos(sol.y[4]))/15637.8, "Kpc")
        print("ymax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.sin(sol.y[4]))/15637.8, "Kpc")
        print("zmax=", np.amax(np.abs(sol.y[1]*np.cos(sol.y[3])))/15637.8, "Kpc")
        print("rmax=", np.amax(sol.y[1])/15637.8, "Kpc")
    else:
        print("status", sol.status)
              
def solver2(u0, uf, y0,ncor, method,max_steps, L, M, b, d, f, b_bar,titles,nomarchvs,ruta):
    def func(t, y):
        return [L**2/(y[1]**3*np.sin(y[3])**2) + y[1]*y[2]**2 - (b_bar/y[1]**2 + b/y[1]**2 + (M*y[1]*(1.+((d*np.cos(y[3])**2)/np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))))/(np.sqrt(d**2 + f**2 + y[1]**2 + 2*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdx(M,a,b,d,f,y),
                y[0],
                -2.*y[0]*y[2]/y[1] + (L**2*np.cos(y[3]))/(y[1]**4*np.sin(y[3])**3) 
                - (- (d*M*np.cos(y[3])*np.sin(y[3]))/(np.sqrt(f**2+y[1]**2*np.cos(y[3])**2) * np.sqrt(d**2 + f**2 + y[1]**2 + 2.*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdtheta(M,a,b,d,f,y),
                y[2],
                L/(y[1]**2*np.sin(y[3])**2)] 
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
    
    print(coord.shape, t.shape )
    
    pts.plotsint_esf(t,coord[:,1],coord[:,3],coord[:,4],
                     coord[:,0],coord[:,2],L,M,b,d,f,b_bar,
                     titles,ruta,nomarchvs,ncor)
       
def solver3(u0,uf,du,y0,ncor, L, M, b, d, f, b_bar,titles,nomarchvs,ruta):
    def func(t, y,M,b,d,f,L,b_bar):
        return [L**2/(y[1]**3*np.sin(y[3])**2) + y[1]*y[2]**2 - (b_bar/y[1]**2 + b/y[1]**2 + (M*y[1]*(1.+((d*np.cos(y[3])**2)/np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))))/(np.sqrt(d**2 + f**2 + y[1]**2 + 2*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdx(M,a,b,d,f,y),
                y[0],
                -2.*y[0]*y[2]/y[1] + (L**2*np.cos(y[3]))/(y[1]**4*np.sin(y[3])**3) 
                - (- (d*M*np.cos(y[3])*np.sin(y[3]))/(np.sqrt(f**2+y[1]**2*np.cos(y[3])**2) * np.sqrt(d**2 + f**2 + y[1]**2 + 2.*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3)),#dVdtheta(M,a,b,d,f,y),
                y[2],
                L/(y[1]**2*np.sin(y[3])**2)] 
    u = np.arange(u0, uf, du)
    y = spi.odeint(func, y0, u, args=(M,b,d,f,L,b_bar), tfirst = True)
    pts.plotsint_esf(u,y[:, 1],y[:, 3],y[:, 4],y[:, 0],y[:, 2],L,M,b,d,f,b_bar,titles,ruta,nomarchvs,ncor)

def solver_green(u0,uf,du,y0,ncor,metodo,L,titles,nomarchvs,ruta,plot3d= False):
#    metodo= 'RK45', 'Radau' o 'LSODA'
    u = np.arange(u0, uf, du)
    def fun(t,y):
#        print(np.shape(dIx(y[1],y[3])))
        return [L**2/(y[1]**3*np.sin(y[3])**2) + y[1]*y[2]**2 - dIx(y[1],y[3]),
                y[0],
                -2.*y[0]*y[2]/y[1] + (L**2*np.cos(y[3]))/(y[1]**4*np.sin(y[3])**3) - dIth(y[1],y[3]),
                y[2],
                L/(y[1]**2*np.sin(y[3])**2)]
        

    sol = spi.solve_ivp(fun, [u0,uf], y0, method=metodo, t_eval=u)
    if sol.status==0:
        if plot3d== True:
            pts.plotsint_esf_green(sol.t,sol.y[1],sol.y[3],sol.y[4],sol.y[0],
                                    sol.y[2],L,titles,ruta,nomarchvs,ncor,
                                    plots3d="Si") 
        elif plot3d== False:
            pts.plotsint_esf_green(sol.t,sol.y[1],sol.y[3],sol.y[4],sol.y[0],
                                    sol.y[2],L,titles,ruta,nomarchvs,ncor,
                                    plots3d="No")
        print("xmax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.cos(sol.y[4]))/15637.8, "Kpc")
        print("ymax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.sin(sol.y[4]))/15637.8, "Kpc")
        print("zmax=", np.amax(np.abs(sol.y[1]*np.cos(sol.y[3])))/15637.8, "Kpc")
        print("rmax=", np.amax(sol.y[1])/15637.8, "Kpc")
#        pts.theanimation(sol.y[1],sol.y[3],sol.y[4],u)
#        print( len(sol.y[1]))

    else:
        print( "status", sol.status)
        
def solver_green2(u0,uf,du,y0,ncor,metodo,L,b,titles,nomarchvs,ruta,plot3d= False):
#    metodo= 'RK45', 'Radau' o 'LSODA'
# Con el potencial calculado con clebsch-gordan
    u = np.arange(u0, uf, du)
    pi = np.pi
    def dpsidx(x,th):
        b2 = b**2
        b4 = b2**2
        b3 = b**3
        b6 = b**6
        return pi*(12. - 18.*b2*x**2 + 4*b4*x**4 -12.*np.cos(2.*b*x)*(1. + 3.*(-2. + b2*x**2)*np.cos(th)**2) - 3.*b*x*np.sin(2.*b*x) + np.cos(th)**2*(18.*(-4. + b2*x**2) + (81.*b*x - 6.*b3*x**3)*np.sin(2*b*x)))/(2*b6*x**5)
    
    def dpsidth(x,th):
        b2 = b**2
        b6 = b**6
        return -3.*pi*np.sin(2.*th)*(6. - 3.*b2*x**2 + (-6. + b2*x**2)*np.cos(2.*b*x) - 5.*b*x*np.sin(2.*b*x))/(2.*b6*x**6)
        
    def fun(t,y):
        return [L**2/(y[1]**3*np.sin(y[3])**2) + y[1]*y[2]**2 - dpsidx(y[1],y[3]),
                y[0],
                -2.*y[0]*y[2]/y[1] + (L**2*np.cos(y[3]))/(y[1]**4*np.sin(y[3])**3) - dpsidth(y[1],y[3]),
                y[2],
                L/(y[1]**2*np.sin(y[3])**2)] 

    sol = spi.solve_ivp(fun, [u0,uf], y0, method=metodo, t_eval=u)
    if sol.status==0:
        if plot3d== True:
            pts.plotsint_esf_green(sol.t,sol.y[1],sol.y[3],sol.y[4],sol.y[0],
                                    sol.y[2],L,titles,ruta,nomarchvs,ncor,
                                    plots3d="Si") 
        elif plot3d== False:
            pts.plotsint_esf_green(sol.t,sol.y[1],sol.y[3],sol.y[4],sol.y[0],
                                    sol.y[2],L,titles,ruta,nomarchvs,ncor,
                                    plots3d="No")
        print("xmax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.cos(sol.y[4]))/15637.8, "Kpc")
        print("ymax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.sin(sol.y[4]))/15637.8, "Kpc")
        print("zmax=", np.amax(np.abs(sol.y[1]*np.cos(sol.y[3])))/15637.8, "Kpc")
        print("rmax=", np.amax(sol.y[1])/15637.8, "Kpc")
#        pts.theanimation(sol.y[1],sol.y[3],sol.y[4],u)
#        print( len(sol.y[1]))

    else:
        print( "status", sol.status)

#mu = 15637.8 #1/pc
#R = 1e2
#n = 1e4
#mu= 1.
#
#ri = 2.
#rf = 300.
#Ra, Th = np.meshgrid(np.linspace(ri,rf, 100), np.linspace(0., np.pi, 100))
#pi = np.pi
#def dpsidx(x,th):
#    return np.exp(-x**2/(R**2*mu**2))*n*(8.*R**4*x**3*mu**4 + 42.*R**6*x*mu**6 - 2.*x*(8.*x**6 + 20.*R**2*x**4*mu**2 + 42.*R**4*x**2*mu**4 + 63.*R**6*mu**6)*np.cos(th)**2 + np.exp(x**2/(R**2*mu**2))*np.sqrt(np.pi)*R**5*mu**5*(10.*x**2 - 21.*R**2*mu**2 + 63.*R**2*mu**2*np.cos(th)**2)*spe.erf(x/(R*mu)))/(10.*np.sqrt(np.pi)*R**5*x**4*mu**2)
#
#def dpsidth(x,th):
#    return np.absolute(-np.exp(-x**2/(R**2*mu**2))*n*np.cos(th)*np.sin(th)*(8.*x**5 + 28.*R**2*x**3*mu**2 - 21.*np.exp(x**2/(R**2*mu**2))*np.sqrt(np.pi)*R**5*mu**5*spe.erf(x/(R*mu)))/(5.*np.sqrt(np.pi)*R**3*x**5))
#def dpsidx(x,th):
#    Rmu2 = R**2*mu**2
#    Rmu4 = Rmu2**2
#    Rmu5 = R**5*mu**5
#    Rmu6 = R**6*mu**6
#    return n*(np.exp(-x**2/Rmu2)*(8.*Rmu4*x**3* + 42.*Rmu6*x - 2.*x*(8.*x**6 + 20.*Rmu2*x**4 + 42.*Rmu4*x**2 + 63.*Rmu6)*np.cos(th)**2) + np.sqrt(np.pi)*Rmu5*(10.*x**2 - 21.*Rmu2 + 63.*Rmu2*np.cos(th)**2)*spe.erf(x/(R*mu)))/(10.*np.sqrt(pi)*Rmu2*R**3*x**4)
#
#def dpsidth(x,th):
#    Rmu2 = R**2*mu**2
#    Rmu4 = Rmu2**2
#    Rmu5 = R**5*mu**5
#    return np.absolute(-n*np.cos(th)*np.sin(th)*(np.exp(-x**2/Rmu2)*(8.*x**5 + 28.*Rmu2*x**3 +42.*Rmu4*x)
#                                  - 21.*np.sqrt(np.pi)*Rmu5*spe.erf(x/(R*mu)))/(5.*np.sqrt(np.pi)*R**3*x**5))
# 
   
#
#pts.densityplot(Ra,Th,-dpsidx(Ra,Th),r'$\mu r$',r'$\theta$',r'$-\partial_r V$','title','name')
#pts.densityplot(Ra,Th,-dpsidth(Ra,Th),r'$\mu r$',r'$\theta$',r'$-\partial_\theta V$','title','name')
#pts.plotfunc3d(Ra,Th,-dpsidx(Ra,Th),r'$r$',r'$\theta$',r'$-\partial_r V$','title')#
#pts.plotfunc3d(Ra,Th,-dpsidth(Ra,Th),r'$r$',r'$\theta$',r'$-\partial_r V$','title')#

def solver_artpaco(u0,uf,du,y0,ncor,metodo,L,mu,R,n,titles,nomarchvs,ruta,plot3d= False):
#    metodo= 'RK45', 'Radau' o 'LSODA'
    u = np.arange(u0, uf, du)
    pi = np.pi
    Rmu2 = R**2*mu**2
    Rmu4 = Rmu2**2
    Rmu5 = R**5*mu**5
    Rmu6 = R**6*mu**6
    
    def dpsidx(x,th):
        return n*(np.exp(-x**2/Rmu2)*(8.*Rmu4*x**3* + 42.*Rmu6*x - 2.*x*(8.*x**6 + 20.*Rmu2*x**4 + 42.*Rmu4*x**2 + 63.*Rmu6)*np.cos(th)**2) + np.sqrt(np.pi)*Rmu5*(10.*x**2 - 21.*Rmu2 + 63.*Rmu2*np.cos(th)**2)*spe.erf(x/(R*mu)))/(10.*np.sqrt(pi)*Rmu2*R**3*x**4)
    
    def dpsidth(x,th):
        return np.absolute(-n*np.cos(th)*np.sin(th)*(np.exp(-x**2/Rmu2)*(8.*x**5 + 28.*Rmu2*x**3 +42.*Rmu4*x)
                                      - 21.*np.sqrt(np.pi)*Rmu5*spe.erf(x/(R*mu)))/(5.*np.sqrt(np.pi)*R**3*x**5))
        

    def fun(t,y):
        return np.array([L**2/(y[1]**3*np.sin(y[3])**2) + y[1]*y[2]**2 - dpsidx(y[1],y[3]),
                y[0],
                -2.*y[0]*y[2]/y[1] + (L**2*np.cos(y[3]))/(y[1]**4*np.sin(y[3])**3) - dpsidth(y[1],y[3]),
                y[2],
                L/(y[1]**2*np.sin(y[3])**2)], dtype='float64')

    sol = spi.solve_ivp(fun, [u0,uf], y0, method=metodo, t_eval=u)
    
    if sol.status==0:
        if plot3d== True:
            pts.plotsint_esf_green(sol.t,sol.y[1],sol.y[3],sol.y[4],sol.y[0],
                                    sol.y[2],L,titles,ruta,nomarchvs,ncor,
                                    plots3d="Si") 
        elif plot3d== False:
            pts.plotsint_esf_green(sol.t,sol.y[1],sol.y[3],sol.y[4],sol.y[0],
                                    sol.y[2],L,titles,ruta,nomarchvs,ncor,
                                    plots3d="No")
        print("xmax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.cos(sol.y[4]))/15637.8, "Kpc")
        print("ymax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.sin(sol.y[4]))/15637.8, "Kpc")
        print("zmax=", np.amax(np.abs(sol.y[1]*np.cos(sol.y[3])))/15637.8, "Kpc")
        print("rmax=", np.amax(sol.y[1])/15637.8, "Kpc")

    else:
        print("status", sol.status)
        
def solver_cart(u0,uf,y0,ncor,metodo,teval,M,b,d,f,titles,nomarchvs,ruta):
    #metodo= 'RK45', 'Radau' o 'LSODA'
    mu = 15.6378 # en pc
    def fun(t,y):
        vx,x,vy,Y,vz,z = y
        return [- b*x/(mu*np.sqrt(x**2 + Y**2 + z**2)**3) 
                - M*x*mu**2/np.sqrt(d**2 + f**2 + (x**2 + Y**2 + z**2)*mu**2 + 2.*d*np.sqrt(f**2 + z**2*mu**2))**3,
                vx,
                - b*Y/(mu*np.sqrt(x**2 + Y**2 + z**2)**3) 
                - M*Y*mu**2/np.sqrt(d**2 + f**2 + (x**2 + Y**2 + z**2)*mu**2 + 2.*d*np.sqrt(f**2 + z**2*mu**2))**3,
                vy,
                - b*z/(mu*np.sqrt(x**2 + Y**2 + z**2)**3) 
                - M*z*mu**2*(1+ d/np.sqrt(f**2 + z**2*mu**2))/np.sqrt(d**2 + f**2 + (x**2 + Y**2 + z**2)*mu**2 + 2.*d*np.sqrt(f**2 + z**2*mu**2))**3,
                vz]

    sol = spi.solve_ivp(fun, [u0,uf], y0, method=metodo, t_eval=teval)
    if sol.status==0:
        t = sol.t
        x = sol.y[1]
        Y = sol.y[3]
        z = sol.y[5]
        vx = sol.y[0]
        vy = sol.y[2]
        vz = sol.y[4]
        pts.parametricplot(t,x,t,r'$\hat{\mu}ct$',r'$\hat\mu x$',titles, "%s/x(t)_%s_%d" % (ruta,nomarchvs,ncor))
        pts.parametricplot(t,Y,t,r'$\hat{\mu}ct$',r'$\hat\mu y$',titles, "%s/y(t)_%s_%d" % (ruta,nomarchvs,ncor))
        pts.parametricplot(t,z,t,r'$\hat{\mu}ct$',r'$\hat\mu z$',titles, "%s/z(t)_%s_%d" % (ruta,nomarchvs,ncor))        
        pts.parametricplot(t,vx,t,r'$\hat{\mu}ct$',r'$v_x/c$',titles, "%s/vx(t)_%s_%d" % (ruta,nomarchvs,ncor))
        pts.parametricplot(t,vy,t,r'$\hat{\mu}ct$',r'$v_y/c$',titles, "%s/vy(t)_%s_%d" % (ruta,nomarchvs,ncor))
        pts.parametricplot(t,vz,t,r'$\hat{\mu}ct$',r'$v_z/c$',titles, "%s/vz(t)_%s_%d" % (ruta,nomarchvs,ncor))
        pts.parametricplot(x,Y,t,r"$x\mu$",r"$y\mu$",
                           titles,"%s/xy_paramt_%s_%d" % (ruta,nomarchvs,ncor)) 
        pts.parametricplot(t, x**2/2. + Y**2/2. + z**2/2. - b/(mu*np.sqrt(x**2 + Y**2 + z**2)) - M/np.sqrt(d**2 + f**2 + (x**2 + Y**2 + z**2)*mu**2 + 2.*d*np.sqrt(f**2 + z**2*mu**2)),
                           t,r"$\mu c t$",r"$E$",
                           titles,"%s/energia_%s_%d" % (ruta,nomarchvs,ncor))

        pts.plot3d(x,Y,z,t,'x','y','z', 45,30,titles,"%s/orbita3d_%s_1_%d" % (ruta,nomarchvs,ncor))
        pts.plot3d(x,Y,z,t,'x','y','z', 20,100,titles,"%s/orbita3d_%s_2_%d" % (ruta,nomarchvs,ncor))

#        print("xmax=", np.amax(x)/15637.8, "Kpc")
#        print( "ymax=", np.amax(Y)/15637.8, "Kpc")
#        print( "zmax=", np.amax(np.abs(z))/15637.8, "Kpc")
#        print( "rmax=", np.amax(np.sqrt(z**2+Y**2+x**2))/15637.8, "Kpc")

    else:
        print( "status", sol.status)
        
def solver_art_luis(u0,uf,du,y0,ncor,metodo,E,titles,nomarchvs,ruta,plot3d= False):
#    metodo= 'RK45', 'Radau' o 'LSODA'
    u = np.arange(u0, uf, du)
 
    def fun(t,y):
        V0,B0,V2,B2,V4,B4,R2,T = y
        r=t
        return [B0/r**2,
                R2**2*r**2/5.,
                B2/r**2,
                6.*V2 + 2.*R2**2*r**2/7.,
                B4/r**2,
                20.*V4 + 18.*R2**2*r**2/35.,
                T/r**2,
                2.*R2*(3. + (V0 + (2./7.)*(V2 + V4) - E)*r**2)]
    a = (ruta,nomarchvs,ncor)    
    sol = spi.solve_ivp(fun, [u0,uf], y0, method=metodo, t_eval=u)
    if sol.status==0:
        if plot3d== False:
            pts.parametricplot(sol.t, sol.y[0],'xlabel', r'$V_0$',titles,"%s/v0(r)_%s_%d" % a,ylim=(0,0))
            pts.parametricplot(sol.t, sol.y[2],'xlabel', r'$V_2$',titles,"%s/v2(r)_%s_%d" % a,ylim=(0,0))
            pts.parametricplot(sol.t, sol.y[4],'xlabel', r'$V_4$',titles,"%s/v4(r)_%s_%d" % a,ylim=(0,0))
            pts.parametricplot(sol.t, sol.y[6]**2,'xlabel', r'$R_2$',titles,"%s/R2(r)_%s_%d" % a,ylim=(0,0))
        elif plot3d== True:
            pts.parametricplot(sol.t, sol.y[0],'xlabel','ylabel','title',nomarchvs,ylim=(0,0))
#        print("xmax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.cos(sol.y[4]))/15637.8, "Kpc")
#        print("ymax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.sin(sol.y[4]))/15637.8, "Kpc")
#        print("zmax=", np.amax(np.abs(sol.y[1]*np.cos(sol.y[3])))/15637.8, "Kpc")
#        print("rmax=", np.amax(sol.y[1])/15637.8, "Kpc")

    else:
        print("status", sol.status)
#ri=1.5
#u0= [-0.0019, 0., 0., -1.*ri**2, 0., -1.*ri**2, 0., 0.5]
#solver_art_luis(ri, 100., 10., u0, 1, 'Radau',
#                -0.0013,'expansion', 'expansion', '/home/jordi/satellite/art_luis')