#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:17:09 2019

@author: jordis
"""
import numpy as np
import scipy.integrate as spi
import scipy.special as spe
import time
import quadpy
#
#import skmonaco
#
import plots_jordi as pts

inf = np.inf
# spe.spherical_jn(1,kmu*xp)**2
# spe.jv(1.5, kmu*xp)**2
# print np.lib.scimath.sqrt(-1)
pi = np.pi
kmu = 0.458263#k/mu
lam = -0.0523222

def plxx(l):
    "integral de x^2 * Pl^2(x) en -1<x<1 "
    return 2.*(2.*l**2 +2.*l - 1.)/((2.*l - 1.)*(2.*l + 1.)*(2.*l + 3.))

def pl(l):
    "integral de Pl^2(x)dx en -1<x<1 "
    return 2./(2.*l + 1.)

def Jx(Nu, P, a):
    "integral de J_nu(a x)^2 x dx de 0<x<P"
    return (P**2/2.)*((spe.jv(-1 + Nu, a*P) - spe.jv(1 + Nu, a*P))**2 /4. + (1. - Nu/(a**2*P**2))*spe.jv(Nu, a*P)**2)

def Jxxx(nu, P, a):
    return spi.quad(lambda x: x**3*spe.jv(nu,a*x)**2, 0.,P)[0]

jx = Jx(1.5,100.,kmu)#=69.0087
jxxx = Jxxx(1.5,100.,kmu) #=227507.
p1xx = plxx(1)
p1 = pl(1)

def dIx(x,th):
    val= -4.*pi*jx/(3.*x**2) + (6.*pi/(3.*x**4) - 9.*pi*(np.cos(th)**2-np.sin(th)**2/2.)*p1xx/x**4. - 3.*pi*np.sin(th)**2/x**4)*jxxx
    return lam*pi*val/(2*kmu)

def dIth(x,th):
    val =  3.*pi*np.cos(th)*np.sin(th)*(-3.*p1xx + p1)*jxxx/x**4
    return lam*pi*val/(2*kmu*x)

def I1(x,th):
    "Integral de green aprox distancias grandes"
    val= 4.*pi*jx/(3.*x) + (-2.*pi/(3.*x**3) + 3.*pi*(np.cos(th)**2 - np.sin(th)**2/2.)*p1xx/x**3. + pi*np.sin(th)**2/x**3)*jxxx
    return lam*pi*val/(2*kmu)

def Il(x,th,l,P):
    val= 4.*pi*Jx(l + 0.5,P,kmu)/(3.*x) + (-2.*pi/(3.*x**3) 
    + 3.*pi*(np.cos(th)**2 - np.sin(th)**2/2.)*plxx(l)/x**3. 
    + pi*np.sin(th)**2/x**3)*Jxxx(l + 0.5,P, kmu)
    return lam*pi*val/(2*kmu)

def psi_green(x,th):
    phi = 0.
    def f(xp,thp,phip):
        return pi*spe.jv(1.5,kmu*xp)**2*spe.legendre(1)(np.cos(thp))**2*xp*np.sin(thp)/(2*kmu*np.sqrt(x**2 + xp**2 - 2.*x*xp*(np.cos(th)*np.cos(thp) + np.sin(th)*np.sin(thp)*np.cos(phi-phip))))

    def opts0(thp,phip):
        return {'points':[x, th, phi]}
    psi = spi.nquad(f, [[0., 100.],[0., pi],[0.,2*pi]],
                              opts=[opts0,{},{}],
                              full_output=True)[0]
    return psi

def psi_green2(x,th):
    phi = 0.
    def f(y):
        xp,thp,phip= y
        return spe.jv(1.5,kmu*xp)**2*spe.legendre(1)(np.cos(thp))**2*xp*np.sin(thp)/np.sqrt(x**2 + xp**2 - 2.*x*xp*(np.cos(th)*np.cos(thp) + np.sin(th)*np.sin(thp)*np.cos(phi-phip)))
    val = skmonaco.mcquad(f, xl=np.zeros(3), xu=[100.,pi,2*pi], npoints=1e5,nprocs = 1)[0]
    return lam*pi*val/(2*kmu)

def psi_green3(xo,th):
    phi = 0.
    dim = 3
    val=quadpy.ncube.integrate(lambda x: spe.jv(1.5,kmu*x[0])**2*spe.legendre(1)(np.cos(x[1]))**2*x[0]*np.sin(x[1])/np.sqrt(xo**2 + x[0]**2 - 2.*xo*x[0]*(np.cos(th)*np.cos(x[1]) + np.sin(th)*np.sin(x[1])*np.cos(phi-x[2]))), 
                               quadpy.ncube.ncube_points([0.0, 100.0], [0., pi], [0., 2*pi]),
                               quadpy.ncube.Stroud(dim, 'Cn 3-3'))
    return lam*pi*val/(2*kmu)

def psi_g3(R,Th):
    "para graficar psi_green3"
    return np.array([[psi_green3(R[j,i], Th[j,i]) for i in range(0,np.shape(R)[1],1)] for j in range(0,np.shape(R)[0])])
    
def derx(x,th):
    phi = 0.  
    def fdx(xp,thp,phip):
        return (-x+xp*(np.cos(th)*np.cos(thp) + np.sin(th)*np.sin(thp)*np.cos(phi-phip)))*spe.spherical_jn(1,kmu*xp)**2*spe.legendre(1)(np.cos(thp))**2*xp**2*np.sin(thp)/np.sqrt(x**2 + xp**2 - 2*x*xp*(np.cos(th)*np.cos(thp) + np.sin(th)*np.sin(thp)*np.cos(phi - phip)))**3
    def opts0(thp,phip):
        return {'points':[x, th, phi]}     
    dpsidx = spi.nquad(fdx, [[0., 100.],[0., pi],[0.,2*pi]],
                              opts=[opts0,{},{}],
                              full_output=True)[0]
    return dpsidx

def derx2(x,th):
    phi = 0.
    def fdx(y):
        xp,thp,phip= y
        return (-x+xp*(np.cos(th)*np.cos(thp) + np.sin(th)*np.sin(thp)*np.cos(phi-phip)))*spe.jv(1.5,kmu*xp)**2*spe.legendre(1)(np.cos(thp))**2*xp*np.sin(thp)/np.sqrt(x**2 + xp**2 - 2*x*xp*(np.cos(th)*np.cos(thp) + np.sin(th)*np.sin(thp)*np.cos(phi - phip)))**3
    val = skmonaco.mcquad(fdx, xl=np.zeros(3), xu=[100.,pi,2*pi],npoints=1e5,nprocs = 1)[0]
    return lam*pi*val/(2*kmu)
    
def derth(x,th):
    phi = 0. 
    def fdth(xp,thp,phip):
        return (-np.sin(th)*np.cos(thp) + np.cos(th)*np.sin(thp)*np.cos(phi-phip))*spe.legendre(1)(np.cos(thp))**2*spe.jv(1.5, kmu*xp)**2*xp**2*np.sin(thp)/np.sqrt(x**2 + xp**2 - 2*x*xp*(np.cos(th)*np.cos(thp) + np.sin(th)*np.sin(thp)*np.cos(phi - phip)))**3
    def opts0(thp,phip):
        return {'points':[x, th, phi]}     
    dpsidth = spi.nquad(fdth, [[0.,100.],[0., pi],[0.,2*pi]],
                              opts=[opts0,{},{}],
                              full_output=True)[0]
    # dpsidth = spi.nquad(fdth, [[0.,inf],[0., pi],[0.,2*pi]],
    #                               full_output=True)[0]
    return dpsidth

def derth2(x,th):
    phi = 0.
    def fdth(y):
        xp,thp,phip= y
        return (-np.sin(th)*np.cos(thp) + np.cos(th)*np.sin(thp)*np.cos(phi-phip))*spe.legendre(1)(np.cos(thp))**2*spe.jv(1.5, kmu*xp)**2*xp**2*np.sin(thp)/np.sqrt(x**2 + xp**2 - 2*x*xp*(np.cos(th)*np.cos(thp) + np.sin(th)*np.sin(thp)*np.cos(phi - phip)))**3
    
    val = skmonaco.mcquad(fdth, xl=np.zeros(3),
                           xu=[100.,pi,2*pi],
                           npoints=1e5,nprocs = 1)[0]
    return lam*pi*val/(2*kmu*x)

def ejem1():
    def func(x0,x1,x2,x3) : 
        return x0**2 + x1*x2 - x3**3 + np.sin(x0) + (1 if (x0 - 0.2*x3 - 0.5 - 0.25*x1>0) else 0)
    points = [[lambda x1,x2,x3 : 0.2*x3 + 0.5 + 0.25*x1], [], [], []]
    def opts0(*args, **kwargs):
        return {'points':[0.2*args[2] + 0.5 + 0.25*args[0]]}
    print(spi.nquad(func, [[0,1], [-1,1], [.13,.8], [-.15,1]],opts=[opts0,{},{},{}], full_output=True))

def ejem2():
    def func2(x0, x1, x2, x3, t0, t1,t2):
        return x0**2 + x1*x2 - x3**3 + np.sin(x0) + (1 if (x0 + t1*x3 - t0 + t2*x1>0) else 0)
    
    def opts0(x1, x2, x3, t0, t1,t2):
        return {'points' : [-t1*x3+ t0 - t2*x1]}
    def opts1(x2, x3, t0, t1,t2):
        return {}
    def opts2(x3, t0, t1,t2):
        return {}
    def opts3(t0):
        return {}
#    print spi.nquad(func, [[0,1], [-1,1], [.13,.8], [-.15,1]], opts=[opts0, opts1, opts2, opts3], full_output=True)

def ejem3():
    def func3(x0,x1,x2,x3) : 
        return x0**2 + x1*x2 - x3**3 + np.sin(x0) + (1 if (x0 - 0.2*x3 - 0.5 - 0.25*x1>0) else 0)
    points = [[lambda x1,x2,x3 : 0.2*x3 + 0.5 + 0.25*x1], [], [], []]
    def opts0(x1,x2,x3):
        return {'points':[0.2*x3 + 0.5 + 0.25*x1]}    
#    print opts0(1,1,1)
#    print spi.nquad(func3, [[0,1], [-1,1], [.13,.8], [-.15,1]],opts=[opts0,{},{},{}], full_output=True)

def ejem_monaco():
    def f(*args):
        # f(x1, x2, ... , xn) = exp(-x1^2 - x2^2 - ... – xn^2)
        return np.exp(-np.sum(np.array(args)**2))
    start = time.time()
#    print spi.nquad(f, [(0,1)] * 5)
    end = time.time()
#    print "tiempo=", (end - start), "seg"
    
    start = time.time()
    val, err = skmonaco.mcquad(f, xl=np.zeros(5),
                               xu=np.ones(5), npoints=100000)
#    print val, err
#    print np.zeros(5)
    end = time.time()
#    print "tiempo=", (end - start), "seg"
    
    start = time.time()
    val, err = skmonaco.mcquad(f, xl=np.zeros(10),
                               xu=np.ones(10), npoints=100000)
#    print val, err
    end = time.time()
#    print "tiempo=", (end - start), "seg"

#############################################################################
#############               Plot Potencial              #####################
### #############################################################################    
#R, Th = np.meshgrid(np.linspace(100,410, 100),np.linspace(0, np.pi, 100))
#
#start = time.time()
#pts.plotfunc3d(R,Th,psi_green2(R,Th),r'$\mu r$',r'$\theta$',
#          r'$\psi(r,\theta)/c^2$',"Potential",
#          name='pot_green_monte2', rot_azimut= -45)
#end = time.time()
#print "tiempo=", (end - start), "seg"
#
#start = time.time()
#pts.plotfunc3d(R,Th,psi_g3(R,Th),r'$\mu r$',r'$\theta$',
#          r'$\psi(r,\theta)/c^2$',"Potential",
#          name='pot_green_quadpy', rot_azimut= -45)
#end = time.time()
#print "tiempo=", (end - start), "seg"
#
#start = time.time()
#pts.plotfunc3d(R,Th,I1(R,Th),r'$\mu r$',r'$\theta$',
#          r'$\psi(r,\theta)/c^2$',"Potential",
#          name ='pot_green_2', rot_azimut= -45)
#end = time.time()
#print "tiempo=", (end - start), "seg"

#start = time.time()
#pts.plotfunc3d(R,Th,derx2(R,Th),r'$\mu r$',r'$\theta$',
#   r'$\frac{1}{\mu c^2} \frac{d \psi(r,\theta)}{dr}$',
#         "Radial derivative of the potential",
#         name='dpotdx_green_monte2')
#end = time.time()
#print "tiempo=", (end - start), "seg"

#start = time.time()
#pts.plotfunc3d(R,Th,dIx(R,Th,0),r'$\mu r$',r'$\theta$',
#   r'$\frac{1}{\mu c^2} \frac{d \psi(r,\theta)}{dr}$',
#         "Radial derivative of the potential",
#         name='dpotdx_green_3')
#end = time.time()
#print "tiempo=", (end - start), "seg"

# start = time.time()
# pts.plotfunc3d(R,Th,derth2(R,Th),r'$\mu r$',r'$\theta$',
#          r'$\frac{1}{\mu^2 r^2 c^2} \frac{d \psi(r,\theta)}{d\theta}$',
#          "Polar derivative of the potential",
#          name='dpotdth_green_monte2')
# end = time.time()
# print "tiempo=", (end - start), "seg"

# start = time.time()
# pts.plotfunc3d(R,Th,dIth(R,Th),r'$\mu r$',r'$\theta$',
#          r'$\frac{1}{\mu^2 r^2 c^2} \frac{d \psi(r,\theta)}{d\theta}$',
#          "Polar derivative of the potential",
#          name='dpotdth_green_2')
# end = time.time()
# print "tiempo=", (end - start), "seg"

# pts.densityplot(R,Th,psi_green2(R,Th),r'$\mu r$',r'$\theta$',
#           r'$\psi(r,\theta)/c^2$',"Potential",
#           name='pot_density_green_monte2')

