#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:40:50 2019

@author: jordis
"""
#####################################################################################
#####                     VER BOSOM_STAR_SHOOTING_SOLVER_N
#####################################################################################
import numpy as np
import scipy.optimize as opt
import scipy.integrate as spi
import plots_jordi as pts
#from solver_ADM import resol_ADM
    
#def rhs(x, y, args = {alpha}):
def rhs(x, y, alpha):
    """     Compute the rhs of y'(t) = f(t, y, params)    """
    V,P,S,F,E = y
    return np.array([P/x**2,
            x**2*S**2,
            F/x**2,
            2.*x**2*S*(V - E - alpha/x),
            0.])   
        
def integrate_over_domain(initial_conditions, integrator, alpha, r0,
                          rf, step, silent=True):
    integrator.set_initial_value(initial_conditions, t=r0)  # Set the initial values of z and x
    integrator.set_f_params(alpha)
    dt = step
    length = rf
    xs, zs = [], []
    while integrator.successful() and integrator.t <= length:
        integrator.integrate(integrator.t + dt)
        xs.append(integrator.t)
        zs.append([integrator.y[0], integrator.y[1],integrator.y[2], integrator.y[3], integrator.y[4]])
        if not silent:
            print("Current x and z values: ", integrator.t, integrator.y)
    zs = np.array(zs)
    xs = np.array(xs)
    return xs, zs

def solve_bvp(ye_bv, yes_at_0_guess, alpha, r0, rf, step):
#    integrator = spi.ode(rhs).set_integrator("dopri5")
    integrator = spi.ode(rhs).set_integrator("dop853")
#    integrator = spi.ode(rhs).set_integrator("lsoda")
    def residuals(yes_at_0,V_at_length, Vprime_at_0, Phi_at_length, Phi_prime_at_0):
         
        z_at_0 = [yes_at_0[0], Vprime_at_0, yes_at_0[1], Phi_prime_at_0, yes_at_0[2]]                
        xs, zs = integrate_over_domain(z_at_0, integrator, alpha, r0, rf, step)
        V_at_length_integrated = zs[-1, 0]
        Phi_at_length_integrated = zs[-1, 2]        
        return [V_at_length - V_at_length_integrated,
                Phi_at_length - Phi_at_length_integrated]    
    V_at_length, Vprime_at_0, Phi_at_length, Phi_prime_at_0 = ye_bv
    lsq = opt.least_squares(residuals, yes_at_0_guess, 
                            args=(V_at_length, Vprime_at_0, Phi_at_length, Phi_prime_at_0), loss="soft_l1")
    V_at_0_estimate = lsq.x[0]
    Phi_at_0_estimate = lsq.x[1]
    E_estimate = lsq.x[2]    
    return [V_at_0_estimate, Phi_at_0_estimate, E_estimate]

def main(ye_bv, yes_at_0_guess, alpha, r0, rf, step):    
    integrator = spi.ode(rhs).set_integrator("dopri5")
#    integrator = spi.ode(rhs).set_integrator("dop853")
#    integrator = spi.ode(rhs).set_integrator("lsoda")
    integrator.set_initial_value([yes_at_0_guess[0], ye_bv[1], yes_at_0_guess[1],
                                  ye_bv[3], yes_at_0_guess[2]], t=r0)  # Set the initial values
    V_at_0_estimate, Phi_at_0_estimate, E_estimate = solve_bvp(ye_bv,
                                                               yes_at_0_guess,
                                                               alpha, r0,rf,step)
    _, Vprime_at_0, _, Phi_prime_at_0 = ye_bv
    xs, zs = integrate_over_domain([V_at_0_estimate, Vprime_at_0, Phi_at_0_estimate,
                                    Phi_prime_at_0, E_estimate],integrator, alpha, r0, rf, step)
    return [xs, zs, E_estimate, alpha]

def plotses(x, ys, tit, ruta, ncor, sv = False):
    pts.parametricplot(x,ys[:,0], r'$\hat\mu r$',r"$\psi/c^2$",tit,
                       '%s/psi_%d' % (ruta, ncor), save = sv)
    pts.parametricplot(x,ys[:,2],r'$\hat\mu r$',r"$\Phi$", tit,
                       '%s/phi_%d' % (ruta, ncor), save = sv)
    pts.parametricplot(x,ys[:,4],r'$\hat\mu r$',r"$E/\mu c$", tit,
                       '%s/E_%d' % (ruta, ncor), save = sv)
    pts.parametricplot(x,ys[:,2]**2,r'$\hat\mu r$',r"$\Phi^2$", tit,
                       '%s/phi_sqr_%d' % (ruta, ncor),logy=True, save = sv)

def Area(f,x,xf,dx):
    "integral definida de f**2 x**2 dr  de 0 a xf"
    A=0.
    elem = int(np.rint(xf/dx))
    for i in range(0,elem,1):
        A+= dx*f[i]**2*x[i]**2
#    return [A, np.sqrt(A)] 
    return A

def Mass(x, Phi, r, dr):
    "integral de Phi(r)**2 r**2 dr  de 0 a x"
    xy = np.arange(0.01, x, dr)
    M = np.zeros([len(xy)])
    for i in range(0, np.shape(xy)[0], 1):
        M[i] = Area(Phi, r, xy[i], dr)
    return [xy,M]

def g_acc(x, Phi, r, dr):
    '-M(x)/x**2'
    xy = np.arange(0.01,x, dr)
    g = np.zeros([len(xy)])
    for i in range(0,np.shape(xy)[0],1):
        g[i] = - Area(Phi, r, xy[i], dr)/xy[i]**2
    return np.array([xy,g])

def Masayaccplot(x,M,x2,g, tit, ncor):
    pts.parametricplot(x,M, r'$\hat\mu r$',r"$M$",tit,
                       '/home/jordi/satellite/schrodinger_poisson_shooting/ground_state/M_%d' % ncor,
                       save = True)
    pts.parametricplot(x2,g,r'$\hat\mu r$',r"$-\hat{g}_h(x)$", tit,
                       '/home/jordi/satellite/schrodinger_poisson_shooting/ground_state/g_%d' % ncor,
                       save = True)
#    
#r0 = 0.01
#rf = 10.
#stp = 0.01
###
#V_at_length = 0.
#Vprime_at_0 = 0.
#Phi_at_length = 0.
#Phi_prime_at_0 = 0.
###
#ye_boundvalue = [V_at_length, Vprime_at_0, Phi_at_length, Phi_prime_at_0]
#V0_guess = -1.3418  
#Phi0_guess = 1. 
##
#alpha = 0. 
#for i in range(0,100,10):
#    E_guess= -i/100
#    al0 = main(ye_boundvalue, [V0_guess, Phi0_guess, E_guess], alpha, r0, rf, stp)
##    a = Area(al0[1][:,2], al0[0],rf,stp)[1]
#    pts.parametricplot(al0[0],al0[1][:,2],'','','','')
#    print('guess=', E_guess, 'E=',al0[2])
##    print(V0_guess, Phi0_guess)
#    print(al0[1][0,:])
    
#E_guess= -0.7
#al0 = main(ye_boundvalue, [V0_guess, Phi0_guess, E_guess], alpha, r0, rf, stp)
##pts.parametricplot(al0[0],al0[1][:,2],'','','','')
#print('E_guess=', E_guess, 'E=',al0[2])
#lam = 1./np.sqrt(al0[1][0,2])
##lam = 3.6*1e-4
#al0[0]= al0[0]*lam
#al0[1][:,2] = lam**2*al0[1][:,2]
#al0[1][:,0] = lam**2*al0[1][:,0]
#print('M=',Area(al0[1][:,2],al0[0],rf,stp))
#print('M_pl^2/m=',2.19*1e12/Area(al0[1][:,2],al0[0],rf,stp))
#cons = 2.19*1e12/Area(al0[1][:,2],al0[0],rf,stp)
#
#lam = 3.6*1e-4
#al0[0]= al0[0]*lam
#al0[1][:,2] = lam**2*al0[1][:,2]
#al0[1][:,0] = lam**2*al0[1][:,0]
#print('M0=',Area(al0[1][:,2],al0[0],rf,stp))
#print('M=',cons*Area(al0[1][:,2],al0[0],rf,stp))
#pts.parametricplot(al0[0],al0[1][:,2],'','','','')
##pts.parametricplot(al0[0],al0[1][:,0],'','','','')
#print('En=', al0[2]*lam**2)
