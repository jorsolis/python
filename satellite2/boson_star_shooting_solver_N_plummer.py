#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:40:50 2019

@author: jordis
"""
import numpy as np
import scipy.optimize as opt
import scipy.integrate as spi
#import plots_jordi as pts
#from solver_ADM import resol_ADM
    
#def rhs(x, y, args = {alpha}):
def rhs(x, y, alpha, alpha2, a0):
    """     Compute the rhs of y'(t) = f(t, y, params)    """
    V,P,S,F,E, N = y ####  potencial V, r^2dV/dr, Phi, r^2 dPhi/dr, eigenvalor, numero particulas
    return np.array([P/x**2,
            x**2*S**2,
            F/x**2,
            2.*x**2*S*(V - E - alpha/x - alpha2/np.sqrt(x**2 + a0**2)),
            0.,
            S**2*x**2])   
        
def integrate_over_domain(initial_conditions, integrator, alpha, alpha2, a0, r0,
                          rf, step, silent=True):
    integrator.set_initial_value(initial_conditions, t=r0)  # Set the initial values of z and x
#    integrator.set_f_params(alpha)
    integrator.set_f_params(alpha,alpha2,a0)
#    integrator.set_f_params(a0)
    dt = step
    length = rf
    xs, zs = [], []
    while integrator.successful() and integrator.t <= length:
        integrator.integrate(integrator.t + dt)
        xs.append(integrator.t)
        zs.append([integrator.y[0], integrator.y[1], integrator.y[2],
                   integrator.y[3], integrator.y[4], integrator.y[5]])
        if not silent:
            print("Current x and z values: ", integrator.t, integrator.y)
    zs = np.array(zs)
    xs = np.array(xs)
    return xs, zs

def solve_bvp(ye_bv, yes_at_0_guess, alpha, alpha2, a0, r0, rf, step):
    integrator = spi.ode(rhs).set_integrator("dopri5")
#    integrator = spi.ode(rhs).set_integrator("dop853")
#    integrator = spi.ode(rhs).set_integrator("lsoda")
    def residuals(yes_at_0, V_at_length, Vprime_at_0,
                  Phi_at_length, Phi_prime_at_0, N_at_0):
        z_at_0 = [yes_at_0[0], Vprime_at_0, yes_at_0[1],
                  Phi_prime_at_0, yes_at_0[2], N_at_0]                
        xs, zs = integrate_over_domain(z_at_0, integrator, alpha, alpha2, a0, r0, rf, step)
        V_at_length_integrated = zs[-1, 0]
        Phi_at_length_integrated = zs[-1, 2]        
        return [V_at_length - V_at_length_integrated,
                Phi_at_length - Phi_at_length_integrated]    
    V_at_length, Vprime_at_0, Phi_at_length, Phi_prime_at_0, N_at_0 = ye_bv
    lsq = opt.least_squares(residuals, yes_at_0_guess, 
                            args=(V_at_length, Vprime_at_0,
                                  Phi_at_length, Phi_prime_at_0, N_at_0), 
                                  loss="soft_l1")
    V_at_0_estimate = lsq.x[0]
    Phi_at_0_estimate = lsq.x[1]
    E_estimate = lsq.x[2]    
    return [V_at_0_estimate, Phi_at_0_estimate, E_estimate]

def main(ye_bv, yes_at_0_guess, alpha, alpha2, a0, r0, rf, step):    
    integrator = spi.ode(rhs).set_integrator("dopri5")
#    integrator = spi.ode(rhs).set_integrator("dop853")
#    integrator = spi.ode(rhs).set_integrator("lsoda")
    integrator.set_initial_value([yes_at_0_guess[0], ye_bv[1], yes_at_0_guess[1],
                                  ye_bv[3], yes_at_0_guess[2]], t=r0)  # Set the initial values
    V_at_0_estimate, Phi_at_0_estimate, E_estimate = solve_bvp(ye_bv,
                                                               yes_at_0_guess,
                                                               alpha, alpha2, a0, r0,rf,step)
    _, Vprime_at_0, _, Phi_prime_at_0,  N_at_0= ye_bv
    xs, zs = integrate_over_domain([V_at_0_estimate, Vprime_at_0, Phi_at_0_estimate,
                                    Phi_prime_at_0, E_estimate, N_at_0], integrator, alpha, alpha2, a0, r0, rf, step)
    return [xs, zs, E_estimate, alpha]

#def plotses(x, ys, tit, ruta, ncor, sv = False):
##    pts.parametricplot(x,ys[:,2]**2,r'$\hat\mu r$',r"$\Phi^2$", tit,
##                       '%s/phi_sqr_%d' % (ruta, ncor),logy=False, save = sv)
#    pts.plotmultiple([x, x, x], [ys[:,0], ys[:,2], ys[:,4]],
#                     [r"$\psi/c^2$", r"$\Phi$",r"$E/\mu c$"],
#                     r'$\hat\mu r$', '', tit, '%s/boson_star_%d'% (ruta, ncor),
#                     save = sv) 
#    pts.plotmultiple([x, x], [ys[:,1], ys[:,5]], 
#                     [r"$r^2 \frac{d\psi}{dr}/c^2$", r"$N(r)$"],
#                     r'$\hat\mu r$', '', tit, '%s/boson_star_der_%d'% (ruta, ncor),
#                     save = sv)    
#
#    a= np.amax(ys[:,2])
#    b = x[np.where(ys[:,2]==a)[0][0]]
#    c = 1.5
#    print(a,b,c)
#    pts.plotmultiple([x, x], [ys[:,2], a*np.exp(-(x+b)**2/(2.*c**2))],
#                     [], r'$\hat{\mu}r$',r'$\Phi (r)$',
#                     "" ,'')
    
def r95(al0, tol = 0.001):
    Nx = al0[1][:,5]
    Nxmax = np.amax(Nx)
    N95 = 0.95*Nxmax
    print('Nmax=', np.amax(al0[1][:,5]), 'N95=', N95)
    try:
        index = np.where((Nx < N95 + tol) & (Nx > N95 -tol))[0][0]
        r95 = al0[0][index] 
        print('r95=', r95)
        print('fi(r95)=', al0[1][index,2])
        print('psi(r95)=', al0[1][index,0])
    
    except IndexError:
        print('subir tolerancia')
    try:
        return Nxmax, N95, r95
    except UnboundLocalError:
        print('subir tolerancia')
            
def write_catidades(al0,nodes, dec = '', N = [0,0,0]):
    f= open("%s/cantidades_%d.txt"%(dec,nodes),"w+")
    f.write(" r0 = %f, rf = %f \r\n " %(al0[0][0], al0[0][-1]))
    print('r0=', al0[0][0], 'rf=', al0[0][-1])
    f.write('V(0)= %f,  N/rf = %f \r\n ' %(al0[1][0,0], -al0[1][-1,5]/al0[0][-1]))
    print('V(0)=',al0[1][0,0], 'N/rf=', -al0[1][-1,5]/al0[0][-1])
    f.write('E = %f, alpha= %f \r\n ' %(al0[2], al0[3]))
    print('E=', al0[2],'alpha=', al0[3])
    a,b,c,d,e,aa = al0[1][0,:]
    f.write('Y(0)= (%f, %f, %f, %f, %f, %f ) \r\n ' % (a,b,c,d,e,aa))
    print('Y(0)=', al0[1][0,:])
    a,b,c,d,e,aa =  al0[1][-1,:]
    f.write('Y(rf)= (%f, %f, %f, %f, %f, %f ) \r\n ' %(a,b,c,d,e,aa))
    print('Y(rf)=', al0[1][-1,:])
    f.write('N = %f, N95 = %f, r95 = %f \r\n ' %(N[0], N[1], N[2]))
    f.close()