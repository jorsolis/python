#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:40:50 2019

RESUELVE ESTADOS BASE + (DIPOLO o TORO) DE SCHRODINGER-POISSON
 USANDO APROX DE ARTICULO 
DE PACO Y LUIS

@author: jordis
"""
import numpy as np
import scipy.optimize as opt
import scipy.integrate as spi
import plots_jordi as pts
from SP_shooting_axysimm_l1_solver_art_pacoluis import integrate_over_domain
from diffeq import pc4, rkf, heun, euler
from tqdm import tqdm
from Os_tools import check_path
dirshoot = '/home/jordi/satellite/mix_shooting_dipole3' 
check_path(dirshoot)
int_method = "dopri5"#'dop853'##"lsoda"###'vode' ### "dopri5"
c = 2./np.sqrt(5)

def rhs(x, y):
    """     Compute the rhs of y'(t) = f(t, y, params)
            para pc4    """
    V0, N0, N2, V2, P2, psi1, F1, psi2, F2, E1, E2 = y
    return np.array([(N0 + N2)/x**2,
                     x**2*psi1**2,
                     x**4*psi2**2,
                     P2,
                     -(6./x)*P2 + c*psi2**2,
                     F1/x**2,
                     2.*x**2*(V0 - E1)*psi1,
                     F2,
                     -4.*F2/x + 2.*(V0 + c*x**2*V2 - E2)*psi2,
                     0.,
                     0.])   
def rhs2(y, x):
    """     Compute the rhs of y'(t) = f(t, y, params)
            para pc4    """
    V0, N0, N2, V2, P2, psi1, F1, psi2, F2, E1, E2 = y
    return np.array([(N0 + N2)/x**2,
                     x**2*psi1**2,
                     x**4*psi2**2,
                     P2,
                     -(6./x)*P2 + c*psi2**2,
                     F1/x**2,
                     2.*x**2*(V0 - E1)*psi1,
                     F2,
                     -4.*F2/x + 2.*(V0 + c*x**2*V2 - E2)*psi2,
                     0.,
                     0.])
def jac(x, y):
    """     Compute the jacobian    """
    V0, N0, N2, V2, P2, psi1, F1, psi2, F2, E1, E2 = y
    return np.array([[0., 1./x**2, 1./x**2, 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 2.*x**2*psi1, 0., 0., 0., 0., 0.]
                     [0., 0., 0., 0., 0., 0., 0., 2.*x**4*psi2, 0.,0., 0.],
                     [0., 0., 0., 0., 1./x**6, 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., -6./x, 0., 0., 2.*c*psi2, 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 1./x**2,0., 0., 0., 0.],
                     [2.*x**2*psi1, 0., 0., 0., 2.*x**2*(V0 - E1),0., 0., 0., - 2.*x**2*psi1,0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                     [2.*psi2, 0., 0., 2.*c*x**2*psi2, 0., 0., 0., 2.*(V0 + c*x**2*V2 - E2), -4./x, 0., -2.*psi2],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
 
def solve_bvp(ye_bv, yes_at_0_guess, r0, rf, step):
#    integrator = spi.ode(rhs,jac).set_integrator(int_method)
    def residuals(yes_at_0_guess, V0_at_length, Nb, eta, Phi0_0):         
        z_at_0 = [yes_at_0_guess[0], 0.,0.,
                  yes_at_0_guess[1], 0.,
                  Phi0_0, 0., 
                  yes_at_0_guess[2], 0.,
                  yes_at_0_guess[3], yes_at_0_guess[4]]
#        xs, zs = integrate_over_domain(z_at_0, integrator, r0, rf, step)       
        xs = np.linspace(r0,rf,int(np.rint((rf-r0)/step))+1)
        zs = pc4(rhs2, z_at_0, xs)        
#        zs = euler(rhs2, z_at_0, xs )
#        zs = heun(rhs2, z_at_0, xs )
              
        V0_at_length_integrated = zs[-1, 0]
        N0_at_length_integrated = zs[-1, 1] 
        N2_at_length_integrated = zs[-1, 2] 
        V2_at_length_integrated = zs[-1, 3]
        psi0_at_length_integrated = zs[-1, 5]
        psi2_at_length_integrated = zs[-1, 7]
        E1_integrated = zs[-1, 9]
        psi1fin = np.exp(-np.sqrt(-2.*E1_integrated)*rf)/rf#0.
        psi2fin = 0.
        return [V0_at_length - V0_at_length_integrated,
                Nb - N0_at_length_integrated,
                eta*Nb - N2_at_length_integrated,
                0. - V2_at_length_integrated,
                psi1fin - psi0_at_length_integrated,
                psi2fin - psi2_at_length_integrated]
        
    V0_at_length, Nb, eta, Phi0_0 = ye_bv
    
    try:
        lsq = opt.least_squares(residuals, yes_at_0_guess, 
                                args=(V0_at_length, Nb, eta, Phi0_0),
                                loss="soft_l1")
        V0_at_0_estimate = lsq.x[0]
        V2_at_0_estimate = lsq.x[1]
        Psi2_at_0_estimate = lsq.x[2]
        E0_estimate = lsq.x[3] 
        E2_estimate = lsq.x[4] 
        return [V0_at_0_estimate,V2_at_0_estimate,Psi2_at_0_estimate, E0_estimate, E2_estimate]
    
    except ValueError:
        return [0,0,0,0,0]    

    

def main(ye_bv, yes_at_0_guess, r0, rf, step):    
    _, _, _, Phi0_0= ye_bv
#    integrator = spi.ode(rhs,jac).set_integrator(int_method)
#    integrator.set_initial_value([yes_at_0_guess[0], 0., 0.,
#                                  yes_at_0_guess[1], 0.,
#                                  Phi0_0, 0., yes_at_0_guess[2], 0.,
#                                  yes_at_0_guess[3], yes_at_0_guess[4]], t=r0)  # Set the initial values
    
    V0_at_0_estimate,V2_at_0_estimate,Phi2_at_0_estimate, E0_estimate, E2_estimate = solve_bvp(ye_bv, yes_at_0_guess,r0,rf,step) 
    

    z0= [V0_at_0_estimate, 0.,0., V2_at_0_estimate, 0.,
         Phi0_0, 0., Phi2_at_0_estimate, 0., E0_estimate, E2_estimate]
    
#    
    xs = np.linspace(r0,rf,int(np.rint((rf-r0)/step))+1)

    if V0_at_0_estimate==0:
        return [xs, np.zeros((400,11)), -10,-10]
    else:
#        xs, zs = integrate_over_domain(z0,integrator, r0, rf, step)
        zs = pc4(rhs2, z0, xs)
#        zs = euler(rhs2, z_at_0, xs )
#    
#        zs = heun(rhs2, z0, xs )
#    
        return [xs, zs, E0_estimate, E2_estimate]
#
def plotses(x, ys, tit, ruta, ncor, sv = False):
    cer = np.zeros(np.shape(ys[:,2])[0])
    one = np.ones(np.shape(ys[:,2])[0])
    savedir = '%s/%d/'%(ruta, int(eta*10))
    check_path(savedir)
    E1 = ys[-1,9]
#    E2 = abs(ys[-1,10])    

    pts.plotmultiple([x, x, x],[ys[:,0], ys[:,3]],[r'$V_{00}$', r'$V_{20}$'],
                     r'$\hat\mu r$','',tit,'%sV_%d'%(savedir,ncor),save = sv)    
#    pts.plotmultiple([x, x, x],[ys[:,0], x**2*ys[:,3]],[r'$V_{00}$', r'$r^2 V_{20}$'],
#                     r'$\hat\mu r$','',tit,'%sV_%d'%(savedir,ncor),save = sv)    

    pts.plotmultiple([x, x, x],[ys[:,1] + ys[:,2], ys[:,1], ys[:,2]],
                     [r'$N_T$',r'$N_{0}$',r'$N_{2}$',],
                     r'$\hat\mu r$',r'$N$',tit,'%sN_%d'%(savedir,ncor),save = sv) 
#    pts.plotmultiple([x,x,x],[ys[:,4], x**2*ys[:,6],cer],
#                     [r'$\psi_{100}$', r"$r^2\psi_{210}$"],r'$\hat\mu r$',
#                     '', tit, '%spsi2_%d'%(savedir,ncor), save = sv)
    pts.plotmultiple([x, x, x, x, x], [ys[:,5], ys[:,7], 
                      one*np.exp(-np.sqrt(-2.*E1)*rf)/rf, cer],
                     [r'$\psi_{100}$',r'$\psi_{210}$',  '1', '2 '],
                     r'$\hat\mu r$',r'$\psi$',tit,'%spsi_%d'%(savedir,ncor),
                     save = sv)

       
Nb = 1.5


eta = 0.1

NTF = Nb*(1. + eta)
svdir = '%s/%d/'%(dirshoot, int(eta*10))
check_path(svdir)

rf = 1.0
      
ncor = 1
r0 = 0.01
stp =0.01

V0_guess = - 2.0
V2_guess = - 0.5
Phi20_guess = 1.5

Phi0_0 = 1.0


ye_boundvalue = [-NTF/rf, Nb, eta, Phi0_0]

EE0 = []
EE2 = []
Evacio = True

pbar = tqdm(total=172)

for n in range(2,20,1):
    E0_guess = - n/10.
    for n2 in range(1,n,1):
        E2_guess = - n2/10.       
        yguess=[V0_guess,V2_guess, Phi20_guess, E0_guess, E2_guess]
        x,y,E0,E2 = main(ye_boundvalue, yguess, r0, rf, stp) 
        plo = True
        for b in y[:-10,5]:
            if abs(b) < 0.01:
                plo = False
                break        
#        for a in y[:-10,7]:
#            if abs(a) < 0.01:
#                plo = False
#                break
        if plo == True:
            if Evacio == True:
                EE0.append(E0); EE2.append(E2)
                Evacio= False
                np.save('%sygues_%d.npy'%(svdir, ncor), np.array(yguess))
                np.save('%sx_%d.npy'%(svdir, ncor), x)
                np.save('%sy_%d.npy'%(svdir, ncor), y)
                plotses(x,y,r'$E_{100}=%f,E_{210}=%f$'%(E0,E2), dirshoot, 
                        ncor, sv= True)
                ncor += 1
            else:
                if abs(EE0[-1]-E0) < 0.01:
                    pass
                else:                        
                    EE0.append(E0); EE2.append(E2)
                    np.save('%sygues_%d.npy'%(svdir, ncor), np.array(yguess))
                    np.save('%sx_%d.npy'%(svdir, ncor), x)
                    np.save('%sy_%d.npy'%(svdir, ncor), y)
                    plotses(x,y,r'$E_{100}=%f,E_{210}=%f$'%(E0,E2), dirshoot, 
                            ncor, sv= True)
                    ncor += 1
        pbar.update(1)
#        ncor += 1
#print(ncor)