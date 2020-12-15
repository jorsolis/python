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
import solver_ABM as abm
from diffeq import pc4, rkf, heun, euler
from tqdm import tqdm
from Os_tools import check_path
from SP_shooting_axysimm_l1_solver_art_pacoluis import integrate_over_domain

dirshoot = '/home/jordi/satellite/mix_shooting_320' 
check_path(dirshoot)
int_method = "dopri5"#'dop853'##"lsoda"###'vode' ### "dopri5"

DC = {0: {'m' : 0, 'c1' :  2.*np.sqrt(5)/7., 'c2' :  6./7.},
      1: {'m' : 1, 'c1' :     np.sqrt(5)/7., 'c2' : -4./7.},
      2: {'m' : 2, 'c1' : -2.*np.sqrt(5)/7., 'c2' :  1./7.}}

m = 0
c1 = DC[m]['c1']  
c2 = DC[m]['c2'] 

def rhs(x, y):
    """     Compute the rhs of y'(t) = f(t, y, params)    """
    V0, P0, V2, P2, V4, P4, psi1, F1, psi3, F3, E1, E3 = y
    return np.array([P0/x**2,
            x**2*psi1**2 + x**6*psi3**2, ##      PO es N
            P2,
            -(6./x)*P2 + c1*x**2*psi3**2,
            P4,
            -(10./x)*P4 + abs(c2)*psi3**2,
            F1/x**2,
            2.*x**2*(V0 - E1)*psi1,
            F3,
            -6.*F3/x + 2.*(V0 + c1*x**2*V2 + c2*x**4*V4 - E3)*psi3,
            0.,
            0.])
    
def jac(x, y):
    """     Compute the jacobian    """
    V0, P0, V2, P2, V4, P4, psi1, F1, psi3, F3, E1, E3 = y
    return np.array([[0., 1./x**2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 2.*x**2*psi1, 0.,2.*x**6*psi3, 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., -(6./x), 0., 0., 0., 0., 2.*c1*x**2*psi3, 0., 0., 0.],
                     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., -10./x, 0., 0., 2.*abs(c2)*psi3, 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 1./x**2, 0., 0., 0., 0.],
                     [2.*x**2*psi1, 0., 0., 0., 0., 0., 2.*x**2*(V0 - E1), 0., 0., 0., -2.*x**2*psi1, 0.]
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]
                     [2.*psi3, 0., 2.*c1*x**2*psi3, 0., 2.*c2*x**4*psi3, 0., 0., 0., 2.*(V0 + c1*x**2*V2 +c2*x**4*V4 - E3), -6./x, 0., -2.*psi3],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    
def rhs2(y, x):
    """     Compute the rhs of y'(t) = f(t, y, params)
            para pc4    """
    V0, P0, V2, P2, V4, P4, psi1, F1, psi3, F3, E1, E3 = y
    return np.array([P0/x**2,
            x**2*psi1**2 + x**6*psi3**2, ##      PO es N
            P2,
            -(6./x)*P2 + c1*x**2*psi3**2,
            P4,
            -(10./x)*P4 + abs(c2)*psi3**2,
            F1/x**2,
            2.*x**2*(V0 - E1)*psi1,
            F3,
            -6.*F3/x + 2.*(V0 + c1*x**2*V2 + c2*x**4*V4 - E3)*psi3,
            0.,
            0.])

def solve_bvp(ye_bv, yes_at_0_guess, r0, rf, step):
#    integrator = spi.ode(rhs,jac).set_integrator(int_method)
    def residuals(yes_at_0_guess, V0_at_length, NTF, Phi0_0):         
        z_at_0 = [yes_at_0_guess[0], 0., yes_at_0_guess[1], 0.,
                  yes_at_0_guess[2], 0.,
                  Phi0_0, 0.,  yes_at_0_guess[3], 0.,
                  yes_at_0_guess[4], yes_at_0_guess[5]]
#        xs, zs = integrate_over_domain(z_at_0, integrator, r0, rf, step)
        
        xs = np.linspace(r0,rf,int(np.rint((rf-r0)/step))+1)

        zs = pc4(rhs2, z_at_0, xs)        
#        zs = euler(rhs2, z_at_0, xs )
#        zs = heun(rhs2, z_at_0, xs )
              
        V0_at_length_integrated = zs[-1, 0]
        N_at_length_integrated = zs[-1, 1]   
        V2_at_length_integrated = zs[-1, 2]
        V4_at_length_integrated = zs[-1, 4]
        psi0_at_length_integrated = zs[-1, 6]
        psi2_at_length_integrated = zs[-1, 8]
        E1_integrated = zs[-1, 10]
        psi1fin = np.exp(-np.sqrt(-2.*E1_integrated)*rf)/rf#0.
        return [V0_at_length - V0_at_length_integrated,
                NTF - N_at_length_integrated,
                0. - V2_at_length_integrated,
                0. - V4_at_length_integrated,
                psi1fin - psi0_at_length_integrated,
                0. - psi2_at_length_integrated]
        
    V0_at_length, NTF, Phi0_0 = ye_bv
    lsq = opt.least_squares(residuals, yes_at_0_guess, 
                            args=(V0_at_length, NTF, Phi0_0),
                            loss="soft_l1")
    V0_at_0_estimate = lsq.x[0]
    V2_at_0_estimate = lsq.x[1]
    V4_at_0_estimate = lsq.x[2]
    Phi2_0_estimate = lsq.x[3]
    E0_estimate = lsq.x[4] 
    E2_estimate = lsq.x[5] 

    return [V0_at_0_estimate,V2_at_0_estimate,V4_at_0_estimate,Phi2_0_estimate, E0_estimate, E2_estimate]

def main(ye_bv, yes_at_0_guess, r0, rf, step):    
    _, _, Phi0_0 = ye_bv
#    integrator = spi.ode(rhs,jac).set_integrator(int_method)
#    integrator.set_initial_value([yes_at_0_guess[0], 0., 
#                                  yes_at_0_guess[1], 0.,
#                                  Phi0_0, 0., Phi2_0, 0.,
#                                  yes_at_0_guess[2], yes_at_0_guess[3]], t=r0)  # Set the initial values
    V0_at_0_estimate, V2_at_0_estimate,V4_at_0_estimate, Phi2_0_estimate, E0_estimate, E2_estimate = solve_bvp(ye_bv, yes_at_0_guess,r0,rf,step) 

    z0= [V0_at_0_estimate, 0., V2_at_0_estimate, 0.,
         V4_at_0_estimate, 0.,
         Phi0_0, 0., Phi2_0_estimate, 0., E0_estimate, E2_estimate]
#    xs, zs = integrate_over_domain(z0,integrator, r0, rf, step)
#    
    xs = np.linspace(r0,rf,int(np.rint((rf-r0)/step))+1)

    zs = pc4(rhs2, z0, xs)
#    zs = euler(rhs2, z_at_0, xs )
#
#    zs = heun( rhs2, z0, xs )
#
    return [xs, zs, E0_estimate, E2_estimate]
#
def plotses(x, ys, tit, ruta, ncor, sv = False):
    cer = np.zeros(np.shape(ys[:,2])[0])
    one = np.ones(np.shape(ys[:,2])[0])
    E1 = ys[-1,10]
    savedir = '%s/%d/'%(ruta, int(NTF*10))
    check_path(savedir)
    pts.plotmultiple([x, x, x],[ys[:,0], ys[:,2], ys[:,4]],
                     [r'$V_{00}$', r'$V_{20}$', r'$V_{40}$'],
                     r'$\hat\mu r$','',tit,'%sV_%d'%(savedir,ncor),save = sv)    
    pts.plotmultiple([x, x, x],[ys[:,2], ys[:,4]],
                     [r'$V_{20}$', r'$V_{40}$'],
                     r'$\hat\mu r$','',tit,'%sV2_%d'%(savedir,ncor),save = sv) 
    pts.plotmultiple([x, x, x],[ys[:,0], x**2*ys[:,2], x**4*ys[:,4]],
                     [r'$V_{00}$', r'$r^2 V_{20}$', r'$r^4 V_{40}$'],
                     r'$\hat\mu r$','',tit,'%sV_%d'%(savedir,ncor),save = sv)    

    pts.plotmultiple([x],[ys[:,1]],[],
                     r'$\hat\mu r$',r'$N_T$',tit,'%sN_%d'%(savedir,ncor),save = sv) 
    pts.plotmultiple([x,x,x],[ys[:,6], x**2*ys[:,8],cer],
                     [r'$\psi_{100}$', r"$r^2\psi_{320}$"],r'$\hat\mu r$',
                     '', tit, '%spsi2_%d'%(savedir,ncor), save = sv)
    pts.plotmultiple([x, x, x, x],[ys[:,6], ys[:,8],
                      one*np.exp(-np.sqrt(-2.*E1)*rf)/rf, cer],
                     [r'$\psi_{100}$', r'$\psi_{320}$',  '1', '2 ' ],
                     r'$\hat\mu r$',r'$\psi$',tit,'%spsi_%d'%(savedir,ncor),
                     save = sv)

       
NTF = 1.9
svdir = '%s/%d/'%(dirshoot, int(NTF*10))
check_path(svdir)

rf = 3.5 
ncor = 1
r0 = 0.01
stp = .01

V0_guess = -1.0
V2_guess = -0.004
V4_guess = -0.002
Phi0_0 = 1.0
Phi3_0_guess = 0.1

ye_boundvalue = [-NTF/rf, NTF, Phi0_0]

EE0 = []
EE3 = []
Evacio = True

pbar = tqdm(total=172)

for n in range(10, 50, 5):
    E0_guess = - n/100.
    for n3 in range(5, 10, 1):
        E3_guess =  n3/10.       
        yguess=[V0_guess,V2_guess,V4_guess, Phi3_0_guess, E0_guess,E3_guess]
        x,y,E0,E3 = main(ye_boundvalue, yguess, r0, rf, stp) 
        plo = True
        for b in y[:-100,6]:
            if abs(b) < 0.01:
                plo = False
                break
        for a in y[:-100,8]:
            if abs(a) < 0.01:
                plo = False
                break
        if plo == True:
            if Evacio == True:
                EE0.append(E0); EE3.append(E3)
                Evacio= False
                np.save('%sygues_%d.npy'%(svdir, ncor), np.array(yguess))
                np.save('%sx_%d.npy'%(svdir, ncor), x)
                np.save('%sy_%d.npy'%(svdir, ncor), y)
                plotses(x,y,r'$E_{100}=%f,E_{320}=%f$'%(E0,E3), dirshoot, 
                        ncor, sv= True)
                ncor += 1
            else:
                if abs(EE0[-1]-E0) < 0.01:
                    break
#                    pass
                else:                        
                    EE0.append(E0); EE3.append(E3)
                    np.save('%sygues_%d.npy'%(svdir, ncor), np.array(yguess))
                    np.save('%sx_%d.npy'%(svdir, ncor), x)
                    np.save('%sy_%d.npy'%(svdir, ncor), y)
                    plotses(x,y,r'$E_{100}=%f,E_{320}=%f$'%(E0,E3), dirshoot, 
                            ncor, sv= True)
                    ncor += 1
        pbar.update(1)