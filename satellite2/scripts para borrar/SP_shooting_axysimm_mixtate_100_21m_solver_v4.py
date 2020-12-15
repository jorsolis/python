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

dirshoot = '/home/jordi/satellite/mix_shooting_dipole4' 
int_method = "dopri5"#'dop853'##"lsoda"###'vode' ### "dopri5"
c = 2./np.sqrt(5)

def rhs(x, y):
    """     Compute the rhs of y'(t) = f(t, y, params)    """
    V0, P0, V2, P2, psi1, F1, psi2, F2, E1, E2 = y
    return np.array([P0/x**2,
                     x**2*psi1**2 + x**4*psi2**2,######    PO es N
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
    V0, P0, V2, P2, psi1, F1, psi2, F2, E1, E2 = y
    return np.array([P0/x**2,
                     x**2*psi1**2 + x**4*psi2**2,######    PO es N
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
    V0, P0, V2, P2, psi1, F1, psi2, F2, E1, E2 = y
    return np.array([[0., 1./x**2, 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0.,2.*x**2*psi1, 0., 2.*x**4*psi2, 0.,0., 0.],
                     [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., -(6./x), 0., 0., 2.*c*psi2, 0., 0., 0.],
                     [ 0., 0., 0., 0., 0., 1./x**2,0., 0., 0., 0.],
                     [2.*x**2*psi1, 0., 0., 0., 2.*x**2*(V0 - E1),0., 0., 0., - 2.*x**2*psi1,0.],
                     [0., 0., 0.,0., 0., 0., 0., 1.,0., 0.],
                     [2.*psi2, 0., 2.*c*x**2*psi2, 0., 0., 0.,2.*(V0 + c*x**2*V2 - E2), -4./x, 0., -2.*psi2],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
       
def solve_bvp(ye_bv, yes_at_0_guess, r0, rf, step):
#    integrator = spi.ode(rhs,jac).set_integrator(int_method)
    def residuals(yes_at_0_guess, V0_at_length):         
        z_at_0 = [yes_at_0_guess[0], 0., yes_at_0_guess[1], 0.,
                  Phi0_0, 0.,  yes_at_0_guess[2], 0.,
                  yes_at_0_guess[3], yes_at_0_guess[4]]
#        xs, zs = integrate_over_domain(z_at_0, integrator, r0, rf, step)
        
        xs = np.linspace(r0,rf,int(np.rint((rf-r0)/step))+1)
#        zs = abm.resol_ABM(rhs, z_at_0, xs, r0, rf, step) 
        zs = pc4(rhs2, z_at_0, xs)        
#        zs = euler(rhs2, z_at_0, xs )

#        zs = heun(rhs2, z_at_0, xs )
#              
        V0_at_length_integrated = zs[-1, 0]
        N_at_length_integrated = zs[-1, 1]   
        V2_at_length_integrated = zs[-1, 2]
        psi0_at_length_integrated = zs[-1, 4]
        psi2_at_length_integrated = zs[-1, 6]
     
        return [V0_at_length - V0_at_length_integrated,
                NTF - N_at_length_integrated,
                0. - V2_at_length_integrated,
                0. - psi0_at_length_integrated,
                0. - psi2_at_length_integrated]
        
    V0_at_length = ye_bv
    lsq = opt.least_squares(residuals, yes_at_0_guess, 
                            args=(V0_at_length),  loss="soft_l1")
    V0_at_0_estimate = lsq.x[0]
    V2_at_0_estimate = lsq.x[1]
    psi2_at_0_estimate = lsq.x[2]
    E0_estimate = lsq.x[3] 
    E2_estimate = lsq.x[4] 
    return [V0_at_0_estimate,V2_at_0_estimate, psi2_at_0_estimate, E0_estimate, E2_estimate]

def main(ye_bv, yes_at_0_guess, r0, rf, step):    
#    integrator = spi.ode(rhs,jac).set_integrator(int_method)
#    integrator.set_initial_value([yes_at_0_guess[0], 0., 
#                                  yes_at_0_guess[1], 0.,
#                                  Phi0_0, 0., yes_at_0_guess[2], 0.,
#                                  yes_at_0_guess[3], yes_at_0_guess[4]], t=r0)  # Set the initial values
    V0_at_0_estimate,V2_at_0_estimate, psi2_at_0_estimate, E0_estimate, E2_estimate = solve_bvp(ye_bv, yes_at_0_guess,r0,rf,step) 
    z0= [V0_at_0_estimate, 0., V2_at_0_estimate, 0.,
         Phi0_0, 0., psi2_at_0_estimate,0., E0_estimate, E2_estimate]
#    xs, zs = integrate_over_domain(z0,integrator, r0, rf, step)
#    
    xs = np.linspace(r0,rf,int(np.rint((rf-r0)/step))+1)
#    zs = abm.resol_ABM(rhs, z0, xs, r0, rf, step) 
    zs = pc4(rhs2, z0, xs)
#    zs = euler(rhs2, z_at_0, xs )

#    zs = heun( rhs2, z0, xs )

    return [xs, zs, E0_estimate, E2_estimate]
#
def plotses(x, ys, tit, ruta, ncor, sv = False):
    cer = np.zeros(np.shape(ys[:,2])[0])
    check_path('%s/%d/'%(ruta, int(NTF*10)))
    pts.plotmultiple([x, x, x],[ys[:,0], ys[:,2]],[r'$V_{00}$', r'$V_{20}$'],
                     r'$\rho$','',tit,'%s/%d/V_%d'%(ruta, int(NTF*10),ncor),save = sv)    
#    pts.plotmultiple([x, x, x],[ys[:,0], x**2*ys[:,2]],[r'$V_{00}$', r'$r^2 V_{20}$'],
#                     r'$\rho$','',tit,'%s/%d/V_%d'%(ruta, int(NTF*10),ncor),save = sv)    

    pts.plotmultiple([x],[ys[:,1]],[],
                     r'$\rho$',r'$N_T$',tit,'%s/%d/N_%d'%(ruta, int(NTF*10),ncor),save = sv) 
    pts.plotmultiple([x,x,x],[ys[:,4], x**2*ys[:,6],cer],
                     [r'$\psi_{100}$', r"$r^2\psi_{210}$"],r'$\hat\mu r$',
                     '', tit, '%s/%d/psi2_%d'%(ruta, int(NTF*10),ncor), save = sv)
    pts.plotmultiple([x,x, x],[ys[:,6], ys[:,4],cer],
                     [r'$\psi_{210}$', r'$\psi_{100}$'],
                     r'$\rho$',r'$\psi$',tit,'%s/%d/psi_%d'%(ruta, int(NTF*10),ncor), save = sv)

       
NTF = 3.5
print(NTF*10)



if int(NTF*10) in (10, 15, 16, 17, 18, 20, 25, 30, 35, 40, 45, 50):
    MM = {'V0': -1.5, 'V2':-0.9, 'Phi2':0.01}
    if int(NTF*10) ==10:
        rf = 2.5
    elif int(NTF*10) ==15:
        rf = 2.8
    else:
        rf =3.0
        
elif int(NTF*10)>50:
    MM = {'V0': -0.5, 'V2':-0.05, 'Phi2':0.25}
    rf = 2.0
elif int(NTF*10) in (5,6,7,8,9):
    MM = {'V0': -0.1, 'V2':-0.05, 'Phi2':1.0}
    rf = 2.0
else:
    MM = {'V0': -0.5, 'V2':-0.05, 'Phi2':1.}
    rf = 2.0
    
ncor = 1
r0 = 0.01
stp = .01
ye_boundvalue = [-NTF/rf ]
V0_guess = MM['V0']
V2_guess = MM['V2']
Phi20_guess = MM['Phi2']

Phi0_0 = 1.

EE0 = []
EE2 = []
Evacio = True
check_path('%s/%d/'%(dirshoot, int(NTF*10)))

pbar = tqdm(total=50)

for n in range(1,20,4):
    E0_guess = - n/10.
    for n2 in range(1,20,2):
        E2_guess = - n2/10       
        yguess=[V0_guess,V2_guess,Phi20_guess,E0_guess,E2_guess]
        x,y,E0,E2 = main(ye_boundvalue, yguess, r0, rf, stp)
        
        plo = True
        for a in y[:-10,6]:
            if abs(a) < 0.01:
                plo = False
                break
            
        for b in y[:-10,4]:
            if abs(b) < 0.01:
                plo = False
                break
        
        if plo == True:
            if Evacio == True:
                EE0.append(E0); EE2.append(E2)
                Evacio= False
                np.save('%s/%d/ygues_%d.npy'%(dirshoot, int(NTF*10), ncor), np.array(yguess))
                np.save('%s/%d/x_%d.npy'%(dirshoot, int(NTF*10), ncor), x)
                np.save('%s/%d/y_%d.npy'%(dirshoot, int(NTF*10), ncor), y)
                plotses(x,y,r'$E_{100}=%f,E_{210}=%f$'%(E0,E2), dirshoot, 
                        ncor, sv= True)
                ncor += 1
            else:
                if abs(EE0[-1]-E0) < 0.01:
                    break
                else:  
                    EE0.append(E0); EE2.append(E2)
                    np.save('%s/%d/ygues_%d.npy'%(dirshoot, int(NTF*10), ncor), np.array(yguess))
                    np.save('%s/%d/x_%d.npy'%(dirshoot, int(NTF*10), ncor), x)
                    np.save('%s/%d/y_%d.npy'%(dirshoot, int(NTF*10), ncor), y)
                    plotses(x,y,r'$E_{100}=%f,E_{210}=%f$'%(E0,E2), dirshoot, 
                            ncor, sv= True)
                    ncor += 1
        pbar.update(1)
