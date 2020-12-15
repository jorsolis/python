#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:40:50 2019

@author: jordis
"""
import numpy as np
import scipy.optimize as opt
import scipy.integrate as spi
import scipy.misc as sm
import plots_jordi as pts
from scipy.interpolate import interp1d, interp2d
from SP_shooting_axysimm_l1_solver_art_pacoluis import (fi, fi2, sph_harm2, pot,
                                                        integral, derivada,
                                                        density3d_files,
                                                        integrate_over_domain)
from SP_shooting_axysimm_l2_solver_art_pacoluis import (vel_DM, pot_sph_harm)

dirshoot = '/home/jordi/satellite/mix_shooting' 
int_method = "dopri5"#'dop853'##"lsoda"###'vode' ### "dopri5"
NTF = 5.

DE = {1:{'rf' : 5., 'Nf' : 3.8, 'V0_guess' : -1.6,
         'V2_guess' : -0.5, 'V4_guess' : -0.3, 
         'Phi10_guess' : 0.1, 'Phi30_guess' : 0.5, 
         'E1_guess' : -1.10, 'E3_guess' : -1.8, 'm' : 0},
      2:{'rf' : 5., 'Nf' : 3.8, 'V0_guess' : -1.6, 
         'V2_guess' : -0.5, 'V4_guess' : -0.3, 
         'Phi10_guess' : 0.1, 'Phi30_guess' : 0.5, 
         'E1_guess' : -1.80, 'E3_guess' : -1.9, 'm' : 0},
      3:{'rf' : 5., 'Nf' : 3.8, 'V0_guess' : -1.6, 
         'V2_guess' : -0.5, 'V4_guess' : -0.3, 
         'Phi10_guess' : 0.1, 'Phi30_guess' : 0.5, 
         'E1_guess' : -1.3, 'E3_guess' : -2.0, 'm' : 0}         }

DC = {0: {'m' : 0, 'c1' :  2.*np.sqrt(5)/7., 'c2' :  6./7.},
      1: {'m' : 1, 'c1' :     np.sqrt(5)/7., 'c2' : -4./7.},
      2: {'m' : 2, 'c1' : -2.*np.sqrt(5)/7., 'c2' :  1./7.}}



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
         
def solve_bvp(ye_bv, yes_at_0_guess, r0, rf, step):
    integrator = spi.ode(rhs,jac).set_integrator(int_method)
    def residuals(yes_at_0_guess, V0_at_length, V0prime_at_0,
                  V2_at_length, V2prime_at_0,
                  V4_at_length, V4prime_at_0,
                  psi1_at_length, psi1_prime_at_0,
                  psi3_at_length, psi3_prime_at_0):         
        z_at_0 = [yes_at_0_guess[0], V0prime_at_0,
                  yes_at_0_guess[1], V2prime_at_0,
                  yes_at_0_guess[2], V4prime_at_0,
                  yes_at_0_guess[3], psi1_prime_at_0,
                  yes_at_0_guess[4], psi3_prime_at_0,
                  yes_at_0_guess[5],
                  yes_at_0_guess[6]]             
        xs, zs = integrate_over_domain(z_at_0, integrator, r0, rf, step)
        V0_at_length_integrated = zs[-1, 0]
        V2_at_length_integrated = zs[-1, 2]
        V4_at_length_integrated = zs[-1, 4]
        psi1_at_length_integrated = zs[-1, 6]
        psi3_at_length_integrated = zs[-1, 8]
        N_at_length_integrated = zs[-1, 1]        
        return [V0_at_length - V0_at_length_integrated,
                V2_at_length - V2_at_length_integrated,
                V4_at_length - V4_at_length_integrated,
                psi1_at_length - psi1_at_length_integrated,
                psi3_at_length - psi3_at_length_integrated,
                NTF - N_at_length_integrated]    
    V0_at_length, V0prime_at_0, V2_at_length, V2prime_at_0, V4_at_length, V4prime_at_0, psi1_at_length, psi1_prime_at_0, psi3_at_length, psi3_prime_at_0 = ye_bv
    lsq = opt.least_squares(residuals, yes_at_0_guess, 
                            args=(V0_at_length, V0prime_at_0, V2_at_length,
                                  V2prime_at_0, V4_at_length, V4prime_at_0,
                                  psi1_at_length, psi1_prime_at_0,
                                  psi3_at_length, psi3_prime_at_0), loss="soft_l1")
    V0_at_0_estimate = lsq.x[0]
    V2_at_0_estimate = lsq.x[1]
    V4_at_0_estimate = lsq.x[2]
    psi1_at_0_estimate = lsq.x[3]
    psi3_at_0_estimate = lsq.x[4]
    E1_estimate = lsq.x[5]
    E3_estimate = lsq.x[6]
    return [V0_at_0_estimate,V2_at_0_estimate,V4_at_0_estimate, 
            psi1_at_0_estimate, psi3_at_0_estimate, E1_estimate, E3_estimate]
##yguess=[V0_guess,V2_guess,V4_guess, Phi1_0_guess, Phi3_0_guess, E1_guess, E3_guess]#    
def main(ye_bv, yes_at_0_guess, r0, rf, step):    
    integrator = spi.ode(rhs,jac).set_integrator(int_method)
    integrator.set_initial_value([yes_at_0_guess[0], ye_bv[1],
                                  yes_at_0_guess[1], ye_bv[3],
                                  yes_at_0_guess[2], ye_bv[5],
                                  yes_at_0_guess[3], ye_bv[7],
                                  yes_at_0_guess[4], ye_bv[9],
                                  yes_at_0_guess[5],
                                  yes_at_0_guess[6]], t=r0)  # Set the initial values
    V0_at_0_estimate,V2_at_0_estimate,V4_at_0_estimate, psi1_at_0_estimate, psi3_at_0_estimate, E1_estimate, E3_estimate = solve_bvp(ye_bv, yes_at_0_guess,r0,rf,step)
    _, V0prime_at_0,_, V2prime_at_0, _, V4prime_at_0, _, psi1_prime_at_0, _, psi3_prime_at_0 = ye_bv
    xs, zs = integrate_over_domain([V0_at_0_estimate, V0prime_at_0,
                                    V2_at_0_estimate, V2prime_at_0,
                                    V4_at_0_estimate, V4prime_at_0,
                                    psi1_at_0_estimate, psi1_prime_at_0,
                                    psi3_at_0_estimate, psi3_prime_at_0,
                                    E1_estimate,
                                    E3_estimate], integrator, r0, rf, step)
    return [xs, zs, E1_estimate, E3_estimate]

def plotses(x, ys, tit, ruta, ncor, sv = False, show = True, m = 10):
    cer = np.zeros(np.shape(ys[:,2])[0])
#    pts.plotmultiple([x, x, x],[ys[:,0], x**2*ys[:,2], x**4*ys[:,4]],
#                     [r'$V_{00}$', r'$r^2 V_{20}$', r'$r^4 V_{40}$'],
#                     r'$r$',r'$V$',tit,'%s/V_%d.png'%(ruta,ncor),save = sv)    
#    pts.plotmultiple([x,x,x,x,x,x,x],
#                     [ys[:,0], x**2*ys[:,2], x**4*ys[:,4],
#                      ys[:,6], ys[:,8], ys[:,10], ys[:,11]],
#                     [r'$V_{00}$',r'$r^2 V_{20}$',r'$r^4 V_{40}$',
#                      r"$\psi_{100}$", r"$\psi_{32%d}$"%m, r'$E_{100}$', r'$E_{32%d}$'%m],
#                     r'$r$','',tit,'%s/todas_%d.png'%(ruta,ncor),save = sv) 
#    pts.parametricplot(x,ys[:,1],r'$\hat\mu r$',r'$N_T(r)$', tit,
#                       '%s/N_%d.png'%(ruta,ncor), save = sv, show = show) 
#    pts.plotmultiple([x, x, x],[ys[:,6] ,ys[:,8], cer], 
#                     [r'$\psi_{100}$', r'$\psi_{32%d}$'%m, r'$0$'],
#                     r'$r$',r'$\psi$',tit,'%s/phi_%d.png'%(ruta,ncor),
#                     save = sv, show = show)
    pts.plotmultiple([x, x, x],[ys[:,6] ,cer], [r'$\psi_{100}$', r'$0$'],
                     r'$r$',r'$\psi$',tit,'%s/phi_1_%d.png'%(ruta,ncor),
                     save = sv, show = show)
    pts.plotmultiple([x, x, x],[ys[:,8], cer], [r'$\psi_{32%d}$'%m, r'$0$'],
                     r'$r$',r'$\psi$',tit,'%s/phi_3_%d.png'%(ruta,ncor),
                     save = sv, show = show)

def cinetica(xs, zs):
    xn, deriv = derivada(xs, zs[:,9])
    dF3dr = interp1d(xn, deriv, kind = 'linear', copy = True,
                     bounds_error = False, fill_value = 'extrapolate')
    int1 = integral(dF3dr(xs)*xs**4*zs[:, 8], xs)
    int2 = 6.*integral(zs[:,8]*zs[:,9]*xs**3, xs)
    return -2.*np.pi*(int1 + int2)    

def potencial(xs, zs):
    int1 = integral(zs[:,8]**2*zs[:,0]*xs**6, xs)
    int2 = integral(zs[:,8]**2*zs[:,2]*xs**8, xs)
    int3 = integral(zs[:,8]**2*zs[:,4]*xs**10, xs)
    return 2.*np.pi*(int1 + c1*int2 + c2*int3)

def virializacion(xs, zs):
    Ke = cinetica(xs, zs)
    We = potencial(xs, zs)
    xn, deriv = derivada(xs, zs[:,7])
    dF1dr = interp1d(xn, deriv, kind ='linear', copy = True,
                     bounds_error = False, fill_value = 'extrapolate')    
    Kb = -2.*np.pi*integral(zs[:,6]*dF1dr(xs), xs)
    Wb = 2.*np.pi*integral(zs[:,6]**2*zs[:,0]*xs**2, xs)
    return Ke, Kb , Wb, We

def r95(xs, zs, tol = 0.001):
    Nx = zs[:,1]
    Nxmax = np.amax(Nx)
    N95 = 0.95*Nxmax
    print('Nmax=', Nxmax, 'N95=', N95)
    try:
        index = np.where((Nx < N95 + tol) & (Nx > N95 -tol))[0][0]
        R95 = xs[index] 
        print('r95=', R95)
    
    except IndexError:
        print('subir tolerancia')
    try:
        return Nxmax, N95, R95
    except UnboundLocalError:
        print('subir tolerancia')
        return 0, 0, 0
        
def write_catidades_file(ncor, x, y, tol = 0.001,
                         dect = '/home/jordi/satellite/mix_shooting'):
    rf = DE[ncor]['rf']
    E1_guess = DE[ncor]['E1_guess']
    E3_guess = DE[ncor]['E3_guess']
    m = DE[ncor]['m']
    N1 = integral(x**2*y[:,6]**2, x)
    N3 = integral(x**6*y[:,8]**2, x)
    NT = N1 + N3
    Ke, Kb , Wb, We = virializacion(x, y)
    ET = (y[-1,10]*N1 + y[-1,11]*N3)/NT
    _, N95, R95 = r95(x, y, tol = tol)
    
    f= open("%s/cantidades_%d.txt"%(dect,ncor),"w+")
    f.write(" N = %f, -N/rf = %f, Rf = %f \r\n " % (NT, -NT/rf, rf))
    f.write(" M_100/M32%d = %f \r\n " % (m, N1/N3))
    f.write(" r95 = %f  N95 = %f \r\n " % (R95, N95))
    f.write(" P0 = %f, V(rf,pi/2) = %f + %f (error?) \r\n " %(y[-1,1],
            y[-1,0], - np.sqrt(5)*rf**2*y[-1,2]*0.5 + 9.*rf**4**y[-1,4]*0.125))
    f.write(" E1_guess = %f, E3_guess = %f \r\n " % (E1_guess, E3_guess))
    f.write(" E_100 = %f \r\n " % y[-1,10])
    f.write(" W1 = %f, K1 = %f, K1/W1 = %f \r\n " %(Wb, Kb, Kb/abs(Wb)))
    f.write(" E_32%d = %f \r\n " % (m, y[-1,11]))
    f.write(" W3 = %f, K3 = %f, K3/W3 = %f \r\n " %(We, Ke, Ke/abs(We)))
    f.write(" ET = %f \r\n " % ET )
    f.write(" WT = %f, KT = %f, KT/WT = %f \r\n " % (Wb+We, Ke+Kb,
                                                     (Ke+Kb)/abs(Wb+We)))
    f.close()

def plots_densidad(x, y, rf,  dec = '/home/jordi/satellite/mix_shooting'):
    densb = interp1d(x, y[:,6]**2, kind = 'linear', copy = True, 
                    bounds_error = False, fill_value = 'extrapolate')
    densex = interp1d(x, x**4*y[:,8]**2, kind = 'linear', copy = True, 
                    bounds_error = False, fill_value = 'extrapolate')
    r = np.linspace(r0, rf, 400)
    th = np.linspace(0, np.pi, 200)
    R, T = np.meshgrid(r, th)
    dens2db = interp2d(r, th, sph_harm2(densb, R, T, l=0, m=0),
                       kind ='linear', copy = True, bounds_error = False)
    dens2dex = interp2d(r, th, sph_harm2(densex, R, T, l=2, m=m),
                        kind = 'linear', copy = True, bounds_error = False)
    x2 = np.linspace(-rf, rf, 400)
    z = np.linspace(-rf, rf, 400)
    X, Z = np.meshgrid(x2, z)
#    pts.densityplot(X, Z, fi(dens2db, x2, z, rf = rf), r"$\hat\mu x$",
#                    r"$\hat\mu z$", r"$\Phi_{100}^2(x,z)$",
#                    r'$E_{100}=%f$'%(E1),
#                    name= '%s/dens_b_xz_%d.png'%(dec, ncor))
#    pts.densityplot(X, Z, fi(dens2dex, x2, z, rf = rf), r"$\hat\mu x$",
#                    r"$\hat\mu z$", r"$\Phi_{32%d}^2(x,z)$"%m,
#                    r'$E_{32%d}=%f$'%(m,E3),                 
#                    name= '%s/dens_ex_xz_%d.png'%(dec, ncor))    
    pts.densityplot2(X, Z, fi(dens2dex, x2,z, rf=rf) + fi(dens2db, x2,z, rf=rf),
                     x2, fi(dens2dex, x2, z, rf = rf)[200, :],
                     fi(dens2db, x2, z, rf = rf)[200, :],
                     r"$\hat\mu x$", r"$\hat\mu z$", 
                    r"$\Phi_{100}^2(x,z) + \Phi_{32%d}^2(x,z)$"%m,
                     r"$\hat\mu x$",
                     [r"$\Phi_{100}^2(x,0)$", r"$\Phi_{32%d}^2(x,0)$"%m],
                     r'$E_{100}=%f, E_{32%d}=%f$'%(E1, m, E3), 
                     name='%s/dens_mix_xz_%d.png'%(dec, ncor))#, extralog = True)
    rho = np.linspace(0., rf, 400)
    z = np.linspace(-rf, rf, 400)
    firhoz = fi2(dens2dex, rho, z) + fi2(dens2db, rho, z)
    firhozinterpol = interp2d(rho, z, firhoz, kind='linear', copy=True, 
                              bounds_error=False)    
    density3d_files(firhozinterpol, rf, dec, ncor)
    
#    Vo = interp1d(x, y[:,0], kind='linear', copy=True, bounds_error=False)
#    V2 = interp1d(x, x**2*y[:,2], kind='linear', copy=True, bounds_error=False)
#    V4 = interp1d(x, x**4*y[:,4], kind='linear', copy=True, bounds_error=False)
#    pot2d = interp2d(r, th, pot_sph_harm(Vo, V2, V4, R, T) , kind = 'linear', 
#                     copy = True, bounds_error = False)
#    pts.densityplot(R,T,pot_sph_harm(Vo, V2, V4, R, T), r"$r$", r"$\theta$",
#                    r"$V(r,\theta)$", r'$E_{21%d}=%f$'%(m, E3),
#                    name = '%s/potrth_%d'%(dec,ncor))
#    pts.plotmultiple([r],[pot_sph_harm(Vo, V2, V4, R, T)[100, :]],
#                      [], r'$r$',r'$V(r, \pi/2)$','','',
#                     save = False)
#    pts.plotmultiple([x],[y[:,0] - np.sqrt(5)*rf**2*y[:,2]*0.5 + 9.*rf**4**y[:,4]*0.125],
#                      [], r'$r$',r'$V(r, \pi/2)$','','',
#                     save = False)
        
#    print(pot_sph_harm(Vo, V2, V4, R, T).shape)
#    print(pot_sph_harm(Vo, V2, V4, R, T)[100, -1])
#    rho = np.linspace(0, rf, 400)
#    z = np.linspace(-rf, rf, 800)
#    Rho, Z = np.meshgrid(rho, z)    
#    pts.densityplot2(Rho, Z, pot(pot2d, rho, z, rf = rf),
#                     rho, pot(pot2d, rho, z, rf = rf)[400,:],
#                      pot(pot2d, rho, z, rf = rf)[600,:], 
#                     r"$\hat\mu \rho$", r"$\hat\mu z$",
#                     r"$\hat V(\hat \rho,\hat z)$", r"$\hat\mu \rho$",
#                     [r"$\hat V(\hat \rho,0)$",r"$\hat V(\hat \rho,2.5)$"],
#                     r'$E_{100}=%f, E_{32%d}=%f$'%(E1, m, E3),
#                     name='%s/potrhoz_%d.png'%(dec,ncor))

def completarV(xs, zs):
    P0 = zs[:, 1]
    V2 = zs[:, 2]
    P2 = zs[:, 3]
    V4 = zs[:, 4]
    P4 = zs[:, 5]
    r = xs
    P00 = interp1d(r, P0, kind = 'linear', copy = True, 
                bounds_error = False, fill_value = 'extrapolate')
    V22 = interp1d(r, V2, kind = 'linear', copy = True, 
                bounds_error = False, fill_value = (V2[0],V2[-1]))
    P22 = interp1d(r, P2, kind = 'linear', copy = True, 
                bounds_error = False, fill_value = 'extrapolate')#(P2[0],0.))
    V44 = interp1d(r, V4, kind = 'linear', copy = True, 
                bounds_error = False, fill_value = 'extrapolate')    
    P44 = interp1d(r, P4, kind = 'linear', copy = True, 
                bounds_error = False, fill_value =  'extrapolate')# (P4[0], 0.)) 
    R = np.linspace(0.0,6, 1000)    
    np.save('%s/ncor%d/r_%d.npy'%(dirshoot, ncor, ncor), R)
    np.save('%s/ncor%d/P0_%d.npy'%(dirshoot, ncor, ncor), P00(R))
    np.save('%s/ncor%d/V2_%d.npy'%(dirshoot, ncor, ncor), V22(R))
    np.save('%s/ncor%d/P2_%d.npy'%(dirshoot, ncor, ncor), P22(R))
    np.save('%s/ncor%d/V4_%d.npy'%(dirshoot, ncor, ncor), V44(R))
    np.save('%s/ncor%d/P4_%d.npy'%(dirshoot, ncor, ncor), P44(R))

def vel_DM2(ncor):
    r = np.load('%s/ncor%d/r_%d.npy'%(dirshoot, ncor, ncor))
    P0 = np.load('%s/ncor%d/P0_%d.npy'%(dirshoot, ncor, ncor))
    V2 = np.load('%s/ncor%d/V2_%d.npy'%(dirshoot, ncor, ncor))
    P2 = np.load('%s/ncor%d/P2_%d.npy'%(dirshoot, ncor, ncor))
    V4 = np.load('%s/ncor%d/V4_%d.npy'%(dirshoot, ncor, ncor))
    P4 = np.load('%s/ncor%d/P4_%d.npy'%(dirshoot, ncor, ncor))   
    c1 =  np.sqrt(5)/2.
    c2 = 9./8.
    P00 = interp1d(r, P0, kind = 'linear', copy = True, 
                bounds_error = False, fill_value = 'extrapolate')
    V22 = interp1d(r, V2, kind = 'linear', copy = True, 
                bounds_error = False, fill_value = (V2[0],V2[-1]))
    P22 = interp1d(r, P2, kind = 'linear', copy = True, 
                bounds_error = False, fill_value = 'extrapolate')#(P2[0],0.))
    V44 = interp1d(r, V4, kind = 'linear', copy = True, 
                bounds_error = False, fill_value = 'extrapolate')    
    P44 = interp1d(r, P4, kind = 'linear', copy = True, 
                bounds_error = False, fill_value =  'extrapolate')
    
    x = np.linspace(0.1, 5, 2000)
    vel2 = []
    for i in range(0, np.shape(x)[0]):
        if x[i] < 6:
            v2 = -(-P00(x[i])/x[i] + c1*x[i]**2*(2.*V22(x[i]) + x[i]*P22(x[i])) - c2*x[i]**4*(4.*V44(x[i]) + x[i]*P44(x[i])))
        else:
            v2 = P0[-1]/x[i]
        vel2.append(v2)
    pts.plotmultiple([x], [vel2], [], 
                          r'$r$',r'$v/c$','rot curve','',save = False)    
         
if __name__ == '__main__': 
    ncor = 1
    m = DE[ncor]['m']
    c1 = DC[m]['c1']  
    c2 = DC[m]['c2'] 
    r0 = 0.1
    stp = .1     
    rf = DE[ncor]['rf']
    NTF = DE[ncor]['Nf']
    V0_guess = DE[ncor]['V0_guess']
    V2_guess = DE[ncor]['V2_guess']
    V4_guess = DE[ncor]['V4_guess']
    Phi10_guess = DE[ncor]['Phi10_guess']
    Phi30_guess = DE[ncor]['Phi30_guess']
    V0_at_length = -NTF/rf
    ye_boundvalue = [V0_at_length,0., 0., 0., 0., 0., 0., 0., 0., 0.] 
#    for i in range(130, 201, 10):
#        print('dir', i)
#        E1_guess = -i/100.  
#        print(E1_guess)
#        for j in range(180, 201, 10):
#            E3_guess = -j/100. 
#            ncor = j
#            print('ncor', ncor)
#            yguess=[V0_guess,V2_guess,V4_guess,Phi10_guess,Phi30_guess,E1_guess,
#                    E3_guess] 
#            x,y,E1, E3 = main(ye_boundvalue, yguess, r0, rf, stp)
#            if E3<0 and E1<0:
#                plotses(x,y,r'$E_{1guess}=%f, E_{100}=%f, E_{3guess}=%f, E_{32%d}=%f$'%(E1_guess, E1, E3_guess,m, E3),
#                        '/home/jordi/satellite/mix_shooting/%d'%i,ncor,m=m, sv= True,show = True)
#      
    E3_guess = DE[ncor]['E3_guess']
    E1_guess = DE[ncor]['E1_guess']
    yguess = [V0_guess, V2_guess, V4_guess, Phi10_guess, Phi30_guess, E1_guess, 
              E3_guess]
    x, y, E1, E3 = main(ye_boundvalue, yguess, r0, rf, stp)
#    plotses(x, y, r'$E_{1guess}=%f, E_{100}=%f, E_{3guess}=%f, E_{32%d}=%f$'%(E1_guess, E1, E3_guess,m, E3),
#            '/home/jordi/satellite/mix_shooting', ncor, m = m, sv = True,
#            show = True)
#    write_catidades_file(ncor, x, y, tol = 0.05)  
#    plots_densidad(x, y, rf)   
    vdm = vel_DM(x, y)
    np.save('%s/ncor%d/vdm_%d.npy'%(dirshoot, ncor,ncor), np.array([x,vdm]))

####    np.save('/home/jordi/satellite/mix_shooting/vdm_%d.npy'%(ncor), np.array([x,vdm]))
####    x, vdm = np.load('%s/vdm_%d.npy'%(dirshoot,ncor))

    x, vdm = np.load('%s/ncor%d/vdm_%d.npy'%(dirshoot,ncor,ncor))

    pts.plotmultiple([x], [vdm], [], 
                      r'$r$',r'$v/c$','rot curve','',save = False)
    
    completarV(x, y)  
    vel_DM2(ncor)
#    