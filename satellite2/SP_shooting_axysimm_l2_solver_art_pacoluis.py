#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:40:50 2019

RESUELVE ESTADOS (3 2 0), (3 2 1), (3 2 2) DE SCHRODINGER-POISSON USANDO 
APROX DE ARTICULO DE PACO Y LUIS

@author: jordi
"""
import numpy as np
import scipy.optimize as opt
import scipy.integrate as spi
import plots_jordi as pts
from scipy.interpolate import interp1d, interp2d
from SP_shooting_axysimm_l1_solver_art_pacoluis import (fi, fi2, sph_harm2, pot,
                                                        integral, derivada,
                                                        density3d_files,
                                                        integrate_over_domain)

ncor = 3 # key de diccionario DE
int_method = "dopri5"#'dop853'##"lsoda"###'vode' ### "dopri5"

DE = {1:{'rf' : 5., 'E_guess' : -1.6, 'Nf' : 3.8, 'V0_guess' : -1.6, 
         'V2_guess' : -0.5, 'V4_guess' : -0.3, 'Phi30_guess' : 0.5, 'm' : 0},
      2:{'rf' : 5., 'E_guess' : -1.5, 'Nf' : 3.8, 'V0_guess' : -1.6, 
         'V2_guess' : -0.5, 'V4_guess' : -0.3, 'Phi30_guess' : 0.5, 'm' : 1},
      3:{'rf' : 5., 'E_guess' : -.40, 'Nf' : 3.4, 'V0_guess' : -1.2, 
         'V2_guess' : -0.5, 'V4_guess' : -0.3, 'Phi30_guess' : 0.5, 'm' : 2}}

DC = {0: {'m' : 0, 'c1' :  2.*np.sqrt(5)/7., 'c2' :  6./7.},
      1: {'m' : 1, 'c1' :     np.sqrt(5)/7., 'c2' : -4./7.},
      2: {'m' : 2, 'c1' : -2.*np.sqrt(5)/7., 'c2' :  1./7.}}

MM = {21:{'mu': 156550.,'rlam': 1.0e-3},
      22:{'mu': 15655.0,'rlam': 1.0e-2},
      23:{'mu': 1565.5, 'rlam': 4.0e-3},
      24:{'mu': 156.55, 'rlam': 1.5e-2},
      25:{'mu': 15.655, 'rlam': 5.5e-2}}

m = DE[ncor]['m']
c1 = DC[m]['c1']  
c2 = DC[m]['c2'] 
c3 = DC[m]['c1'] # c3 = c1
c4 = DC[m]['c2'] # c4 = c2

def rhs(x, y):
    """     Compute the rhs of y'(t) = f(t, y, params)    """
    V0, P0, V2, P2, V4, P4, psi3, F3, E3 = y
    return np.array([P0/x**2,
            psi3**2*x**6,
            P2,
            -(6./x)*P2 + c1*x**2*psi3**2,
            P4,
            -(10./x)*P4 + c2*psi3**2,
            F3,
            -6.*F3/x + 2.*(V0 +c3*x**2*V2 + c4*x**4*V4 - E3)*psi3,
            0.])
    
def jac(x, y):
    """     Compute the jacobian    """
    V0, P0, V2, P2, V4, P4, psi3, F3, E3 = y
    return np.array([[0., 1./x**2, 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 2.*x**6*psi3, 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                     [0., 0., 0., -(6./x), 0., 0., 2.*c1*x**2*psi3, 0., 0.],
                     [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                     [0., 0., 0., 0., 0., -10./x, 2.*c2*psi3, 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                     [2.*psi3, 0., 2.*c3*x**2*psi3, 0., 2.*c4*x**4*psi3, 0.,
                      2.*(V0 + c3*x**2*V2 +c4*x**4*V4 - E3), -6./x, -2.*psi3],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
 
def solve_bvp(ye_bv, yes_at_0_guess, r0, rf, step):
    integrator = spi.ode(rhs,jac).set_integrator(int_method)
    def residuals(yes_at_0_guess, V0_at_length, V0prime_at_0,
                  V2_at_length, V2prime_at_0,
                  V4_at_length, V4prime_at_0,
                  psi3_at_length, psi3_prime_at_0):         
        z_at_0 = [yes_at_0_guess[0], V0prime_at_0,
                  yes_at_0_guess[1], V2prime_at_0,
                  yes_at_0_guess[2], V4prime_at_0,
                  yes_at_0_guess[3], psi3_prime_at_0,
                  yes_at_0_guess[4]]             
        xs, zs = integrate_over_domain(z_at_0, integrator, r0, rf, step)
        V0_at_length_integrated = zs[-1, 0]
        V2_at_length_integrated = zs[-1, 2]
        V4_at_length_integrated = zs[-1, 4]
        psi3_at_length_integrated = zs[-1, 6]
        N_at_length_integrated = zs[-1, 1]        
        return [V0_at_length - V0_at_length_integrated,
                V2_at_length - V2_at_length_integrated,
                V4_at_length - V4_at_length_integrated,
                psi3_at_length - psi3_at_length_integrated,
                NTF - N_at_length_integrated]    
    V0_at_length, V0prime_at_0, V2_at_length, V2prime_at_0, V4_at_length, V4prime_at_0, psi3_at_length, psi3_prime_at_0 = ye_bv
    lsq = opt.least_squares(residuals, yes_at_0_guess, 
                            args=(V0_at_length, V0prime_at_0,V2_at_length,
                                  V2prime_at_0, V4_at_length, V4prime_at_0,
                                  psi3_at_length, psi3_prime_at_0), loss="soft_l1")
    V0_at_0_estimate = lsq.x[0]
    V2_at_0_estimate = lsq.x[1]
    V4_at_0_estimate = lsq.x[2]
    psi3_at_0_estimate = lsq.x[3]
    E0_estimate = lsq.x[4]
    return [V0_at_0_estimate,V2_at_0_estimate,V4_at_0_estimate, psi3_at_0_estimate, E0_estimate]

##yguess=[V0_guess,V2_guess,V4_guess, Phi3_0_guess, E3_guess]
    
def main(ye_bv, yes_at_0_guess, r0, rf, step):    
    integrator = spi.ode(rhs,jac).set_integrator(int_method)
    integrator.set_initial_value([yes_at_0_guess[0], ye_bv[1],
                                  yes_at_0_guess[1], ye_bv[3],
                                  yes_at_0_guess[2], ye_bv[5],
                                  yes_at_0_guess[3], ye_bv[7],
                                  yes_at_0_guess[4]], t=r0)  # Set the initial values
    V0_at_0_estimate,V2_at_0_estimate,V4_at_0_estimate, psi3_at_0_estimate, E3_estimate = solve_bvp(ye_bv, yes_at_0_guess,r0,rf,step)
    _, V0prime_at_0,_, V2prime_at_0, _, V4prime_at_0, _, psi3_prime_at_0 = ye_bv
    xs, zs = integrate_over_domain([V0_at_0_estimate, V0prime_at_0,
                                    V2_at_0_estimate, V2prime_at_0,
                                    V4_at_0_estimate, V4prime_at_0,
                                    psi3_at_0_estimate, psi3_prime_at_0,
                                    E3_estimate],
            integrator, r0, rf, step)
    return [xs, zs, E3_estimate]

def plotses(x, ys, tit, ruta, ncor, sv = False, show = True, m = 10):
    cer = np.zeros(np.shape(ys[:,2])[0])
    pts.plotmultiple([x, x, x],[ys[:,0], x**2*ys[:,2], x**4*ys[:,4]],
                     [r'$V_{00}$', r'$r^2 V_{20}$', r'$r^4 V_{40}$'],
                     r'$\rho$',r'$V$',tit,'%s/V_%d.png'%(ruta,ncor),save = sv)    
    pts.plotmultiple([x,x,x,x,x],
                     [ys[:,0], x**2*ys[:,2], x**4*ys[:,4],
                      ys[:,6], ys[:,8]],
                     [r'$V_{00}$',r'$r^2 V_{20}$',r'$r^4 V_{40}$',
                      r"$\psi_{32%d}$"%m,r'$E$'],
                     r'$\rho$','',tit,'%s/todas_%d.png'%(ruta,ncor),save = sv) 
    pts.parametricplot(x,ys[:,1],r'$\hat\mu r$',r'$N_T(r)$', tit,
                       '%s/N_%d.png'%(ruta,ncor), save = sv, show = show) 
    pts.plotmultiple([x, x],[ys[:,6], cer], [r'$\psi_{32%d}$'%m, r'$0$'],
                     r'$\rho$',r'$\psi$',tit,'%s/phi_%d.png'%(ruta,ncor),
                     save = sv, show = show)
def pot_sph_harm(V0, V2, V4, r, th):
    return V0(r) + V2(r)*np.sqrt(5./(4.))*(-1. + 3.*np.cos(th)**2) + V4(r)*3.*(3. - 30.*np.cos(th)**2 + 35.*np.cos(th)**4)/(4.)

def cinetica(xs, zs):
    xn, deriv = derivada(xs, zs[:,7])
    dFdr = interp1d(xn, deriv, kind='linear', copy=True, bounds_error=False, 
                    fill_value = 'extrapolate')
    int1 = integral(dFdr(xs)*xs**4*zs[:, 6], xs)
    int2 = 6.*integral(zs[:,6]*zs[:,7]*xs**3, xs)
    return -2.*np.pi*(int1 + int2)    

def potencial(xs, zs):
    int1 = integral(zs[:,6]**2*zs[:,0]*xs**6, xs)
    int2 = integral(zs[:,6]**2*zs[:,2]*xs**8, xs)
    int3 = integral(zs[:,6]**2*zs[:,4]*xs**10, xs)
    return 2.*np.pi*(int1 + c1*int2 + c2*int3)
    
def virializacion(xs, zs):
    K = cinetica(xs, zs)
    W = potencial(xs, zs)
    return K, W, K/abs(W)

def write_catidades_file(ncor, x, y, E_guess,
                         dec = '/home/jordi/satellite/l2_shooting'):
    rf = DE[ncor]['rf']
    N2 = integral(x**6*y[:,6]**2, x)
    Ke, We, _ = virializacion(x, y)
    f= open("%s/cantidades_%d.txt"%(dec,ncor),"w+")
    f.write(" N= %f, -N/rf= %f, Rf= %f \r\n " % (N2, -N2/rf, rf))
    f.write(" E_guess = %f, E_t= %f \r\n " % (E_guess, y[0,8]))
    f.write(" We= %f, Ke= %f, Ke/We= %f \r\n " % (We, Ke, Ke/abs(We)))
    f.write(" phif= %.30f \r\n " %(y[-1,6]))
    f.close()

def plots_densidad(x, y, rf,  dec = '/home/jordi/satellite/l2_shooting'):
    dens = interp1d(x, x**4*y[:,4]**2, kind='linear', copy=True, 
                    bounds_error=False)
    r = np.linspace(r0, rf, 400)
    th = np.linspace(0, np.pi, 400)
    R, T = np.meshgrid(r, th)
    dens2d = interp2d(r, th, sph_harm2(dens, R,T, l=2, m =m), kind='linear',
                      copy=True, bounds_error=False)
    x2 = np.linspace(-rf, rf, 400)
    z = np.linspace(-rf, rf, 400)
    X, Z = np.meshgrid(x2, z)
    pts.densityplot(X,Z,fi(dens2d, x2,z, rf = rf),r"$x$",r"$z$",
                    r"$\Phi_{32%d}^2(x,z)$"%m,
                    r'$E_{32%d}=%f$'%(m,E3), aspect = '1/1',
                    name= '%s/densxz_%d.png'%(dec, ncor))
    
    rho = np.linspace(0., rf, 400)
    z = np.linspace(-rf, rf, 400)
    firhoz = fi2(dens2d,rho,z)
    firhozinterpol = interp2d(rho, z, firhoz, kind='linear', copy=True, 
                              bounds_error=False)    
    density3d_files(firhozinterpol, rf, dec, ncor)
    #
    Vo = interp1d(x, y[:,0], kind='linear', copy=True, bounds_error=False)
    V2 = interp1d(x, x**2*y[:,2], kind='linear', copy=True, bounds_error=False)
    V4 = interp1d(x, x**4*y[:,4], kind='linear', copy=True, bounds_error=False)
    pot2d = interp2d(r, th, pot_sph_harm(Vo, V2, V4, R, T) , kind='linear', 
                     copy=True, bounds_error=False)
    pts.densityplot(R,T,pot_sph_harm(Vo, V2, V4, R, T) ,r"$r$",r"$\theta$",
                    r"$V(r,\theta)$", r'$E_{21%d}=%f$'%(m, E3),
                    name= '%s/potrth_%d'%(dec,ncor))    
    rho = np.linspace(0, rf, 400)
    Rho, Z = np.meshgrid(rho, z)    
    pts.densityplot(Rho,Z,pot(pot2d, rho,z, rf = rf),r"$\rho$",r"$z$",
                    r"$V(\rho,z)$", r'$E_{32%d}=%f$'%(m,E3), aspect = '1/2',
                    name= '%s/potrhoz_%d.png'%(dec,ncor))    
def vel_DM(xs, zs):
    P0 = zs[:, 1]
    V2 = zs[:, 2]
    P2 = zs[:, 3]
    V4 = zs[:, 4]
    P4 = zs[:, 5]
    r = xs
    c1 =  np.sqrt(5)/2.
    c2 = 9./8.
    v2 = -(-P0/r + c1*r**2*(2.*V2 + r*P2) - c2*r**4*(4.*V4 + r*P4))
    return np.sqrt(v2)



if __name__ == '__main__': 

    r0 = 0.1
    stp = .1 
    
    rf = DE[ncor]['rf']
    NTF = DE[ncor]['Nf']
    V0_guess = DE[ncor]['V0_guess']
    V2_guess = DE[ncor]['V2_guess']
    V4_guess = DE[ncor]['V4_guess']
    Phi30_guess = DE[ncor]['Phi30_guess']
    V0_at_length = -NTF/rf
    ye_boundvalue =[V0_at_length,0.,0.,0.,0.,0., 0., 0.] 
    #for i in range(10,201, 10):
    #    E3_guess = -i/100.  
    #    print(E3_guess)
    #    ncor = i 
    #    yguess=[V0_guess,V2_guess,V4_guess,Phi30_guess,E3_guess] 
    #    x,y,E3 = main(ye_boundvalue, yguess, r0, rf, stp)
    #    if E3<0:
    #        plotses(x,y,r'$E_{guess}=%f, E_{32%d}=%f$'%(E3_guess,m, E3),
    #                '/home/jordi/satellite/l2_shooting',ncor,m=m, sv= True,show = True)
     
    
    E3_guess = DE[ncor]['E_guess']
    yguess=[V0_guess,V2_guess,V4_guess,Phi30_guess,E3_guess] 
    x,y,E3 = main(ye_boundvalue, yguess, r0, rf, stp)
    #plotses(x,y,r'$E_{guess}=%f, E_{32%d}=%f$'%(E3_guess,m, E3),
    #        '/home/jordi/satellite/l2_shooting',ncor,m=m, sv= True,show = True)
    #write_catidades_file(ncor, x, y, E3_guess)  
    #plots_densidad(x, y, rf)   
    
    vdm = vel_DM(x, y)
    np.save('/home/jordi/satellite/l2_shooting/vdm_%d'%ncor, np.array([x,vdm]))
    mu = MM[21]['mu']/1000. #en pc
    rlam = MM[21]['rlam']
    c = 2.9e5
    pts.plotmultiple([x/(mu*rlam)], [rlam**2*vdm*c], [], 
                      r'$r$(pc)',r'$v$(km/s)','rot curve','',save = False)