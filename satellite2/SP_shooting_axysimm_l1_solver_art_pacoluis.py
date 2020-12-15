#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:40:50 2019

RESUELVE ESTADOS DIPOLO Y TORO DE SCHRODINGER-POISSON USANDO APROX DE ARTICULO 
DE PACO Y LUIS

@author: jordi
"""
import numpy as np
import scipy.optimize as opt
import scipy.integrate as spi
import plots_jordi as pts
from scipy.interpolate import interp1d, interp2d

ncor = 1 # key de diccionario DE
int_method = "dopri5"#'dop853'##"lsoda"###'vode' ### "dopri5"

DE = {1:{'rf' : 5., 'E2guess': .81, 'Nf': 4.04, 'V0_guess': -1.6, 
         'V2_guess': -0.9, 'Phi20_guess': 0.5, 'm':0},
      2:{'rf' : 5., 'E2guess': .88, 'Nf': 4.18, 'V0_guess': -1.6, 
         'V2_guess': -0.9, 'Phi20_guess': 0.5, 'm':1}} 

DC = {0: {'m':0, 'c':2./np.sqrt(5)}, 1: {'m':1, 'c':-1./np.sqrt(5)}}

NTF = DE[ncor]['Nf']
m = DE[ncor]['m']
c = DC[m]['c']

def rhs(x, y):
    """     Compute the rhs of y'(t) = f(t, y, params)    """
    V0, P0, V2, P2, psi2, F2, E2, N = y
    return np.array([P0/x**2,
            psi2**2*x**4,
            P2,
            -(6./x)*P2 + c*psi2**2,
            F2,
            -4.*F2/x + 2.*(V0 + c*x**2*V2 - E2)*psi2,
            0.,
            psi2**2*x**4])
    
def jac(x, y):
    """     Compute the jacobian    """
    V0, P0, V2, P2, psi2, F2, E2, N = y
    return np.array([[0., 1./x**2, 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 2.*x**4*psi2, 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0., 0.],
                     [0., 0., 0., -(6./x), 2.*c*psi2, 0., 0., 0.],
                     [0., 0., 0., 0., 0., 1., 0., 0.],
                     [2.*psi2, 0., 2.*c*x**2*psi2, 0., 2.*(V0 + c*x**2*V2 - E2), -4./x, -2.*psi2, 0.],
                     [0., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 2.*psi2*x**4,0., 0., 0.]])
#         
def integrate_over_domain(initial_conditions, integrator, r0,
                          rf, step, silent=True):
    integrator.set_initial_value(initial_conditions, t=r0)  # Set the initial values of z and x
#    integrator.set_f_params(alpha)
    dt = step
    length = rf
    xs, zs = [], []
    while integrator.successful() and integrator.t <= length:
        integrator.integrate(integrator.t + dt)
        xs.append(integrator.t)
        zs.append(integrator.y)
        if not silent:
            print("Current x and z values: ", integrator.t, integrator.y)
    zs = np.array(zs)
    xs = np.array(xs)
    return xs, zs

def solve_bvp(ye_bv, yes_at_0_guess, r0, rf, step):
    integrator = spi.ode(rhs,jac).set_integrator(int_method)
#    integrator = spi.ode(rhs).set_integrator(int_method)
    def residuals(yes_at_0_guess, V0_at_length, V0prime_at_0, 
                  V2_at_length, V2prime_at_0,
                  psi2_at_length, psi2_prime_at_0):         
        z_at_0 = [yes_at_0_guess[0], V0prime_at_0,
                  yes_at_0_guess[1], V2prime_at_0,
                  yes_at_0_guess[2], psi2_prime_at_0,
                  yes_at_0_guess[3],
                  0.]                
        xs, zs = integrate_over_domain(z_at_0, integrator, r0, rf, step)
        V0_at_length_integrated = zs[-1, 0]
        V2_at_length_integrated = zs[-1, 2]
        psi2_at_length_integrated = zs[-1, 4]
        N_at_length_integrated = zs[-1, 7]        
        return [V0_at_length - V0_at_length_integrated,
                V2_at_length - V2_at_length_integrated,
                psi2_at_length - psi2_at_length_integrated,
                NTF - N_at_length_integrated]    
    V0_at_length, V0prime_at_0, V2_at_length, V2prime_at_0, psi2_at_length, psi2_prime_at_0 = ye_bv
    lsq = opt.least_squares(residuals, yes_at_0_guess, 
                            args=(V0_at_length, V0prime_at_0,V2_at_length,
                                  V2prime_at_0,psi2_at_length, psi2_prime_at_0), loss="soft_l1")
    V0_at_0_estimate = lsq.x[0]
    V2_at_0_estimate = lsq.x[1]
    psi2_at_0_estimate = lsq.x[2]
    E0_estimate = lsq.x[3] 
    return [V0_at_0_estimate,V2_at_0_estimate, psi2_at_0_estimate, E0_estimate]

##yguess=[V0_guess,V2_guess,Phi2_0_guess,E2_guess]

def main(ye_bv, yes_at_0_guess, r0, rf, step):    
    integrator = spi.ode(rhs,jac).set_integrator(int_method)
#    integrator = spi.ode(rhs).set_integrator(int_method)
    integrator.set_initial_value([yes_at_0_guess[0], ye_bv[1],
                                  yes_at_0_guess[1], ye_bv[3],
                                  yes_at_0_guess[2], ye_bv[5],
                                  yes_at_0_guess[3],
                                  0.], t=r0)  # Set the initial values
    V0_at_0_estimate,V2_at_0_estimate, psi2_at_0_estimate, E2_estimate = solve_bvp(ye_bv, yes_at_0_guess,r0,rf,step)
    _, V0prime_at_0,_, V2prime_at_0, _, psi2_prime_at_0 = ye_bv
    xs, zs = integrate_over_domain([V0_at_0_estimate, V0prime_at_0,
                                    V2_at_0_estimate, V2prime_at_0,
                                    psi2_at_0_estimate,psi2_prime_at_0,
                                    E2_estimate,
                                    0.],integrator, r0, rf, step)
    return [xs, zs, E2_estimate]

def plotses(x, ys, tit, ruta, ncor, sv = False, m=m):
    cer = np.zeros(np.shape(ys[:,2])[0])
    pts.parametricplot(x,x**2*ys[:,2], r'$\hat\mu r$',r"$r^2 V_{20}$",tit,
                       '%s/r2V20_%d' % (ruta, ncor), save = sv)
    pts.parametricplot(x,ys[:,2], r'$\hat\mu r$',r"$ V_{20}$",tit,
                       '%s/V20_%d' % (ruta, ncor), save = sv)
    pts.plotmultiple([x,x],[ys[:,0],x**2*ys[:,2]],[r'$V_{00}$',r'$r^2 V_{20}$'],
                     r'$\rho$',r'$V$',tit,'%s/V_%d.png'%(ruta,ncor),save = sv)    
    pts.plotmultiple([x,x,x,x],[ys[:,0], x**2*ys[:,2], x**2*ys[:,4]**2, ys[:,6]],
                     [r'$V_{00}$',r'$r^2 V_{20}$',r"$r^2\psi_{21%d}^2$"%m,r'$E$'],
                     r'$\rho$','',tit,'%s/todas_%d.png'%(ruta,ncor),save = sv) 
    pts.parametricplot(x,ys[:,7],r'$\hat\mu r$',r'$N_T(r)$', tit,
                       '%s/N_%d.png'%(ruta,ncor), save = sv)
   
    pts.plotmultiple([x],[ys[:,4]], [r'$\psi_{21%d}$'%m],
                     r'$\rho$',r'$\psi$',tit,'%s/phi_%d.png'%(ruta,ncor), save = sv)

def write_catidades_file(ncor, x, y, E_guess,
                         dec = '/home/jordi/satellite/l1_shooting'):
    rf = DE[ncor]['rf']
    N2 = Area(y[:,4]*x,x,rf,stp)[0]
    Ke, We, _ = virializacion(x, y)
    f= open("%s/cantidades_%d.txt"%(dec,ncor),"w+")
    f.write(" N= %f, -N/rf= %f, Rf= %f \r\n " % (N2, -N2/rf, rf))
    f.write(" E_guess = %f, E_t= %f \r\n " % (E_guess, y[0,6]))
    f.write(" We= %f, Ke= %f, Ke/We= %f \r\n " % (We, Ke, Ke/abs(We)))
    f.write(" phif= %.30f \r\n " %(y[-1,4]))
    f.close()

def Area(f,x,xf,dx):
    "integral definida de f**2 x**2 dr  de 0 a xf"
    A=0.
    elem = int(np.rint(xf/dx))
    for i in range(0,elem - 1,1):
        A+= dx*f[i]**2*x[i]**2
    return [A, np.sqrt(A)] 

def fi(dens2d, x, z, r0 = 0.1, rf=5.):
    Phi = []
    tol = 0.02
    for i in range(0,np.shape(x)[0]):
        Phii=[]
        for j in range(0,np.shape(z)[0]):
            if z[j]**2 + x[i]**2 > rf**2:
                Phii.append(np.nan)
#            elif z[j]**2 + x[i]**2 < r0**2 + tol:
#                Phii.append(0.)
            else:
                if z[j]<0:
                   Phii.append(dens2d(np.sqrt(x[i]**2 + z[j]**2), np.arccos(-z[j]/np.sqrt(x[i]**2 + z[j]**2)))[0])
                else:
                    Phii.append(dens2d(np.sqrt(x[i]**2 + z[j]**2), np.arccos(z[j]/np.sqrt(x[i]**2 + z[j]**2)))[0]) 
        Phi.append(Phii)
    Phi = np.array(Phi)
    return Phi.T

def fi2(dens2d, x, z, r0 = 0.1, rf=5.):
    Phi = []
    tol = 0.02
    for i in range(0,np.shape(x)[0]):
        Phii=[]
        for j in range(0,np.shape(z)[0]):
            if z[j]**2 + x[i]**2 < r0**2 + tol:
                Phii.append(0.)
            else:
                if z[j]<0:
                   Phii.append(dens2d(np.sqrt(x[i]**2 + z[j]**2), np.arccos(-z[j]/np.sqrt(x[i]**2 + z[j]**2)))[0])
                else:
                    Phii.append(dens2d(np.sqrt(x[i]**2 + z[j]**2), np.arccos(z[j]/np.sqrt(x[i]**2 + z[j]**2)))[0]) 
        Phi.append(Phii)
    Phi = np.array(Phi)
    return Phi.T

def pot(pot2d, x, z, r0 = 0.1, rf=5.):
    V = []
    tol = 0.01
    for i in range(0,np.shape(x)[0]):
        Vi=[]
        for j in range(0,np.shape(z)[0]):
#            if z[j]**2 + x[i]**2 > rf**2:
#                Vi.append(np.nan)
#            elif z[j]**2 + x[i]**2 < r0**2 + tol:
#                Vi.append(np.nan)
#            else:
            if z[j]<0:
               Vi.append(pot2d(np.sqrt(x[i]**2 + z[j]**2), np.arccos(-z[j]/np.sqrt(x[i]**2 + z[j]**2)))[0])
            else:
                Vi.append(pot2d(np.sqrt(x[i]**2 + z[j]**2), np.arccos(z[j]/np.sqrt(x[i]**2 + z[j]**2)))[0]) 
        V.append(Vi)
    V = np.array(V)
    return V.T

def sph_harm2(dens, r, th, l=1, m=0):
    if l==0:
        return dens(r)*(1./(4.*np.pi))*np.ones(th.shape)
    elif l == 1:
        if m==0:
            return dens(r)*(3./(4.*np.pi))*np.cos(th)**2
        elif m==1:
            return dens(r)*(3./(8.*np.pi))*np.sin(th)**2
    elif l ==2:
        if m==0:
            return dens(r)*(5./(64.*np.pi))*(1. + 3.*np.cos(2.*th))**2
        elif m==1:
            return dens(r)*(15.*np.sin(th)**2*np.cos(th)**2)/(8.*np.pi)
        elif m==2:
            return dens(r)*(15.*np.sin(th)**4)/(32.*np.pi)

def pot_sph_harm(V0, V2, r, th):
    return V0(r)/(4.*np.pi) + V2(r)*np.sqrt(5./(16.*np.pi))*(-1. + 3.*np.cos(th)**2)

def derivada(r, f):
    df = []
    rn = []
    for j in range(0,np.shape(r)[0] - 1):
        df.append((f[j + 1] - f[j])/(r[j + 1] - r[j]))
        rn.append(r[j] )
    dfdr = np.array(df)
    rnn = np.array(rn)
    return rnn, dfdr

def integral(f, r):
    "integral definida de f dr"
    A=0.
    dx = r[1]-r[0]
    for i in range(0, np.shape(r)[0],1):
        A+= dx*f[i]
    return A 

def cinetica(xs, zs):
    xn, deriv = derivada(xs, zs[:,5])
    dFdr = interp1d(xn, deriv, kind='linear', copy=True, bounds_error=False, 
                    fill_value = 'extrapolate')
    int1 = integral(dFdr(xs)*xs**4*zs[:, 4], xs)
    int2 = 4.*integral(zs[:,5]*zs[:,4]*xs**3,xs)
    return -2.*np.pi*(int1 + int2)    

def potencial(xs, zs):
    int1 = integral(zs[:,4]**2*zs[:,0]*xs**4, xs)
    int2 = c*integral(zs[:,4]**2*zs[:,2]*xs**6, xs)
    return 2.*np.pi*(int1 + int2)
    
def virializacion(xs, zs):
    K = cinetica(xs, zs)
    W = potencial(xs, zs)
    return K, W, K/abs(W)

def vel_DM(xs, zs):
    P0 = zs[:, 1]
    V2 = zs[:, 2]
    P2 = zs[:, 3]
    r = xs
    return np.sqrt(-(-P0/r + np.sqrt(5)*r**2*(2.*V2 + r*P2)/2. ))

###############################################################################
###### Funcion para crear archivos density 3d para graficar en mathematica ####################
###############################################################################    
def density3d_files(phiex, Rf , rut, ncor):    
    def ficuad3D(phiex, x, y, z):
        Phi = []
        for i in range(0,np.shape(x)[0]):
            Phii=[]
            for j in range(0,np.shape(y)[0]):
                Phiii=[]
                for k in range(0,np.shape(z)[0]):
                    R,Z = np.sqrt(x[i]**2 + y[j]**2), z[k]
                    Phiii.append(phiex(R,Z))
                Phii.append(Phiii)
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.transpose(Phi)    
    p = 80   
    pp= p**3
    x = np.linspace(-Rf , Rf , p)
    y = np.linspace(-Rf , Rf , p)
    z = np.linspace(-Rf , Rf , p)    
    a= ficuad3D(phiex, x, y, z)
    b = a.reshape(1,pp)
    np.savetxt("%s/dens_%d.CSV"%(rut,ncor),b,delimiter=',')    

    
if __name__ == '__main__':        
    r0 = 0.1
    stp = .1
    rf = DE[ncor]['rf']
    E2_guess = DE[ncor]['E2guess']   
    NTF = DE[ncor]['Nf']
    V0_guess = DE[ncor]['V0_guess'] 
    V2_guess = DE[ncor]['V2_guess'] 
    Phi20_guess = DE[ncor]['Phi20_guess'] 
    
    
    V0_at_length = -NTF/rf
    ye_boundvalue =[V0_at_length,0.,0.,0.,0.,0.] #ye_boundvalue=[V0_at_length, V0prime_at_0, V2_at_length, V2prime_at_0, psi2_at_length, psi2_prime_at_0]
    
    yguess=[V0_guess,V2_guess,Phi20_guess,E2_guess]
    x,y,E2 = main(ye_boundvalue, yguess, r0, rf, stp)
    
    #plotses(x,y,r'$E_{guess}=%f, E_{210}=%f$'%(E2_guess, E2),
    #        '/home/jordi/satellite/l1_shooting',ncor, sv= True)
#    write_catidades_file(ncor, x, y, E2_guess)
 
    vdm = vel_DM(x, y)
    np.save('/home/jordi/satellite/l1_shooting/vdm_%d'%ncor, np.array([x,vdm]))

    pts.plotmultiple([x], [vdm], [], 
                      r'$r$',r'$v$','rot curve','',save = False)    
#    N2 = Area(y[:,4]*x,x,rf,stp)[0]
#    
#    print('N2=',N2)
#    print("-Nf/rf=", -NTF/rf, "V0(rf)=",y[-1,0])
#    
#    V0_at_0 = y[0,0]
#    V2_at_0 = y[0,2]
#    psi2_at_0 = y[0,4]
#    
#    
#    integrator = spi.ode(rhs,jac).set_integrator(int_method)
#    xs, zs = integrate_over_domain([V0_at_0, 0., V2_at_0, 0., psi2_at_0,0., E2, 0.],
#                                   integrator, r0, rf, stp)
##    plotses(xs,zs,r'$E_{21%d}=%f$'%(m, E2),
##            '/home/jordi/satellite/l1_shooting',ncor, sv= True, m= m) 
# 
#    print('K/W=', virializacion(xs, zs)[2])
#   
#       
#    dens = interp1d(xs, xs**2*zs[:,4]**2, kind='linear', copy=True,
#                    bounds_error=False)  
#    r = np.linspace(r0, rf, 400)
#    th = np.linspace(0, np.pi, 400)
#    R, T = np.meshgrid(r, th)
#         
##    pts.densityplot(R,T,sph_harm2(dens, R,T, m=m),r"$r$",r"$\theta$",
##                    r"$\Phi_{21%d}^2(r,\theta)$"%m,
##                    r'$E_{21%d}=%f$'%(m, E2),
##                    name= '/home/jordi/satellite/l1_shooting/densrth_%d'%ncor)
#    
#    dens2d = interp2d(r, th, sph_harm2(dens,R,T, l=1, m=m), kind='linear', 
#                      copy=True, bounds_error=False, )
#
#
#    
#    x = np.linspace(-rf, rf, 400)
#    z = np.linspace(-rf, rf, 400)
#    X, Z = np.meshgrid(x, z)
#    
##    pts.densityplot(X,Z,fi(dens2d,x,z),r"$x$",r"$z$",r"$\Phi_{21%d}^2(x,z)$"%m,
##                    r'$E_{21%d}=%f$'%(m, E2), aspect = '1/1',
##                    name= '/home/jordi/satellite/l1_shooting/densxz_%d'%ncor)
#
#    x = np.linspace(0., rf, 400)
#    z = np.linspace(-rf, rf, 400)
#    firhoz = fi2(dens2d,x,z)
#    firhozinterpol = interp2d(x, z, firhoz, kind='linear', copy=True, 
#                              bounds_error=False)    
#    density3d_files(firhozinterpol, rf ,'/home/jordi/satellite/l1_shooting', ncor)
#    
#    Vo = interp1d(xs, zs[:,0], kind='linear', copy=True, bounds_error=False)
#    V2 = interp1d(xs, xs**2*zs[:,2], kind='linear', copy=True,
#                  bounds_error=False)
#    pot2d = interp2d(r, th, pot_sph_harm(Vo, V2, R, T) , kind='linear',
#                     copy=True, bounds_error=False)
#    
#    rho = np.linspace(0, rf, 400)
#    z = np.linspace(-rf, rf, 400)
#    Rho, Z = np.meshgrid(rho, z)
#    
#    pts.densityplot(Rho,Z,pot(pot2d, rho,z, rf = rf),r"$\rho$",r"$z$",r"$V(\rho,z)$",
#                    r'$E_{210}=%f$'%(E2), aspect = '1/2',
#                    name= '/home/jordi/satellite/l1_shooting/potrhoz_%d.png'%ncor)    
#    
    #r0 = 0.1
    #rf = 5.
    #stp = .1    
    #for i in range(88, 89, 1):
    ##for i in (67, 72, 73, 74, 77,81, 82,83,84,85,86,88, 92):
    #    E2_guess = i/100.     
    #    ncor = i
    #    V0_at_length = -NTF/rf
    #    ye_boundvalue =[V0_at_length,0.,0.,0.,0.,0.] ##ye_boundvalue=[V0_at_length, V0prime_at_0, V2_at_length, V2prime_at_0, psi2_at_length, psi2_prime_at_0]
    #    
    #    V0_guess = -1.6
    #    V2_guess = -0.9
    #    Phi20_guess = 0.5
    #    
    #    yguess=[V0_guess,V2_guess,Phi20_guess,E2_guess]
    #    x,y,E2 = main(ye_boundvalue, yguess, r0, rf, stp)
    #    
    ##    plotses(x,y,r'$E_{guess}=%f, E_{210}=%f$'%(E2_guess, E2),
    ##            '/home/jordi/satellite/l1_shooting',ncor, sv= True)
    #    
    #
    #    N2 = Area(y[:,4]*x,x,rf,stp)[0]
    # 
    #    print('N=', NTF)
    #    print('N2=',N2)
    #    print("-1/rf=", -NTF/rf, "V0(rf)=",y[-1,0])
