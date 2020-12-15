#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:32:47 2020

Velocidades circulares de diferentes potenciales y/o distribuciones
Potenciales
Masa(r)
densidades
@author: jordi
"""
#from galpy import potential
import numpy as np
from scipy import special as sp
from scipy import integrate
from scipy.interpolate import interp1d
from constants_grav import c

################################################################################
#####################      VELOCIDADES CIRCULARES      ###########################3
################################################################################
def f(r, Mc, Rc):
    return Mc*(-2.*np.exp(-r**2/Rc**2)*r + np.sqrt(np.pi)*Rc*sp.erf(r/Rc))/(Rc*np.sqrt(np.pi))  

def g(r, rs):
    return  1./(1. + r/rs) + np.log(1. + r/rs)

def v2_DM(r,G, Rc, Mc, re, rs):## M es entre 10^10 M_sol
    "soliton + NFW"
    zeroval = 1.
    rhos = Mc*re*np.exp(-re**2/Rc**2)*(1. + re/rs)**2/(rs*np.sqrt(np.pi)**3*Rc**3)
    Mh = f(r, Mc, Rc)*np.heaviside(re - r, zeroval) + f(re, Mc, Rc)*np.heaviside(r - re, zeroval) + 4.*np.pi*rs**3*rhos*(g(r, rs) - g(re, rs))*np.heaviside(r - re, zeroval)
    ve2 = G*Mh/r
    return ve2

def v2_GaussDM(r,G, Rc, Mc):## M es entre 10^10 M_sol
    "soliton (Gaussian aprox)"
    Mh = f(r, Mc, Rc)
    ve2 = G*Mh/r
    return ve2

#def RC_miyamoto(rho, G, M, a, b):
#    "Miyamoto-Nagai"
#    rp= potential.MiyamotoNagaiPotential(amp = G*M, a = a, b = b)
#    vcc = rp.vcirc(rho)
#    return vcc

def RC_NFW(r, G, rho0, rs):
    Mh = 4.*np.pi*rs**3*rho0*(-r/(r + rs) - np.log(rs) + np.log(r+rs))
    vc = G*Mh/r
    return np.sqrt(vc)

def vnagai(rho, G=1., M = 1., a = 1., b = 2.):
    "Miyamoto-Nagai"
    vel = np.sqrt(G*M)*rho/np.sqrt(np.sqrt(rho**2 + (a + b)**2)**3)
    return vel

def vHernquist(rho, G, M, a):
    v = np.sqrt(G*M*rho)/(rho + a)
    return v

#def Miyamoto_Nagai_3(rho, G, S0, hr, hz):
#    "3 discs Miyamoto-Nagai"
#    A = G*S0
#    dp= potential.MN3ExponentialDiskPotential(amp=A,hr=hr,hz=hz, sech = True,
#                                              normalize=False)
#    vcc = dp.vcirc(rho)
#    return vcc

def vBH(r, G, MBH):
    ve2 = G*MBH/r
    return np.sqrt(ve2)

def RC_razor(rho, A, hr):
    print(' AHORA se llama RC_exponential')
    
def RC_exponential(rho, G, M, hr):
    "Razor thin exponential disc"
    Sigma = M/(2.*np.pi*hr**2)
    A = G*Sigma
    v2 = np.pi*A*rho**2*(sp.i0(rho/(2.*hr))*sp.k0(rho/(2.*hr)) - sp.i1(rho/(2.*hr))*sp.k1(rho/(2.*hr)))/hr
    return np.sqrt(v2)

#def RC_exponential(rho, G, S0, hr):
#    A = G*S0
#    dp= potential.RazorThinExponentialDiskPotential(amp=A,hr=hr,normalize=False)
#    vcc = dp.vcirc(rho)
#    return vcc

#def RC_double_exponential(rho, G, S0, hr, hz):
#    A = G*S0
#    dp= potential.DoubleExponentialDiskPotential(amp = A, hr = hr, hz = hz,
#                                                 normalize = False)
#    vcc = dp.vcirc(rho)
#    return vcc

def RC_double_exponential_aprox(rho, G, S0, hr, hz):
    A = G*S0
    v2 = np.pi*A*rho**2*(sp.i0(rho/(2.*hr))*sp.k0(rho/(2.*hr)) - sp.i1(rho/(2.*hr))*sp.k1(rho/(2.*hr)))/hr - 2.*np.pi*A*rho*hz*np.exp(-rho/hr)/hr
    return np.sqrt(v2)

def F(x):
    return 1. - np.exp(-x)*(1. + x + 0.5*x**2)

def exp_bulge(r,G, M, a):
    "velocidad circular de bulge exponencial esferico"
    vcuad = G*M*F(r/a)/r
    return np.sqrt(vcuad)

#def v_multi_SFDM(r, rlam, mu, ncor):
#    paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'
#    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))    
#    xn = x2/(mu*rlam)
#    if rlam ==1:
#        C = 1
#    else:
#        C = c
#    vdmn = rlam*vdm*C
#    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
#                  fill_value = "extrapolate")
#    """para corregir la extrapolacion (NEWTONIAN TALE) """
#    res = ve(r)
#    rsd = r*rlam*mu
#    N = x2[-1]*vdm[-1]**2 
#    res = np.where(rsd>x2[-1], np.sqrt(N/rsd)*rlam*C, res)
#    return res

def integralesferica(f,r,R):
    "integral definida de r**2*f(r) dr r de 0 a R "
    A=0.
    dr = r[1]-r[0]
    for i in range(0,np.shape(r)[0],1):
        A+= dr*f[i]*r[i]**2
    return A
def integral(f,r,R):
    "integral definida de f(r) dr r de 0 a R "
    A=0.
    dr = r[1]-r[0]
    for i in range(0,np.shape(r)[0],1):
        A+= dr*f[i]
    return A

class multistate:
    def __init__(self, state):
        self.state = state
        self.C = np.sqrt(5)/2.# constante potencial para RC
        self.dirshoot = '/home/jordi/satellite/mix_shooting_%d'%(self.state)
        if self.state == 210:
            self.nsols = tuple(range(12,19,1))+tuple(range(20,101,5))
#            self.nsols = tuple(range(12,19,1))+tuple(range(20,35,5)) +tuple(range(40,101,5))
        elif self.state == 211:
            self.nsols = tuple(range(12,21,1))+tuple(range(25,101,5))
        elif self.state == 200:    
#            self.nsols = tuple(range(120,180,5))
            self.nsols = (180, 360, 380, 400)
        elif self.state == 320:    
            self.nsols = tuple(range(15,20,1))+tuple(range(20,101,5))
            self.C2 = 9./8.# constante potencial para RC

    def calc_cir_vel(self):
        for i in self.nsols:
            r = np.load('%s/%d/x_1.npy'%(self.dirshoot, i)) 
            if self.state == 200: 
                _, P0, _, _, _, _, _, _= np.load('%s/%d/y_1.npy'%(self.dirshoot,i)).T
                vdm = np.sqrt(P0/r)
                np.save('%s/%d/vdm_1.npy'%(self.dirshoot,i), np.array([r, vdm]))
            elif self.state == 320:
                _, P0, V20, P2, V40, P4, _, _, _, _, _, _= np.load('%s/%d/y_1.npy'%(self.dirshoot,i)).T
                vdm = np.sqrt(P0/r - self.C*r**2*(r*P2 + 2.*V20) + self.C2*r**4*(r*P4 + 4.*V40))
                np.save('%s/%d/vdm_1.npy'%(self.dirshoot,i), np.array([r, vdm]))
            else: #los 21m
                _, P0, V20, P2, _, _, _, _,_, _= np.load('%s/%d/y_1.npy'%(self.dirshoot,i)).T
                vdm = np.sqrt(P0/r - self.C*r**2*(r*P2 + 2.*V20))
                np.save('%s/%d/vdm_1.npy'%(self.dirshoot,i), np.array([r, vdm]))

    def plot_family(self):
        import plots_jordi as pts
        psis2 = []; psis1 = []; V0s = []; V2s=[]; E2s = []; E1s = []; Ns = []
        Ets = []; vdms = []
        erres = []; labelsn = []
        for i in self.nsols:
            r = np.load('%s/%d/x_1.npy'%(self.dirshoot, i)) 
            _, vdm = np.load('%s/%d/vdm_1.npy'%(self.dirshoot,i))
            if self.state == 200: 
                V00, P0, Psi1, _, Psi2, _, E1, E2= np.load('%s/%d/y_1.npy'%(self.dirshoot,i)).T
                V20 = V00
                N2 = integralesferica(Psi2**2, r, r[-1])
            elif self.state == 320:
                V00, P0, V20, _, V40, _, Psi1, _, Psi2, _, E1, E2= np.load('%s/%d/y_1.npy'%(self.dirshoot,i)).T
                N2 = integralesferica(r**4*Psi2**2, r, r[-1])
            else: # los 21m
                V00, P0, V20, _, Psi1, _, Psi2, _,E1, E2= np.load('%s/%d/y_1.npy'%(self.dirshoot,i)).T
                N2 = integralesferica(r**2*Psi2**2, r, r[-1])
            V0s.append(V00)
            V2s.append(V20)
            Ns.append(P0)  
            psis1.append(Psi1)
            psis2.append(Psi2) 
            E1s.append(E1)
            E2s.append(E2)    
            labelsn.append(P0[-1])
            erres.append(r)
            N1 = integralesferica(Psi1**2, r, r[-1])
            Et = (E1[0]*N1 + E2[0]*N2)/P0[-1]
            Ets.append(Et*np.ones(np.shape(r)))
            vdms.append(vdm)
            
        pts.multiplot_colorbar(erres, V0s, labelsn, r'$\hat r$', r'$V_{00}$', 
                         '', '%s/V00.png'%self.dirshoot, r'$N_T$', ticks =labelsn)
        pts.multiplot_colorbar(erres, V2s, labelsn, r'$\hat r$', r'$V_{20}$', 
                         '', '%s/V20.png'%self.dirshoot, r'$N_T$', ticks =labelsn)
        pts.multiplot_colorbar(erres, Ns, labelsn, r'$\hat r$', r'$N_T$', 
                         '', '%s/NT.png'%self.dirshoot, r'$N_T$', ticks =labelsn)      
        pts.multiplot_colorbar(erres, psis1, labelsn, r'$\hat r$', r'$\psi_{100}$', 
                         '', '%s/Phib%d.png'%(self.dirshoot, self.state), r'$N_T$', ticks =labelsn) 
        pts.multiplot_colorbar(erres, psis2, labelsn, r'$\hat r$', r'$\psi_{%d}$'%(self.state), 
                         '', '%s/Phi%d.png'%(self.dirshoot, self.state), r'$N_T$', ticks =labelsn) 
        pts.multiplot_colorbar(erres, E1s, labelsn, r'$\hat r$', r'$E_{100}$', 
                         '', '%s/E1.png'%self.dirshoot, r'$N_T$', ticks =labelsn)
        pts.multiplot_colorbar(erres, E2s, labelsn, r'$\hat r$', r'$E_{%d}$'%(self.state), 
                         '', '%s/E2.png'%self.dirshoot, r'$N_T$', ticks =labelsn)
        pts.multiplot_colorbar(erres, Ets, labelsn, r'$\hat r$', r'$E_{T}$', 
                         '', '%s/ET.png'%self.dirshoot, r'$N_T$', ticks =labelsn)    
        pts.multiplot_colorbar(erres, vdms, labelsn, r'$\hat r$', r'$v_h/c$', 
                         '', '%s/vcirc.png'%self.dirshoot, r'$N_T$',
                         ticks =labelsn, ylim=(0,2.3))        
class SFDM(multistate):
    def __init__(self, state, NT):
        super().__init__(state)
        self.NT = NT
        self.x = np.load('%s/%d/x_1.npy'%(self.dirshoot,self.NT)) 
        self.funcs = np.load('%s/%d/y_1.npy'%(self.dirshoot,self.NT)).T         

        if self.state == 200: 
            self.V00, self.P0, self.Psi1, _, self.Psi2, _,self.E1, _= self.funcs 
            self.V20 = self.V00
        elif self.state == 320:
            self.V00, self.P0, self.V20, _, self.V40, _, self.Psi1, _, self.Psi2, _,self.E1, _= self.funcs 
        else: # los 21m
            self.V00, self.P0, self.V20, _, self.Psi1, _, self.Psi2, _,self.E1, _= self.funcs 
            
class multi_SFDM(SFDM):
    def __init__(self, state, r, rlam, mu, NT):
        super().__init__(state, NT)
        self.r = r
        self.rlam = rlam
        self.mu = mu
        if self.rlam ==1:
            self.cc = 1
        else:
            self.cc = c     
            
    def circ_vel(self):
        x2, vdm = np.load('%s/%d/vdm_1.npy'%(self.dirshoot,self.NT))    
        xn = x2/(self.mu*self.rlam)
        vdmn = self.rlam*vdm*self.cc
        ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
                      fill_value ="extrapolate")
        """para corregir la extrapolacion (NEWTONIAN TALE) """
        res = ve(self.r)
        rsd = self.r*self.rlam*self.mu
        N = x2[-1]*vdm[-1]**2 
        res = np.where(rsd>x2[-1], np.sqrt(N/rsd)*self.rlam*self.cc, res)
#        return res 
        return np.where(np.isnan(res), 0, res)
    
    
    def potential(self): 
        v00 = interp1d(self.x, self.V00, kind = 'linear', copy = True,
                       bounds_error = False, fill_value ="extrapolate")
        """para corregir la extrapolacion (NEWTONIAN TALE) """
        res = v00(self.r)
        N = self.P0[-1]
        res = np.where(self.r>self.x[-1], -N/self.r, res)
        
        v20 = interp1d(self.x, self.V20, kind = 'linear', copy = True, 
                       bounds_error = False, fill_value ="extrapolate")
        res2 = v20(self.r)
        N2 = -self.V20[-1]*self.x[-1]
        res2 = np.where(self.r>self.x[-1], -N2/self.r, res2) 
        
        if self.state==320:
            v40 = interp1d(self.x, self.V40, kind = 'linear', copy = True, 
                           bounds_error = False, fill_value ="extrapolate")
            res3 = v40(self.r)
            N3 = -self.V40[-1]*self.x[-1]
            res3 = np.where(self.r>self.x[-1], -N3/self.r, res3) 
            return res, res2, res3
        else:
            return res, res2   
    
    def density(self):
        xn = self.x/(self.mu*self.rlam)
        Psi1n = self.Psi1*self.rlam**2
        Psi2n = self.Psi2*self.rlam**2
        psi1 = interp1d(xn, Psi1n, kind = 'linear', copy = True,
                       bounds_error = False, fill_value ="extrapolate")        
        """para corregir la extrapolacion (NEWTONIAN TALE) """
        res = psi1(self.r)
        E = self.E1[-1]
        res = np.where(self.r>self.x[-1], np.exp(-np.sqrt(-2.*E)*self.r)/self.r, res)         
        psi2 = interp1d(xn, Psi2n, kind = 'linear', copy = True,
                       bounds_error = False, fill_value = (Psi2n[0], 0.))        
        res2 = psi2(self.r)
        return res**2, res2**2

################################################################################
########################            DENSIDADES      ###########################3
################################################################################
def dens_Miyamoto(rho, z, M, a, b):
    dens = b**2*M*(a**3 + 5.*a**2*np.sqrt(b**2 + z**2) + 3.*np.sqrt(b**2 + z**2)**3 + a*(7.*b**2 + 7.*z**2 +rho**2))/(4.*np.pi*np.sqrt(b**2 + z**2)**3*np.sqrt((a + np.sqrt(b**2 + z**2))**2 +rho**2)**5)
    return 4.*np.pi*dens*rho

#def dens_Miyamoto_Nagai(rho,z, G, Md, ad, bd):
#    print("SE MODIFICO ESTA FUNCION SE AGREGO G A PARAMS")
#    rp= potential.MiyamotoNagaiPotential(amp = G*Md, a = ad, b = bd)
#    dens = rp.dens(rho, z = z)
#    return dens

def dens_doble_exp(rho,z,G, A,a,b):
    print("SE MODIFICO ESTA FUNCION SE AGREGO G A PARAMS")    
    return A*np.exp(-rho/a)*np.exp(-abs(z)/b)/(2.*G*b)

def dens_gaus(r, Rc, Mc):
    rho = Mc*np.exp(-r**2/Rc**2)/(Rc**3*np.sqrt(np.pi)**3)
    return rho

def dens_DM(r, Rc, Mc, re, rs):
    "soliton + NFW"
    zeroval = 1.
    rhos = Mc*re*np.exp(-re**2/Rc**2)*(1. + re/rs)**2/(rs*np.sqrt(np.pi)**3*Rc**3)
#    dens = Mc*np.exp(-r**2/Rc**2)/(Rc**3*np.sqrt(np.pi)**3)*np.heaviside(re - r, zeroval) 
    dens2 = rhos*np.heaviside(r - re, zeroval)*rs/(r*(1. + r/rs)**2)
    return dens2

################################################################################
########################            MASAS      ###########################3
################################################################################
def M_Miyamoto(r, M, a, b):
    integral = integrate.nquad(dens_Miyamoto, [[0., r], [0., r]],
                               args = (M, a, b))
    return integral[0]

def integ(x, rho, M, a,b):
    Z = np.sqrt(b**2 + x**2)
    return -b**2*M*((a+Z)**3 + a*rho**2)/(Z**3*np.sqrt( (a+Z)**2 + rho**2)**3)
    
def M_Miyamoto2(r, z, M, a, b):
    integral = integrate.nquad(integ, [[0., r]], args = (r, M, a, b))
    inte2 = M*z/np.sqrt(b**2 + z**2)
    return integral[0] + inte2

def M_hernquist(r, M, a):
    return M*r**2/(r + a)**2

def M_doble_disk(r, M, a, b):
    integral = integrate.nquad(dens_doble_exp, [[0., r], [0., r]], 
                               args = (M, a, b))
    return integral[0]

def M_two_bulge(x, Mi, ai, Mo, ao):
    M = Mi + Mo - (np.exp(-(x/ai))*Mi*(2.*ai**2 + 2.*ai*x + x**2))/(2.*ai**2) - (np.exp(-(x/ao))*Mo*(2.*ao**2 + 2.*ao*x + x**2))/(2.*ao**2)
    return M

def M_exp_bulge(x, M, a):
    Mass = M - (np.exp(-(x/a))*M*(2.*a**2 + 2.*a*x + x**2))/(2.*a**2)
    return Mass

def M_exp_disk(x, M, a):
    S = M/(2.*np.pi*a**2)     
    Masa = 2.*a*np.pi*(a - np.exp(-x/a)*(a + x))*S
    return Masa

def M_CNFW(r, Rc, Mc, re, rs):
    "soliton + NFW"
    zeroval = 1.
    rhos = Mc*re*np.exp(-re**2/Rc**2)*(1. + re/rs)**2/(rs*np.sqrt(np.pi)**3*Rc**3)
    Mh = f(r, Mc, Rc)*np.heaviside(re - r, zeroval) + f(re, Mc, Rc)*np.heaviside(r - re, zeroval) + 4.*np.pi*rs**3*rhos*(g(r, rs) - g(re, rs))*np.heaviside(r - re, zeroval)
    return Mh

def M_NFW(r, rho0, rs):
    Mh = 4.*np.pi*rs**3*rho0*(-r/(r + rs) - np.log(rs) + np.log(r+rs))
    return Mh
################################################################################
#####################      POTENCIALES      ###################################
################################################################################
def POT_multi_SFDM(r, nsol, di = 'baja_dens/',
                   rut = "/home/jordi/satellite/schrodinger_poisson/potpaco"):
    xn, V00 = np.load("%s/%spot_%d/V00_%d.npy"%(rut,di,nsol,nsol))
    _, r2V20 = np.load("%s/%spot_%d/r2V20_%d.npy"%(rut,di,nsol,nsol))
    v0 = interp1d(xn, V00, kind = 'linear', copy = True, bounds_error = False,
                  fill_value = "extrapolate")    
    r2v2 = interp1d(xn, r2V20, kind = 'linear', copy = True, bounds_error = False,
                  fill_value = "extrapolate")
    """para corregir la extrapolacion (NEWTONIAN LIMIT) """  
    resV0 = v0(r)
    N = -xn[-1]*V00[-1]   
    resV0 = np.where(r>xn[-1], -N/r, resV0)
    """para corregir la extrapolacion (NEWTONIAN LIMIT) """  
    resr2V2 = r2v2(r)
    N2 = -xn[-1]*r2V20[-1]  
    resr2V2 = np.where(r>xn[-1], -N2/r, resr2V2)
    return resV0, resr2V2

def POT_multi_SFDM2(r, nsol, di = 'baja_dens/',
                   rut = "/home/jordi/satellite/schrodinger_poisson/potpaco"):
    xn, V00 = np.load("%s/%spot_%d/V00_ext_%d.npy"%(rut,di,nsol,nsol))
    _, r2V20 = np.load("%s/%spot_%d/r2V20_ext_%d.npy"%(rut,di,nsol,nsol))
    v0 = interp1d(xn, V00, kind = 'linear', copy = True, bounds_error = False)    
    r2v2 = interp1d(xn, r2V20, kind = 'linear', copy = True, bounds_error = False)
    return v0, r2v2