#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020

@author: jordi
"""
import pymc3 as pm3
import plots_jordi as pts
import numpy as np
from MW_M31_rotation_curve_padila import v2_DM, M_t
import matplotlib.pyplot as plt
import arviz as az
#plt.style.use('seaborn')
from def_potenciales import g
from scipy.interpolate import interp1d
from SPARC_desc import tipos, data_dict
from def_potenciales import f as ff
sat = '/home/jordi/satellite'
dirdata = '/home/jordi/SPARC'
dirfitsG = '/home/jordi/SPARC/Fits/mcmc'

c = 2.99792458e5 ## km/s
Gentc2 = 4.799e-7 ##G/c^2 en kpc/(10^10 M_sol)
G = Gentc2*c*c
np.random.seed(12345)

def f(r, Mc, Rc):
    return Mc*(-2.*np.exp(-r**2/Rc**2)*r + np.sqrt(np.pi)*Rc*pm3.math.erf(r/Rc))/(Rc*np.sqrt(np.pi))  

def v2_DMatter(r, Rc, Mc, re, rs):## M es entre 10^10 M_sol
    rhos = Mc*re*np.exp(-re**2/Rc**2)*(1. + re/rs)**2/(rs*np.sqrt(np.pi)**3*Rc**3)
    Mh = pm3.math.switch(r>re, f(re, Mc, Rc) + 4.*np.pi*rs**3*rhos*(g(r, rs) - g(re, rs)),
                    f(r, Mc, Rc))
    ve2 = G*Mh/r
    return ve2

def M(r, data, Rc, Mc, re, rs, ML):
    vgas = data[:,3]
    vdisk = data[:,4]
    vbul = data[:,5]     
    rhos = Mc*re*np.exp(-re**2/Rc**2)*(1. + re/rs)**2/(rs*np.sqrt(np.pi)**3*Rc**3)
    Mh = ff(r, Mc, Rc)*np.heaviside(re - r, 0.5) + ff(re, Mc, Rc)*np.heaviside(r - re, 0.5) + 4.*np.pi*rs**3*rhos*(g(r, rs) - g(re, rs))*np.heaviside(r - re, 0.5)
    return np.array(Mh + vgas**2*r/G + ML*vdisk**2*r/G + 1.4*ML*vbul**2*r/G)

def fit_mcmc(data, i):
    MO = {'Rc' : [0.001, 1., 0.01], 'Mc' : [0.0001, 0.1, 8.], 
          're' : [0.001, 20., 0.001], 'rs' : [0.002, 100., 2.], 
          'ML' : [0.1, 5., 1.5]}
    namep =   [r'$R_c$', r'$M_c$', r'$r_e$', r'$r_s$', r'$M/L$']
    
    nsamples = 10000
    
    rad, v, v_error, vgas, vdisk, vbul, _, _ = data.T
    with pm3.Model() as model: 
        Rc = pm3.Uniform(r'$R_c$', MO['Rc'][0], MO['Rc'][1])#, testval = MO['Rc'][2])
        Mc = pm3.Uniform(r'$M_c$', MO['Mc'][0], MO['Mc'][1])#, testval = MO['Mc'][2])
        re = pm3.Uniform(r'$r_e$', MO['re'][0], MO['re'][1])#, testval = MO['re'][2])
        rs = pm3.Uniform(r'$r_s$', MO['rs'][0], MO['rs'][1])#, testval = MO['rs'][2])  
        ML = pm3.Uniform(r'$M/L$', MO['ML'][0], MO['ML'][1])#, testval = MO['ML'][2])
        rot_vel = pm3.Deterministic('rot_vel', 
                                    np.sqrt(v2_DMatter(rad, Rc, Mc, re, rs) + 
                                    vgas**2 + ML*vdisk**2 + 1.4*ML*vbul**2))

        y = pm3.Normal('y', mu = rot_vel, tau = 1.0/v_error**2, observed = v)
        start = pm3.find_MAP()
        step = pm3.NUTS()
        trace = pm3.sample(nsamples, start=start)
    
    #############         extract and plot results         ########################
    pm3.summary(trace).to_csv('%s/summary_%d.csv' % (dirfitsG, i))
    plt.show()
    y_min = np.percentile(trace.rot_vel,2.5,axis=0)
    y_max = np.percentile(trace.rot_vel,97.5,axis=0)
    y_fit = np.percentile(trace.rot_vel,50,axis=0)    
    popt = pm3.summary(trace, var_names=namep)['mean'].values 
    import matplotlib as mpl
    mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 1
    az.rcParams['plot.max_subplots']= 80
    
    az.plot_pair(trace, var_names = namep, kind = 'kde',# kind = ["scatter", "kde"],
                      kde_kwargs={"fill_last": False}, marginals=True,
                      point_estimate="mode", figsize=(15, 12))
    plt.savefig('%s/DM_fit_MCMC_pars_%d.png'%(dirfitsG, i), bbox_inches='tight')
    plt.show()
    return y_fit, y_min, y_max, popt

if __name__ == '__main__':
    
    fil= open("%s/cantidades_head.txt"%dirfitsG,"w+")
    fil.write('Nfile \t Name \t Type \t Rc \t errRc \t Mc \t errMc \t M/L \t errM/L \t re \t errre \t rs \t errrs \t r2  \r\n')
    fil.write('Nfile \t Name \t Type \t (kpc) \t (kpc) \t (10^7)M_sun \t (10^7)M_sun  \t M/L \t errM/L \t (kpc) \t (kpc) \t r2 \r\n')
    fil.close()
    fil = open("%s/cantidades.txt"%dirfitsG,"w+")
    PAR = []
       
    ID = 'all'
    for i in range(50,52,1):
        data = np.loadtxt("%s/%d.dat"%(dirdata,i))   
        
        effrad = data_dict['Effective Radius at [3.6]'][i - 1]
        name = data_dict['Name'][i - 1]
        tipo = tipos[data_dict['Type'][i - 1]]
       
        fit, y_min, y_max, popt= fit_mcmc(data, i)
        
        Rc, Mc, re, rs, MtL = popt 
#       fil.write('%d \t %s \t %s \t %.3f \t %.3f \t %.1f \t %.1f \t %.2f \t %.2f  \t %.2f \t %.2f  \t %.2f \t %.2f \t %.2f \r\n'%(i, name, tipo, Rc, perr[0], Mc*1e3, perr[1]*1e3, MtL, perr[2], re, perr[3], rs, perr[4], r2))
        rad, v, v_error, vgas, vdisk, vbul, _, _ = data.T
        rmin, rmax = np.amin(rad), np.amax(rad)
        r99 = 2.3816 *Rc
        r = np.linspace(rmin, r99, 110) 
        parss = r'$R_c = %.1f $kpc, $M_c = %.1f \times 10^7 M_\odot$, $M/L=%.2f$, $r_e=%.2f$kpc '%(Rc, Mc*1e3, MtL, re)
        pts.plotmultiple([rad], [fit], [r'Fit %s'%parss, 'Data'],
                         r'$r$(kpc)', r'$v$(km/s)', 
                         r'%d, %s : %s, $R_{eff}=$ %.2f kpc'%(i, tipo, name, effrad), 
                         '%s/DM_%d_fit.png'%(dirfitsG,i),
                         save = True, data=True, xd = rad, yd = v,
                         err = True, yerr = v_error, show=True,
                         xv = [Rc, effrad, re, rs],xlim = (rmin,rmax),
                         fill_between = True, fbx = rad, fby1 = y_min, 
                         fby2 = y_max)
                         
        PAR.append(popt)
                  
    fil.close()
    PAR = np.array(PAR)
    np.save( '%s/pars.npy'%dirfitsG, PAR)
    
    
#    PARS = np.load('%s/pars.npy'%dirfitsG)
#    MU = []
#    MU2 = []
#    MC = []
#    RC = []
#    RE = []    
#    for i in range(1,9,1):
#        data = np.loadtxt("%s/%d.dat"%(dirdata,i))
#        popt = PARS[i-1,:]
#        rad, v, v_error, vgas, vdisk, vbul, _, _ = data.T
#        Mint = interp1d(rad, M(rad, data, *popt), kind='linear', 
#                        bounds_error=False,
##                        fill_value = (M(rad,data, *popt)[0], M(rad,data, *popt)[-1])
#                        fill_value = 'extrapolate'
#                    )
#        Rc, Mc, re, rs, MtL = popt 
#        cons = Mint(Rc)**2*Mc*1.0e9/1.45e4
#        mu2 = 1./np.sqrt(np.sqrt(cons))
#        mu = 1./(Rc*Gentc2*Mint(Rc))
#        r = np.linspace(0,rad[-1],500)
#        pts.plotmultiple([r], [Mint(r)], [], r'$r$(kpc)', r'$M$','','')        
#        MU.append(mu/1565.5)#10^-23
#        MU2.append(mu2)
#        MC.append(Mc)
#        RC.append(Rc)
#        RE.append(re)
#
#    bins = 20
#    MU = np.array(MU)
#    MU2 = np.array(MU2)    
#    MC = np.array(MC)
#    RC = np.array(RC)
#    RE = np.array(RE)        
#    pts.histo(MU,r'$\hat\mu (\times 10^{-23}) \rm{eV}/c^2$',
##              bins =bins,
#              nom_archivo ='%s/mu.png'%(dirfitsG), fit = False,
##              rang = (0, 1.3e5)
#              )
#    pts.histo(MU2*10.,r'$\hat\mu (\times 10^{-23}) \rm{eV}/c^2$', 
##              bins = bins,
#              nom_archivo ='%s/mu_rel_McMt.png'%(dirfitsG), fit = False,
##              logx=True,
##              rang = (0, 100)
#              )
##############################################################################