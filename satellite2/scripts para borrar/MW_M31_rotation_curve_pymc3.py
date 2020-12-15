#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:43:11 2020

No funciona porque el potencial es interpolado

@author: jordi
"""
import plots_jordi as pts
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from galpy import potential
from scipy import special as sp
import matplotlib.pyplot as plt
from MW_M31_rotation_curve import RC_miyamoto, plot_data, ord_data
import pymc3 as pm3
import arviz as az

paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'
dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
dataMW = np.loadtxt("%s/tab_rcmwall.dat.txt"%dirdata)
errMW = np.array([-dataMW[:,4] + dataMW[:,3], dataMW[:,5] - dataMW[:,3]])
dataMW2 = np.loadtxt("%s/tab_rcmw.dat.txt"%dirdata) 
errMW2 = np.array([-dataMW2[:,4] + dataMW2[:,3], dataMW2[:,5] - dataMW2[:,3]])
modelMW = np.loadtxt("%s/tab_rcmw-model.dat.txt"%dirdata) 
dire = '/home/jordi/satellite'
dirfits = '/home/jordi/satellite/MW_rotation_curve_data/Fits/mixSFDM'
dataM31 = np.loadtxt('%s/M31_rotation_curve_data.txt'%dire)

dataGrandMW= np.loadtxt("%s/Table_GrandRC.dat.txt"%dirdata) 
c = 2.9e5
Gentc2 = 4.799e-7 ##G/c^2 en kpc/(10^10 M_sol)
G = Gentc2*c*c

ncor = 6  ## 4 y 5 m = 1
n = 2
l = 1
m = 0
    
def vdm(x, rlam, mu):
    rlam2 = rlam.value()
    if not isinstance(rlam2, float):
        raise TypeError("Name needs to be a float but got: {}".format(rlam2))
        
    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))
    xn = x2[:-150]/(mu*rlam)
    vdmn = rlam**2*vdm[:-150]*c
#    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
#                  fill_value = "extrapolate" )
    ve = pm3.distributions.continuous.Interpolated(xn, vdmn)
    return ve(x)

def func(x, rlam, mu, Md, ad, bd, Mb, bb):## M es entre 10^10 M_sol
    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))    
    xn = x2[:-150]/(mu*rlam)
    vdmn = rlam**2*vdm[:-150]*c
    vh = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
                  fill_value = "extrapolate" )
    vd = RC_miyamoto(x, G, Md, ad, bd)
    vb = RC_miyamoto(x, G, Mb, 0., bb)
    V = vh(x)**2 + vd**2 + vb**2
#    V = vdm(x,rlam,mu)**2 + vd**2 + vb**2
    return np.sqrt(V)
    
def fit_mcmc(rad, v, v_error, MO, nsamples = 1000):
    namep =   [r'$\lambda$', r'$\mu$', r'$M_d$', r'$a_d$', r'$b_d$',
              r'$M_b$',r'$b_b$']    

    with pm3.Model() as model: 
        rlam=pm3.Uniform(namep[0], MO['lamb'][0], MO['lamb'][1],testval = MO['lamb'][2])
        mu = pm3.Uniform(namep[1], MO['mu'][0], MO['mu'][1],testval = MO['mu'][2])
        Md = pm3.Uniform(namep[2], MO['Md'][0], MO['Md'][1], testval = MO['Md'][2])
        ad = pm3.Uniform(namep[3], MO['ad'][0], MO['ad'][1], testval = MO['ad'][2])
        bd = pm3.Uniform(namep[4], MO['bd'][0], MO['bd'][1], testval = MO['bd'][2])
        Mb = pm3.Uniform(namep[5], MO['Mb'][0], MO['Mb'][1], testval = MO['Mb'][2])
        bb = pm3.Uniform(namep[6], MO['bb'][0], MO['bb'][1], testval = MO['bb'][2])
    
        rot_vel = pm3.Deterministic('rot_vel', 
                                    np.sqrt(vdm(rad, rlam, mu) + 
                                            RC_miyamoto(rad, G, Md, ad, bd)**2 +
                                            RC_miyamoto(rad, G, Mb, 0., bb)**2))

        y = pm3.Normal('y', mu = rot_vel, tau = 1.0/v_error**2, observed = v)
        start = pm3.find_MAP()
        step = pm3.NUTS()
        trace = pm3.sample(nsamples, start=start)
    
    #############         extract and plot results         ########################
    pm3.summary(trace).to_csv('%s/summary.csv'%(dirfits))
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
    plt.savefig('%s/DM_fit_MCMC_pars.png'%(dirfits), bbox_inches='tight')
    plt.show()
    return y_fit, y_min, y_max, popt

if __name__ == '__main__':
    #plot_data(dataMW, dataM31, modelMW) 
#    rad, v, errord = ord_data(dataMW)    
    rad, v, v_error = dataGrandMW.T    
    rad, v, v_error =rad[:118], v[:118], v_error[:118]
    
    MO = {'lamb' : [1e-10, 1e-1,  1e-8], 'mu' : [1.5655, 156.55,  15.655],
          'Md' : [1e-4, 10., 5.0], 'ad' : [10., 50., 20.],'bd' : [1., 5.0, 1.15],
          'Mb' : [1e-4, 1., 0.3], 'bb' : [1e-1, 2, 0.8]}

    fit, y_min, y_max, popt= fit_mcmc(rad, v, v_error, MO, nsamples = 1000)
    rlam, mu, Md, ad, bd, Mb, bb = popt
    rmin, rmax = np.amin(rad), 20.   
    r = np.linspace(rmin, rmax, 1000)   
    pts.plotmultiple([rad, rad, r, r, r],
                     [func(rad, *popt), fit, RC_miyamoto(r, G, Md, ad, bd),
                      RC_miyamoto(r, G, Mb, 0., bb),vdm(r,rlam,mu)],
                     [r'Disk+Bulge+mixSFDM', r'Disk+Bulge+mixSFDM',
                      'Disk $M_d = %.2f$, $a_d = %.2f$, $b_d=%.2f$'%(Md, ad, bd),
                      'Bulge $M_b = %.2f$, $b_b=%.2f$'%(Mb,  bb), 
                      'DM  $\lambda = %.3f$, $\mu = %.1f \mu_{25}$ '%(rlam, mu/15.655),
                      'Data'],
                     r'$r$(kpc)', r'$v$(km/s)', 
                     'mixSFDM $\Phi_{100} + \Phi_{%d%d%d}$'%(n, l, m), 
                     '%s/DM_fit_ncor_%d.png'%(dirfits,ncor), ylim = (0,360),
                     xlim = (0, rmax), save = True, data=True, xd = rad, yd = v,
                     err = True, yerr = v_error, fill_between= True, fbx = rad,
                     fby1 = y_min, fby2 = y_max)
