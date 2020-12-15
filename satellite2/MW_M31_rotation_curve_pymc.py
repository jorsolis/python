#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:43:11 2020
  
Fit MCMC de la via lactea 

DM          mixSFDM             pars        mu, lambda
Disco       Miyamoto-Nagai      pars        Md, ad, bd
Bulge       Hernquist           pars        Mb, bb
BH          Newtonian           pars        MBH
   
@author: jordi
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pymc
import arviz as az
from def_potenciales import (RC_miyamoto, vnagai, vHernquist, Miyamoto_Nagai_3,
                             RC_exponential, RC_double_exponential,
                             RC_double_exponential_aprox,vBH)
from subprocess import call
from constants_grav import G, c, hc
from pymc_tools import todas_las_trazas

paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'

galaxy = 'Milky Way'

if galaxy == 'Milky Way':
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfits = '%s/Fits/mixSFDM'%dirdata

#    dataGrandMW= np.loadtxt("%s/Table_GrandRC.dat.txt"%dirdata) 
#    rad, v, v_error = dataGrandMW.T
#    rad, v, v_error =rad[:118], v[:118], v_error[:118]

    datalogRCMW= np.loadtxt("%s/LogRC_data.dat.txt"%dirdata) 
    rad,_, v, v_error = datalogRCMW.T    
    rad, v, v_error =  rad[:53], v[:53], v_error[:53]
    
elif galaxy == 'M31':
    dirdata = '/home/jordi/satellite/M31_rotation_curve_data'
    dirfits = '%s/Fits/mixSFDM'%dirdata
    dataM31 = np.loadtxt('%s/M31_rotation_curve_data.txt'%dirdata)
    _, rad, v, v_error = dataM31.T

ncor = 6  ## 4 y 5 m = 1
n = 2
l = 1
m = 0
    
def vdm(x, rlam, mu):
    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))
    xn = x2[:-150]/(mu*rlam)
    vdmn = rlam**2*vdm[:-150]*c
    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
#                  fill_value = "extrapolate" 
                  )
    return ve(x)

def fit_mcmc(rad, v, v_error, MO, nsamples = 1000):
    namep =   [r'$\lambda$', r'$\mu$', r'$M_d$', r'$a_d$', r'$b_d$',
              r'$M_b$',r'$b_b$', r'$M_{BH}$']    
    def model(rad, v, v_error):
        rlam=pymc.Uniform(r'$\lambda$',MO['lamb'][0],MO['lamb'][1])
        mu = pymc.Uniform(r'$\mu$', MO['mu'][0], MO['mu'][1])
        Md = pymc.Uniform(r'$M_d$', MO['Md'][0], MO['Md'][1])
#        ad = pymc.Normal(r'$a_d$', mu=MO['ad'][2], tau = 1./0.4**2)   
        bd = pymc.Normal(r'$b_d$', mu=MO['bd'][2], tau = 1./0.4**2)
        ad = pymc.Uniform(r'$a_d$', MO['ad'][0], MO['ad'][1])   
#        bd = pymc.Uniform(r'$b_d$', MO['bd'][0], MO['bd'][1])
        Mb = pymc.Uniform(r'$M_b$', MO['Mb'][0], MO['Mb'][1])
        bb = pymc.Uniform(r'$b_b$', MO['bb'][0], MO['bb'][1])
        MBH = pymc.Uniform(r'$M_{BH}$', MO['MBH'][0], MO['MBH'][1])

        @pymc.deterministic(plot = False)
        def rot_vel(r = rad, rlam = rlam, mu = mu,
                    Md = Md, ad = ad, bd = bd, ## M es entre 10^10 M_sol
                    Mb = Mb, bb = bb,
                    MBH = MBH):
            vh = vdm(rad, rlam, mu)
            vdisk = vnagai(r, G=G, M = Md, a = ad, b = bd)
            vbul = vHernquist(r, G, Mb, bb)
            velBH = vBH(r, G, MBH)
            return np.sqrt(vh**2 + vdisk**2 + vbul**2 + velBH**2) 
        y = pymc.Normal('y', mu = rot_vel, tau = 1.0/v_error**2, value = v,
                        observed = True)
        return locals()
    MDL = pymc.MCMC(model(rad, v, v_error))
    MDL.sample(nsamples)
    y_min = MDL.stats()['rot_vel']['quantiles'][2.5]
    y_max = MDL.stats()['rot_vel']['quantiles'][97.5]
    y_fit = MDL.stats()['rot_vel']['mean']
    MDL.write_csv("%s/data_%d.csv"%(dirfits, ncor), variables=namep)
    trace= {r'$\lambda$':np.array(MDL.trace(r'$\lambda$')[:]),
               r'$\mu$':np.array(MDL.trace(r'$\mu$')[:]),
               r'$M_d$':np.array(MDL.trace(r'$M_d$')[:]),
               r'$a_d$':np.array(MDL.trace(r'$a_d$')[:]),
               r'$b_d$':np.array(MDL.trace(r'$b_d$')[:]),
               r'$M_b$':np.array(MDL.trace(r'$M_b$')[:]),
               r'$b_b$':np.array(MDL.trace(r'$b_b$')[:]),
               r'$M_{BH}$':np.array(MDL.trace(r'$M_{BH}$')[:])}
    popt = []
    for i in namep:
        print('%s='%i, MDL.stats()['%s'%i]['mean'])
        popt.append(MDL.stats()['%s'%i]['mean']) 
        az.plot_density(trace, var_names=i, shade=0.1, point_estimate='mean',
                        figsize = [4,2], textsize = 14)
        plt.savefig('%s/DM_fit_MCMC_ncor_%d_%s.png'%(dirfits, ncor, i), bbox_inches='tight')   
    todas_las_trazas(trace, namep,'%s/DM_fit_MCMC2_pars_ncor_%d.png'%(dirfits, ncor),
                     point_estimate="mode" )
#    print(pymc.gelman_rubin(trace))#,varnames=namep))
#    pymc.Matplot.plot(MDL)
    return y_fit, y_min, y_max, popt, trace

if __name__ == '__main__':
    nsam = 100000
    MO = {'lamb' : [1e-6, 1e-1,  3.0e-3], 'mu' : [1.565, 156.550,  15.5655],
          'Md' : [1e-5, 400., 5.0], 'ad' : [1., 100., 20.],'bd' : [.1, 3., 1.15],
          'Mb' : [1e-5, 10., 0.3], 'bb' : [1e-3, 1., 1.6],
          'MBH':[1e-4, 1e-3]}

    fit, y_min, y_max, popt, trace= fit_mcmc(rad, v, v_error, MO, nsamples = nsam)
    
    np.savetxt('%s/params.txt'%(dirfits), np.array(popt))
    np.save('%s/params.npy'%(dirfits), np.array(popt))
    rlam, mu, Md, ad, bd, Mb, bb, Mbh = popt
    rmin, rmax = np.amin(rad), np.amax(rad)#20.
    r = np.linspace(rmin, rmax, 1000)  
    vdisk = vnagai(r, G = G, M = Md, a = ad, b = bd)
    vbulge = vHernquist(r, G, Mb, bb)
    veldm = vdm(r, rlam,mu)
    velBH = vBH(r, G, Mbh)
    np.save('%s/fit.npy'%(dirfits), np.array([r, fit, vdisk, vbulge, veldm]))
    r, fit, vdisk, vbulge, veldm = np.load('%s/fit.npy'%(dirfits))
    rlam, mu, Md, ad, bd, Mb, bb, Mbh = np.load('%s/params.npy'%(dirfits))
    import plots_jordi as pts
    pts.plotmultiple([rad, r, r, r, r,r ], [fit, np.sqrt(vdisk**2 + vbulge**2), 
                      vdisk, vbulge, veldm, velBH],
                     [r'Disk+Bulge+mixSFDM', r'Disk+Bulge',
                      r'Disk $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc, $b_d=%.2f$kpc'%(Md,ad,bd),
                      r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $b_b=%.2f$kpc'%(Mb,bb), 
                      r'DM  $\lambda = %.3f$, $\mu = %.1f \times 10^{-25}eV/c^2$ '%(rlam, mu*hc*1e25),
                      r'BH  $M_{BH}=%f \times 10^7 M_\odot$'%(Mbh*10**3),
                      'Data'], r'$r$(kpc)', r'$v$(km/s)', 
                     'mixSFDM $\Phi_{100} + \Phi_{%d%d%d}$'%(n, l, m), 
                     '%s/DM_fit2_ncor_%d.png'%(dirfits,ncor),
#                     ylim = (0,360), 
#                     xlim = (0, rmax),
                     logx =True, logy=True,
                     data=True, xd = rad, yd = v, err = True, yerr = v_error, 
                     fill_between= True, fbx = rad, fby1 = y_min, fby2 = y_max)
    
    pts.plotmultiple([rad, r, r, r, r], [fit, np.sqrt(vdisk**2 + vbulge**2), 
                      vdisk, vbulge, veldm],
                     [r'Disk+Bulge+mixSFDM', r'Disk+Bulge',
                      r'Disk $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc, $b_d=%.2f$kpc'%(Md,ad,bd),
                      r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $b_b=%.2f$kpc'%(Mb,bb), 
                      r'DM  $\lambda = %.3f$, $\mu = %.1f \times 10^{-25}eV/c^2$ '%(rlam, mu*hc*1e25),
                      'Data'], r'$r$(kpc)', r'$v$(km/s)', 
                     'mixSFDM $\Phi_{100} + \Phi_{%d%d%d}$'%(n, l, m), 
                     '%s/DM_fit2_ncor_%d.png'%(dirfits,ncor),
                     ylim = (0,360), 
#                     xlim = (0, rmax),
                     data=True, xd = rad, yd = v, err = True, yerr = v_error, 
                     fill_between= True, fbx = rad, fby1 = y_min, fby2 = y_max)
    pts.residual(v, fit, datalabel=r'$v$(km/s)', lowess=True)
#    print('wolfram script....')
#    call(["wolframscript", "-file", "Plot_galaxy_densities_mathematica2.wls"])    
#    call(["eog", "%s/bulgedens_hern.png"%dirfits])    
#    call(["eog", "%s/discdens.png"%dirfits])     
#    call(["eog", "%s/galaxydens.png"%dirfits])     