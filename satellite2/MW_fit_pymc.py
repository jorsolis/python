#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020

Fit MCMC de la via lactea 

DM      Core + NFW              pars      Rc, Mc, re, rs
Disco   Razor exponential       pars      Md, ad
Bulge   Exponential             pars      Mb, bb
BH      Newtonian               pars      MBH

@author: jordi
"""
import pymc
import plots_jordi as pts
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from pymc_tools import todas_las_trazas
from def_potenciales import(vHernquist, vnagai, Miyamoto_Nagai_3,
                            M_Miyamoto2,M_hernquist, exp_bulge,M_exp_bulge,
                            M_exp_disk, M_two_bulge, vBH, RC_exponential,
                            v2_DM, M_CNFW)
from constants_grav import G, Gentc2, mu22, hc
from rotation_curves_SPARC_padilla import fitting
###############################################################################
def M_t(r, Rc, Mc, re, rs, Md, ad, Mb, bb, MBH):
    Mhalo = M_CNFW(r, Rc, Mc, re, rs)
    Mdisk =  M_exp_disk(r, Md, ad)
    Mbulge = M_exp_bulge(r, Mb, bb)
    return Mhalo + Mdisk + Mbulge + MBH

def proba_eq(Rc, Mc, MctRc, MctR99, 
             dirfitsG='' , nomb = 'cantidades_pymc'):
    fil= open("%s/%s.txt"%(dirfitsG, nomb),"w+")

    cons2 = MctR99**2*Mc*1.0e9/1.636e5
    mu3 = 1./np.sqrt(np.sqrt(cons2)) # ecuacion 51
    cons = MctRc**2*Mc*1.0e9/9.355e3
    mu2 = 1./np.sqrt(np.sqrt(cons)) ## ec 51 con Rc en vez de R99
    mu = 1./np.sqrt(Rc*Gentc2*MctRc) ### ecuacion 26

    muDM = Mc/(Rc**2 * np.sqrt(np.pi)**3)

    fil.write('Rc = %.2f pc, \r\n'%(Rc*1e3))
    fil.write('M(Rc) = %.2f x10^{7} M_sun \r\n'%(MctRc*1e3))
    fil.write('mu = %f eV/c^2 \r\n'%(mu*hc))
    fil.write('mu2 = %f x10^{-22} eV/c^2 \r\n' %mu2)
    fil.write('mu3 = %f x10^{-22} eV/c^2 \r\n'%mu3)    
    fil.write('rho_halo(0) = %f M_sun/pc^3 \r\n'%(Mc/(np.sqrt(np.pi)**3*Rc**3)*10.))
    fil.write('muDM = %f 10^{2} M_\odot pc^{-2} \r\n'%(muDM*100))
    fil.close()
    
    print('muDM = ', muDM*100,r'10^{2} M_\odot pc^{-2}')
    print('Rc = %.2f pc'%(Rc*1e3))
    print('M(Rc) = %.2f x10^{7} M_sun'%(MctRc*1e3))
    print('mu =',mu*hc, 'eV/c^2  ecuacion 26')
#    print('mu2 =',mu2, 'x10^{-22} eV/c^2  ecuacion 51 con Rc')
    print('mu3 =',mu3, 'x10^{-22} eV/c^2  ecuacion 51')
    print('rho_halo(0) = ',Mc/(np.sqrt(np.pi)**3*Rc**3)*10., 'M_sun/pc^3')
    
if __name__ == '__main__':  
    np.random.seed(12345)
    galaxy = 'Milky Way'
    nsamples = 1e6
    
    if galaxy == 'Milky Way':
        dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
        dirfitsG = '%s/Fits/Gaussian/'%dirdata
        datalogRCMW= np.loadtxt("%s/LogRC_data.dat.txt"%dirdata) 
        rad,_, v, v_error = datalogRCMW.T
        rad,_, v, v_error = rad[:57],_, v[:57], v_error[:57]
        
    
    
    #MO = {'Rc' : [0.004, 0.007, 6.0e-3], 'Mc' : [0.003, 0.005], 
    #      're' : [0.01, 0.02, 8.5],'rs' : [2., 15., 4.], 
    #      'Md' : [1e-2, 30., 5.0], 'ad' : [0.1, 10., 5.],'bd' : [.8, 3., 1.15],
    #      'Mb': [.1, 1.5, 0.3], 'bb' : [0.05, 1., 1.6],
    #      'BH' : [1e-4, 1e-3]}# el mejor hasta ahora
    #MO = {'Rc' : [0.004, 0.0095, 6.0e-3], 'Mc' : [0.002, 0.008], 
    #      're' : [0.01, 0.025, 8.5],'rs' : [1.5, 15., 4.], 
    #      'Md' : [1e-2, 30., 5.0], 'ad' : [0.1, 10., 5.],'bd' : [.8, 3., 1.15],
    #      'Mb': [.1, 1.5, 0.3], 'bb' : [0.05, 0.5, 1.6],
    #      'BH' : [1e-4, 1e-3]}#
    
    MO = {'Rc' : [0.004, 0.0120, 6.0e-3], 'Mc' : [0.002, 0.008], 
          're' : [0.01, 0.029, 8.5],'rs' : [1.5, 15., 4.], 
          'Md' : [1e-2, 30., 5.0], 'ad' : [0.1, 10., 5.],'bd' : [.8, 3., 1.15],
          'Mb': [.1, 1.5, 0.3], 'bb' : [0.05, 0.5, 1.6],
          'BH' : [1e-4, 1e-3]}#
    def model(rad, v): 
        Rc = pymc.Uniform(r'$R_c$', MO['Rc'][0], MO['Rc'][1])
        Mc = pymc.Uniform(r'$M_c$', MO['Mc'][0], MO['Mc'][1])
        re = pymc.Uniform(r'$r_e$', MO['re'][0], MO['re'][1])
        rs = pymc.Uniform(r'$r_s$', MO['rs'][0], MO['rs'][1])  
        Md = pymc.Uniform(r'$M_d$', MO['Md'][0], MO['Md'][1])
        ad = pymc.Uniform(r'$a_d$', MO['ad'][0], MO['ad'][1])  
    #    bd = pymc.Uniform(r'$b_d$', MO['bd'][0], MO['bd'][1])   
    #    bd = pymc.Normal(r'$b_d$', mu=MO['bd'][2], tau = 1./0.4**2)
        Mb = pymc.Uniform(r'$M_b$', MO['Mb'][0], MO['Mb'][1])
        bb = pymc.Uniform(r'$b_b$', MO['bb'][0], MO['bb'][1])
        MBH = pymc.Uniform(r'$M_{BH}$', MO['BH'][0], MO['BH'][1])
        
        @pymc.deterministic(plot = False)
        def rot_vel(r = rad, Rc = Rc, Mc = Mc, re = re, rs = rs, 
                    Md = Md, ad = ad, Mb = Mb, bb = bb, MBH = MBH):## M es entre 10^10 M_sol
            ve2 = v2_DM(r, G, Rc, Mc, re, rs)
    #        vdisk = RC_miyamoto(r, G, Md, ad, bd)
            vdisk = RC_exponential(r, G, Md, ad)
            vbul = exp_bulge(r, G, Mb, bb)
            velBH = vBH(r, G, MBH)
            return np.sqrt(ve2 + vdisk**2 + vbul**2 +  velBH**2) 
        y = pymc.Normal('y', mu = rot_vel, tau = 1.0/v_error**2, value = v,
                        observed = True)
        return locals()
    #
    MDL = pymc.MCMC(model(rad, v))
    #                , db='pickle', dbname='%s/MW.pickle'%dirfitsG)
    MDL.sample(nsamples, burn = int(0.3*nsamples))
    ##MDL.db.close()
    
    ##db = pymc.database.pickle.load('%s/MW.pickle'%dirfitsG)
    
    
    y_min = MDL.stats()['rot_vel']['quantiles'][2.5]
    y_max = MDL.stats()['rot_vel']['quantiles'][97.5]
    y_fit = MDL.stats()['rot_vel']['mean']
    name =   [r'$R_c$', r'$M_c$', r'$r_e$', r'$r_s$', r'$M_d$', r'$a_d$', #r'$b_d$',
              r'$M_b$',r'$b_b$', r'$M_{BH}$']
    MDL.write_csv("%s/%s.csv"%(dirfitsG, 'data_MCMC'), variables=name)
    popt = []
    poptsd= []
    for i in name:
        print('%s='%i, MDL.stats()['%s'%i]['mean'])
        popt.append(MDL.stats()['%s'%i]['mean'])
        poptsd.append(MDL.stats()['%s'%i]['standard deviation'])
    Rc, Mc, re, rs, Md, ad, Mb, bb, MBH = popt 
    rmin, rmax = 1e-3, np.amax(rad)
    #rmin, rmax = np.amin(rad), np.amax(rad)
    r = np.linspace(rmin, rmax, 100000)
    vdisk = RC_exponential(r, G, Md, ad)
    vbulge = exp_bulge(r, G, Mb, bb)
    
    pts.plotmultiple([rad,r, r, r, r,r], 
                     [y_fit,
                      np.sqrt(vdisk**2 + vbulge**2),
                     vdisk, vbulge, np.sqrt(v2_DM(r, G, Rc, Mc, re, rs)),
                     vBH(r, G, MBH)],
                     [r'Disk+Bulge+BH+SFDM',
                      r'Disk+Bulge',
                      r'Disk $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad),
                      r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $b_b=%.2f$pc'%(Mb,bb*1e3), 
                      r'DM  $R_c = %.3f$ pc, $M_c = %.1f \times 10^{7} M_\odot$, $r_e=%.2f$pc, $r_s=%.1f$kpc'%(Rc*1e3, Mc*1e3, re*1e3, rs),
                      r'BH, $M_{BH}=%.4f\times 10^{6} M_\odot$'%(MBH*1.0e4),'Observed'],
                     r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                     '%sDM_fit_MCMC_log.png'%(dirfitsG),
                     ylim = (20,500),
                     xlim = (rmin,rmax), data = True, xd = rad, yd = v, err = True,
                     yerr = v_error, fill_between = True, logx = True,logy = True,
                     fbx = rad, fby1 = y_min, fby2 = y_max, xv=[Rc, re, rs])
    pts.residual(v, y_fit, lowess=False)
    
    r = np.linspace(rmin, rmax, 1000)
    vdisk = RC_exponential(r, G, Md, ad)
    vbulge = exp_bulge(r, G, Mb, bb)
    
    pts.plotmultiple([rad,r, r, r, r,r], [y_fit, np.sqrt(vdisk**2 + vbulge**2),
                     vdisk, vbulge, np.sqrt(v2_DM(r, G, Rc, Mc, re, rs)), 
                     vBH(r, G, MBH)],
                     [r'Disk+Bulge+BH+SFDM',
                      r'Disk+Bulge',
                      r'Disk $M_d = %.1f\times 10^{10} M_\odot$, $a_d = %.2f $ kpc'%(Md,ad),
                      r'Bulge $M_b = %.1f\times 10^{10} M_\odot$, $b_b=%.2f$pc'%(Mb,bb*1e3), 
                      r'DM  $R_c = %.3f$ pc, $M_c = %.1f \times 10^{7} M_\odot$, $r_e=%.2f$pc, $r_s=%.1f$kpc'%(Rc*1e3, Mc*1e3, re*1e3, rs),
                      r'BH, $M_{BH}=%.4f\times 10^{6} M_\odot$'%(MBH*1.0e4),'Observed'],
                     r'$r$(kpc)', r'$v$(km/s)', 'Miky Way',
                     '%sDM_fit_MCMC.png'%(dirfitsG),ylim = (0,300.),
                     xlim = (rmin,rmax), data = True, xd = rad, yd = v, err = True,
                     yerr = v_error, fill_between = True,
                     fbx = rad, fby1 = y_min, fby2 = y_max, xv=[Rc, re])
    pts.residual(v, y_fit, lowess=False)
 
    MctRc = M_t(Rc, *popt)
    R99 = 2.38167*Rc
    MctR99= M_t(R99, *popt)
    proba_eq(Rc, Mc, MctRc, MctR99, dirfitsG= dirfitsG, nomb = 'cantidades_pymc')
           
    ##pymc.Matplot.plot(MDL, path = dirfitsG,)
    ##print(MDL.summary())
    ##print(MDL.trace(r'$R_c$')[:])
    ##az.plot_kde(MDL.trace(r'$R_c$')[:], values2=MDL.trace(r'$M_c$')[:], contour=False)
    ##plt.show()
    #
    traces= {r'$R_c$':np.array(MDL.trace(r'$R_c$')[:]),
             r'$M_c$':np.array(MDL.trace(r'$M_c$')[:]),
               r'$r_e$':np.array(MDL.trace(r'$r_e$')[:]),
               r'$r_s$':np.array(MDL.trace(r'$r_s$')[:]),
               r'$M_d$':np.array(MDL.trace(r'$M_d$')[:]),
               r'$a_d$':np.array(MDL.trace(r'$a_d$')[:]),
    #           r'$b_d$':np.array(MDL.trace(r'$b_d$')[:]),
               r'$M_b$':np.array(MDL.trace(r'$M_b$')[:]),
               r'$b_b$':np.array(MDL.trace(r'$b_b$')[:]),           
               r'$M_{BH}$':np.array(MDL.trace(r'$M_{BH}$')[:])}
    
    todas_las_trazas(traces, name, '%sDM_fit_MCMC_pars.png'%(dirfitsG),
                     point_estimate="mode")
    
    
    #az.plot_density(traces, var_names=name, shade=0.1, point_estimate='mean',
    #                textsize=10. , figsize=(7, 5))
    #plt.savefig('%sDM_fit_MCMC2.png'%(dirfitsG), bbox_inches='tight')
    #for i in range(0, 11):
    #    az.plot_density(traces, var_names=name[i], shade=0.1, point_estimate='mean' )
    #    plt.savefig('%sDM_fit_MCMC2_%s.png'%(dirfitsG, name[i]), bbox_inches='tight')