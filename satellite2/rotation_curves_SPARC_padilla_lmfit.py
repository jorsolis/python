#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:20:13 2020

Fit MINIMOS CUADRADOS de SPARC

DM        core + NFW      pars       Rc, Mc, re, rs
baryons   Obervational    pars       M/L
 
@author: jordi
"""
import numpy as np
import plots_jordi as pts
from scipy.interpolate import interp1d
from SPARC_desc import tipos, data_dict
from constants_grav import G, Gentc2, hc
from def_potenciales import f, v2_DM, M_CNFW
from pymc_tools import todas_las_trazas
#from rotation_curves_SPARC_padilla import v_bar
import lmfit as lm
sat = '/home/jordi/satellite'
dirdata = '/home/jordi/SPARC'
dirfitsG = '/home/jordi/SPARC/Fits/Gaussian'
a= ['Distance','Mean error on D','Distance Method',
    'Inclination','Mean error on Inc','Total Luminosity at [3.6]',
    'Mean error on L[3.6]','Effective Radius at [3.6]',
    'Effective Surface Brightness at [3.6]','Disk Scale Length at [3.6]',
    'Disk Central Surface Brightness at [3.6]','Total HI mass',
    'HI radius at 1 Msun over pc2','Asymptotically Flat Rotation Velocity',
    'Mean error on Vflat','Quality Flag']
def plot_trace(res):
    nam = [r'$R_c$',  r'$M_c$', r'$r_e$', r'$r_s$',  r'$\Upsilon$']
    traces= {r'$R_c$':np.array(res.flatchain['Rc']),
             r'$M_c$':np.array(res.flatchain['Mc']),
               r'$r_e$':np.array(res.flatchain['re']),
               r'$r_s$':np.array(res.flatchain['rs']),
               r'$\Upsilon$':np.array(res.flatchain['ML'])} 
    todas_las_trazas(traces, nam, '%sDM_fit_MCMC_emcee_pars_%d.png'%(dirfitsG, i),
                     point_estimate="mode")
            
def cantidades(i,rad, popt, Bulge = True):
    Rc, Mc, MtL, re, rs = popt
    if Bulge == True:
        Mint = interp1d(rad, M_bar(rad,i, *popt), kind='linear', 
                        bounds_error=False, fill_value = 'extrapolate')
    else:
        Mint = interp1d(rad, M_bar2(rad,i, *popt), kind='linear', 
                        bounds_error=False, fill_value = 'extrapolate')        
    rmax = np.amax(rad)
    if Mint(Rc)>0:
        Mtot = Mint(Rc) + M_CNFW(Rc, Rc, Mc, re, rs)
    else:
        Mtot = M_CNFW(Rc, Rc, Mc, re, rs)

    mu = 1./np.sqrt(Rc*Gentc2*Mtot)### ecuacion 26
    cons = Mtot**2*Mc*1.0e9/9.41044e3## ec 51
    mu2 = 1./np.sqrt(np.sqrt(cons)) ## ec 51
    R99 = 2.38167*Rc
    if Mint(R99)>0:
        Mt99 = Mint(R99) + M_CNFW(R99, Rc, Mc, re, rs)
    else:
        Mt99 = M_CNFW(R99, Rc, Mc, re, rs)
    cons2 = Mt99**2*Mc*1.0e9/1.636e5# ecuacion 50
    mu3 = 1./np.sqrt(np.sqrt(cons2)) # ecuacion 50
    muDM = Mc/(Rc**2 *np.sqrt(np.pi)**3)   
    Mtrf = M_CNFW(rmax, Rc, Mc, re, rs)/1e2
    Mt997 = Mt99*1e3
    mu4 = 140 * Mtrf**(1./3.)/Mt997 #ecuacion 42
    rc = 2.38167*Rc*1e3/3.77#en parcecs
    mu5 = 521.*Mtrf**(1./3.)/rc #ecuacion 43
#    print(mu3)
    return mu, mu2, mu3, mu4, mu5, muDM
      
def filtro(r0, tho, phio, value, dejamos):
    mask = []
    if dejamos == 'menores a':
        for i in range(0, np.shape(r0)[0]):
            if r0[i]>value:
                mask.append(1) ## quitamos estos
            else:
                mask.append(0)## dejamos estos
    elif dejamos == 'mayores a':
        for i in range(0, np.shape(r0)[0]):
            if r0[i]<value:
                mask.append(1) ## quitamos estos
            else:
                mask.append(0)## dejamos estos  
    elif dejamos == 'intervalo':
        for i in range(0, np.shape(r0)[0]):
            if r0[i]<value[0]:
                mask.append(1) ## quitamos estos
            elif r0[i]>value[1]:
                mask.append(1) ## quitamos estos           
            else:
                mask.append(0)## dejamos estos  
    nr0 = np.ma.masked_array(r0, mask = mask, fill_value=np.nan)
    ntho = np.ma.masked_array(tho, mask = mask, fill_value=np.nan)
    nphio = np.ma.masked_array(phio, mask = mask, fill_value=np.nan)
    return nr0, ntho, nphio

def M_bar(r,i, Rc, Mc, ML, re, rs):
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    vbul = data[:,5]     
    return vgas**2*r/G + ML*vdisk**2*r/G + 1.4*ML*vbul**2*r/G

def v_bar(r, Rc, Mc, ML, re, rs):
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    vbul = data[:,5]     
    return np.sqrt(vgas**2 + ML*vdisk**2 + 1.4*ML*vbul**2)

def v_bar2(r, Rc, Mc, ML, re, rs):
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]   
    return np.sqrt(vgas**2 + ML*vdisk**2)

def M_C(r, Rc, Mc, ML, re, rs):
    zeroval = 1.
    Mh = f(r, Mc, Rc)*np.heaviside(re - r, zeroval) + f(re, Mc, Rc)*np.heaviside(r - re, zeroval) 
    return Mh

def M_bar2(r,i, Rc, Mc, ML, re, rs):
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]    
    return vgas**2*r/G + ML*vdisk**2*r/G
def func_CNFW(r, Rc, Mc, ML, re, rs):
    ve2 = v2_DM(r,G, Rc, Mc, re, rs)
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    vbul = data[:,5] 
    return np.sqrt(ve2 + vgas**2 + ML*vdisk**2 + 1.4*ML*vbul**2) 

def residual(params, r, data, eps_data, i):
    Rc = params['Rc']
    Mc = params['Mc']
    re = params['re']
    rs = params['rs']
    ML = params['ML']
    ve2 = v2_DM(r, G, Rc, Mc, re, rs)
    dat = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vdisk = dat[:,4]
    vgas = dat[:,3]
    vbul = dat[:,5] 
    model =  np.sqrt(ve2 + vgas**2 + ML*vdisk**2 + 1.4*ML*vbul**2) 
    return (data-model) / eps_data
def func_CNFW2(r, Rc, Mc, ML, re, rs):
    ve2 = v2_DM(r,G, Rc, Mc, re, rs)
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    return np.sqrt(ve2 + vgas**2 + ML*vdisk**2) 
def residual2(params, r, data, eps_data, i):
    Rc = params['Rc']
    Mc = params['Mc']
    re = params['re']
    rs = params['rs']
    ML = params['ML']
    ve2 = v2_DM(r, G, Rc, Mc, re, rs)
    dat = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vdisk = dat[:,4]
    vgas = dat[:,3]
    model =  np.sqrt(ve2 + vgas**2 + ML*vdisk**2) 
    return (data-model) / eps_data

def fitting(residual, params, args, MCMC = False, print_res = False):
    out = lm.minimize(residual, params, args=args,
                      method='nelder')
    if print_res==True:
        print(lm.printfuncs.report_fit(out.params, min_correl=0.5))
        print(out.params.pretty_print())
    Rc, Mc, re, rs, MtL = list(out.params.valuesdict().values())
    popt = [Rc, Mc, MtL, re, rs]
    Chi2red = out.redchi  
    if MCMC==True:
        res = lm.minimize(residual, args=args, method='emcee',
                          nan_policy='omit', burn=int(.3*nsamples),
                          nwalkers = 500,
                          steps=nsamples, #thin=20,
                          params=out.params, 
                          is_weighted=True)
        plot_trace(res)      
        Rc, Mc, re, rs, MtL = list(res.params.valuesdict().values())
        popt = [Rc, Mc, MtL, re, rs]
        Chi2red = res.redchi
    return popt, Chi2red

if __name__ == '__main__':   
    nsamples = 1000
    ID = '145galaxies'
    withbulge= (31, 40, 46, 47, 50, 67, 73, 74, 77, 80, 81, 86, 88, 90, 92, 93, 95, 107, 108, 109, 110, 111, 112, 113, 120, 132, 135, 136, 160, 163, 169)
    bulgeless = (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18, 20, 21, 22, 23, 24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 45, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 70, 71, 72, 75, 79, 82, 83, 84, 85, 87, 89, 94, 97, 98, 100, 103, 105, 114,  116, 117, 118, 119, 122, 123, 124, 125, 126, 127, 128, 130, 131, 134, 137, 138, 140, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 159, 161, 162, 165, 166, 167, 168, 170, 171, 172, 174, 175)

#    MO = {'Rc' : [0.003, 0.020, 76e-4], 'Mc' : [0.002, 0.016, 46e-4], 
#          're' : [0.01, 0.05, 0.020],'rs' : [1., 15., 5.630]}# el mejor hasta ahora  

#    MO = {'Rc' : [0.001, 0.036, 0.015], 'Mc' : [0.001, 0.0328, 0.026], 
#          're' : [0.005, 0.10, 0.05],'rs' : [1., 30., 12.]} 
#     para Type 0, 1, 11

    MO = {'Rc' : [0.001, 0.036, 0.015], 'Mc' : [0.0001, 0.050, 0.026], 
          're' : [0.005, 0.10, 0.05],'rs' : [1., 60., 12.]} 
    # para Type 2, 3, 4,5, 6   
    
    fil= open("%s/Fits/Gaussian/cantidades_head.txt"%dirdata,"w+")
    fil.write('Nfile \t Name \t Type \t Rc \t errRc \t Mc \t errMc \t M/L \t errM/L \t re \t errre \t rs \t errrs \t r2  \r\n')
    fil.write('Nfile \t Name \t Type \t (kpc) \t (kpc) \t (10^7)M_sun \t (10^7)M_sun  \t M/L \t errM/L \t (kpc) \t (kpc) \t r2 \r\n')
    fil.close()         
    MU = []; MU2 = []; MU3 = []; MU4 = []; MU5 = []; MUDM = []
    MC = []; RC = []; RE = []; RS = [] ; MTL = []; X2 = []     
    params = lm.Parameters()
    params.add('Rc', value = MO['Rc'][2], min = MO['Rc'][0], max = MO['Rc'][1])
    params.add('Mc', value = MO['Mc'][2], min = MO['Mc'][0], max = MO['Mc'][1])
    params.add('re', value = MO['re'][2], min = MO['re'][0], max = MO['re'][1])
    params.add('rs', value = MO['rs'][2], min = MO['rs'][0], max = MO['rs'][1])
    params.add('ML', value = 1.000, min = 0.1, max = 5.0)
    fil = open("%s/Fits/Gaussian/cantidades.txt"%dirdata,"w+")
    
#    for i in range(110, 111, 2):
    for i in (withbulge + bulgeless):  
        dat = np.loadtxt("%s/%d.dat"%(dirdata,i))  
        rad, vobs, err, vgas, vdisk, vbul, _, _ = dat.T
        name = data_dict['Name'][i - 1]
        tipo = tipos[data_dict['Type'][i - 1]]          
        rmin, rmax = np.amin(rad), np.amax(rad)
#        estosno = (2.,3,4,5,6)
        estosno = (0,1,2, 5,6,11, 7, 8,9,10)
        if data_dict['Type'][i - 1] in estosno:
            pass
        else:
            if i in withbulge:
                popt, Chi2red = fitting(residual, params, (rad, vobs, err, i),
    #                                        MCMC = True,
                                        )
                fit = func_CNFW(rad, *popt)
                v_baryons = v_bar(rad, *popt)
                mu, mu2,mu3,mu4,mu5, muDM = cantidades(i,rad, popt, Bulge=True)
                labbar = r'Disc + Bulge + Gas $\Upsilon =%.2f$'%(popt[2])
            if i in bulgeless:
                popt, Chi2red = fitting(residual2, params, (rad, vobs, err, i),
                                        MCMC = False)
                fit = func_CNFW2(rad, *popt)              
                v_baryons = v_bar2(rad, *popt)              
                mu, mu2,mu3,mu4,mu5, muDM = cantidades(i,rad, popt, Bulge=False)
                labbar = r'Disc + Gas $\Upsilon=%.2f$'%(popt[2])
                    
            Rc, Mc, MtL, re, rs = popt    
            X2.append(Chi2red)
            fil.write('%d \t %s \t %s \t %.3f\t %.1f \t %.2f  \t %.2f \t %.2f \r\n'%(i, name, tipo, Rc, Mc*1e3, MtL, re, rs))
            parss = r'$R_c = %.1f $pc, $M_c = %.1f \times 10^7 M_\odot$, $r_s=%.2f$, $r_e=%.2f$kpc '%(Rc*1e3, Mc*1e3,rs, re)                                 
            pts.plotmultiple([rad, rad,rad], [fit, v_baryons,
                             np.sqrt(v2_DM(rad,G, Rc, Mc, re, rs))],
                             [r'fit: %f'%Chi2red, labbar,
                              'DM %s'%parss,'data'],
                             r'$r$(kpc)',r'$v$(km/s)',tipo + name,
                             '%s/%d.png'%(dirfitsG,i), data=True, 
                             xd=rad, yd=vobs, err=True, yerr=err, xv=[Rc, re, rs])
            MU.append(mu*hc); MU2.append(mu2); MU3.append(mu3)
            MU4.append(mu4); MU5.append(mu5); MC.append(Mc); RC.append(Rc)
            RE.append(re); RS.append(rs); MUDM.append(muDM); MTL.append(MtL)    
            if Chi2red>20:
                print(i,'CHi =', Chi2red, data_dict['Type'][i - 1])
    fil.close()
      
    MU = np.array(MU); MU2 = np.array(MU2); MU3 = np.array(MU3)
    MU4 = np.array(MU4); MU5 = np.array(MU5); MUDM = np.array(MUDM)
    MC = np.array(MC); RC = np.array(RC); RE = np.array(RE)
    RS = np.array(RS); X2 = np.array(X2); MTL= np.array(MTL)
    bins = 50
#  
    chimax = 20.
    _, MC, RC = filtro(X2, MC, RC, chimax, 'menores a')
    _, RE, RS = filtro(X2, RE, RS, chimax, 'menores a')
    _, MTL, MU = filtro(X2, MTL, MU, chimax, 'menores a')
    _, MU2, MU3= filtro(X2, MU2, MU3, chimax, 'menores a')
    _, _, MUDM= filtro(X2, MUDM, MUDM, chimax, 'menores a')
    X2, MU4, MU5 = filtro(X2, MU4, MU5, chimax, 'menores a')
#
#    chimax = 70.
#    _, MC, RC = filtro(RS, MC, RC, chimax, 'menores a')
#    _, RE, X2 = filtro(RS, RE, X2, chimax, 'menores a')
#    _, MTL, MU = filtro(RS, MTL, MU, chimax, 'menores a')
#    _, MU2, MU3= filtro(RS, MU2, MU3, chimax, 'menores a')
#    _, _, MUDM= filtro(RS, MUDM, MUDM, chimax, 'menores a')
#    RS, MU4, MU5 = filtro(RS, MU4, MU5, chimax, 'menores a')
    
    pts.histo(MC,r'$M_c (\times 10^{10} M_\odot)$', bins =bins,
              normalized = False, #rang = (.0001, .0003),
              nom_archivo ='%s/Mc_%s.png'%(dirfitsG, ID))
    pts.histo(RC,r'$R_c$(kpc)', bins =bins, normalized=False,
              nom_archivo ='%s/Rc_%s.png'%(dirfitsG, ID))
    pts.histo(RE,r'$r_e (kpc)$', bins =bins,normalized=False,
              nom_archivo ='%s/Re_%s.png'%(dirfitsG, ID))
    pts.histo(RS,r'$r_s (kpc)$', bins =bins,normalized=False,
              nom_archivo ='%s/Rs_%s.png'%(dirfitsG, ID))     
    pts.histo(MTL,r'$\gamma$', bins =bins,normalized=False,
              nom_archivo ='%s/MtL_%s.png'%(dirfitsG, ID))      
    pts.histo(X2,r'$\chi^2_{red}$', bins =bins,normalized = False,
              nom_archivo ='%s/chi_%s.png'%(dirfitsG, ID),
#              rang = (0, 7)
              ) 
#    pts.histo(MU,r'$m (\rm{eV}/c^2)$', bins =bins,normalized=False,
#              nom_archivo ='%s/mu_%s.png'%(dirfitsG, ID), #fit = True,
#              title='ecuacion 26',
##              logx=True,
##              rang = (0, 3e-20)
#              )
##    pts.histo(MU2,r'$m (\times 10^{-22} \rm{eV}/c^2)$', 
##              bins = bins,title='ecuacion 51',normalized=False,
##              nom_archivo ='%s/mu_rel_McMt_%s.png'%(dirfitsG, ID), 
###              logx=True,
###              rang = (10, 100)
##              )
    pts.histo(MU3,r'$m (\times 10^{-22} \rm{eV}/c^2)$', 
              bins = bins,title='ecuacion 50',normalized=False,
              nom_archivo ='%s/mu_3_%s.png'%(dirfitsG, ID), 
#              logx=True,
#              rang = (0, 100)
              )
##    pts.histo(MU4,r'$m (\times 10^{-22} \rm{eV}/c^2)$', 
#              bins = bins,title='ecuacion 42',normalized=False,
#              nom_archivo ='%s/mu_4_%s.png'%(dirfitsG, ID), 
##              logx=True,
##              rang = (10, 100)
#              )
#    pts.histo(MU5,r'$m (\times 10^{-22} \rm{eV}/c^2)$',
#              bins = bins,normalized = False,title='ecuacion 43',
#              nom_archivo ='%s/mu_5_%s.png'%(dirfitsG, ID), 
##              logx=True,
##              rang = (10, 100)
#              )    
    pts.histo(MUDM,r'$\mu_\psi (10^{4} M_\odot pc^{-2})$', bins =bins, 
              normalized = False,
#              logx = True,
              nom_archivo ='%s/mudm_%s.png'%(dirfitsG, ID))  
