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
import scipy.stats as ss
from SPARC_desc import tipos, data_dict
from constants_grav import G, Gentc2, hc
from def_potenciales import f, v2_DM, M_CNFW
from scipy.optimize import curve_fit
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
def fitting(func, rad, v, bounds, paramname, error = False, err = [],
            printing = True, loss = 'linear'):
    if error == False:
        popt, pcov = curve_fit(func, rad, v, bounds = (bounds[0], bounds[1]),
                               loss = loss)
    else:
        popt, pcov = curve_fit(func, rad, v,
                               bounds = (bounds[0], bounds[1]),
                               sigma = err,
                               loss = loss, 
                               absolute_sigma = True
                               )       
    perr = np.sqrt(np.diag(pcov)) #one standard deviation errors on the parameters
    nstd = 5. # to draw 1-sigma intervals
    popt_up = popt + nstd * perr  ##Fitting parameters at 5 sigma
    popt_dw = popt - nstd * perr  ##Fitting parameters at 5 sigma
    fit = func(rad, *popt)
    ss_res = np.sum((v - fit) ** 2) # residual sum of squares
    ss_tot = np.sum((v - np.mean(v)) ** 2)  # total sum of squares
    r2 = 1. - (ss_res / ss_tot) # r-squared

    if error ==True:
        sigma = err
        m = len(popt)#number of params
#        Chi2 =  np.sum(((v - fit)/sigma)**2)
        Chi2 =ss.chisquare(v, f_exp=fit)[0]
        Chi2red = Chi2/(np.shape(rad)[0]-m)
    if printing ==True:
        print('r-squared=', r2)   
        print('Chi2=', Chi2)
        print('Chi2red=', Chi2red)
        print('fit parameters and 1-sigma error')
        for j in range(len(popt)):
            print(paramname[j],'=', str(popt[j])+' +- '+str(perr[j])) 
    return popt, popt_up, popt_dw, r2, perr, Chi2, Chi2red

def M_bar(r, Rc, Mc, ML, re, rs):
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

def func_CNFW(r, Rc, Mc, ML, re, rs):
    ve2 = v2_DM(r,G, Rc, Mc, re, rs)
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    vbul = data[:,5] 
    return np.sqrt(ve2 + vgas**2 + ML*vdisk**2 + 1.4*ML*vbul**2) 

def M_bar2(r, Rc, Mc, ML, re, rs):
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]    
    return vgas**2*r/G + ML*vdisk**2*r/G

def func_CNFW2(r, Rc, Mc, ML, re, rs):
    ve2 = v2_DM(r,G, Rc, Mc, re, rs)
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    return np.sqrt(ve2 + vgas**2 + ML*vdisk**2) 

if __name__ == '__main__':   
 
    MO = {'Rc' : [0.004, 0.007], 'Mc' : [0.003, 0.005], 
          're' : [0.01, 0.02],'rs' : [2., 15.], 'BH' : [1e-4, 1e-3]}# el mejor hasta ahora
    
    bound = [[MO['Rc'][0], MO['Mc'][0], 0.1, MO['re'][0], MO['rs'][0]],
              [MO['Rc'][1], MO['Mc'][1], 5.0, MO['re'][1], MO['rs'][1]]]
    nameb = ['Rc',  'Mc',  'M/L', 're',  'rs']
##      
#    fil= open("%s/Fits/Gaussian/cantidades_head.txt"%dirdata,"w+")
#    fil.write('Nfile \t Name \t Type \t Rc \t errRc \t Mc \t errMc \t M/L \t errM/L \t re \t errre \t rs \t errrs \t r2  \r\n')
#    fil.write('Nfile \t Name \t Type \t (kpc) \t (kpc) \t (10^7)M_sun \t (10^7)M_sun  \t M/L \t errM/L \t (kpc) \t (kpc) \t r2 \r\n')
#    fil.close()
#    fil = open("%s/Fits/Gaussian/cantidades.txt"%dirdata,"w+")
#    
#    MU = []
#    MU2 = []
#    MU3 = []
#    MU4 = []
#    MU5 = []
#    MUDM = []
#    MC = []
#    RC = []
#    RE = []
#    RS = [] 
#    MTL = []
#    R2 = np.zeros(175)
#    INDEX =[]
#    INDEX2 =[]
#    X2 = []
#    
#    ID = '145galaxies'
#    withbulge= (31, 40, 46, 47, 50, 67, 73, 74, 77, 80, 81, 86, 88, 90, 92, 93, 95, 107, 108, 109, 110, 111, 112, 113, 120, 132, 135, 136, 160, 163, 169)
#    bulgeless = (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18, 20, 21, 22, 23, 24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 45, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 70, 71, 72, 75, 79, 82, 83, 84, 85, 87, 89, 94, 97, 98, 100, 103, 105, 114,  116, 117, 118, 119, 122, 123, 124, 125, 126, 127, 128, 130, 131, 134, 137, 138, 140, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158, 159, 161, 162, 165, 166, 167, 168, 170, 171, 172, 174, 175)
#
#    for i in withbulge:  
#        data = np.loadtxt("%s/%d.dat"%(dirdata,i))  
#        rad, vobs, err, vgas, vdisk, vbul, _, _ = data.T
#        name = data_dict['Name'][i - 1]
#        tipo = tipos[data_dict['Type'][i - 1]]          
#        try:    
#            popt, popt_up, popt_dw, r2, perr, Chi2, Chi2red = fitting(func_CNFW, rad, vobs,
#                                                           bound,
#                                           nameb,error = True, err = err,
#                                           printing = False, loss = 'cauchy')       
#            fit = func_CNFW(rad, *popt)
#            X2.append(Chi2red)
#
#            Rc, Mc, MtL, re, rs = popt        
#            fil.write('%d \t %s \t %s \t %.3f \t %.3f \t %.1f \t %.1f \t %.2f \t %.2f  \t %.2f \t %.2f  \t %.2f \t %.2f \t %.2f \r\n'%(i, name, tipo, Rc, perr[0], Mc*1e3, perr[1]*1e3, MtL, perr[2], re, perr[3], rs, perr[4], r2))
#            R2[i - 1]= r2
#            if r2>0:
#                rmin, rmax = np.amin(rad), np.amax(rad)
#                r99 = 2.3816 *Rc
#                r = np.linspace(rmin, r99, 110) 
#                parss = r'$R_c = %.1f $pc, $M_c = %.1f \times 10^7 M_\odot$, $r_s=%.2f$, $r_e=%.2f$kpc '%(Rc*1e3, Mc*1e3,rs, re)                  
#                parsbar = r'$\gamma=%.2f$'%(MtL) 
##                pts.plotmultiple([rad, rad,rad], [fit, v_bar(rad, *popt),
##                                 np.sqrt(v2_DM(rad,G, Rc, Mc, re, rs))],
##                                 [r'fit: $R^2 = %.2f$'%r2,'Disc + Bulge %s'%parsbar,
##                                  'DM %s'%parss,'data'],
##                                 r'$r$(kpc)',r'$v$(km/s)',tipo + name,
##                                 '%s/%d.png'%(dirfitsG,i), data=True, 
##                                 xd=rad, yd=vobs, err=True, yerr=err)
#                Mint = interp1d(rad, M_bar(rad, *popt), kind='linear', 
#                                bounds_error=False, fill_value = 'extrapolate')
#                
#                if Mint(Rc)>0:
#                    Mtot = Mint(Rc) + M_CNFW(Rc, Rc, Mc, re, rs)
#                else:
#                    Mtot = M_CNFW(Rc, Rc, Mc, re, rs)
##                cons = Mtot**2*Mc*1.0e9/1.45e4
#                cons = Mtot**2*Mc*1.0e9/9.41044e3
#                mu = 1./np.sqrt(Rc*Gentc2*Mtot)### ecuacion 26
#                mu2 = 1./np.sqrt(np.sqrt(cons)) ## ec 51
#                R99 = 2.38167*Rc
#                if Mint(R99)>0:
#                    Mt99 = Mint(R99) + M_CNFW(R99, Rc, Mc, re, rs)
#                else:
#                    Mt99 = M_CNFW(R99, Rc, Mc, re, rs)
#                cons2 = Mt99**2*Mc*1.0e9/2.536e5
#                mu3 = 1./np.sqrt(np.sqrt(cons2)) # ecuacion 50
#                
##                muDM = Mc/(Rc**2 * np.sqrt(np.pi)**3) ##10^10M_sun/kpc^2 = 10^4M_sun/pc^2
#                H = 2.38167/3.77
#                muDM = H*Mc/(Rc**2 *np.sqrt(np.pi)**3)
#                
#                Mtrf = M_CNFW(rmax, Rc, Mc, re, rs)/1e2
#                Mt997 = Mt99*1e3
#                mu4 = 140 * Mtrf**(1./3.)/Mt997 #ecuacion 42
#                rc = 2.38167*Rc*1e3/3.77#en parcecs
#                mu5 = 521.*Mtrf**(1./3.)/rc #ecuacion 43
#                MU.append(mu*hc)
#                MU2.append(mu2)
#                MU3.append(mu3)
#                MU4.append(mu4)
#                MU5.append(mu5)
#                MC.append(Mc)
#                RC.append(Rc)
#                RE.append(re)
#                RS.append(rs)
#                MUDM.append(muDM)
#                MTL.append(MtL)
#    
#                if mu2<10.:
#                    INDEX.append(i)
#                else:
#                    INDEX2.append(i)
#    #                MU3.append(mu3)
#        except RuntimeError:
#            print(i,'ups, minimizacion falló')
#            print('-------------------------------------------------------')
#            
#    for i in bulgeless:  
#        data = np.loadtxt("%s/%d.dat"%(dirdata,i))   
#        rad, vobs, err, vgas, vdisk, vbul, _, _ = data.T
#        effrad = data_dict['Effective Radius at [3.6]'][i - 1]
#        name = data_dict['Name'][i - 1]
#        tipo = tipos[data_dict['Type'][i - 1]]
#        try:    
#            popt, popt_up, popt_dw, r2, perr, Chi2, Chi2red = fitting(func_CNFW2, rad, vobs, bound,
#                                           nameb,error = True, err = err,
#                                           printing = False, loss = 'cauchy' ) 
#            X2.append(Chi2red)
#
#            fit = func_CNFW2(rad, *popt)        
#            Rc, Mc, MtL, re, rs = popt        
#            fil.write('%d \t %s \t %s \t %.3f \t %.3f \t %.1f \t %.1f \t %.2f \t %.2f  \t %.2f \t %.2f  \t %.2f \t %.2f \t %.2f \r\n'%(i, name, tipo, Rc, perr[0], Mc*1e3, perr[1]*1e3, MtL, perr[2], re, perr[3], rs, perr[4], r2))
#            R2[i - 1]= r2
#            if r2>0:
#                rmin, rmax = np.amin(rad), np.amax(rad)
#                r99 = 2.3816 *Rc
#                parss = r'$R_c = %.1f $pc, $M_c = %.1f \times 10^7 M_\odot$, $r_s=%.2f$, $r_e=%.2f$kpc '%(Rc*1e3, Mc*1e3, rs, re)
#                parsbar = r'$\gamma=%.2f$'%(MtL) 
##                pts.plotmultiple([rad,rad,rad],
##                                 [fit, v_bar2(rad, *popt),np.sqrt(v2_DM(rad,G, Rc, Mc, re, rs))],
##                                 [r'fit: $R^2 = %.2f$'%r2,'Disc %s'%parsbar,
##                                  'DM %s'%parss, 'data'],
##                                 r'$r$(kpc)',r'$v$(km/s)',tipo + name,
##                                 '%s/%d.png'%(dirfitsG,i),data=True, 
##                                 xd=rad, yd=vobs, err=True, yerr=err)                  
#                Mint2 = interp1d(rad, M_bar2(rad, *popt), kind='linear', 
#                                bounds_error=False, fill_value = 'extrapolate')
#
#                if Mint2(Rc)>0:
#                    Mtot = Mint2(Rc) + M_CNFW(Rc, Rc, Mc, re, rs)
#                else:
#                    Mtot = M_CNFW(Rc, Rc, Mc, re, rs)
#
#
#                mu = 1./np.sqrt(Rc*Gentc2*Mtot)### ecuacion 26
#                cons = Mtot**2*Mc*1.0e9/9.355e3## ec 52
#                mu2 = 1./np.sqrt(np.sqrt(cons)) ## ec 52
#                R99 = 2.38167*Rc
#
#                if Mint2(R99)>0:
#                    Mt99 = Mint2(R99) + M_CNFW(R99, Rc, Mc, re, rs)
#                else:
#                    Mt99 = M_CNFW(R99, Rc, Mc, re, rs)
#                cons2 = Mt99**2*Mc*1.0e9/2.536e5
#                mu3 = 1./np.sqrt(np.sqrt(cons2)) # ecuacion 50
#                muDM = Mc/(Rc**2 * np.sqrt(np.pi)**3)
#                Mtrf = M_CNFW(rmax, Rc, Mc, re, rs)/1e2#ecuacion 42
#                Mt997 = Mt99*1e3#ecuacion 42
#                mu4 = 140 * Mtrf**(1./3.)/Mt997 #ecuacion 42
#                rc = 2.38167*Rc*1e3/3.77#en parcecs
#                mu5 = 521.*Mtrf**(1./3.)/rc #ecuacion 43
#
##                muDM = Mc/(Rc**2 * np.sqrt(np.pi)**3)
#                H = 2.38167/3.77
#                muDM = H*Mc/(Rc**2 *np.sqrt(np.pi)**3)
#                MU.append(mu*hc)
#                MU2.append(mu2)
#                MU3.append(mu3)
#                MU4.append(mu4)
#                MU5.append(mu5)
#                MC.append(Mc)
#                RC.append(Rc)
#                RS.append(rs)
#                RE.append(re)
#                MUDM.append(muDM)
#                MTL.append(MtL)
#                if mu2<10.:
#                    INDEX.append(i)
#                else:
#                    INDEX2.append(i)
#        except RuntimeError:
#            print(i,'ups, minimizacion falló')
#            print('-------------------------------------------------------')                   
#    fil.close()
#      
    MU = np.array(MU)
    MU2 = np.array(MU2) 
    MU3 = np.array(MU3)
    MU4 = np.array(MU4) 
    MU5 = np.array(MU5)
    MUDM = np.array(MUDM)
    MC = np.array(MC)
    RC = np.array(RC)
    RE = np.array(RE)
    RS = np.array(RS)
    X2 = np.array(X2)
    bins = 50
    INDEX = np.array(INDEX)
    INDEX2 = np.array(INDEX2)
    MTL= np.array(MTL)
    
#    for A in a:
#        temp=[]
#        
#        val = 'no'
#        for i in INDEX:
##            
##        val = 'si'
##        for i in INDEX2:
#
#            data = np.loadtxt("%s/%d.dat"%(dirdata,i))
#            temp.append(data_dict[A][i - 1])
#        temp = np.array(temp)
#        pts.histo(temp,'', bins =bins,
#                  nom_archivo ='%s/%s_%s.png'%(dirfitsG, A, val),
#                  fit = False,  normalized = False,title=A)
#    temp2 =[]
#    for i in INDEX:
#        temp2.append(R2[i - 1])
#    temp2 = np.array(temp2)
#    pts.histo(temp2,r'$r^2$', bins =bins,
#              nom_archivo ='%s/r2_no.png'%(dirfitsG),
#              fit = False,  normalized = False)
#    temp2 =[]
#    for i in INDEX2:
#        temp2.append(R2[i - 1])
#    temp2 = np.array(temp2)
#    pts.histo(temp2,r'$r^2$', bins =bins,
#              nom_archivo ='%s/r2_si.png'%(dirfitsG),
#              fit = False,  normalized = False)
    

    pts.histo(MC,r'$M_c (\times 10^{10} M_\odot)$', bins =bins, normalized = False,
              nom_archivo ='%s/Mc_%s.png'%(dirfitsG, ID))
    pts.histo(RC,r'$R_c$(kpc)', bins =bins, 
              nom_archivo ='%s/Rc_%s.png'%(dirfitsG, ID))
    pts.histo(RE,r'$r_e (kpc)$', bins =bins,
              nom_archivo ='%s/Re_%s.png'%(dirfitsG, ID))
    pts.histo(RS,r'$r_s (kpc)$', bins =bins,
              nom_archivo ='%s/Rs_%s.png'%(dirfitsG, ID))     
    pts.histo(MTL,r'$\gamma$', bins =bins,
              nom_archivo ='%s/MtL_%s.png'%(dirfitsG, ID))      
    pts.histo(MU,r'$m (\rm{eV}/c^2)$', bins =bins,
              nom_archivo ='%s/mu_%s.png'%(dirfitsG, ID), #fit = True,
              title='ecuacion 26',
#              logx=True,
#              rang = (0.1e-21, 0.1e-20)
              )
    pts.histo(MU2,r'$m (\times 10^{-22} \rm{eV}/c^2)$', 
              bins = bins,title='ecuacion 51',
              nom_archivo ='%s/mu_rel_McMt_%s.png'%(dirfitsG, ID), 
#              logx=True,
#              rang = (10, 100)
              )
    pts.histo(MU3,r'$m (\times 10^{-22} \rm{eV}/c^2)$', 
              bins = bins,title='ecuacion 50',
              nom_archivo ='%s/mu_3_%s.png'%(dirfitsG, ID), 
#              logx=True,
#              rang = (10, 100)
              )
    pts.histo(MU4,r'$m (\times 10^{-22} \rm{eV}/c^2)$', 
              bins = bins,title='ecuacion 42',
              nom_archivo ='%s/mu_4_%s.png'%(dirfitsG, ID), 
#              logx=True,
#              rang = (10, 100)
              )
    pts.histo(MU5,r'$m (\times 10^{-22} \rm{eV}/c^2)$', 
              bins = bins,normalized = False,title='ecuacion 43',
              nom_archivo ='%s/mu_5_%s.png'%(dirfitsG, ID), 
#              logx=True,
#              rang = (10, 100)
              )    
    pts.histo(MUDM,r'$\mu_\psi (10^{4} M_\odot pc^{-2})$', bins =bins, 
              normalized = False,
#              logx = True,
              nom_archivo ='%s/mudm_%s.png'%(dirfitsG, ID))  
    pts.histo(X2,r'$\chi^2_{red}$', bins =bins,normalized = False,
              nom_archivo ='%s/chi_%s.png'%(dirfitsG, ID),
#              rang = (0, 250)
              )  
    pts.histo(R2,r'$R^2_{red}$', bins =bins,normalized = False,
              nom_archivo ='%s/R_%s.png'%(dirfitsG, ID),
#              rang = (0, 250)
              )