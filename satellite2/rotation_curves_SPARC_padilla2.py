#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:20:13 2020

Fit MINIMOS CUADRADOS de SPARC

DM        core + NFW      pars       Rc, Mc, re, rs
baryons   Obervational    pars       M/L(bulge), M/L(disc)
      
@author: jordi
"""
import numpy as np
import plots_jordi as pts
from scipy.interpolate import interp1d
from SPARC_desc import tipos, data_dict
from rotation_curves_SPARC_padilla import fitting
from constants_grav import G, Gentc2, hc
from def_potenciales import f, v2_DM, M_CNFW
sat = '/home/jordi/satellite'
dirdata = '/home/jordi/SPARC'
dirfitsG = '/home/jordi/SPARC/Fits/Gaussian'

def M(r, Rc, Mc, re, rs, MLb, MLd):
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    vbul = data[:,5]     
    Mh = M_CNFW(r, Rc, Mc, re, rs)
    return Mh + vgas**2*r/G + MLd*vdisk**2*r/G + MLb*vbul**2*r/G
def M_bar(r, Rc, Mc, re, rs, MLb, MLd):
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    vbul = data[:,5]     
    return vgas**2*r/G + MLd*vdisk**2*r/G + MLb*vbul**2*r/G
def M_bar2(r, Rc, Mc, re, rs, MLd):
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    return vgas**2*r/G + MLd*vdisk**2*r/G
def v_bar(r, Rc, Mc, re, rs, MLb, MLd):
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    vbul = data[:,5]     
    return np.sqrt(vgas**2 + MLd*vdisk**2 + MLb*vbul**2)
def v_bar2(r, Rc, Mc, re, rs, MLd):
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]   
    return np.sqrt(vgas**2 + MLd*vdisk**2)
def func_CNFW(r, Rc, Mc, re, rs, MLb, MLd):
    ve2 = v2_DM(r,G, Rc, Mc, re, rs)
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    vbul = data[:,5] 
    return np.sqrt(ve2 + vgas**2 + MLd*vdisk**2 + MLb*vbul**2) 

def M2(r, Rc, Mc, re, rs, MLd):
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]    
    Mh = M_CNFW(r, Rc, Mc, re, rs)
    return Mh + vgas**2*r/G + MLd*vdisk**2*r/G

def func_CNFW2(r, Rc, Mc, re, rs, MLd):
    ve2 = v2_DM(r,G, Rc, Mc, re, rs)
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]
    return np.sqrt(ve2 + vgas**2 + MLd*vdisk**2) 
if __name__ == '__main__':   
    
    
    fil= open("%s/Fits/Gaussian/cantidades_head.txt"%dirdata,"w+")
    fil.write('Nfile \t Name \t Type \t Rc \t errRc \t Mc \t errMc \t M/L \t errM/L \t re \t errre \t rs \t errrs \t r2  \r\n')
    fil.write('Nfile \t Name \t Type \t (kpc) \t (kpc) \t (10^7)M_sun \t (10^7)M_sun  \t M/L \t errM/L \t (kpc) \t (kpc) \t r2 \r\n')
    fil.close()
    fil = open("%s/Fits/Gaussian/cantidades.txt"%dirdata,"w+")
    
    MU = []
    MU2 = []
    MU3 = []
    MUDM = []
    MC = []
    RC = []
    RE = []
    RS = [] 
    MTL = []
    R2 = np.zeros(175)
    INDEX =[]
    INDEX2 =[]
    X2 = []
    
    ID = '145galaxies'
    withbulge= (31, 40, 46, 47, 50, 67, 73, 74, 77, 80, 81, 86, 88, 90, 92, 93, 95, 107, 108, 109, 110, 111, 112, 113, 120, 132, 135, 136, 141, 160, 163, 169)
    bulgeless = (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 70, 71, 72, 75, 76, 79, 82, 83, 84, 85, 87, 89, 94, 97, 98, 100, 102, 103, 105, 106, 114, 115, 116, 117, 118, 119, 122, 123, 124, 125, 126, 127, 128, 130, 131, 133, 134, 137, 138, 140, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 161, 162, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175)

    for i in withbulge:   
        data = np.loadtxt("%s/%d.dat"%(dirdata,i))   
        rad, vobs, err, vgas, vdisk, vbul, _, _ = data.T
        effrad = data_dict['Effective Radius at [3.6]'][i - 1]
        name = data_dict['Name'][i - 1]
        tipo = tipos[data_dict['Type'][i - 1]]

        try:     
#            bound = [[0.01, 1.0e-4, 0.01,  0.2,   0.1,   0.1],
#                     [0.1,    10.,   10.,  100.,    5.,   5. ]]             
            bound=[0.001, np.inf]
            nameb = ['Rc',  'Mc',  're',  'rs',  'M/Lb', 'M/Ld']
            popt, _, _, r2, perr, Chi2, Chi2red = fitting(func_CNFW, rad, vobs,
                                                           bound,
                                           nameb,error = True, err = err,
                                           printing = False, loss = 'cauchy')       
            fit = func_CNFW(rad, *popt)
            X2.append(Chi2)
            Rc, Mc, re, rs , MtLb, MtLd= popt        
            fil.write('%d \t %s \t %s \t %.3f \t %.3f \t %.1f \t %.1f \t %.2f \t %.2f  \t %.2f \t %.2f  \t %.2f \t %.2f \t %.2f \r\n'%(i, name, tipo, Rc, perr[0], Mc*1e3, perr[1]*1e3, MtLd, perr[2], re, perr[3], rs, perr[4], r2))

            if r2>0:
                rmin, rmax = np.amin(rad), np.amax(rad)
                r99 = 2.3816 *Rc
                r = np.linspace(rmin, r99, 110) 
                parss = r'$R_c = %.1f $pc, $M_c = %.1f \times 10^7 M_\odot$, $r_e=%.2f$kpc '%(Rc*1e3, Mc*1e3, re) 
                parsbar = r'$M/Ld=%.5f$, $M/Lb=%.5f$'%(MtLd, MtLb)                 
                pts.plotmultiple([rad, rad,rad], [fit, v_bar(rad, *popt),np.sqrt(v2_DM(rad,G, Rc, Mc, re, rs))],
                                 ['fit %s'%parss,'baryons %s'%parsbar,'DM','data'],
                                 r'$r$(kpc)',r'$v$(km/s)',tipo + name,'%s/%d.png'%(dirfitsG,i),
                                 data=True, 
                                 xd=rad, yd=vobs, err=True, yerr=err)
                Mint = interp1d(rad, M_bar(rad, *popt), kind='linear', 
                                bounds_error=False, fill_value = 'extrapolate')
                R2[i - 1]= r2
                if Mint(Rc)>0:
                    Mtot = Mint(Rc) + M_CNFW(Rc, Rc, Mc, re, rs)
                else:
                    Mtot = M_CNFW(Rc, Rc, Mc, re, rs)
                cons = Mtot**2*Mc*1.0e9/1.45e4
                mu = 1./np.sqrt(Rc*Gentc2*Mtot)### ecuacion 26
                mu2 = 1./np.sqrt(np.sqrt(cons)) ## ec 51
                R99 = 2.38167*Rc
#                cons2 = Mint(R99)**2*Mc*1.0e9/2.536e5
#                mu3 = 1./np.sqrt(np.sqrt(cons2)) # ecuacion 50
                muDM = Mc/(Rc**2 * np.sqrt(np.pi)**3)
                MU.append(mu*hc)
                MU2.append(mu2)
                MC.append(Mc)
                RC.append(Rc)
                RE.append(re)
                RS.append(rs)
                MUDM.append(muDM)
                MTL.append(MtLd)
    
                if mu2<10.:
                    INDEX.append(i)
                else:
                    INDEX2.append(i)
    #                MU3.append(mu3)
        except RuntimeError:
            print(i,'ups, minimizacion falló')
            print('-------------------------------------------------------')
                  

    for i in bulgeless:
        data = np.loadtxt("%s/%d.dat"%(dirdata,i))   
        rad, vobs, err, vgas, vdisk, vbul, _, _ = data.T
        effrad = data_dict['Effective Radius at [3.6]'][i - 1]
        name = data_dict['Name'][i - 1]
        tipo = tipos[data_dict['Type'][i - 1]]

        try:     
            popt, _, _, r2, perr, Chi2, Chi2red = fitting(func_CNFW2, rad, vobs, bound,
                                           nameb,error = True, err = err,
                                           printing = False, loss = 'cauchy' ) 
            X2.append(Chi2)
            fit = func_CNFW2(rad, *popt)        
            Rc, Mc, re, rs , MtLd= popt        
            fil.write('%d \t %s \t %s \t %.3f \t %.3f \t %.1f \t %.1f \t %.2f \t %.2f  \t %.2f \t %.2f  \t %.2f \t %.2f \t %.2f \r\n'%(i, name, tipo, Rc, perr[0], Mc*1e3, perr[1]*1e3, MtLd, perr[2], re, perr[3], rs, perr[4], r2))

            if r2>0:
                rmin, rmax = np.amin(rad), np.amax(rad)
                r99 = 2.3816 *Rc
                parss = r'$R_c = %.1f $pc, $M_c = %.1f \times 10^7 M_\odot$, $r_e=%.2f$kpc '%(Rc*1e3, Mc*1e3, re)
                parsbar = r'$M/Ld=%.5f$'%(MtLd) 
                pts.plotmultiple([rad,rad,rad],
                                 [fit, v_bar2(rad, *popt),np.sqrt(v2_DM(rad,G, Rc, Mc, re, rs))],
                                 ['fit %s'%parss,'baryons %s'%parsbar,'DM', 'data'],
                                 r'$r$(kpc)',r'$v$(km/s)',tipo + name,'%s/%d.png'%(dirfitsG,i),data=True, 
                                 xd=rad, yd=vobs, err=True, yerr=err)                  
                Mint2 = interp1d(rad, M_bar2(rad, *popt), kind='linear', 
                                bounds_error=False, fill_value = 'extrapolate')
                R2[i - 1]= r2
                if Mint2(Rc)>0:
                    Mtot = Mint2(Rc) + M_CNFW(Rc, Rc, Mc, re, rs)
                else:
                    Mtot = M_CNFW(Rc, Rc, Mc, re, rs)
                cons = Mtot**2*Mc*1.0e9/1.45e4
                mu = 1./np.sqrt(Rc*Gentc2*Mtot)### ecuacion 26
                mu2 = 1./np.sqrt(np.sqrt(cons)) ## ec 51
                R99 = 2.38167*Rc
                muDM = Mc/(Rc**2 * np.sqrt(np.pi)**3)
                MU.append(mu*hc)
                MU2.append(mu2)
                MC.append(Mc)
                RC.append(Rc)
                RS.append(rs)
                RE.append(re)
                MUDM.append(muDM)
                MTL.append(MtLd)
                if mu2<10.:
                    INDEX.append(i)
                else:
                    INDEX2.append(i)
        except RuntimeError:
            print(i,'ups, minimizacion falló')
            print('-------------------------------------------------------')                   
    fil.close()
      
    MU = np.array(MU)
    MU2 = np.array(MU2) 
    MU3 = np.array(MU3)
    MUDM = np.array(MUDM)
    MC = np.array(MC)
    RC = np.array(RC)
    RE = np.array(RE)
    RS = np.array(RS)
    X2 = np.array(X2)
    bins = 20
    INDEX = np.array(INDEX)
    INDEX2 = np.array(INDEX2)
    MTL= np.array(MTL)
#    for A in a:
#        temp=[]
#        
#        val = 'no'
#        for i in INDEX:
#            
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
    
    pts.histo(MU,r'$m (\rm{eV}/c^2)$', bins =bins,
              nom_archivo ='%s/mu%s.png'%(dirfitsG, 'all'), fit = False,
              normalized = False,title='ecuacion 26',
#              logx=True,
#              rang = (0.1e-21, 0.1e-20)
              )
    pts.histo(MU2,r'$m (\times 10^{-22} \rm{eV}/c^2)$', 
              bins = bins,normalized = False,title='ecuacion 51',
              nom_archivo ='%s/mu_rel_McMt_%s.png'%(dirfitsG, ID), 
              fit = False,
#              logx=True,
#              rang = (10, 100)
              )
    pts.histo(MC,r'$M_c (\times 10^{10} M_\odot)$', bins =bins,
              nom_archivo ='%s/Mc_%s.png'%(dirfitsG, ID), fit = False, 
#              logx=True,
#              rang = (0, 100)
              )
    pts.histo(RC,r'$R_c$(kpc)', bins =bins, normalized = False,
              nom_archivo ='%s/Rc_%s.png'%(dirfitsG, ID))
    pts.histo(MUDM,r'$\mu_\psi (10^{4} M_\odot pc^{-2})$', bins =bins, 
              normalized = False,
              nom_archivo ='%s/Rc_%s.png'%(dirfitsG, ID))    
    pts.histo(RE,r'$r_e (kpc)$', bins =bins,normalized = False,
              nom_archivo ='%s/Re_%s.png'%(dirfitsG, ID))
    pts.histo(RS,r'$r_s (kpc)$', bins =bins,normalized = False,
              nom_archivo ='%s/Rs_%s.png'%(dirfitsG, ID))    
    pts.histo(X2,r'$X^2$', bins =bins,normalized = False,
              nom_archivo ='%s/chi_%s.png'%(dirfitsG, ID),
              rang = (0, 250)
              )     
    pts.histo(MTL,r'$\gamma$', bins =bins,normalized = False,
              nom_archivo ='%s/MtL_%s.png'%(dirfitsG, ID))      
    

            