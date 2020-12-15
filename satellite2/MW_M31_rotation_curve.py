#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:43:11 2020


Fit MINIMOS CUADRADOS de la via lactea 

DM         mixSFDM        pars       mu, lambda
Disco     Doble_exp      pars      Md, ad, bd
Bulge     Miyamoto-Nagai   pars      Mb, bb

NO BH
TAL VEZ SERIA BUENO PONER BH    

      
@author: jordi
"""
import plots_jordi as pts
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from galpy import potential
from def_potenciales import (RC_miyamoto, vnagai, vHernquist,Miyamoto_Nagai_3,
                             RC_exponential, RC_double_exponential,RC_double_exponential_aprox)
import numpy as np
import matplotlib.pyplot as plt
from constants_grav import c, G, Gentc2
paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'
dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
dataMW = np.loadtxt("%s/tab_rcmwall.dat.txt"%dirdata)
errMW = np.array([-dataMW[:,4] + dataMW[:,3], dataMW[:,5] - dataMW[:,3]])
dataMW2 = np.loadtxt("%s/tab_rcmw.dat.txt"%dirdata) 
errMW2 = np.array([-dataMW2[:,4] + dataMW2[:,3], dataMW2[:,5] - dataMW2[:,3]])
modelMW = np.loadtxt("%s/tab_rcmw-model.dat.txt"%dirdata) 
dire = '/home/jordi/satellite'
dataM31 = np.loadtxt('%s/M31_rotation_curve_data.txt'%dire)

dataGrandMW= np.loadtxt("%s/Table_GrandRC.dat.txt"%dirdata) 

ncor = 6  ## 4 y 5 m = 1
n = 2
l = 1
m = 0
    
def vdm(x, rlam, mu):
    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))    
    xn = x2[:-150]/(mu*rlam)
    vdmn = rlam**2*vdm[:-150]*c
    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
                  fill_value = "extrapolate" )
    return ve(x)

def plot_data(dataMW, dataM31, modelMW):
    pts.plotmultiple([], [], ['Data'], r'$r$(kpc)', r'$v$(km/s)',
                     'Andromeda M31', '%s/M31data.png'%dire,
                     ylim=(0,300), save = True,
                     data=True, xd=dataM31[:,1], yd=dataM31[:,2],
                     err=True, yerr=dataM31[:,3])
    
    pts.plotmultiple([], [], ['Data'], r'$r$(kpc)', r'$v$(km/s)',
                     'Milky Way', '%s/MWdata.png'%dire,
                     ylim=(0,300), save = True,
                     data=True,  xd=dataMW[:,0], yd=dataMW[:,3],
                     err=True, yerr=errMW)

    pts.plotmultiple([modelMW[:,0],modelMW[:,0],modelMW[:,0],modelMW[:,0]],
                     [modelMW[:,1],modelMW[:,2],modelMW[:,3],modelMW[:,4]],
                     ['Bulge','disk','total','DM','Data'], r'$r$(kpc)', r'$v$(km/s)',
                     'Milky Way', '%s/MWdata.png'%dire,
                     save = True,
                     data=True,  xd=dataMW[:,0], yd=dataMW[:,3],
                     err=True, yerr=errMW)

def ord_data(dataMW):
    r = dataMW[:,0]
    vel = dataMW[:,3]
    err = errMW[0,:]    
    sortindex = np.argsort(r, axis = 0)
    rord = np.sort(r, axis= 0)    
    velord = vel[sortindex]
    errord = err[sortindex]
    return rord, velord, errord

def fitting(func, rad, v, bounds, paramname, error = False, err = [],
            printing = True, loss = 'linear'):
    if error == False:
        popt, pcov = curve_fit(func, rad, v, bounds = (bounds[0], bounds[1]),
                               loss = loss)
    else:
        popt, pcov = curve_fit(func, rad, v, bounds = (bounds[0], bounds[1]),
                               sigma = 1./(err*err), loss = loss,
                               absolute_sigma = True)       
    perr = np.sqrt(np.diag(pcov)) 
    if printing ==True:
        print('fit parameters and 1-sigma error')
        for j in range(len(popt)):
            print(paramname[j],'=', str(popt[j])+' +- '+str(perr[j])) 
    nstd = 1. # to draw 1-sigma intervals
    popt_up = popt + nstd * perr  ##Fitting parameters at 1 sigma
    popt_dw = popt - nstd * perr  ##Fitting parameters at 1 sigma
    fit = func(rad, *popt)
    ss_res = np.sum((v - fit) ** 2) # residual sum of squares
    ss_tot = np.sum((v - np.mean(v)) ** 2)  # total sum of squares
    r2 = 1 - (ss_res / ss_tot) # r-squared
    if error ==True:
        sigma = 1./(err*err)
        m = len(popt)#number of params
        Chi2 =  np.sum(((v - fit)/sigma)**2)
        print('Chi2', Chi2, 'Chi2_red', Chi2/(np.shape(rad)[0]-m),)
    if printing ==True:
        print('r-squared=', r2)   
    return popt, popt_up, popt_dw, r2, perr

def func(x, rlam, mu, Md, ad, bd, Mb, bb):## M es entre 10^10 M_sol
    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))    
    xn = x2[:-150]/(mu*rlam)
    vdmn = rlam**2*vdm[:-150]*c
    vh = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
                  fill_value = "extrapolate" )

    vd = RC_double_exponential_aprox(x, Md, ad, bd)
#    vd = RC_miyamoto(x, G, Md, ad, bd)
    vb = RC_miyamoto(x, G, Mb, 0., bb)
    V = vh(x)**2 + vd**2 + vb**2
    return np.sqrt(V)


#def func(x, rlam, mu, Md, ad, bd, Mb, bb, Md1, ad1, bd1, Md2, ad2, bd2, Md3, ad3, bd3):## M es entre 10^10 M_sol
#    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))    
#    xn = x2[:-150]/(mu*rlam)
#    vdmn = rlam**2*vdm[:-150]*c
#    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
#                  fill_value = "extrapolate" )
#    vb = RC_miyamoto(x, G, Mb, 0., bb)
#    vd = RC_miyamoto(x, G, Md, ad, bd)
#    vd1 = RC_miyamoto(x, G, Md1, ad1, bd1)
#    vd2 = RC_miyamoto(x, G, Md2, ad2, bd2)
#    vd3 = RC_miyamoto(x, G, Md3, ad3, bd3)
#    vtot2 = ve(x)**2 + vb**2 + (vd + vd1 + vd2 + vd3)**2
#    return np.sqrt(vtot2)

if __name__ == '__main__':
    #plot_data(dataMW, dataM31, modelMW)
    
    #Md = [0.606, 0.013, 0.546, 0.218]    
    #ad = [3.859, 0.993, 9.021, 6.062] #original paper
    #bd = [0.243, 0.776, 0.168, 0.128] #original paper   
        
#    bound = [[1e-8, 1.5655, 0.5,  1., 0.09, 0.1,  0.05, 0.008, 0.6, 0.4, 0.09, 6.,  0.09, 0.09, 1., 0.08], 
#             [1e-1, 156.55, 0.8,  5.,  0.4, 1.0,  0.3,  0.018, 1.0,  0.9, 0.6, 10.,  0.4, 0.4, 8., 0.15]]    
#    name = ['rlam',  'mu', 'Md','ad', 'bd', 'Mb', 'bb','Md1','ad1','bd1','Md2','ad2','bd2','Md2','ad2','bd2']
    
    bound = [[1e-10, 1.5655,  0.0,   0.,  0.0,  0., 0.0], 
             [1e-1,  156.55,  500.,  25., 0.5,  2.0, 0.3]]
    name =   ['rlam',  'mu',  'Md', 'ad', 'bd','Mb', 'bb']    

#    rad, v, errord = ord_data(dataMW)
    
    rad, v, errord = dataGrandMW.T    
#    rad, v, errord =rad[:119], v[:119], errord[:119]
    rmin, rmax = np.amin(rad), 20.
    
    r = np.linspace(rmin, rmax, 1000)
    loss = ['linear', 'soft_l1','huber','cauchy' ,'arctan']  
#    for i in loss:
    try:        
        popt, popt_up, poptdw, r2, _ = fitting(func, rad, v, bound, name, 
                                            error = True,
#                                                loss = i,
                                            err= errord)     
        pts.plotmultiple([rad, r, r, r],
                         [func(rad, *popt),
                          RC_miyamoto(r, G, popt[2], popt[3], popt[4]),
                          RC_miyamoto(r, G, popt[5], 0., popt[6]),
                          vdm(r,popt[0],popt[1])],
                         [r'Disk+Bulge+Gas+mixSFDM',
                          'Disk $M_d = %.2f$, $a_d = %.2f$, $b_d=%.2f$'%(popt[2],popt[3],popt[4]),
                          'Bulge $M_b = %.2f$, $b_b=%.2f$'%(popt[5],popt[6]), 
                          'DM  $\lambda = %.3f$, $\mu = %.1f \mu_{25}$ '%(popt[0],popt[1]/15.655),
                          'Data'],
                         r'$r$(kpc)', r'$v$(km/s)', 
                         'mixSFDM $\Phi_{100} + \Phi_{%d%d%d}$'%(n, l, m), 
                         '%s/DM_fit_ncor_%d.png'%(dirdata,ncor),
                         ylim = (0,360),
                         xlim = (0, 20),
                         save = True, data=True, xd = rad, yd = v,
                         err = True, yerr = errord, 
                         fill_between= True, fbx = rad,
                         fby1 = func(rad, *popt_up), fby2 = func(rad, *poptdw),
                         text =r'$R^2 = %f $'%r2)
    except RuntimeError:
#        print(i, 'Optimal parameters not found')
        print('Optimal parameters not found')