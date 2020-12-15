#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 23:20:13 2020

Fit MINIMOS CUADRADOS de SPARC

DM         mixSFDM        pars       mu, lambda
baryons   Obervational    pars       M/L

@author: jordi
"""
import numpy as np
import plots_jordi as pts
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from SPARC_desc import tipos, data_dict
from MW_M31_rotation_curve import fitting

sat = '/home/jordi/satellite'
dirdata = '/home/jordi/SPARC'
paco = '/home/jordi/satellite/schrodinger_poisson/potpaco'
dirmixshoot = '/home/jordi/satellite/mix_shooting'

c = 2.9e5
Gentc2 = 4.799e-7 ##G/c^2 en kpc/(10^10 M_sol)
G = Gentc2*c*c

#lm = { 0:{'ncor':301}, 1:{'ncor':302}, 2:{'ncor':303} }
#l = 1
#m = 0
#def func_DM(x, rlam, mu):
##    mu = MM[mm]['mu'] #en kpc  
#    ncor = lm[m]['ncor']
#    x2, vdm = np.load('%s/l%d_shooting/vdm_%d.npy'%(sat,l,ncor))    
#    xn = x2/(mu*rlam)
#    vdmn = rlam**2*vdm*c
#    ve = interp1d(xn, vdmn, kind='linear', copy=True, bounds_error=False,
#                  fill_value="extrapolate" )
#    return ve(x)
###############################################################################
###################            POT PACO        ################################
###############################################################################
######  mix21m = { 0:{'ncor':1,2,6}, 1:{'ncor':3,4,5}}
ncor = 6  ## 4 y 5 m = 1
n = 2
l = 1
m = 0
def func(r, rlam, mu, ML):
    x2, vdm = np.load('%s/baja_dens/pot_%d/vdm_%d.npy'%(paco,ncor,ncor))    
    xn = x2[:-150]/(mu*rlam)
    vdmn = rlam**2*vdm[:-150]*c
    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
                  fill_value = "extrapolate" )
    ve2 = ve(r)**2
    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
    vgas = data[:,3]
    vdisk = data[:,4]  
    vbul = data[:,5] 
    return np.sqrt(ve2 + vgas**2 + ML*vdisk**2 + 1.4*ML*vbul**2)

###############################################################################
################          MULTISTATE 32m        #############################
###############################################################################
#ncor = 103
#n = 3
#l = 2
#m = 0
#def func(x, rlam, mu, ML):
#    x2, vdm = np.load('%s/vdm_%d.npy'%(dirmixshoot, ncor-100))    
#    xn = x2/(mu*rlam)
#    vdmn = rlam**2*vdm*c
#    ve = interp1d(xn, vdmn, kind = 'linear', copy = True, bounds_error = False,
#                  fill_value = "extrapolate" )
#    ve2 = ve(x)**2
#    data = np.loadtxt("%s/%d.dat"%(dirdata,i)) 
#    vgas = data[:,3]
#    vdisk = data[:,4]  
#    vbul = data[:,5] 
#    return np.sqrt(ve2 + vgas**2 + ML*vdisk**2 + 1.4*ML*vbul**2) 
###############################################################################
###############################################################################
#f= open("%s/Fits/mix/%d/cantidades.txt"%(dirdata,ncor),"w+")
#f.write('Nfile \t Name \t Type \t lambda(10^-3) \t errlam \t mu(10^-25) \t errmu \t M/L \t errM/L \t r2 \r\n')

for k in range(0, 12,1):
    
    R = []
    MMM = [] 
    M =[]
    
#    MU=[]
    for i in range(1,176,1):
        if data_dict['Type'][i - 1]== k :    
#        if data_dict['Type'][i - 1]== 10 : 
    #        pass
    #    elif data_dict['Type'][i - 1]== 11: #BCD
    #        pass
    #    elif data_dict['Type'][i - 1]== 0: #SO
    #        pass
    #    else:
            
    #        if data_dict['Quality Flag'][i - 1]== 1:        
            data = np.loadtxt("%s/%d.dat"%(dirdata,i))   
            rad = data[:,0]
            vobs = data[:,1]
            vgas = data[:,3]
            vdisk = data[:,4]
            vbul = data[:,5]
            err = data[:,2]
            vtot = np.sqrt(vgas**2 + vdisk**2 + vbul**2) 
            vDM2 = vobs**2 - vtot**2
            name = data_dict['Name'][i - 1]
            tipo = tipos[data_dict['Type'][i - 1]]
            distance = data_dict['Distance'][i - 1]
            effrad = data_dict['Effective Radius at [3.6]'][i - 1]
            
            M.append(rad*vobs**2/G)
            MMM.append((rad*vobs**2/G)**2*rad*vDM2/G)
            R.append(rad)
        
    #    pts.plotmultiple([rad, rad, rad, rad, rad],
    #                     [vobs, vgas, vdisk, vbul , vtot],
    #                     ['obs', 'gas', 'disk', 'bulge', 'total'],
    #                     r'$r$(kpc)', r'$v$(km/s)',
    #                     'Name: %s, Type: %s, distance=%.2f Mpc'%(name, tipo, distance),
    #                     '%s/%d.png'%(dirdata,i), save = True, xv = [effrad],
    #                     data=True, xd=rad, yd=vobs,err=True, yerr=err) 
    
    #    try:
    #        bound = [[1e-6, 15.655, 1.], [1e-1, 1565.50, 10.]]       
    #        nameb = ['lambda',  'mu', 'M/L']
    #        popt, popt_up, popt_dw, r2, perr = fitting(func, rad, vobs, bound, 
    #                                                   nameb, error = True, 
    #                                                   err = err)        
    #        fit = func(rad, *popt)
    #        fit_up = func(rad, *popt_up)
    #        fit_dw = func(rad, *popt_dw)
    #    
    #        f.write('%d \t %s \t %s \t %.3f \t %.3f \t %.1f \t %.1f \t %.2f \t %.2f \t %.2f \r\n'%(i, name, tipo, popt[0]*1000., perr[0]*1000, popt[1]/15.655, perr[1]/15.655, popt[2], perr[2], r2))
    #        if r2 > 0.7:
    #            pts.plotmultiple([rad, rad],[ fit],
    #                             [r'$\lambda = %.3f \times10^{-3}$, $\mu = %.1f \mu_{25}$, $M/L=%.2f$ '%(popt[0]*1000.,popt[1]/15.655, popt[2]),
    #                            'Data'],
    #                             r'$r$(kpc)', r'$v$(km/s)', 
    #                             '%d, Name: %s, Type: %s, mixSFDM $\Phi_{100} + \Phi_{%d%d%d}$'%(i,name, tipo, n, l, m), 
    #                             '%s/Fits/mix/%d/DM_%d_fit_%d.png'%(dirdata,ncor,i,ncor),
    #                             save = True, data=True, xd = rad, yd = vobs,
    #                             err = True, yerr = err, show=False,
    ##                             fill_between= True, 
    ##                             fbx = rad, fby1 = fit_up, fby2 = fit_dw,
    #                             text =r'$R^2 = %f $'%r2)
#                Mint = interp1d(rad, M(rad, *popt), kind='linear', 
#                                copy=True, bounds_error=False, fill_value = 'extrapolate')
#                mu = popt[1]
#                print(mu)
#                MU.append(mu/15655.)
    #
    #    except RuntimeError:
    #        print(i,'ups, minimizacion falló')
    #        print('-------------------------------------------------------')
    #    except ValueError:
    #        print(i, 'array must not contain infs or NaNs')
    #        print('-------------------------------------------------------')
    #f.close()
    
    print(np.shape(MMM))
    pts.plotmultiple(R, MMM, [], 'r(kpc)','$M_{obs}^2 M_{DM}(10^{10} M_{\odot})^3$', 
                     '%s'%tipos[k],
                     '%s/%s.png'%(dirdata,tipos[k]), 
                     xlim = (0,1), 
    #                 ylim = (0, 1e6),
#                     logx=True,
                     logy=True
                     )
#    pts.plotmultiple(R, M, [], 'r(kpc)','$M_{obs} (10^{10} M_{\odot})$', 
#                     '%s'%tipos[k],
#                     '%s/%s.png'%(dirdata,tipos[k]),
#                      xlim = (0,1), 
##                      ylim = (0, 1),
##                     logx=True,
#                     logy=True
#                     )    
#MU = np.array(MU)
#print(MU, 'x10^-22')
##print(MU.shape)
#maximo = np.nanmax(MU) ### DATOS NUMERADOR
#moda = stats.mode(MU, axis=None)
#print('promedio',MU.mean(), 'x10^-22')
#print('max',maximo, 'x10^-22')
#print('moda',moda, 'x10^-22')
#plt.hist(MU,bins=60)#, range=[0,20])
#plt.xlabel(r'$\hat\mu (\times 10^{-22}) \rm{eV}/c^2$')