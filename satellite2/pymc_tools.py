#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:38:54 2020

@author: jordi
"""
import matplotlib as mpl
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from constants_grav import hc
import plots_jordi as pts
import lmfit as lm
import pybroom as br
import pandas as pd


def todas_las_trazas(trace, var_names, filename, point_estimate="mean"):
    az.rcParams['plot.max_subplots']= 80        
    plt.style.use('default')
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 0.9
    mpl.rcParams['text.usetex'] = True
    az.plot_pair(trace, var_names = var_names, kind = "kde",
                 kde_kwargs={"fill_last": False}, marginals=True,
                 point_estimate=point_estimate, figsize=(15, 12))
    plt.savefig(filename, bbox_inches='tight')
    plt.show()   
    
def sigmas(res, dirfitsG, ncor, ID='version'):
    # lets work out a 1 and 2-sigma error estimate for 'Rc'
    print('parameter   1 sigma spread  2 sigma spread')   
    fil= open("%s/cantidades_emcee_nsol%d_%s.txt"%(dirfitsG,ncor, ID),"w+")   
    fil.write('name \t mean \t 1sigma \t 2sigma  \r\n')
    unsigma=[]
    dossigma=[]
#   print(res.var_names)
    for ii in res.var_names:
        quantiles = np.percentile(res.flatchain[ii], [2.28, 15.9, 50, 84.2, 97.7])      
        sigma1 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2 = 0.5 * (quantiles[4] - quantiles[0])
        prom = res.params[ii].value
        unsigma.append(sigma1)
        dossigma.append(sigma2)

        print(ii, prom, sigma1, sigma2)
        fil.write('%s \t %.4f \t %.5f \t %.5f  \r\n'%(ii, prom, sigma1, sigma2))
        if ii== 'rlam':
            print(ii, '(',prom*1e3,sigma1*1e3, sigma2*1e3, ') 10^-3')    
            fil.write('%s \t %.4f \t %.4f \t %.4f  \r\n'%(ii, prom*1e3, sigma1*1e3, sigma2*1e3))
        if ii == 'mu':
            print(ii, '(',prom*hc*1e25,sigma1*hc*1e25, sigma2*hc*1e25, ') 10^-25 eV/c^2')
            fil.write('%s \t %.4f \t %.4f \t %.4f  \r\n'%(ii, prom*hc*1e25, sigma1*hc*1e25, sigma2*hc*1e25))
    fil.close()
    unsigma = np.array(unsigma)
    np.save('%sunsigma_nsol%d_%s.npy'%(dirfitsG,ncor, ID), unsigma)
    dossigma = np.array(dossigma)
    np.save('%sdossigma_nsol%d_%s.npy'%(dirfitsG,ncor, ID), dossigma)
    return unsigma, dossigma

def save_flatchain(res, dirfitsG, ncor, ID='version'):
    res.flatchain.to_pickle("%sflatchain_nsol%d_%s.pkl"%(dirfitsG,ncor, ID))
    np.save("%schain_nsol%d_%s.npy"%(dirfitsG,ncor, ID),res.chain)
    
def popt_up_dw(res, dirfitsG, ncor, ID='version'):
    popt_dw = []; popt_up = []
    print(res.var_names[:])
    for ii in res.var_names[:]:
        per = np.percentile(res.flatchain[ii], [2.5])
        per2 = np.percentile(res.flatchain[ii], [97.5])
        popt_dw.append(per[0])
        popt_up.append(per2[0])
    popt_dw = np.array(popt_dw); popt_up = np.array(popt_up)
    np.save('%spoptup_nsol%d_%s.npy'%(dirfitsG,ncor, ID), popt_up)
    np.save('%spoptdw_nsol%d_%s.npy'%(dirfitsG,ncor, ID), popt_dw)
    popt = np.array(list(res.params.valuesdict().values()))
    np.save('%spopt_nsol%d_%s.npy'%(dirfitsG,ncor, ID), popt)
    return popt_up, popt_dw

def max_like_sol(res):
### find the maximum likelihood solution
#    highest_prob = np.argmax(res.lnprob)
#    hp_loc = np.unravel_index(highest_prob, res.lnprob.shape)
#    mle_soln = res.chain[hp_loc]
#    print("\nMaximum likelihood Estimation")
#    print('-----------------------------')
#    for ix, param in enumerate(res.params):
#        print(param + ': ' + str(mle_soln[ix]))
    p = res.params
    highest_prob = np.argmax(res.lnprob)
    hp_loc = np.unravel_index(highest_prob, res.lnprob.shape)
    mle_soln = res.chain[hp_loc]
    for i, par in enumerate(p):
        p[par].value = mle_soln[i]
    
    
    print('\nMaximum Likelihood Estimation from emcee       ')
    print('-------------------------------------------------')
    print('Parameter  MLE Value   Median Value   Uncertainty')
    fmt = '  {:5s}  {:11.5f} {:11.5f}   {:11.5f}'.format
    for name, param in p.items():
        print(fmt(name, param.value, res.params[name].value,
                  res.params[name].stderr))

def gelman_rubin(chain):
    ssq = np.var(chain, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    θb = np.mean(chain, axis=1)
    θbb = np.mean(θb, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1) * np.sum((θbb - θb)**2, axis=0)
    var_θ = (n - 1) / n * W + 1 / n * B
    R = np.sqrt(var_θ / W)
    return R

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n
    # Optionally normalize
    if norm:
        acf /= acf[0]
    return acf

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def plot_tau_estimates(chain, namefile ='', dirfit = ''):
# Compute the estimators for a few different chain lengths
    N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
    gw2010 = np.empty(len(N))
    new = np.empty(len(N))
    for i, n in enumerate(N):
        gw2010[i] = autocorr_gw2010(chain[:, :n])
        new[i] = autocorr_new(chain[:, :n])   
    # Plot the comparisons
    plt.loglog(N, gw2010, "o-", label="G\&W 2010")
    plt.loglog(N, new, "o-", label="new")
    ylim = plt.gca().get_ylim()
    plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
    plt.ylim(ylim)
    plt.xlabel("number of steps, $N$")
    plt.ylabel(r"$\tau$ estimates")
    plt.legend(fontsize=14)
    if namefile!='':
        plt.savefig('%sautocorr_time_%s.png'%(dirfit, namefile), bbox_inches='tight')
    plt.show()

def autocorr_time(res):
    if hasattr(res, "acor"):
        print("Autocorrelation time for the parameters:")
        print("----------------------------------------")
    #        for i, p in enumerate(res.params):
    #            print(p, res.acor[i])
        for i in range (0, np.shape(res.var_names[:])[0]):
            print(res.var_names[i], res.acor[i])

def walkers_plot(res, name_pars,MO, namefile=''):
    pts.jordi_style()
    fig, ax = plt.subplots(np.shape(name_pars)[0],1, sharex=True,
                           figsize=(10,20))
    for i in range(np.shape(name_pars)[0]):
        ax[i].plot(res.chain[:,:,i].T)
        ax[i].set_ylabel(name_pars[i])
#        mini = MO[res.var_names[i]][0]
#        maxi = MO[res.var_names[i]][1]
#        ax[i].set_ylim(mini, maxi)
    plt.xlabel('steps')
    if namefile!='':
        plt.savefig('%s_%s.png'%(namefile,name_pars[i]), bbox_inches='tight')
    plt.show() 
        
def plot_aceptance(res, namefile=''):
    import matplotlib.pyplot as plt
    plt.plot(res.acceptance_fraction)
    plt.xlabel('walker')
    plt.ylabel('acceptance fraction')    
    if namefile!='':
        plt.savefig('%s.png'%(namefile), bbox_inches='tight')
    plt.show()    
    
def reports(result, dirfit = '', ID = ''):
    print(lm.fit_report(result))
    print(result.params.pretty_print())    
    dg = br.glance(result)
    dg.to_pickle('%sfitstat_%s.pkl'%(dirfit, ID))  # where to save it, usually as a .pkl
    dg = pd.read_pickle('%sfitstat_%s.pkl'%(dirfit, ID)) 
#    print(dg.AIC)
    dt = br.tidy(result)    
    dt.to_pickle('%sfitparamsstat_%s.pkl'%(dirfit, ID))  # where to save it, usually as a .pkl
    dt = pd.read_pickle('%sfitparamsstat_%s.pkl'%(dirfit, ID))         
#    print(dt)
    
    