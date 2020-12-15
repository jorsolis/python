#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020

@author: jordi
"""
import pymc
import plots_jordi as pts
import numpy as np
# set random seed for reproducibility
np.random.seed(12345)

x = np.arange(5,400,10)*1e3

# Parameters for gaussian
amp_true = 0.2
size_true = 1.8
ps_true = 0.1

# Gaussian function
gauss = lambda x,amp,size,ps: amp*np.exp(-1*(np.pi**2/(3600.*180.)*size*x)**2/(4.*np.log(2.)))+ps
f_true = gauss(x=x,amp=amp_true, size=size_true, ps=ps_true )

# add noise to the data points

noise = np.random.normal(size=len(x)) * .02 
f = f_true + noise 
f_error = np.ones_like(f_true)*0.05*f.max()

# define the model/function to be fitted.
def model(x, f): 
    amp = pymc.Uniform('amp', 0.05, 0.4, value= 0.15)
    size = pymc.Uniform('size', 0.5, 2.5, value= 1.0)
    ps = pymc.Normal('ps', 0.13, 40, value=0.15)

    @pymc.deterministic(plot=False)
    def gauss(x=x, amp=amp, size=size, ps=ps):
        e = -1*(np.pi**2*size*x/(3600.*180.))**2/(4.*np.log(2.))
        return amp*np.exp(e)+ps
    y = pymc.Normal('y', mu=gauss, tau=1.0/f_error**2, value=f, observed=True)
    return locals()

MDL = pymc.MCMC(model(x,f))
MDL.sample(1e4)

# extract and plot results
y_min = MDL.stats()['gauss']['quantiles'][2.5]
y_max = MDL.stats()['gauss']['quantiles'][97.5]
y_fit = MDL.stats()['gauss']['mean']

pts.plotmultiple([x, x], [f_true, y_fit],
                 ['True','Fit', 'Observed'], 'x', 'y','' ,'',
                 data=True, xd=x, yd=f, err=True, yerr=f_error,
                 fill_between=True, fbx=x, fby1=y_min, fby2=y_max)
pymc.Matplot.plot(MDL)