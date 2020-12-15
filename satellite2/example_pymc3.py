#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 22:59:08 2020

@author: jordi
"""

import numpy as np
import pymc3 as pm3
import plots_jordi as pts
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
with pm3.Model() as model3:
    amp = pm3.Uniform('amp', 0.05, 0.4, testval= 0.15)
    size = pm3.Uniform('size', 0.5, 2.5, testval= 1.0)
    ps = pm3.Normal('ps', 0.13, 40, testval=0.15)

    gauss=pm3.Deterministic('gauss', 
                            amp*np.exp(-1*(np.pi**2*size*x/(3600.*180.))**2/(4.*np.log(2.)))+ps)

    y =pm3.Normal('y', mu=gauss, tau=1.0/f_error**2, observed=f)

    start=pm3.find_MAP()
    step=pm3.NUTS()
    trace=pm3.sample(2000,start=start)

# extract and plot results
y_min = np.percentile(trace.gauss,2.5,axis=0)
y_max = np.percentile(trace.gauss,97.5,axis=0)
y_fit = np.percentile(trace.gauss,50,axis=0)

pts.plotmultiple([x, x], [f_true, y_fit],
                 ['True','Fit', 'Observed'], 'x', 'y','' ,'',
                 data=True, xd=x, yd=f, err=True, yerr=f_error,
                 fill_between=True, fbx=x, fby1=y_min, fby2=y_max)