#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:38 2020

@author: jordi
"""
import pymc
import plots_jordi as pts
import numpy

class test():
    # create some test data
    x = numpy.arange(100) * 0.3
    f = 0.1 * x**2 - 2.6 * x - 1.5
    numpy.random.seed(76523654)
    noise = numpy.random.normal(size=100) * .5     # create some Gaussian noise
    f2 = f + noise                                # add noise to the data
    #priors
    sig = pymc.Uniform('sig', 0.0, 100.0, value=1.)
    a = pymc.Uniform('a', -10.0, 10.0, value= 0.0)
    b = pymc.Uniform('b', -10.0, 10.0, value= 0.0)
    c = pymc.Uniform('c', -10.0, 10.0, value= 0.0)
    #model
    @pymc.deterministic(plot=False)
    def mod_quadratic(x=x, a=a, b=b, c=c):
          return a*x**2 + b*x + c
    #likelihood
    y = pymc.Normal('y', mu=mod_quadratic, tau=1.0/sig**2, value=f2, observed=True)


R = pymc.MCMC(test)    #  build the model
R.sample(10000)        # populate and run it

#print('a', R.a.stats())
#print('b', R.b.stats())
#print(R.stats())
def quadratic(x, a, b, c):
      return a*x**2 + b*x + c
  
x = test.x
f2 = test.f2
z = numpy.polyfit(x, f2, 2)   # the traditional chi-square fit
y_min = R.stats()['mod_quadratic']['quantiles'][2.5]
y_max = R.stats()['mod_quadratic']['quantiles'][97.5]
y_fit = R.stats()['mod_quadratic']['mean']

pts.plotmultiple([x, x], [quadratic(x, *z), y_fit],
                 ['Traditional chi-square fit','Bayesian fit'], 'x', 'y','' ,'',
                 data=True, xd=x, yd=test.f, err=True, yerr=test.noise,
                 fill_between=True, fbx=x, fby1=y_min, fby2=y_max, text= '', xv=[])

pymc.Matplot.plot(R)