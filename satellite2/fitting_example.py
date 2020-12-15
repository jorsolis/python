#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:10:46 2020

@author: jordi
"""
import numpy as np
from pylab import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plots_jordi as pts

def func(x, s, mu):
    return np.exp(-(x - mu)**2/(2.*s**2))/(s*np.sqrt(2.*np.pi))
#def func(x, a, b, c):
#    return a * x *x + b*x + c

# test data and error
x = np.linspace(-10, 10, 100)


y0 = func(x, -0.07, 0.5, 2.)
#y0 = func(x, -0.07, 0.5)


np.random.seed(1729)
noise = np.random.normal(0.0, 1.0, len(x))
y = y0 + noise

#y = y0

# curve fit [with only y-error]
#popt, pcov = curve_fit(func, x, y, sigma=1./(noise*noise))

popt, pcov = curve_fit(func, x, y)
perr = np.sqrt(np.diag(pcov))

#print fit parameters and 1-sigma estimates

print('fit parameter 1-sigma error')
print('———————————–')
for i in range(len(popt)):
    print(str(popt[i])+' +- '+str(perr[i]))

# prepare confidence level curves
nstd = 5. # to draw 5-sigma intervals
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

fit = func(x, *popt)
fit_up = func(x, *popt_up)
fit_dw = func(x, *popt_dw)
    
pts.plotmultiple([x, x],[fit, y0],
                 ['best fit curve', 'True curve', 'data'],
                 r'$x$', r'$y$','fit with only Y-error', 
                 '', save = False,
                 data=True, xd = x, yd = y0,
                 err = True, yerr=noise,
                 fill_between= True, fbx=x, fby1=fit_up,
                 fby2=fit_dw, text ='jaja')

#
#
#
#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
#import numpy as np
#
#
#def func(x, a, b, c):
#    return a * np.exp(-b * x) + c
#
#
##Define the data to be fit with some noise:
#
#xdata = np.linspace(0, 4, 50)
#y = func(xdata, 2.5, 1.3, 0.5)
#np.random.seed(1729)
#y_noise = 0.2 * np.random.normal(size=xdata.size)
#ydata = y + y_noise
#plt.plot(xdata, ydata, 'b-', label='data')
#
##Fit for the parameters a, b, c of the function func:
#
#popt, pcov = curve_fit(func, xdata, ydata)
#print(popt)
#
#plt.plot(xdata, func(xdata, *popt), 'r-',
#         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#
##Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1
## and 0 <= c <= 0.5:
#popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
#print(popt)
#
#plt.plot(xdata, func(xdata, *popt), 'g--',
#         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

print(np.shape(Zp))
#plt.xlabel('x')
#plt.ylabel('y')
#plt.legend()
#plt.show()
##The estimated covariance of popt. The diagonals provide the variance of 
##the parameter estimate. To compute one standard deviation errors 
##on the parameters use 
#
#perr = np.sqrt(np.diag(pcov))
#print('1 Sigma error on parameters',perr)