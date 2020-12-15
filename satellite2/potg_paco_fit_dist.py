#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:34:05 2020

@author: jordi
"""
import zfit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from pot_paco_orbitas_analisis import filtro

te = 975
rut = "/home/jordi/satellite/schrodinger_poisson/potpaco"
de = 'baja_dens/'
nsol = 6
carp = 'orbitas_random_vel_new'
t =  np.load("%s/%spot_%d/%s/1/tiemp_1.npy"%(rut,de,nsol,carp))
X = np.load("%s/%spot_%d/%s/X_%d.npy"%(rut,de,nsol,carp,t[te]))
Y = np.load("%s/%spot_%d/%s/Y_%d.npy"%(rut,de,nsol,carp,t[te]))
Z = np.load("%s/%spot_%d/%s/Z_%d.npy"%(rut,de,nsol,carp,t[te])) 

tho = np.load("%s/%spot_%d/%s/Th_%d.npy"%(rut,de,nsol,carp,t[te]))
tho = np.load("%s/%spot_%d/%s/Th_%d.npy"%(rut,de,nsol,carp,t[te]))
phio = np.load("%s/%spot_%d/%s/Ph_%d.npy"%(rut,de,nsol,carp,t[te]))
r0 = np.sqrt(X[:]**2 + Y[:]**2 + Z[:]**2)

#r0, tho, phio = filtro(r0, tho, phio, 150.,'mayores a')
#r0, tho, phio = filtro(r0, tho, phio, [0., 40.],'intervalo')
#r0, tho, phio = filtro(r0, tho, phio,  [100.,600.],'intervalo')
  
dataarray = tho
df = pd.DataFrame({'xlab': dataarray})

##
#################           create space             ##########################
#obs = zfit.Space("x", limits=(0., np.pi))
obs = zfit.Space("x", limits=(-10, 10))
#################          model building, pdf creation             ###########
#frac = zfit.Parameter("fraction", 0.3)#, 0., 3.)

mu = zfit.Parameter("mu", 1.5)#, -4, 6)
sigma = zfit.Parameter("sigma", 1., 0.1, 10)
gauss = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

mui = zfit.Parameter("mui", 0.1)#, 0., 1.)
sigi = zfit.Parameter("sigmai", 1., 0.1, 10)
lowi = zfit.Parameter("lowi", 0.)
highi = zfit.Parameter("highi", 1.)
truncated_left = zfit.pdf.TruncatedGauss(mui, sigi, lowi, highi, obs)

mud = zfit.Parameter("mud", 3.)#, 2, np.pi)
sigd = zfit.Parameter("sigmad", 1., 0.1, 10)
lowd = zfit.Parameter("lowd", 2.)
highd = zfit.Parameter("highd", np.pi)
truncated_right = zfit.pdf.TruncatedGauss(mud, sigd, lowd, highd, obs)

frac = zfit.Parameter("fraction", 0.5, 0.1, 1.)
frac2 = zfit.Parameter("fraction2", 0.5, 0.1, 1.)
frac3 = zfit.Parameter("fraction3", 0.5, 0.1, 1.)
sum_pdf = zfit.pdf.SumPDF([gauss,truncated_left, truncated_right], 
                          fracs=[frac, frac2, frac3])

####################             data             ############################
#dataarray = np.random.normal(loc=2., scale=3., size=10000)
data = zfit.Data.from_numpy(obs=obs, array=dataarray)
######################              create NLL             ####################
nll = zfit.loss.UnbinnedNLL(model=sum_pdf, data=data)
#nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)

minimizer = zfit.minimize.Minuit()# create a minimizer

result = minimizer.minimize(nll)

#param_errors, _ = result.errors()# do the error calculations, here with minos

print(result.params)
#print(result.valid)

mu = 1.588
sigma = 0.8326

b = (0.536795)# - muiz)/sigi
a = (0.6846)# - muiz)/sigi
muiz = 0.07374
sigi = 0.9842

b2 = (2.352)#- muder)/sigd
a2 = (2.505)# - muder)/sigd
muder = 3.014
sigd = 0.9989

frac = 0.672
fraci =9.537e-7
fracd =9.537e-7
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
df.plot.hist(density=True)
#plt.plot(x, stats.norm.pdf(x, mu, sigma))
#plt.plot(x, stats.norm.pdf(x, muiz, sigi))
#plt.plot(x, stats.norm.pdf(x, muder, sigd))
#plt.plot(x, stats.truncnorm(a, b).pdf(x))
#plt.plot(x, stats.truncnorm(a2, b2).pdf(x))
#plt.plot(x, stats.norm.pdf(x, mu, sigma) + stats.truncnorm(a, b).pdf(x) + stats.truncnorm(a2, b2).pdf(x))
plt.plot(x, frac*stats.norm.pdf(x, mu, sigma) + fraci*stats.norm.pdf(x, muiz, sigi) + fracd*stats.norm.pdf(x, muder, sigd))
plt.show()
