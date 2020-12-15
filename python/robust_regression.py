#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 22:26:01 2018
@author: https://scipy-cookbook.readthedocs.io/items/robust_regression.html
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
#
rcParams['figure.figsize'] = (10, 6)
rcParams['legend.fontsize'] = 16
rcParams['axes.labelsize'] = 16
rcParams['text.usetex'] = True  # or plt.rc('text', usetex=True)
rcParams['font.family'] = 'serif' #or plt.rc('font', family='serif')
#
r = np.linspace(0, 5, 100)
#
linear = r**2
#
huber = r**2
huber[huber > 1] = 2 * r[huber > 1] - 1
#
soft_l1 = 2 * (np.sqrt(1 + r**2) - 1)
#
cauchy = np.log1p(r**2)
#
arctan = np.arctan(r**2)
#   Plot
plt.plot(r, linear, label='linear')
plt.plot(r, huber, label='huber')
plt.plot(r, soft_l1, label='soft l1')
plt.plot(r, cauchy, label='cauchy')
plt.plot(r, arctan, label=r'$\arctan$')
plt.xlabel('r')
plt.ylabel(r'$\rho(r^2)$')
plt.legend(loc='upper left');
#
def generate_data(t, A, sigma, omega, noise=0, n_outliers=0, random_state=0):
    y = A * np.exp(-sigma * t) * np.sin(omega * t)
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 35
    return y + error
#Define model parameters:
A = 2 #amplitud
sigma = 0.1 #fase
omega = 0.1 * 2 * np.pi #frecuencia
x_true = np.array([A, sigma, omega])
noise = 0.1
t_min = 0
t_max = 30
#Data for fitting the parameters will contain 3 outliers:
t_train = np.linspace(t_min, t_max, 30)
y_train = generate_data(t_train, A, sigma, omega, noise=noise, n_outliers=4)
#
#Define the function computing residuals for least-squares minimization:
def fun(x, t, y):
    return x[0] * np.exp(-x[1] * t) * np.sin(x[2] * t) - y
#Use all ones as the initial estimate.
x0 = np.ones(3)
from scipy.optimize import least_squares
#
#Run standard least squares:
res_lsq = least_squares(fun, x0, args=(t_train, y_train))
#Run robust least squares with loss='soft_l1', set f_scale to 0.1 
#which means that inlier residuals are approximately lower than 0.1.
res_robust = least_squares(fun, x0, loss='soft_l1', f_scale=0.1,
                           args=(t_train, y_train))
#Define data to plot full curves.
t_test = np.linspace(t_min, t_max, 300)
y_test = generate_data(t_test, A, sigma, omega)
#Compute predictions with found parameters:
y_lsq = generate_data(t_test, *res_lsq.x)
y_robust = generate_data(t_test, *res_robust.x)
fig = plt.figure()
plt.plot(t_train, y_train, 'o', label='data')
plt.plot(t_test, y_test, label='true')
plt.plot(t_test, y_lsq, label='lsq')
plt.plot(t_test, y_robust, label='robust lsq')
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.legend();