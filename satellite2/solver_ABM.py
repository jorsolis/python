#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def resol_ABM(func, y0, t, ti, tf, d):
    NP = int((tf-ti)/d)
       
    def rk4(func, y_0, t):
        # Inicia el arreglo de las aproximaciones
        y = np.zeros([4, len(y_0)])
        y[0] = y_0
        for i, t_i in enumerate(t[:3]): #Revisar el contenido del enumerate
    
            h = d #t[i+1] - t_i
            k_1 = func(t_i, y[i])
            k_2 = func(t_i+h/2., y[i]+h/2.*k_1)
            k_3 = func(t_i+h/2., y[i]+h/2.*k_2)
            k_4 = func(t_i+h, y[i]+h*k_3)
    
            y[i+1] = y[i] + h/6.*(k_1 + 2.*k_2 + 2.*k_3 + k_4) # RK4 step
    
        return y
     
    #Adams-Bashforth 4/Moulton 4 Step Predictor/Corrector
    def ABM3(func, y_0, t):
        y = np.zeros([len(t), len(y_0)])
    	#Se calcularan los primeros pasos con rk4
        y[0:4] = rk4(func,y_0, t)
        k_1 = func(t[2], y[2])
        k_2 = func(t[1], y[1])
        k_3 = func(t[0], y[0])
        for i in range(3,NP-1):
            h = d
            k_4 = k_3
            k_3 = k_2
            k_2 = k_1
            k_1 = func(t[i], y[i])
            #Adams Bashforth predictor
            y[i+1] = y[i] + h*(55.*k_1 - 59.*k_2 + 37.*k_3 - 9.*k_4)/24.
            k_0 = func(t[i+1],y[i+1])
            #Adams Moulton corrector
            y[i+1] = y[i] + h*(9.*k_0 + 19.*k_1 - 5.*k_2 + k_3)/24.
        return y 
        
#    V, P, S, F, E = ABM3(RHS, y0, t).T
    return ABM3(func, y0, t)
#    return ABM3(func, y0, t)
