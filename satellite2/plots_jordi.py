#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 01:58:27 2019

@author: jordis
"""
import numpy as np
#import matplotlib.animation as pltani
#from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from matplotlib import cm, rcdefaults, rcParams
import seaborn as sns
#from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import *
import pandas as pd
import scipy.special as spe
import matplotlib.colors as mcolors
from scipy.stats import norm
import scipy.stats as sstats
#from integral import I1
#from fit_pdf import best_fit_distribution
import scipy.stats as st

def jordi_style():
    plt.style.use('default')
    rcParams['figure.figsize'] = (11, 6) # (6, 5)#(11, 6)
    rcParams['savefig.transparent'] = False
    rcParams['axes.formatter.limits'] = (-3,4)
    rcParams['axes.titlesize'] =24
    rcParams['lines.markersize'] =10
    rcParams['legend.fontsize'] =20
    rcParams['legend.framealpha'] =0.3
    rcParams['axes.labelsize'] =22
    rcParams['axes.linewidth'] =2
    rcParams['lines.linewidth'] =2
    rcParams['text.usetex'] =True
    rcParams['font.family'] ='serif'
#    rcParams['savefig.transparent'] =True
    rcParams['savefig.dpi'] =100
    rcParams['ytick.labelsize'] =22
    rcParams['xtick.labelsize'] =22

jordi_style()
#plt.style.use('ggplot')
pi = np.pi

def scater(x,y,xlab,ylab,title,ylim=(0,0),xlim=(0,0), xangular = False, 
           z3D=False, z=[],angular =False, color = False, c=[], clab = '',
           zlab='', t=[], errorbar= False, yerr= [], initialview=[45,-60],
           name = '', dpi = 250, R = 200, save = True, s = 1., extra_data=False,
           x_extra = [], y_extra= [], extra_err=False, yerr_extra=None,
           xerr_extra= None, extratext = False, texts = []):  
    if z3D==True:
        plt.figure(figsize=(10,10))
        plt.style.use('ggplot')
        ax = plt.axes(projection='3d')
#        ax.set_aspect(aspect=1.)
        if color == False:
            ax.scatter(x, y, z, marker='*', s = s)
        else:
            sc = ax.scatter(x, y,z, marker='*', s = s, c= c, cmap=cm.jet)
            plt.colorbar(sc).ax.set_ylabel(clab) 

        ax.set_zlabel(zlab,rotation=45,fontsize=20)
        ax.view_init(initialview[0],initialview[1])
        ax.set_zlim(-R, R) 
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R) 
    else:
        jordi_style()
        plt.figure(figsize=(7,4))
        ax = plt.axes()
        if errorbar==True:
            ax.errorbar(x,y,yerr=yerr, fmt='o',alpha = 0.95,
                         ms = 2.5, capsize = 2.5, elinewidth = 0.5)
        else:
            if color == False:
                ax.scatter(x, y, marker=',', s = s)
            else:
                sc = ax.scatter(x, y, marker=',', s = s, c= c, cmap=cm.jet)
                plt.colorbar(sc).ax.set_ylabel(zlab)           
    if extra_data ==True:
        if extratext == True:
            for i in range(0, np.shape(x_extra)[0]):
                ax.text(x_extra[i], y_extra[i], texts[i], fontsize=16)

        if extra_err==False:
            ax.scatter(x_extra, y_extra, marker= 'X', s = 50, c = 'r')
        else:
            ax.errorbar(x_extra, y_extra, yerr=yerr_extra, xerr = xerr_extra,
                        fmt='o',alpha = 0.95,c = 'r',
                         ms = 2.5, capsize = 2.5, elinewidth = 2.5)# linestyle="None")
  

    if angular==True:
        if ylim!=(0,0):
            plt.ylim(ylim)
        else:
            ylim= (0,np.pi)
        plt.ylim(ylim)
#        plt.yticks(np.arange(-np.pi - 0.2, np.pi+0.2 , step=np.pi/8),
#                   (r"$-\pi$",r"$-7\pi/8$",r"$-3\pi/4$",r"$-5\pi/8$",r"$-\pi/2$",r"$-3\pi/8$",r"$-\pi/4$",r"$-\pi/8$", 0,r"$\pi/8$",r"$\pi/4$",r"$3\pi/8$",r"$\pi/2$",r"$5\pi/8$",r"$3\pi/4$",r"$7\pi/8$",r"$\pi$"),
#                   fontsize=16, rotation=0)   
        plt.yticks(np.arange(-np.pi , np.pi + 0.2 , step=np.pi/4),
                   (r"$-\pi$",r"$-3\pi/4$",r"$-\pi/2$",r"$-\pi/4$", 0,r"$\pi/4$",r"$\pi/2$",r"$3\pi/4$",r"$\pi$"),
                   fontsize=16, rotation=0)
        
    if xangular == True:
        plt.xticks(np.arange(0, np.pi+0.2 , step=np.pi/3),
                   (0,r"$\pi/3$",r"$2\pi/3$",r"$\pi$"),
                   fontsize=18, rotation=0) 
    else:        
        plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    ax.set_xlabel(xlab ,fontsize=20)
    ax.set_ylabel(ylab,fontsize=20)
    plt.title(title,fontsize=20)
    if xlim!=(0,0):
        plt.xlim(xlim)
    if ylim!=(0,0):
        plt.ylim(ylim)
    if save ==True:
        plt.savefig(name, dpi=dpi, bbox_inches='tight')
    plt.show()
    jordi_style()

def multiplot_colorbar(X, Y, labels, xlab, ylab, title, nom_archivo, barlab,
                       ticks = None, ylim=(0,0)):
    
    N = np.shape(labels)[0]
    cmap = plt.get_cmap('viridis', N)#'jet',N)
    plt.figure(figsize=(11, 6))
    for i,j in enumerate(labels):
        x = X[i]
        y = Y[i]
        plt.plot(x,y,c=cmap(i))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    norm = mcolors.Normalize(vmin=labels[0],vmax=labels[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if ticks ==  None:
        plt.colorbar(sm, label = barlab).ax.tick_params(labelsize=10)
    else:
        plt.colorbar(sm, ticks=labels, label = barlab).ax.tick_params(labelsize=10)
    if ylim!=(0,0):
        plt.ylim(ylim)
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)    
    plt.title(title,fontsize=20)
    plt.savefig(nom_archivo, bbox_inches='tight')
    plt.show()
    
def plotmultiple(x, y, legends, xlabel, ylabel, title, nom_archivo, ylim=(0,0),
                 xlim=(0,0), logy= False, save = True, loc_leg='best',
                 angular=False, xangular=False, logx = False, show = True,
                 data=False, xd=[], yd=[], err=False, yerr=[], errx = False,
                 xerr = [], markersize = 20, fill_between=False, fbx=[],
                 fby1=1, fby2=0, text= '', xv=[]):
    "x=[x1,x2,x3,...]; y=[y1,y2,y3,...]; x1,x2,...,y1,y2,y3,... son arrays; xlabel,ylabel,title son strings; legends touple of strings"
    sh =['-','--','-.',':','-k','--k','-.k',':k','-b','--b','-.b',':b']
    sc = list(mcolors.XKCD_COLORS)
    np.random.seed(12555)
#    marc = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d"]
#    sh=['r','--b','r','r','r','r','r','r','r','r','--b','--b','--b','--b','--b','--b','--b','--b']
#    sh=['r','--b','r','r','r','r','--b','--b','--b','--b']
    for i in range(0, np.shape(y)[0],1):
        if logx==True:
            plt.semilogx(x[i], y[i],sh[i],markersize= 9)#,markersize= 9)
#            plt.semilogx(x[i], y[i],color = sc[np.random.randint(0, 147)],markersize= 9)#,markersize= 9)
        else:
#            plt.plot(x[i], y[i],sh[i],markersize= 9)          
            plt.plot(x[i], y[i],color = sc[np.random.randint(0, 147)],markersize= 9)
#            plt.scatter(x[i], y[i],color = sc[random.randint(0, 147)],
#                        marker = '*',s = markersize)  
#            plt.scatter(x[i], y[i], marker = 'o', s = 1, c = 'b')
    if data==True:
        if err==False:
            plt.scatter(xd, yd, marker='*', color= 'Red')
        else:
            if errx==True:
                xerr = xerr
            else:
                xerr = None
            plt.errorbar(xd, yd, yerr=yerr, xerr = xerr,fmt='o',alpha = 0.95,
                         ms = 2.5, capsize = 2.5, elinewidth = 0.5)# linestyle="None")
            
    if angular==True:
        if ylim!=(0,0):
            plt.ylim(ylim)
        else:
            ylim= (0,np.pi)
        plt.ylim(ylim)
        plt.yticks(np.arange(0, np.pi+0.2 , step=np.pi/8),
                   (0,r"$\pi/8$",r"$\pi/4$",r"$3\pi/8$",r"$\pi/2$",r"$5\pi/8$",r"$3\pi/4$",r"$7\pi/8$",r"$\pi$"),
                   fontsize=16, rotation=0)    
    if ylim!=(0,0):
        plt.ylim(ylim)
    if xlim!=(0,0):
        plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logy == True :
        plt.yscale('log')
    if xangular == True:
        plt.xticks(np.arange(0, np.pi+0.2 , step=np.pi/8),
                   (0,r"$\pi/8$",r"$\pi/4$",r"$3\pi/8$",r"$\pi/2$",r"$5\pi/8$",r"$3\pi/4$",r"$7\pi/8$",r"$\pi$"),
                   fontsize=16, rotation=0)
    else:      
        plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.title(title,fontsize=20)
    plt.legend(legends, loc= loc_leg, frameon = False)
    if text != '':
        plt.text(0., 0., text, size=20, rotation=30., 
                 ha = "left", va = "bottom", bbox = dict(boxstyle="round", 
                                                         alpha = 0.5,
                                                           ec=(1., 0.5, 0.5), 
                                                           fc=(1., 0.8, 0.8),))
#        plt.text(np.amin(xd), np.amax(yd)/2., text, size=20, rotation=30., 
#                 ha = "left", va = "bottom", bbox = dict(boxstyle="round", 
#                                                         alpha = 0.5,
#                                                           ec=(1., 0.5, 0.5), 
#                                                           fc=(1., 0.8, 0.8),))
    if fill_between==True:
        plt.fill_between(fbx, fby1, y2=fby2, alpha=.25, label='5-sigma interval')
    if xv !=[]:
        for j in range(0, np.shape(xv)[0],1):
            plt.axvline(xv[j], color = sc[j] , ls='-.')
    if save == True:
        plt.savefig(nom_archivo, bbox_inches='tight')
    if show == True:
        plt.show()
    plt.close()

def residual(data, fit, datalabel=r'$v$(km/s)', lowess=True):
    plt.figure(figsize= (11, 3))
    sns.residplot(data, fit, lowess=lowess, color="g")
    plt.xlabel(datalabel)
    plt.ylabel(r'Residuals')
    plt.show()
    
def histo(data,xlab, bins = 120, rang=(0,0),nom_archivo ='', fit = False,
          dist = 'norm', normalized = False, title='',
          logx = False, dens = False, xangular =False, xv =[]):
    print(data.shape)
    data = data[~np.isnan(data)]# we use the logical-not operator ~
    data = data[~np.isinf(data)]# we use the logical-not operator ~  
    print(data.shape)
    from scipy import stats   
    df = pd.DataFrame({xlab: data})
    if rang!=(0,0):
        n , bins, patches = plt.hist(data, bins=bins, align ='mid',
                                     range=[rang[0], rang[1]])
    else:
        n, bins, patches = plt.hist(data, bins=bins, align ='mid')
    moda = stats.mode(data, axis=None)
    print('moda',moda)  
    print(df.describe())
    plt.close()
    plt.figure(figsize=(6,5))
    if dens == True:
        df.plot.density(logx=logx)    
    else:
        df.plot.hist(bins=bins, logx=logx, density=normalized)
    plt.xlabel(xlab, size=22 ) 
    plt.title(title,fontsize=20)
    if xv !=[]:
        sc = list(mcolors.XKCD_COLORS)
        for j in range(0, np.shape(xv)[0],1):
            plt.axvline(xv[j], color = sc[j] , ls='-.')
    if fit==True:
#        best_fit_name, best_fit_params = best_fit_distribution(data, bins)
#        best_dist = getattr(st, best_fit_name)
#        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
#        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
#        dist_str = '{}({})'.format(best_fit_name, param_str)
#        plt.title(dist_str,fontsize=20)
#        arg = best_fit_params[:-2]
#        loc = best_fit_params[-2]
#        scale = best_fit_params[-1]
#        best_fit_line = best_dist.pdf(bins, loc=loc, scale=scale, *arg)
        if dist == 'norm':      
            pars= norm.fit(data)
#            best_fit_line = norm.pdf(bins, *pars)* sum(n * np.diff(bins))
            best_fit_line = norm.pdf(bins, *pars)
            plt.title(r'$\mu=$%f, $\sigma=%f$'%pars,fontsize=20)
        elif dist == 'weibull':
            pars = sstats.weibull_min.fit(data)
            print(pars)
            best_fit_line = sstats.weibull_min.pdf(bins, *pars)
            plt.title('weibull %f,%f,%f'%pars,fontsize=20)
        elif dist == 'dweibull':
            pars = sstats.dweibull.fit(data, floc = np.pi/2.)##mean, var, skew, kurt
            print(pars)
            best_fit_line = sstats.dweibull.pdf(bins, *pars)
            plt.title('dweibull $c = $%f, %f,%f'%pars,fontsize=20)
        elif dist == 'alpha':
            pars = sstats.alpha.fit(data)##mean, var, skew, kurt
            print(pars)
            best_fit_line = sstats.alpha.pdf(bins, *pars)
            plt.title('alpha %f,%f,%f'%pars,fontsize=20)            
       
        plt.plot(bins, best_fit_line)
        
    if xangular == True:
        plt.xticks(np.arange(0, np.pi+0.2 , step=np.pi/8),
                   (0,r"$\pi/8$",r"$\pi/4$",r"$3\pi/8$",r"$\pi/2$",r"$5\pi/8$",r"$3\pi/4$",r"$7\pi/8$",r"$\pi$"),
                   fontsize=20, rotation=0)
    if nom_archivo != '':
        plt.savefig(nom_archivo, bbox_inches='tight')
    plt.show()
    plt.close()

def histo2d(datax, datay, xlab, ylab, bins = '',density= False,
            nom_archivo ='', fit = True, cmin = None, cmax = None):
    df = pd.DataFrame({xlab: datax, ylab: datay})
    if bins == '':
        N, binsx, binsy =plt.hist2d(datax, datay, density = density, cmin=cmin,
                                    cmax=cmax)
    else:
        plt.hist2d(datax, datay, bins = [bins[0], bins[1]], density = density,
                   cmin=cmin, cmax=cmax) 
    plt.xlabel(xlab, size=22)
    plt.ylabel(ylab, size=22)
    plt.colorbar()
    if nom_archivo != '':
        plt.savefig(nom_archivo, bbox_inches='tight')
    plt.show()        
    plt.close()
    print(df.describe())       
    
def coordsplotang(x,y,xlab,ylab,title,nom_archivo, ylim=(0,np.pi)):
    "x1 y t son arrays,x2label,title son strings"
    plt.plot(x, y,'b-')
    plt.grid()
    plt.xlabel(xlab)#,fontsize=16)
    plt.ylabel(ylab)
    if ylim==(0,np.pi):
        plt.ylim(ylim)
        plt.yticks(np.arange(0, np.pi+0.2 , step=np.pi/8),
                   (0,r"$\pi/8$",r"$\pi/4$",r"$3\pi/8$",r"$\pi/2$",r"$5\pi/8$",r"$3\pi/4$",r"$7\pi/8$",r"$\pi$"),
                   fontsize=16, rotation=0)
    else:        
        plt.ylim(ylim)
        plt.yticks(np.arange(0, 2*np.pi+0.5 , step=np.pi/4),
                   (0,r"$\pi/4$",r"$\pi/2$",r"$3\pi/4$",r"$\pi$",r"$5\pi/4$",r"$3\pi/2$",r"$7\pi/4$",r"$2\pi$"),
                   fontsize=16, rotation=0)
    plt.title(title, fontsize=20, color='black')
    plt.savefig(nom_archivo, bbox_inches='tight')
    plt.show()  
 
#def plot_mollweide(th, phi, s = 1, amp = False,amplitudes = [], name = '', 
#                   title = '' ):
#    fig = plt.figure(figsize=(8,5))
#    ax = fig.add_subplot(111, projection="aitoff")
#    ax.grid(True)
#    if amp == True:    
#        sc = ax.scatter(phi, np.pi/2. - th, s = s, c=amplitudes)
#        plt.colorbar(sc)
#    else:    
#        ax.scatter(phi, np.pi/2. - th, s = s)
#    plt.savefig(name, dpi=100, bbox_inches='tight')
#    plt.title(title, fontsize=20, color='black')
#    plt.show()
    
def parametricplot(x,y,xlabel,ylabel,title,nom_archivo,ylim=(0,0), logy= False,
                   save = True, data=False, xd=[],yd=[],yerr=False, ls = '-',
                   legends = '', show = True):
    "x e y son arrays. xlabel,ylabel,title son strings"
    print('use plotmultiple()')
        
def plot3d(x,y,z,t,xlab,ylab,zlab,grad_elev,grad_azimut,title,nom_archivo,
           galaxy= True, DM = False, R = 0, save = True,legend = []):
    fig=plt.figure(figsize=(13.3,10))
    ax = plt.axes(projection='3d')
    #     Plot the galaxy
    if galaxy==True:
        if R!=0:
            rad = R/5 #50 #real 50kpc
            rad2 = R/500 #0.6 #real 0.6kpc
        else:
            rad = 50.
            rad2= 0.6
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xb = rad * np.outer(np.cos(u), np.sin(v))# mu*rad * np.outer(np.cos(u), np.sin(v))
        yb = rad * np.outer(np.sin(u), np.sin(v))#mu*rad * np.outer(np.sin(u), np.sin(v))
        zb = rad2 * np.outer(np.ones(np.size(u)), np.cos(v))#mu*rad2 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xb, yb, zb, cmap='viridis')#cmap=cm.coolwarm)
#   Plot the DM
    if DM == True:
        th, phi = np.meshgrid(np.linspace(0, pi, 91),np.linspace(0, 2*pi, 181))
        xyz = np.array([np.sin(th)*np.sin(phi),np.sin(th)*np.cos(phi),
                        np.cos(th)]) 
        l=1
        m=0
        Y_lm = spe.sph_harm(m,l, phi, th)
#        ro = ckpc**2*(3.*w**2 - mu**2)/(16*pi)
        ro = 100
        r = abs(Y_lm.real)*ro*xyz
        ax.scatter(r[0], r[1], r[2], marker='_')
#   Plot the trayectory
    ax.scatter(x, y, z, marker='*', color= 'Red')
#    ax.scatter3D(x, y, z, c=-t, cmap='Blues');
    plt.legend(legend, loc= 'lower left')
    plt.title(title,fontsize=20)
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    ax.set_xlabel(xlab)#,fontsize=16)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab,rotation=45)   
    if R!=0:
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_zlim(-R, R)
#        xs = [17.1, -0.6, 16.5, -43., -22.2, -5.2, -36.7, -25.0, -41.3, -77.3, -123.6]
#        ys = [2.5, -41.8, -38.5, 62.2, 52.0, -9.8, -56.9, -95.9, -51.0, -58.3, -119.3]
#        zs = [-6.4, -27.5, -44.7, 43.2, 53.5, -85.3, 57.8, -39.8, -134.1, 215.2, 191.7]
#        pos = np.array([xs,ys,zs])
#        pos = pos/215.2*R    
#        ax.scatter(pos[0], pos[1], pos[2], marker='*', color= 'Red');
#    ax.text3D(0, 0, 0, 'Galaxy', zdir=None, fontsize='xx-large',
#              fontweight='bold', bbox=dict(facecolor='red', alpha=0.5))    
    ax.view_init(grad_elev,grad_azimut)# elevation of grad_elev degrees (that is, grad_elev degrees above the x-y plane) and an azimuth of grad_azimut degrees (that is, rotated grad_azimut degrees counter-clockwise about the z-axis)
    if save == True:
        plt.savefig(nom_archivo, bbox_inches='tight')
    plt.show()

def plotfunc3d(x1,x2,f,xlabel,ylabel,zlabel,title,name= None,elev=45,
               rot_azimut=-60):
    "Plot de superficie x1 y x2 son de la forma x1, x2 = np.meshgrid(np.linspace(limx_inf,limx_sup, num_puntos), np.linspace(limy_inf,limy_sup, num_puntos)) y f debe ser f(x1,x2)"
    plt.figure(figsize=(13.3,10))
    ax = plt.axes(projection='3d')
    ax.contour3D(x1, x2, f)
    ax.plot_surface(x1, x2, f, rstride=1, cstride=1, cmap='viridis',
                    edgecolor='none')
    plt.title(title,fontsize=20)
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel,rotation=45)
#    ax.set_zlim(-0.175, 0.175)
    ax.view_init(elev,rot_azimut)# elevation of grad_elev degrees (that is, grad_elev degrees above the x-y plane) and an azimuth of grad_azimut degrees (that is, rotated grad_azimut degrees counter-clockwise about the z-axis)
    if name == None:
        plt.show()
    else:
        plt.savefig(name, dpi=200, bbox_inches='tight')
        plt.show()
        
def densityplot(x1,x2,f,xlabel,ylabel,zlabel,title,name=None, aspect='1/1',
                log = False, rang = False, rango = [0.01, 1.], show = True):
    " Density plot where x1 y x2 son de la forma x1, x2 = np.meshgrid(np.linspace(limx_inf,limx_sup, num_puntos), np.linspace(limy_inf,limy_sup, num_puntos))"" y f debe ser f(x1,x2)"
    if aspect == '1/2':
        plt.figure(figsize=(6.5, 10))
    elif aspect == '1/1':
        plt.figure(figsize=(6, 5))       
    if log==True:
        import matplotlib.colors as colors
        plt.pcolormesh(x1, x2, f,
                       norm=colors.LogNorm(vmin=rango[0], vmax=rango[1]))
    else:
        if rang == True:
            plt.pcolormesh(x1, x2, f, vmin = rango[0], vmax = rango[1])
        else:
            plt.pcolormesh(x1, x2, f)   
    plt.title(title,fontsize=20)
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar().ax.set_ylabel(zlabel)
    if name == None:
        plt.show()
    else:
        if show == True:
            plt.savefig(name, dpi=200, bbox_inches='tight')
            plt.show()
        else:
            plt.savefig(name, dpi=200, bbox_inches='tight')

def densityplot2(x1, x2, f, x3, f1, f2, xlabel, ylabel, zlabel,
                 extralabelx, extralabely, title, name=None, aspect='1/1',
                log = False, rang = False, rango = [0.01, 1.], show = True,
                extralog = False):
    " Density plot where x1 y x2 son de la forma x1, x2 = np.meshgrid(np.linspace(limx_inf,limx_sup, num_puntos), np.linspace(limy_inf,limy_sup, num_puntos))"" y f debe ser f(x1,x2)"
    if aspect == '1/2':
        fig = plt.figure(figsize=(8, 6))
        ax = [plt.subplot2grid((1, 3), (0, 0), colspan=2),
              plt.subplot2grid((1, 3), (0, 2))]
    elif aspect == '1/1':
        fig = plt.figure(figsize=(8, 10))
        ax = [plt.subplot2grid((11, 1), (0, 0), rowspan=7),
              plt.subplot2grid((11, 1), (8, 0), rowspan=2)]
    if log==True:
        import matplotlib.colors as colors
        pcm1 = ax[0].pcolormesh(x1, x2, f,
                 norm=colors.LogNorm(vmin=rango[0], vmax=rango[1]))
    else:
        if rang == True:
            pcm1 = ax[0].pcolormesh(x1, x2, f, vmin = rango[0], vmax = rango[1])
        else:
            pcm1 = ax[0].pcolormesh(x1, x2, f)
    ax[1].plot(x3, f1)
    ax[1].plot(x3, f2)
    if extralog ==True:
        ax[1].set_yscale('log')
    ax[0].set_title(title,fontsize=20)
    ax[1].set_xlabel(extralabelx)
    ax[1].legend(extralabely, loc= 'best')
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    fig.colorbar(pcm1, ax=ax[0], extend='max').ax.set_ylabel(zlabel)
    if name == None:
        plt.show()
    else:
        if show == True:
            plt.savefig(name, dpi=200, bbox_inches='tight')
            plt.show()
        else:
            plt.savefig(name, dpi=200, bbox_inches='tight')
                   


def plotmultiple3(x,y,z,title,nom_archivo,Rho=[],Z=[], fi=[],R=2,
                  zlabel=''):    
    plt.figure(figsize=(5.5, 5))
    ax = plt.axes()
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    ax.set_xlabel(r'$x$(kpc)')#,fontsize=16)
    ax.set_ylabel(r'$z$(kpc)')
    plt.title(title,fontsize=20)
    
#    pcm=ax.pcolormesh(Rho, Z, fi)#,norm=colors.LogNorm(vmin=fi.min(), vmax=fi.max()))
#    ax.colorbar(pcm, ax=ax[0], extend='max').ax.set_ylabel(zlabel)

    ax.scatter(x, z, marker=',', s = 1)
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.grid(True)
    ax.set_title(title,fontsize=20)
    plt.savefig(nom_archivo, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(5.5, 5))
    ax = plt.axes()
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    ax.set_xlabel(r'$y$(kpc)')#,fontsize=16)
    ax.set_ylabel(r'$z$(kpc)')
    plt.title(title,fontsize=20)    

#    pcm=ax.pcolormesh(Rho, Z, fi)#,norm=colors.LogNorm(vmin=fi.min(), vmax=fi.max()))
#    ax.colorbar(pcm, ax=ax[0], extend='max').ax.set_ylabel(zlabel)
    ax.scatter(y, z, marker=',', s = 1)
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.grid(True)
    ax.set_title(title,fontsize=20)
    plt.savefig(nom_archivo+'_2', bbox_inches='tight')    
    plt.show()
    
def scater3d(x,y,z):
    print("use scater()")
    
def plotmultiple2(x,y,z,title,nom_archivo,Rho=[],Z=[], fi=[],R=2, units =False,                  zlabel=''): 
    if units == False:
        figsize = (10.5,5.5)
    else :
        figsize = (12.5,5.5)
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=figsize)
    sh = list(mcolors.XKCD_COLORS)
    import matplotlib.colors as colors
    if Rho!=[]:
        print('0:')
        pcm=ax[0].pcolormesh(Rho, Z, fi)#,norm=colors.LogNorm(vmin=fi.min(), vmax=fi.max()))
#        pcm=ax[0].pcolormesh(Rho, Z, fi,norm=colors.LogNorm(vmin=0.1, vmax=1))#,
#              cmap='PuBu_r')
        fig.colorbar(pcm, ax=ax[0], extend='max')
        pcm1= ax[1].pcolormesh(Rho, Z, fi)
#        pcm1= ax[1].pcolormesh(Rho, Z, fi,norm=colors.LogNorm(vmin=0.1, vmax=1))
        if units==True:        
            fig.colorbar(pcm1, ax=ax[1], extend='max').ax.set_ylabel(zlabel)

    ax[0].set_xlim(-R, R)
    ax[0].set_ylim(-R, R)
    ax[0].grid(True)
    if units ==True:    
        ax[0].set_xlabel("$x$(kpc)")
        ax[0].set_ylabel("$z$(kpc)")
    else:
        ax[0].set_xlabel("$\hat{\mu}x$")
        ax[0].set_ylabel("$\hat{\mu}z$")
    for i in range(0, np.shape(y)[0],1):
        ax[0].scatter(x[i], z[i], marker='*', color= 'red')#sh[random.randint(0, 147)])
    ax[1].set_xlim(-R, R)
    ax[1].set_ylim(-R, R)
    ax[1].grid(True)
    if units ==True: 
        ax[1].set_xlabel("$y$(kpc)")
    else:
        ax[1].set_xlabel("$\hat{\mu}y$")
    for i in range(0, np.shape(y)[0],1):
        ax[1].scatter(y[i], z[i], marker='*', color=  'red')#sh[np.random.randint(0, 147)])
    ax[0].set_title(title,fontsize=20)
    plt.savefig(nom_archivo, bbox_inches='tight')
    plt.show()

def multiplot3d(x,y,z,legends,xlab,ylab,zlab,grad_elev,grad_azimut,title,nom_archivo, R = 0):
    fig=plt.figure(figsize=(13.3,10))
    ax = plt.axes(projection='3d')
#    sh = list(mcolors.CSS4_COLORS)
    sh = list(mcolors.XKCD_COLORS)
#    sh =['Red','Blue','Green','Black','Yellow', 'Cyan', 'Magenta']
#   Plot the trayectory
    for i in range(0, np.shape(y)[0],1):
        ax.scatter(x[i], y[i], z[i], marker='*', color= sh[np.random.randint(0, 147)])
    
    plt.title(title,fontsize=20)
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    ax.set_xlabel(xlab)#,fontsize=16)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab,rotation=45)   
    if R!=0:
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_zlim(-R, R)
    ax.view_init(grad_elev,grad_azimut)# elevation of grad_elev degrees (that is, grad_elev degrees above the x-y plane) and an azimuth of grad_azimut degrees (that is, rotated grad_azimut degrees counter-clockwise about the z-axis)
    plt.legend(legends, loc= 'best')
    plt.savefig(nom_archivo, bbox_inches='tight')
    plt.show()
    
def sphericalplot3D_harmonics(l,m,grad_elev=20,grad_azimut=45):    
    plt.figure(figsize=(13.3,10))
    ax = plt.axes(projection='3d')
    th, phi = np.meshgrid(np.linspace(0, np.pi, 191),np.linspace(0, 2*np.pi, 181))
    xyz = np.array([np.sin(th)*np.sin(phi),np.sin(th)*np.cos(phi),
                    np.cos(th)]) 
#    Y_lm = spe.legendre(l)(np.cos(th))
    ro = 1.
    r = abs(spe.sph_harm(m,l, phi, th).real)*ro*xyz
    ax.scatter(r[0], r[1], r[2], marker='_')
    ax.view_init(grad_elev,grad_azimut)
    plt.show()

def sphericalplot3D(f, ro,grad_elev=20,grad_azimut=45):
    pi = np.pi
    plt.figure(figsize=(13.3,10))
    ax = plt.axes(projection='3d')
    th, phi = np.meshgrid(np.linspace(0, pi, 191),np.linspace(0, 2*pi, 181))
    xyz = np.array([np.sin(th)*np.sin(phi),np.sin(th)*np.cos(phi),
                    np.cos(th)]) 
    r = abs(f(ro,th))*ro*xyz
    ax.scatter(r[0], r[1], r[2], marker='_')
    ax.view_init(grad_elev,grad_azimut)
    plt.show()    