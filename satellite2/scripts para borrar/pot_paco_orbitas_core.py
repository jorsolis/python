#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:01:22 2019

@author: jordi
"""
import numpy as np
from scipy.integrate import solve_ivp
import plots_jordi as pts
from pot_paco import DE
#from plots_orbitas_schrodinger_poisson import (plotscoordscyl, los2plots3dcyl)
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from pot_paco_orbitas_analisis import filtro    

nsol = 1

tol = 0.01
Rf = 10

ruta = '/home/jordi/satellite/schrodinger_poisson_shooting_N'

#ruta = '/home/jsolis/satellite/schrodinger_poisson/schrodinger_poisson_shooting_N'


nodes = 0
drho = np.load("%s/derrho_%d.npy"%(ruta,nodes))
dz = np.load("%s/derz_%d.npy"%(ruta,nodes))

r0 = 0.01
rf = 10.
rho = np.linspace(r0, rf, 200)
z = np.linspace(-rf ,rf, 400)

derrho = interp2d(rho, z, drho, kind='linear', copy=True, bounds_error=False)
derz = interp2d(rho, z, dz, kind='linear', copy=True, bounds_error=False)


def principal(Rf, tiem, ncor, y0a=[], plotting=True,
              direct = "/home/jordi/satellite/schrodinger_poisson/potpaco",
              k = 1):
  
    def resolvedor(u0,uf,y0,method,teval,L,ncor, conds):
        #method= 'RK45', 'Radau' or 'LSODA'        
        def event(t,y):
            vrho, rho, vz, z, phi = y
            return z

        def fun(t,y):
            vrho, rho, vz, z, phi = y
            return [L**2/rho**3 - derrho(rho,z),
                    vrho,
                    - derz(rho,z),
                    vz,
                    L/rho**2]   
        sol = solve_ivp(fun, [u0,uf], y0, method=method, t_eval=teval,
                            events=event, dense_output=True)
        if sol.status==0:
#            if plotting ==True:
#                plotscoordscyl(sol.t, sol.y[1], sol.y[3], sol.y[4], sol.y[0], sol.y[2],
#                               uf, "%s/orbitas"%(direct), ncor,
#                               labcond % conds)
#                los2plots3dcyl(sol.y[1], sol.y[3], sol.y[4], sol.t,
#                               "%s/orbitas/%d"%(direct,k), ncor,
#                               labcond % conds, Rf, galaxia=False, MO=False)
            return [sol.t, sol.y[1], sol.y[3],sol.y[4], sol.y[0], sol.y[2]]
        else:
            print("status", sol.status)
            f= open("%s/orbitas_random_vel/%d/error_%d.txt"%(direct,k, ncor),"w+")
            f.write(" No integro  %d \r\n " %(ncor))
            f.write(" condiciones iniciales v_rho= %f, rho = %f, vz = %f, z = %f, phi = %f  \r\n " %(y0[0], y0[1], y0[2], y0[3], y0[4]))
            f.close()
            ceros = np.zeros(1000)
            return [ceros, ceros,ceros, ceros, ceros, ceros]

#    labcond = r"$l\hat{\mu}/c=$ %.5f, $v_\rho(0)=$ %.5f $c$, $\hat{\mu}\rho(0)=$ %.1f, $v_z(0)=$ %.5f $c$,\
#    $\hat{\mu} z(0)=$ %.2f"
    
    du = tiem/1000
    uev = np.arange(0., tiem, du)

    vphi0= y0a[5]
    rho0 = y0a[1]
    L = abs(rho0*vphi0)
    conds = (L, y0a[0], rho0, y0a[2], y0a[3])
    y0aa = [y0a[0], rho0, y0a[2], y0a[3], y0a[4]]#vrho, rho, vz, z, phi
#    print(y0aa)
#    print('l=' ,L)
    Tt, Rrho, Zz, Pphi , Vvrho, Vvphi = resolvedor(0,tiem,y0aa,'RK45',uev,L,ncor, conds)
    
    return Tt, Rrho, Zz, Pphi, Vvrho, Vvphi

rut = "/home/jordi/satellite/schrodinger_poisson/potpaco/baja_dens/potcore"

#rut = "/home/jsolis/satellite/schrodinger_poisson/potpaco/baja_dens/potcore"

DU = {2:{'t':600, 'rf':8}}#para boson star shooting
###############################################################################
###############        Random positions & velocities        ###################
###############################################################################

nsol = 2
rf = DU[nsol]['rf']
tiemp= DU[nsol]['t'] * 4.

ncor = 1
nump= 1000
ri  = 0.1 
vesc = np.sqrt(0.5) # = sqrt(2 N(8)/8) 
#c = 2.99e5 #km/s


#print('vescape/c= ', vesc)
#######           Escala de tiempo       ######################################
#vr0 = vesc/2.
#vt0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
#vf0 = np.random.uniform(0.,2.*np.pi)    
#vrho0 = vr0*np.sin(vt0)*np.cos(vf0)
#vphi0 = vr0*np.sin(vt0)*np.sin(vf0)
#vz0 = vr0*np.cos(vt0)
#tiemp = tiemp/20.
#vphi0 = vesc/2.
#y0= [0., 4., 0., 0., 0., vphi0] #vrho, rho, vz, z, phi, vphi
#k = 1
#t, rho, z, phi, vrho, vphi = principal(rf, tiemp, ncor,
#                                       y0a=y0, plotting = True,
#                                       k = k,direct = rut)
###############################################################################
#x00 = []
#y00 = []
#z00= []


#vrh=[]
#vph=[]  
#vz = []
#
#da2 = float(input(''))
#k = int(da2)
####
###for k in range(1,101,1):
#
#print(k)
#ncor = 1        
#for i in range(1,nump,1):
#########      esfera de posiciones aleatorias de radio rf 
#    r0 = np.random.uniform(ri, rf)
#    th0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
#    phi0 = np.random.uniform(0.,2.*np.pi) 
#
#########      posiciones iniciales
#    rho0 = r0*np.sin(th0)
#    z0 = r0*np.cos(th0)
#    
#    x0 = r0*np.sin(th0)*np.cos(phi0)
#    y0 = r0*np.sin(th0)*np.sin(phi0)
#########      esfera de velocidades aleatorias de radio vrf  
#    vrf = vesc/2.
#    vr0 = np.random.uniform(0., vrf)
#    vt0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
#    vf0 = np.random.uniform(0.,2.*np.pi)    
##    ########      velocidades iniciales
#    vrho0 = vr0*np.sin(vt0)*np.cos(vf0)
#    vphi0 = vr0*np.sin(vt0)*np.sin(vf0)
#    vz0 = vr0*np.cos(vt0)
##        vrh.append(vrho0)
##        vph.append(vphi0)
##        vz.append(vz0)
##        print(rlam*vrho0*c, rlam*vphi0*c ,rlam*vz0*c)
##        print(rlam*np.sqrt(vrho0**2 + vphi0**2 + vz0**2)*c )
########      con1diciones  iniciales
#    y0= [vrho0, rho0, vz0, z0, phi0, vphi0]
########      Programa   
#    t, rho, z, phi, _, _ = principal(rf, tiemp, ncor, y0a=y0,
#                                     plotting = False,k = k, direct = rut)
#
#    x= rho*np.cos(phi)
#    y= rho*np.sin(phi)
#    cord= np.array([x,y,z])
##    cord_cyl= np.array([rho,z,phi,vrho,vphi])
#    np.save("%s/orbitas_random_vel/%d/cords_%d.npy"%(rut,k,ncor), cord)
##    np.save("%s/orbitas_random_vel/%d/cords_cyl_%d.npy"%(rut,k,ncor), cord_cyl)
##    np.save("%s/orbitas_random_vel/%d/tiemp_%d.npy"%(rut,k,ncor), t)
#    ncor += 1
#        
###############################################################################
#####################     Plot initial positions     ####################################
###########################################################################
def initial_pos_plot3D(te=950):
    carp = 'orbitas_random_vel'
    t =  np.load("%s/%s/1/tiemp_1.npy"%(rut,carp))
    x00 = np.load("%s/%s/X_%d.npy"%(rut, carp, t[te]))
    y00 = np.load("%s/%s/Y_%d.npy"%(rut, carp, t[te]))
    z00 = np.load("%s/%s/Z_%d.npy"%(rut, carp, t[te])) 
    
    pts.scater(x00,y00,r'$x$',r'$y$','Initial positions', z3D=True, z=z00,
               zlab=r'$z$', initialview=[0,45]) #initialview=[45,-60])

def mptodos(te=0, carp = 'orbitas_random_vel', kmax = 101):
    mue = 25
    mu = DE[2][mue]['mu']
    rlam = DE[2][mue]['rlam']
    lanp = 100./10.
    xo2= []
    yo2= []
    zo2= []
#    tho= []
#    phio= []
    t =  np.load("%s/%s/1/tiemp_1.npy"%(rut,carp))
    print(te, t[te])
    for k in range(1,kmax,1):
        for ncor in range(1,1000 ,1):
            xi,yi,zi =  np.load("%s/%s/%d/cords_%d.npy"%(rut,carp,k, ncor))      
            xo2.append(lanp*xi[te]/(rlam*mu))
            yo2.append(lanp*yi[te]/(rlam*mu))
            zo2.append(lanp*zi[te]/(rlam*mu))
#            if abs(zi[te])< 1e-8 :
#                thi = np.pi/2.
#            else:
#                thi = np.arccos(zi[te] / np.sqrt(xi[te]**2 + yi[te]**2 + zi[te]**2))
#            phii = np.arctan2(yi[te],xi[te])
#            tho.append(thi)
#            phio.append(phii)
    X2, Y2, Z2 = np.array(xo2), np.array(yo2), np.array(zo2)
#    TH, Ph = np.array(tho), np.array(phio)   
    np.save("%s/%s/X_%d.npy"%(rut, carp, t[te]), X2)
    np.save("%s/%s/Y_%d.npy"%(rut, carp, t[te]), Y2)
    np.save("%s/%s/Z_%d.npy"%(rut, carp, t[te]), Z2)
#    np.save("%s/%s/Th_%d.npy"%(rut, carp, t[te]), TH)
#    np.save("%s/%s/Ph_%d.npy"%(rut, carp, t[te]), Ph)
#
def mptodos_plot(te=0, carp = 'orbitas_random_vel', numpa = 99900):    
    print(te)
    t =  np.load("%s/%s/1/tiemp_1.npy"%(rut,carp))
    X = np.load("%s/%s/X_%d.npy"%(rut, carp, t[te]))
    Y = np.load("%s/%s/Y_%d.npy"%(rut, carp, t[te]))
    Z = np.load("%s/%s/Z_%d.npy"%(rut, carp, t[te])) 
    mue = 25
    mu = DE[2][mue]['mu']
    rlam = DE[2][mue]['rlam']
    lanp = 100./10.    
    tho = np.load("%s/%s/Th_%d.npy"%(rut, carp, t[te]))
    phio = np.load("%s/%s/Ph_%d.npy"%(rut, carp, t[te]))
    tho = tho[:numpa]
    phio = phio[:numpa] 
    r0 = np.sqrt(X[:numpa]**2 + Y[:numpa]**2 + Z[:numpa]**2)
#    pts.scater(tho,r0,r'$\theta$',r"$r$(kpc)",
##               r'$t=20 \tau_s$',
#               '$t=$%.2f'%t[te],
#               xangular=True, ylim=(0, lanp*Rf/(rlam*mu)),
#               name= "%s/%s/tho_ro_N%d_t%d"%(rut,carp,numpa,te))
#    pts.scater(tho,phio,r'$\theta$',r"$\phi$",
##               r'$t=20 \tau_s$',
#               '$t=$%.2f'%t[te],
#               xangular=True,
#               name= "%s/%s/tho_phio_N%d_t%d"%(rut,carp,numpa,te))
#    r0, tho, phio = filtro(r0, tho, phio, 150.,'mayores a')
#    r0, tho, phio = filtro(r0, tho, phio, [0., 30.],'intervalo')
#    r0, tho, phio = filtro(r0, tho, phio,  [200.,600.],'intervalo') 
    pts.histo(tho, r'$\theta$', bins = 80, #rang=(2.,np.pi),
              nom_archivo ="%s/hist_tho_N%d_t%d"%(rut,numpa,te),
#                  fit = True, #dist = 'dweibull',
#                  normalized=False,
              logx = False, xangular =True)
    pts.histo(r0, r'$r$(kpc)', bins = 80, 
#              rang=(0,2),
              nom_archivo ="%s/hist_ro_N%d_t%d"%(rut,numpa,te),
#                  fit = True,
#                  normalized=False,
              logx = False)
#    pts.scater(phio, r0, r"$\phi$", r'$r$(kpc)','$t=$%.2f'%t[te], xangular=True,
#               ylim=(0, lanp*Rf/(rlam*mu)),
#               name= "%s/%s/phio_ro_N%d_t%d"%(rut,carp,numpa,te))    


#for i in range(0,999, 25):  
#for i in range(0,999,1000): 
#    mptodos_plot(te=i, carp = 'orbitas_random_vel', numpa = 99900)
#    mptodos(te=i) 
mptodos_plot(te=975, carp = 'orbitas_random_vel', numpa = 99900)
#mptodos(te=975)   

#initial_pos_plot3D(te=950)

def exportdata():
    numpa = 99900
    carp = 'orbitas_random_vel'
    t =  np.load("%s/%s/1/tiemp_1.npy"%(rut,carp))    
    data = []    
    data2 = []
    data3 = []
    #
    #for te in range(0,999,25):
    for te in range(0,999,10):
        tho = np.load("%s/%s/Th_%d.npy"%(rut, carp, t[te]))
        phio = np.load("%s/%s/Ph_%d.npy"%(rut, carp, t[te]))
        X = np.load("%s/%s/X_%d.npy"%(rut, carp, t[te]))
        Y = np.load("%s/%s/Y_%d.npy"%(rut, carp, t[te]))
        Z = np.load("%s/%s/Z_%d.npy"%(rut, carp, t[te])) 
        r0 = np.sqrt(X[:numpa]**2 + Y[:numpa]**2 + Z[:numpa]**2)    
        tho = tho[:numpa]
        phio = phio[:numpa] 
        data.append([tho,phio])
        data2.append([tho,r0])
        data3.append([phio,r0])    
    data3 = np.array(data3)
    data2 = np.array(data2)
    data = np.array(data)
    return data, data2, data3
#
def update_plot(i, data, scat):
#    scat.set_array(data[i])
    scat.set_offsets(data[i][:].T)
    return scat,

def video(data,xlabel, ylabel,title='prueba', fps = 3, dpi = 500, numframes = 39):
    xy_data = data  
    fig = plt.figure(figsize=(7,5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    scat = plt.scatter(data[0][0], data[0][1], marker=',', s=1)
    ani = animation.FuncAnimation(fig, update_plot, frames= range(numframes),
                                  fargs=(xy_data, scat))
    writer = FFMpegWriter(fps = fps)
    ani.save('%s.mp4'%title, writer=writer, dpi = dpi) 
    plt.show()

#_, data2, data3 = exportdata()
#video(data,r'$\theta$',r'$\phi$', title="%s/orbitas_random_vel/tho_phio"%(rut),
#      numframes = 100, fps = 10)
#video(data2,r'$\theta$',r'$r$(kpc)', title="%s/orbitas_random_vel/tho_ro"%(rut),
#      numframes = 100, fps = 10)
#video(data3,r'$\phi$',r'$r$(kpc)', title="%s/orbitas_random_vel/phio_ro"%(rut),
#      numframes = 100, fps = 10)
