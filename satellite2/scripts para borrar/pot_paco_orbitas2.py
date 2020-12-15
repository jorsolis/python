#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:01:22 2019

@author: jordi
"""
from time import time

import numpy as np
#from numpy import (arange, sqrt)
#from scipy.integrate import solve_ivp
from scipy.interpolate import interp2d
#from pot_paco import fuerza_interpolada, plotdensitySFDM_units, num_partic_rf
#from pot_paco import plotdensitySFDM_units
#from plots_orbitas_schrodinger_poisson import (plotscoordscyl, los2plots3dcyl)
#import matplotlib.animation as animation
#from matplotlib.animation import FFMpegWriter
import gala.potential as gp
import gala.dynamics as gd
#from gala.units import galactic
#from pot_paco_orbitas_analisis import mptodos_plot, mptodos

class SFDM_Potential(gp.PotentialBase):
    def __init__(self,A,nsol, units=None):
#    def __init__(self,A, nsol, units=galactic):
        pars = dict(A=A, nsol=nsol)
        super(SFDM_Potential, self).__init__(units=units,parameters =pars,ndim=3)

    def _energy(self, xyz, t):
        A = self.parameters['A'].value
        x,y,z = xyz.T
        rho = np.sqrt(x**2 + y**2)
        pot_cyl = potencial_interpolado(nsol, refina = 3, di = di)
        return A*pot_cyl(rho, z)
   
    def _gradient(self, xyz, t):
        A = self.parameters['A'].value
        x,y,z = xyz.T
        rho = np.sqrt(x**2 + y**2)
        grad = np.zeros_like(xyz)
        derrho, derz = fuerza_interpolada(nsol, refina = 3, di = di) 
        grad[:,0] = A*derrho(rho,z)*x/rho
        grad[:,1] = A*derrho(rho,z)*y/rho
        grad[:,2] = A*derz(rho, z)
        return grad
    
#DE = {24:{'mu' : 156.55, 'rlam' : 3.8e-3, 'limb' : 90},
#      25:{'mu' : 15.655, 'rlam' : 2.9e-2, 'limb' : 100}}   ######## para di 2
rmin = .1
stps = 3000
def fuerza_interpolada(nsol, refina = 3, di = ''):
    dVdrho = np.load("%sdVdrho_%d_%d.npy"%(di,nsol,refina))
    dVdz = np.load("%sdVdz_%d_%d.npy"%(di,nsol,refina))
    rho = np.load("%scoordrho_%d_%d.npy"%(di,nsol,refina))
    z = np.load("%scoordz_%d_%d.npy"%(di,nsol,refina))
    rhor = rho[:50]
    derrho = interp2d(rhor, z, dVdrho, kind='linear', copy=True, bounds_error=False)
    zr = z[:100]
    derz = interp2d(rho, zr, dVdz, kind='linear', copy=True, bounds_error=False)    
    return derrho, derz

def potencial_interpolado(nsol, refina = 3, di = ''):
    rho = np.load("%scoordrho_%d_%d.npy"%(di,nsol,refina))
    z = np.load("%scoordz_%d_%d.npy"%(di,nsol,refina))
    potxz = np.load("%spotxz_%d_%d.npy"%(di,nsol,refina))    
    Vu = interp2d(rho, z, potxz, kind='linear', copy=True, bounds_error=False)
    return Vu   
        
def principal(derrho, derz, nsol,  Rf, tiem, ncor, y0a=[], plotting=True,
              direct = "/home/jordi/satellite/schrodinger_poisson/potpaco",
              de = '', k = 1):
    def resolvedor2(y0a, du, tiem):    
        potdm = SFDM_Potential(A=-1., nsol = nsol)#,units=galactic)
    
        vrho0, rho0, vz0, z0, phi0, vphi0 = y0a
        L = rho0*vphi0
      
        x0 = rho0*np.cos(phi0)
        y0 = rho0*np.sin(phi0)
        vx0 = vrho0*np.cos(phi0) - L*np.sin(phi0)/rho0
        vy0 = vrho0*np.sin(phi0) + L*np.cos(phi0)/rho0
        
        w0 = gd.PhaseSpacePosition(pos=[x0,y0,z0], vel=[vx0,vy0,vz0])
        orbit = gp.Hamiltonian(potdm).integrate_orbit(w0, dt=du, n_steps=stps)  
#        figs = orbit.plot(marker=',', linestyle='none')
#        sixplots_orbits(orbit[0:1000])    
        w =  orbit.w()  
        t = uev
#        x, y, z, vx,vy,vz = w
        
#        rho = np.sqrt(x**2 + y**2)
#        phi = np.arctan2(y,x)
#        vrho = x*vx/rho + y*vy/rho
#        vphi = x*vy/rho - y*vx/rho       
#        wcyl = np.array([rho,z,phi, vrho,vz,vphi])
#        R2 = np.sqrt( x**2 + y**2 + z**2)
#        if np.any(R2<rmin)==True:
#            print(i,'r min !!!!!')
#        if np.any(R2>12)==True:
#            print(i,'r max !!!!!')
#            print('rmax=', np.amax(R2))
#            print('r(0)=', R2[0])        
#        if plotting ==True:
#            plotscoordscyl(t, rho, z, phi, vrho, vz, tiem, 
#                           "%s/pot_%d/orbitas"%(direct,nsol),
#                           ncor, 'title')
#            los2plots3dcyl(rho, z, phi, t,
#                           "%s/%spot_%d/orbitas/%d"%(direct,de,nsol,k), ncor,
#                           '', np.amax(R2), galaxia=False, MO=False)

        return t, w
    
    du = tiem/stps
    uev = np.arange(0., tiem, du)
    t, fs = resolvedor2(y0a, du, tiem)
    return t, fs

if __name__ == '__main__':
    time0 = time()
    rut = "/home/jordi/satellite/schrodinger_poisson/potpaco"
    #rut = "/home/jsolis/satellite/schrodinger_poisson/potpaco"
    
    DU = {1:{'ref': 3, 't':50, 'rf': 3},
          2:{'ref': 3, 't':800, 'rf': 8, 'N': 2.822160530609563},#para di para vmax = ves/20 y vesc/2
          3:{'ref': 3, 't':50, 'rf': 3},
          4:{'ref': 1, 't':300, 'rf': 10},
          5:{'ref': 1, 't':300, 'rf': 20},
          6:{'ref': 3, 't':600, 'rf': 8, 'N':4.809892726429224}} 
    
    #DU = {2:{'ref': 3, 't':1200, 'rf': 8}} #para vamx = vesc
    
    ###############################################################################
    ###############        Random positions & velocities        ###################
    ###############################################################################
    di = 'baja_dens/'
    nsol = 2
    carp = 'orbitas_random_vel_new3'### vesc/10
#    carp = 'orbitas_random_vel_new2'### vesc/20    
    refina = DU[nsol]['ref']
    rf = DU[nsol]['rf']
    #tiemp= DU[nsol]['t']
    tiemp = 3200

#    rlam = DE[25]['rlam']

    nump= 1000
    
    lanp = 100./7.5
    ri  = 0.2 #1.
    #Nb, Ne = num_partic_rf(2, ref= 3, rf = 4., di= di)
    #N = Nb + Ne
    N = DU[nsol]['N']
    vesc = np.sqrt(2.*N/4.)
    c = 2.99e5 #km/s
    #print('vesc=', rlam*vesc*c/lanp,'km/s')
#    vmax = vesc/20.
    vmax = vesc/10.
    #######           Escala de tiempo       ######################################
#    tiemp= 60.
##    print('vel_max= vesc/2=', rlam*vmax*c/lanp,'km/s')
#    vrho0 = 0.
#    vz0 = 0.
#    vphi0 = vmax/2.
##    print('vel_r4 = vel_max/2 =', rlam*vphi0*c/lanp,'km/s')
#    y0= [vrho0, 4., vz0, 0, 0., vphi0]
#    derrho, derz = fuerza_interpolada(nsol, refina = 3, di = di)
#    ncor = 1
#    _, _ = principal(derrho,derz,nsol, rf, tiemp, ncor, 
#                                           y0a=y0, plotting = False, 
#                                           de=di)
    ##############################################################################
#    da2 = float(input('k'))
#    k = int(da2)
##    np.random.seed(12345)    
#    
#    derrho, derz = fuerza_interpolada(nsol, refina = 3, di = di)    
#    print(k)
#    ncor = 1
#    for i in range(1,nump,1):
##        print(i)
#    ########      esfera de posiciones aleatorias de radio rf 
#        r0 = np.random.uniform(ri, rf)
#        th0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
#        phi0 = np.random.uniform(0.,2.*np.pi) 
#    ########      posiciones iniciales
#        rho0 = r0*np.sin(th0)
#        z0 = r0*np.cos(th0)      
#    ########      esfera de velocidades aleatorias de radio vmax  
#        vr0 = np.random.uniform(0., vmax)
#        vt0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
#        vf0 = np.random.uniform(0.,2.*np.pi)    
#    ########      velocidades iniciales
#        vrho0 = vr0*np.sin(vt0)*np.cos(vf0)
#        vphi0 = vr0*np.sin(vt0)*np.sin(vf0)
#        vz0 = vr0*np.cos(vt0)
#    #######      condiciones  iniciales
#        y0= [vrho0, rho0, vz0, z0, phi0, vphi0]
#    ########      Programa   
#        _, cord = principal(derrho,derz, nsol, rf, tiemp, ncor, y0a=y0,
#                                   plotting = True ,de=di, k = k)
#
#        np.save("%s/%spot_%d/%s/%d/cords_%d.npy"%(rut,di,nsol,carp,k,ncor), cord)
##        np.save("%s/%spot_%d/%s/%d/tiemp_%d.npy"%(rut,di,nsol,carp,k,ncor), t)
#        ncor += 1
#
#    timef = time()
#    print(timef-time0,'sec')    

    ###############################################################################
    #####################     Plot 3D positions     ####################################
    ###########################################################################
    #te = 999
    #t =  np.load("%s/%spot_%d/orbitas_random_vel/1/tiemp_1.npy"%(rut,di,nsol))    
    #x00 = np.load("%s/%spot_%d/orbitas_random_vel/X_%d.npy"%(rut,di,nsol,t[te]))
    #y00 = np.load("%s/%spot_%d/orbitas_random_vel/Y_%d.npy"%(rut,di,nsol,t[te]))
    #z00 = np.load("%s/%spot_%d/orbitas_random_vel/Z_%d.npy"%(rut,di,nsol,t[te])) 
    ###
    #pts.scater(x00,y00,r'$x$(kpc)',r'$y$(kpc)',r'$t = %f$'%t[te], z3D=True, z=z00,
    #           zlab=r'$z$(kpc)', initialview=[10, 45], #initialview=[45,-60]
    #           R = 300,
    #           name = "%s/%spot_%d/orbitas_random_vel/particles_%d.png"%(rut,di,nsol,t[te]))
    
    #pts.scater(vrh,vph,r'$v_\rho$',r'$v_phi$','initial velocities', z3D=True, z=vz, 
    #           zlab=r'$v_z$', initialview=[45,-60])
    #pts.scater(vrh,vph,r'$v_\rho$',r'$v_\phi$','initial velocities', z3D=True, z=vz,
    #           zlab=r'$v_z$', initialview=[0,45])
    ###############################################################################
    ##############     plots fase space       ##################################
    ###############################################################################
#    for i in range(2975, 3000, 2500):     
#        print(nsol, i)
#        mu = 25
##        mptodos(nsol,te= i,carp = carp,rut = rut, de=di, kmax=10, mue= mu)
#        mptodos_plot(nsol,te= i, Rf = rf, carp = carp, de=di, mue= mu,
#                      rut = rut, numpa = 99900,
#                      histo = True, 
#                      tresD=True,
##                      Animation= True,
##                 Fase = True
#                 )
#        plotdensitySFDM_units(nsol,mue=mu, de = di, ref = 3)
#  
    ###############################################################################
    #####################      ANIMACIONES     ####################################
    ###############################################################################
    #de = 'baja_dens/'
    ##numpa = 999*100
    #t =  np.load("%s/%spot_%d/orbitas_random_vel/1/tiemp_1.npy"%(rut,de,nsol))     
    ##data = []
    ##data3 = []
    ##data2 = []
    ##
    ##for i in range(0,999,25):
    ##    tho = np.load("%s/%spot_%d/orbitas_random_vel/Th_%d.npy"%(rut,de,nsol,t[i]))
    ##    phio =np. load("%s/%spot_%d/orbitas_random_vel/Ph_%d.npy"%(rut,de,nsol,t[i]))
    ##    X = np.load("%s/%spot_%d/orbitas_random_vel/X_%d.npy"%(rut,de,nsol,t[i]))
    ##    Y = np.load("%s/%spot_%d/orbitas_random_vel/Y_%d.npy"%(rut,de,nsol,t[i]))
    ##    Z = np.load("%s/%spot_%d/orbitas_random_vel/Z_%d.npy"%(rut,de,nsol,t[i])) 
    ##    r0 = np.sqrt(X[:numpa]**2 + Y[:numpa]**2 + Z[:numpa]**2)
    ##    tho = tho[:numpa]
    ##    phio = phio[:numpa] 
    ##    data2.append([tho,r0])    
    ##    data.append([tho,phio])
    ##    data3.append([X, Y,Z])
    ##data = array(data)
    ##data2= array(data2)
    ##data3 = array(data3)
    #
    #def video(data,xlabel, ylabel,title='prueba', fps = 3, dpi = 500):
    #    numframes = 39  
    #
    #    xy_data = data  
    #    fig = plt.figure()
    #    plt.xlabel(xlabel)#,fontsize=16)
    #    plt.ylabel(ylabel)
    #    scat = plt.scatter(data[0][0], data[0][1], marker=',', s=1)
    ##    scat.set_title('')
    #    ani = animation.FuncAnimation(fig, update_plot, frames= range(numframes),
    #                                  fargs=(xy_data, scat))
    #    writer = FFMpegWriter(fps = fps)
    #    ani.save('%s.mp4'%title, writer=writer, dpi = dpi) 
    #    plt.show()
    #    
    #def update_plot(i, data, scat):
    #    scat.set_offsets(data[i][:].T)
    #    return scat,
    #
    #def video3D(data,xlabel, ylabel,zlabel, title='prueba', fps = 3, dpi = 500,
    #            R = 310):
    #    numframes = 40  
    #    xy_data = data  
    #
    #    fig = plt.figure(figsize=(10,10))
    #    plt.style.use('ggplot')
    #    ax = plt.axes(projection='3d')
    #    ax.set_aspect(aspect=1.)
    #    scat = ax.scatter(data[0][0][:], data[0][1][:], data[0][2][:], marker=',', s = 1)
    #    ax.set_zlabel(zlabel,rotation=45,fontsize=20)
    #    ax.set_zlim(-R, R) 
    #    ax.set_xlim(-R, R)
    #    ax.set_ylim(-R, R) 
    #    ax.set_xlabel(xlabel ,fontsize=20)
    #    ax.set_ylabel(ylabel,fontsize=20)
    ##    title = ax.set_title('3D Test')
    #    ani = animation.FuncAnimation(fig, update_plot3D, frames= range(numframes),
    #                                  fargs=(xy_data, scat))
    #    writer = FFMpegWriter(fps = fps)
    #    ani.save('%s.mp4'%title, writer=writer, dpi = dpi) 
    #    plt.show()
    #    
    #def update_plot3D(i, data, scat):
    #    scat._offsets3d = (data[i][0][:], data[i][1][:], data[i][2][:])
    ##    title.set_text(r'time=%f$\tau_s$'%i)
    #    return scat,
        
    
    #a = np.random.rand(2000, 3)*10
    #t = np.array([np.ones(100)*i for i in range(20)]).flatten()
    #df = pd.DataFrame({"time": t ,"x" : a[:,0], "y" : a[:,1], "z" : a[:,2]})
    #
    
    #video(data,r'$\theta$',r'$\phi$', title="%s/%spot_%d/orbitas_random_vel/tho_phio"%(rut,de,nsol))
    #video(data2,r'$\theta$',r'$r$(kpc)', title="%s/%spot_%d/orbitas_random_vel/tho_ro"%(rut,de,nsol))
    #video3D(data3,r'$x$(kpc)',r'$y$(kpc)',r'$z$(kpc)',
    #        title="%s/%spot_%d/orbitas_random_vel/orbitas"%(rut,de,nsol), R = 200)
    ###############################################################################
    #####################      ANIMACIONES     ####################################
    ###############################################################################
    #for ncor in range(1,7,1):
    #    x,y,z =  np.load("%s/pot_%d/orbitas/cords_%d.npy"%(rut,nsol,ncor))
    #    t = np.load("%s/pot_%d/orbitas/tiemp_%d.npy"%(rut,nsol,ncor))
    #    Rho, Z, fi = np.load("%s/pot_%d/rhozfi.npy"%(rut,nsol))           
    #    anim2d2(x, y, z, t, Rho, Z, fi, r'$\hat\mu x$', r'$\hat\mu y$', r'$\hat\mu z$',
    #            "%s/pot_%d/orbitas/orbita_2d_%d.mp4"%(rut,nsol,ncor), titulo1=r'$xz$-plane projection',
    #            titulo2=r'$yz$-plane projection',Rmax = 3.5)
