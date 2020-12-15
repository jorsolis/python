#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:01:22 2019

@author: jordi
"""
from time import time

import numpy as np
from scipy.interpolate import interp2d
#import matplotlib.animation as animation
#from matplotlib.animation import FFMpegWriter
import gala.potential as gp
import gala.dynamics as gd
from pot_paco import plots_SFDM_density_interpolado, num_partic_rf
#from gala.units import galactic
from pot_paco_orbitas_analisis import mptodos_plot, mptodos, mptodos_plot_l, mptodos_l
from constants_grav import ckpc, c, Gentc2
from def_potenciales import v_multi_SFDM
#from pot_paco_rapido import SFDM_Potential
#from torsion import calc_orbital_poles

class SFDM_Potential(gp.PotentialBase):
    def __init__(self,A,nsol, units=None):
        pars = dict(A=A, nsol=nsol)
        super(SFDM_Potential, self).__init__(units=units,parameters =pars,ndim=3)

    def _energy(self, xyz, t):
        A = self.parameters['A'].value
        x,y,z = xyz.T
        rho = np.sqrt(x**2 + y**2)
        return A*pot_cyl(rho, z)
   
    def _gradient(self, xyz, t):
        A = self.parameters['A'].value
        x,y,z = xyz.T
        rho = np.sqrt(x**2 + y**2)
        grad = np.zeros_like(xyz)
        grad[:,0] = A*derrho(rho,z)*x/rho
        grad[:,1] = A*derrho(rho,z)*y/rho
        grad[:,2] = A*derz(rho, z)
        return grad

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
        
def principal(derrho, derz, nsol, tiem, y0a=[], plotting=True,
              direct = "/home/jordi/satellite/schrodinger_poisson/potpaco",
              de = '', k = 1):
    def resolvedor(y0a, du, tiem):    
        potdm = SFDM_Potential(A=-1., nsol = nsol)#,units=galactic)       
        x0, y0,z0, vx0, vy0, vz0 = y0a
        
        w0 = gd.PhaseSpacePosition(pos=[x0,y0,z0], vel=[vx0,vy0,vz0])
        orbit = gp.Hamiltonian(potdm).integrate_orbit(w0, dt=du, n_steps=stps)  
        if plotting ==True:
            orbit.plot(marker=',', linestyle='none') 
        w =  orbit.w()  
        t = uev
        x, y, z, vx,vy,vz = w
        R2 = np.sqrt(x**2 + y**2 + z**2)
#        V2 = np.sqrt(vx**2 + vy**2 + vz**2)
        if np.any(R2>12)==True:
#            print(i,'r max !!!!!')
            print('rmax=', np.amax(R2))
#            print('r(0)=', R2[0])  
#            print('v(0)=', V2[0]) 
            return False, 0
        else:
            return t, w
    du = tiem/stps
    uev = np.arange(0., tiem, du)
    t, fs = resolvedor(y0a, du, tiem)
    return t, fs

if __name__ == '__main__':
    mass = 25
    
    time0 = time()
    rut = "/home/jordi/satellite/schrodinger_poisson/potpaco" 
    DIC = { 6:{'R':4, 'N(R)': 4.89, 'tiemp': 87, 'rf': 10.5}} ## mu = 10**-25
    ###############################################################################
    ###############        Random positions & velocities        ###################
    ###############################################################################
    di = 'baja_dens/'
#    nsol = 2 # dipole dominated M1/M2 = 0.36
    nsol = 6 # monopole dominated
    carp = 'orbitas_random_vel_%d'%mass
    nump= 1000    
    ri  = 0.2 
    stps = 3000    
    
#    N = DIC[nsol]['N(R)']
#    tiemp = DIC[nsol]['tiemp']
#    rf = DIC[nsol]['rf']
    R = DIC[nsol]['R']
    
    dirdata = '/home/jordi/satellite/MW_rotation_curve_data'
    dirfitsG = '%s/Fits/mixSFDMmu%d/'%(dirdata,mass)
    popt = np.load('%spopt_nsol%d_%s.npy'%(dirfitsG, nsol, 'v5'))
    rlam, mu, Md, ad, Mb, bb = popt

    print(nsol)
    print('lam =', rlam*1e3, 'x 1e-3' )
    print('mu = ', mu)
    tiemp = 13e9*rlam**2*mu*ckpc
    print('tiemp_edaduniv=', tiemp)
#    
#    R = 20.
#    
    print('muR = %d, R ='%R, R/(rlam*mu), 'kpc')
    Nb, Ne = num_partic_rf(nsol, ref= 1, rf = R, di= di)
    N = Nb + Ne
    print('N(%d)='%R, N)
    
    vesc = np.sqrt(2.*N/R)    
    vmax = vesc
    print('v_esc(%d)='%R, vesc)  
    print('vesc(R=%d)='%R, rlam*vesc*c,'km/s') 
    
    print('v_circ(%d)='%R, v_multi_SFDM( 4./(rlam*mu) , rlam, mu, nsol)/(rlam*c))
    print('v_circ(R=%d)='%R, v_multi_SFDM( 4./(rlam*mu) , rlam, mu, nsol),'km/s')
    
#    R2 = 89.
#    print('R=300kpc, muR = ', 300.*rlam*mu)
#    Nb, Ne = num_partic_rf(nsol, ref= 1, rf = R2, di= di)
#    N = Nb + Ne
#    print('N(%d)='%R2, N)
#    print('v_esc(%d)='%R2, np.sqrt(2.*N/R2)) 
#    print('vesc(R=%d)='%R2, rlam*np.sqrt(2.*N/R2)*c,'km/s')     
#    print('v_circ(R=%d)='%R2, v_multi_SFDM(R2/(rlam*mu) , rlam, mu, nsol),'km/s')     

#    plots_SFDM_density_interpolado(nsol, di = 'baja_dens/', ref = 2, m=0,
#                               lam = rlam**2, mu = mu
#                               )
    #######           Escala de tiempo       ######################################
#    tiemp= 1200.
    vphi0 = v_multi_SFDM(R/(rlam*mu) , rlam, mu, nsol)/(rlam*c)
    y0= np.array([R, 0. ,0., 0., vphi0, 0.])
    derrho, derz = fuerza_interpolada(nsol, refina = 1, di = di)
    pot_cyl = potencial_interpolado(nsol, refina = 1, di = di)
    _, _ = principal(derrho,derz,nsol, tiemp, y0a=y0, plotting = True, 
                     de=di)
    ##############################################################################
#    da2 = float(input('k = '))
#    k = int(da2)
##    np.random.seed(12345)    
#    
#    derrho, derz = fuerza_interpolada(nsol, refina = 1, di = di)
#    pot_cyl = potencial_interpolado(nsol, refina = 1, di = di)
#    ncor = 1
#    for i in range(1,nump + 1,1):
#    ########      esfera de posiciones aleatorias de radio rf 
#        r0 = np.random.uniform(ri, rf)
#        th0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
#        phi0 = np.random.uniform(0.,2.*np.pi) 
#    ########      posiciones iniciales
#        x0 = r0*np.sin(th0)*np.cos(phi0)
#        y0 = r0*np.sin(th0)*np.sin(phi0)
#        z0 = r0*np.cos(th0)      
#    ########      esfera de velocidades aleatorias de radio vmax  
#        vr0 = np.random.uniform(0., vmax)
#        vt0 = np.arccos(2.*np.random.uniform(0.,1.) - 1.) #correct
#        vf0 = np.random.uniform(0.,2.*np.pi)    
#    ########      velocidades iniciales
#        vx0 = vr0*np.sin(vt0)*np.cos(vf0)
#        vy0 = vr0*np.sin(vt0)*np.sin(vf0)
#        vz0 = vr0*np.cos(vt0)
#    #######      condiciones  iniciales
#        y0= np.array([x0, y0, z0, vx0, vy0, vz0,])
#    ########      Programa   
#        t, cord = principal(derrho,derz, nsol, tiemp, y0a = y0,
#                            plotting = False, de = di, k = k)
#        if type(t) == bool:
#            nump +=1
#        else:
#            np.save("%s/%spot_%d/%s/%d/cords_%d.npy"%(rut,di,nsol,carp,k,ncor), cord)
##            np.save("%s/%spot_%d/%s/%d/tiemp_%d.npy"%(rut,di,nsol,carp,k,ncor), t)
#            ncor += 1
#
#    timef = time()
#    print((timef-time0)/60.,'min')    

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
    ################################# ##############################################   
#    for i in range(0, 10000, 500000):     
#        print(nsol, i)
#        mu = 25
#        mptodos(nsol,te= i,carp = carp,rut = rut, de=di, kmax=100, mue= mu,
#                ncormax = 450)
#        mptodos_plot(nsol,te= i, Rf = rf, carp = carp, de=di, mue= mu,
#                      rut = rut, 
#                      histo = True, 
##                      histo2D = True,
#                      tresD=True,
##                      astro = True, #cord = 'Heliocentric',
##                      Animation= True,
##                 Fase = True
#                 )
#        calc_orbital_poles(nsol, kmax = 100, nump = 450, carp = carp)
        
#        mptodos_l(nsol,te= i,carp = carp,rut = rut, de=di, kmax=100, 
#                  ncormax = 450)
#        mptodos_plot_l(nsol,te= i, carp = carp, de=di, rut = rut,
##                      histo = True, 
##                      histo2D = True,
##                      tresD=True,
#                 Fase = True,
##                 astro = True, cord = 'Galactocentric'
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