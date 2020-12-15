#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:32:37 2019

@author: jordi
"""
import matplotlib.pyplot as plt
from dolfin import plot, grad
import numpy as np
import plots_jordi as pts

def graf(func, titulo, directorio, name='', zlabel= 'solution', rango = False,
         vmin=0, vmax=10, show = True):
    plt.figure()
    if rango == True:
        c = plot(func, mode='color', title = titulo, vmin=vmin, vmax=vmax)
    else:
        c = plot(func, mode='color', title = titulo)
    plt.title(titulo,fontsize=20)
    plt.colorbar(c).ax.set_ylabel(zlabel)
    plt.xlabel(r'$\hat{\mu} \rho$')
    plt.ylabel(r'$\hat{\mu} z$')
    if name != '':
        plt.savefig("%s/%s.png"%(directorio,name))
    if show == True:
        plt.show()
    else:
        plt.close()
        
def plotses(u, En, expEn, ncor,
            direct = '/home/jordi/satellite/schrodinger_poisson'):
    psi_1, phi_1 = u.split()     
#    graf(psi_1, r"$\hat{E}=%.4f \times 10^{%d}$ " % (En,expEn), direct,
#         'psi_%d'%ncor, zlabel = r"$\psi (\rho,z)$")  
#    graf(phi_1, r"$\hat{E}=%.4f \times 10^{%d}$ " % (En,expEn), direct, 
#         'phi_%d'%ncor, zlabel = r"$\Phi (\rho,z)$")
    graf(phi_1**2, r"$\hat{E}=%.4f \times 10^{%d}$ " % (En,expEn), direct, 
         'phi_cuad_%d'%ncor, zlabel = r"$\Phi (\rho,z)^2$")
#    graf(grad(psi_1)[0], r"$\hat{E}=%.4f \times 10^{%d}$ " % (En,expEn), direct,
#         'dpsidrho_%d'%ncor, zlabel = r"$\frac{d\psi}{d\rho}(\rho,z)$")
#    graf(grad(psi_1)[1], r"$\hat{E}=%.4f \times 10^{%d}$ " % (En,expEn), direct,
#         'dpsidz_%d'%ncor, zlabel = r"$\frac{d\psi}{dz}(\rho,z)$")
#    graf(grad(phi_1)[0], r"$\hat{E}=%.4f \times 10^{%d}$ " % (En,expEn), direct,
#         'dphidrho_%d'%ncor, zlabel = r"$\frac{d\Phi}{d\rho}(\rho,z)$")
#    graf(grad(phi_1)[1], r"$\hat{E}=%.4f \times 10^{%d}$ " % (En,expEn), direct,
#         'dphidz_%d'%ncor, zlabel = r"$\frac{d\Phi}{dz}(\rho,z)$")
#    plt.show()
    
def plots_pyplot(u, Rf, En, expon, ncor, R0= 0.01, m=0.,
                 ruta='/home/jordi/satellite/schrodinger_poisson',
                 show= True, dens=True, pot=False, otras = False, 
                 cc = 1):
#################       PLOTS with pyplot         #############################  
    tol = 0.01 # avoid hitting points outside the domain
    u.set_allow_extrapolation(True)
    rho = np.linspace( R0, Rf - tol, 201)
    rhol = np.linspace(-Rf + tol, -R0, 201)
    points = [(abs(a), 0.) for a in rho] 
##    psiline = np.array([u[0](point) for point in points])
##    pts.parametricplot(rho,psiline,r'$\hat{\mu}\rho$',r'$\psi (\rho,0)/c^2 $',
##                       r"$\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
##                       "%s/psiz0_%d.png"%(ruta,ncor))    
    philine = np.array([u[1](point) for point in points])
#    densline = np.array([u[1](point)**2 for point in points])
#    print(np.amax(philine))
    bb = R0
    aa = philine[0]
    pts.plotmultiple([rho, rho],
                     [philine, aa*np.exp(-(rho+bb)**2/(2.*cc**2))],
                     ['sol',r'$a \exp{(-(\rho + b)^2/(2 c^2))}$'],
                     r'$\hat{\mu}\rho$',r'$\Phi (\rho,0)$',
                     r"$\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
                     "%s/phiz0_%d.png"%(ruta,ncor), show = show)
    pts.plotmultiple([rho], [philine], ['sol'], r'$\hat{\mu}\rho$',
                     r'$\Phi (\rho,0)$',
                     r"$\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
                     "%s/phiz0_%d.png"%(ruta,ncor), show = show)
    
#    pts.plotmultiple([rho], [densline], [], r'$\hat{\mu}x$',r'$\Phi^2 (\rho,0)$',
#                     r"$\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
#                     "%s/dens_rho0_%d.png"%(ruta,ncor), show = show)
#
#    z = np.linspace(tol, Rf - tol, 201)
#    points = [(0., abs(a)) for a in z]
##    densline = np.array([u[1](point)**2 for point in points])
#    Philine = np.array([u[1](point) for point in points])
##    a= np.amax(Philine)
##    b = z[np.where(Philine==np.amax(Philine))[0][0]]
##    c = 1.5
##    print(a,b,c)
#    pts.plotmultiple([z, z], [Philine, aa*np.exp(-(z+bb)**2/(2.*cc**2))],
#                     ['sol',r'$a \exp{(-(z + b)^2/(2 c^2))}$'], r'$\hat{\mu}z$',r'$\Phi (0,z)$',
#                     r"$\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
#                     "%s/phirho0_%d.png"%(ruta,ncor), show = show)
##    pts.plotmultiple([z], [Philine], [], r'$\hat{\mu}z$',r'$\Phi (0,z)$',
##                     r"$\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
##                     "%s/phirho0_%d.png"%(ruta,ncor), show = show)
#    print(aa*np.exp(-(z[-1]+bb)**2/(2.*cc**2)))
    def fi(rho,z, n=1):
        Phi = []
        for i in range(0,np.shape(rho)[0]):
            Phii=[]
            for j in range(0,np.shape(z)[0]):
                point = (rho[i], z[j]) 
                if z[j]**2 + rho[i]**2 > Rf**2:
                    Phii.append(np.nan)
                elif z[j]**2 + rho[i]**2 < R0**2:
                    Phii.append(np.nan)
                else:
                    if z[j]<0:
                        if rho[i]<0:
                            point = (-rho[i], -z[j])
                            Phii.append((-u[1](point))**n)
                        else:
                            point = (rho[i], -z[j])
                            Phii.append((-u[1](point))**n)
                    else:
                        if rho[i]<0:
                            point = (-rho[i], z[j])
                            Phii.append((u[1](point))**n)
                        else:
                            Phii.append((u[1](point))**n)    
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.transpose(Phi)     
    def psi(rho,z):    
        psi = []
        for i in range(0,np.shape(rho)[0]):
            psii=[]
            for j in range(0,np.shape(z)[0]):
                point = (rho[i], abs(z[j]))     
                if z[j]**2 + rho[i]**2 > Rf**2:
                    psii.append(np.nan)
                else:
                    psii.append(u[0](point))    
            psi.append(psii)
        psi = np.array(psi)
        return np.transpose(psi)

    z = np.linspace(-Rf + tol, Rf - tol, 201)
    rho = np.linspace(-Rf + tol, Rf - tol, 201)    
    Rho, Z = np.meshgrid(rho, z)
    from matplotlib import rcParams
    rcParams['figure.figsize'] = (6, 5)
    if m==0:
        nom = "%s/phi_yz_ot_%d.png"%(ruta,ncor)
        nom2 = "%s/psi_yz_%d.png"%(ruta,ncor)
    else:
        nom =  "%s/phi_yz_ot_m%d_%d.png"%(ruta,m,ncor)
        nom2 = "%s/psi_yz_m%d_%d.png"%(ruta,m,ncor)
    if dens ==True:
        pts.densityplot(Rho,Z,fi(rho,z,n=2),r'$\hat{\mu} y$',r'$\hat{\mu} z$',r'$\Phi(0,y,z)^2$',
                        r"$\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
                        nom, show=show, aspect='1/1')#, rango = [0.0, 1.])
    if pot ==True:
        rho = np.linspace(tol, Rf - tol, 201)
        Rho, Z = np.meshgrid(rho, z)    
        pts.densityplot(Rho,Z,psi(rho,z),r'$\hat{\mu} \rho$',r'$\hat{\mu} z$',r'$\psi(\rho,z)/c^2$',
                        r"$\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
                        nom2, show=show, aspect='1/1')
    if otras==True:
        z = np.linspace(-Rf + tol, Rf - tol, 201)
        rho = np.linspace(tol, Rf - tol, 201)
        Rho, Z = np.meshgrid(rho, z) 
        pts.plotfunc3d(Rho,Z,fi(rho,z),r'$\hat{\mu} \rho$',r'$\hat{\mu} z$',r'$\Phi(\rho,z)$',
                       r"$\Phi (\rho,z)$, $\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
                        "%s/Phi3d_%d.png"%(ruta,ncor))
        pts.plotfunc3d(Rho,Z,psi(rho,z),r'$\hat{\mu} \rho$',r'$\hat{\mu} z$',r'$\psi(\rho,z)/c^2$',
                       r"$\psi (\rho,z)$, $\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
                       "%s/psi3d_%d.png"%(ruta,ncor),rot_azimut=-120)

    
def plot_sf_sph(u, Rf, ncor, direct='/home/jordi/satellite/schrodinger_poisson'):
    def sf(r,theta):
        rho = r*np.sin(theta)
        z = r*np.cos(theta)
        Phi = []
        for j in range(0,np.shape(rho)[0]):
            Phii=[]
            for i in range(0,np.shape(rho)[1]):
                point = (rho[j][i], z[j][i]) 
                if z[j][i] < 0:
                    point = (rho[j][i], -z[j][i])
                    Phii.append(-u[1](point))
                else:
                    Phii.append(u[1](point))
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.array(Phi)
    
    def pot(r,theta):
        rho = r*np.sin(theta)
        z = abs(r*np.cos(theta))
        R1 = np.array([[u[0]((rho[j][i],z[j][i])) for i in range(0,np.shape(rho)[1])] for j in range(0,np.shape(rho)[0])])
        return R1
    
#    R, Th = np.meshgrid(np.linspace(0.,Rf, 150), np.linspace(0., np.pi/2, 100))
    R, Th = np.meshgrid(np.linspace(0.,Rf, 150), np.linspace(0., np.pi, 100))
    pts.plotfunc3d(R,Th,pot(R,Th),r'$\hat{\mu} r$',r'$\theta$',r'$\psi(r,\theta)/c^2$',
                   "Potential",name='%s/psi_sphericalcoords_%d'%(direct,ncor))
#    pts.densityplot(R,Th,pot(R,Th),r'$\hat{\mu} r$',r'$\theta$',r'$\psi(r,\theta)/c^2$',
#                "Potential",name='%s/psi_dens_sphericalcoords_%d'%(direct,ncor))
#    pts.densityplot(R,Th,sf(R,Th),r'$\hat{\mu} r$',r'$\theta$',r'$\Phi(r,\theta)/c^2$',
#                   "Scalar Field",name='%s/Phi_dens_sphericalcoords_%d'%(direct,ncor))
#    pts.densityplot(R,Th,sf(R,Th)**2,r'$\hat{\mu} r$',r'$\theta$',r'$\Phi(r,\theta)/c^2$',
#                   "Scalar Field",name='%s/Phi_cuad_dens_sphericalcoords_%d'%(direct,ncor))    
    pts.plotfunc3d(R,Th,sf(R,Th),r'$\hat{\mu} r$',r'$\theta$',r'$\Phi(r,\theta)/c^2$',
                   "Scalar Field",name='%s/Phi_sphericalcoords_%d'%(direct,ncor))
#    pts.sphericalplot3D(sf, Rf/2)
    
def plot_sf_sph2():
    def campo(r,th):
        return (np.sin(r)/r**2 - np.cos(r)/r)**2*np.cos(th)**2
    R, Th = np.meshgrid(np.linspace(0.5,10., 150), np.linspace(0., np.pi/2, 100))
#    pts.plotfunc3d(R,Th,campo(R,Th),'$r$',r'$\theta$','',"Scalar field")
    pts.densityplot(R,Th,campo(R,Th),'$r$',r'$\theta$','',"Scalar field")

def plotbessel(nu, rf= 30., A = 1e-4/2., kmu = 1):
    from scipy.special import jv,legendre
    x = np.linspace(0, rf, 100)
    y = np.linspace(0, np.pi/2, 100)    
    X, Y = np.meshgrid(x, y)
    def f(x, y):
        return A*jv(nu+0.5,kmu*x) * legendre(nu)(np.cos(y))
    #    return spe.spherical_jn(nu, x) * spe.legendre(nu)(np.cos(y))
    
    pts.plotfunc3d(X,Y,f(X,Y),r"$r$",r"$\theta$",r"$\phi_0 j_{%d}(r)P_%d (\theta)$"%(nu,nu),"",elev=45,
                   rot_azimut=-60, name= '/home/jordi/satellite/schrodinger_poisson/legendre_%d'%nu)
    pts.densityplot(X,Y,f(X,Y),r"$r$",r"$\theta$",r"$\phi_0 j_{%d}(r)P_%d (\theta)$"%(nu,nu),"",name= '/home/jordi/satellite/schrodinger_poisson/dens_legendre_%d'%nu)
    
def plots_pyplot_2(u, Rf, ncor, nt, ruta='/home/jordi/satellite/gross_evolution'):
#################       PLOTS with pyplot         #############################  
    tol = 0.001 # avoid hitting points outside the domain
    u.set_allow_extrapolation(True)
#    z = np.linspace(-Rf + tol, Rf - tol, 201)
#    rho = np.linspace(tol, Rf - tol, 201)
##    points = [(a, 0.) for a in rho]
#    psiline = np.array([u[0](point) for point in points])
#    pts.parametricplot(rho,psiline,r'$\hat{\mu}\rho$',r'$\psi (\rho,0)/c^2 $',r"$\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
#                       "%s/psiz0_%d_t%d.png"%(ruta,ncor,nt))    
    def fic(rho,z):
        Phi = []
        for i in range(0,np.shape(rho)[0]):
            Phii=[]
            for j in range(0,np.shape(z)[0]):
                point = (rho[i], z[j]) 
                if z[j]**2 + rho[i]**2 > Rf**2:
                    Phii.append(np.nan)
                else:
                    if z[j]<0:
                        if rho[i]<0:
                            point = (-rho[i], -z[j])
                            Phii.append((-u[1](point))**2 + (-u[2](point))**2)
                        else:
                            point = (rho[i], -z[j])
                            Phii.append((-u[1](point))**2 + (-u[2](point))**2)
                    else:
                        if rho[i]<0:
                            point = (-rho[i], z[j])
                            Phii.append((u[1](point))**2 + (u[2](point))**2)
                        else:
                            Phii.append((u[1](point))**2 + (u[2](point))**2)    
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.transpose(Phi)     
    def ficb(rho,z):
        Phi = []
        for i in range(0,np.shape(rho)[0]):
            Phii=[]
            for j in range(0,np.shape(z)[0]):
                point = (rho[i], z[j]) 
                if z[j]**2 + rho[i]**2 > Rf**2:
                    Phii.append(np.nan)
                else:
                    if z[j]<0:
                        if rho[i]<0:
                            point = (-rho[i], -z[j])
                            Phii.append((-u[2](point))**2)
                        else:
                            point = (rho[i], -z[j])
                            Phii.append((-u[2](point))**2)
                    else:
                        if rho[i]<0:
                            point = (-rho[i], z[j])
                            Phii.append((u[2](point))**2)
                        else:
                            Phii.append((u[2](point))**2)    
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.transpose(Phi) 
    def fice(rho,z):
        Phi = []
        for i in range(0,np.shape(rho)[0]):
            Phii=[]
            for j in range(0,np.shape(z)[0]):
                point = (rho[i], z[j]) 
                if z[j]**2 + rho[i]**2 > Rf**2:
                    Phii.append(np.nan)
                else:
                    if z[j]<0:
                        if rho[i]<0:
                            point = (-rho[i], -z[j])
                            Phii.append((-u[1](point))**2)
                        else:
                            point = (rho[i], -z[j])
                            Phii.append((-u[1](point))**2)
                    else:
                        if rho[i]<0:
                            point = (-rho[i], z[j])
                            Phii.append((u[1](point))**2)
                        else:
                            Phii.append((u[1](point))**2)    
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.transpose(Phi)
    def psi(rho,z):    
        psi = []
        for i in range(0,np.shape(rho)[0]):
            psii=[]
            for j in range(0,np.shape(z)[0]):
                point = (rho[i], abs(z[j]))     
                if z[j]**2 + rho[i]**2 > Rf**2:
                    psii.append(np.nan)
                else:
                    psii.append(u[0](point))    
            psi.append(psii)
        psi = np.array(psi)
        return np.transpose(psi)
    
    Rf = Rf/200.
    z = np.linspace(-Rf + tol, Rf - tol, 101)
    rho = np.linspace(-Rf + tol, Rf - tol, 101)    
    Rho, Z = np.meshgrid(rho, z)
    from matplotlib import rcParams
    rcParams['figure.figsize'] = (6, 5)
    tiem = nt*0.2
    pts.densityplot(Rho,Z,fic(rho,z),r'$\hat{\mu} y$',r'$\hat{\mu} z$',r'$\Phi(0,y,z)^2$',
                    "$\hat{\mu} c t=%f$"%tiem,
                    "%s/phi_yz_ot_%d_t%d.png"%(ruta,ncor,nt))
    pts.densityplot(Rho,Z,ficb(rho,z),r'$\hat{\mu} y$',r'$\hat{\mu} z$',r'$\Phi(0,y,z)^2$',
                    "$\hat{\mu} c t=%f$"%tiem,
                    "%s/phib_yz_ot_%d_t%d.png"%(ruta,ncor,nt))
    rho = np.linspace(tol, Rf - tol, 201)
    Rho, Z = np.meshgrid(rho, z) 

    pts.densityplot(Rho,Z,psi(rho,z),r'$\hat{\mu} \rho$',r'$\hat{\mu} z$',r'$\psi(\rho,z)/c^2$',
                    "$t=%f$"%tiem,
                    "%s/psi_yz_%d_t%d.png"%(ruta,ncor,nt))
    Rf = 7.5
    z = np.linspace(-Rf + tol, Rf - tol, 101)
    rho = np.linspace(-Rf + tol, Rf - tol, 201)    
    Rho, Z = np.meshgrid(rho, z) 
    pts.densityplot(Rho,Z,fice(rho,z),r'$\hat{\mu} y$',r'$\hat{\mu} z$',r'$\Phi(0,y,z)^2$',
                    "$\hat{\mu} c t=%f$"%tiem,
                    "%s/phie_yz_ot_%d_t%d.png"%(ruta,ncor,nt))
    z = np.linspace(-Rf + tol, Rf - tol, 101)
    rho = np.linspace(tol, Rf - tol, 201)    
    Rho, Z = np.meshgrid(rho, z) 
    pts.parametricplot(Rho[50,:],psi(rho,z)[50,:],r'$\hat{\mu} \rho$',
                       r'$\psi(\rho,0)/c^2$',"$t=%f$"%tiem,
                       "%s/psi_z0_%d_t%d.png"%(ruta,ncor,nt))    

def plots_pyplot_mix(u, Rf, ncor, data, R0 = 0.01,
                     ruta='/home/jordi/satellite/schrodinger_poisson_mix',
                     show= True):
#################       PLOTS with pyplot         #############################  
    tol = 0.01 # avoid hitting points outside the domain
    u.set_allow_extrapolation(True)
#    z = np.linspace(-Rf + tol, Rf - tol, 201)
#    rho = np.linspace(tol, Rf - tol, 201)
#    points = [(a, 0.) for a in rho]
#    psiline = np.array([u[0](point) for point in points])
#    pts.parametricplot(rho,psiline,r'$\hat{\mu}\rho$',r'$\psi (\rho,0)/c^2 $',r"$\hat{E}=%.4f \times 10^{%d}$" % (En,expon),
#                       "%s/psiz0_%d_t%d.png"%(ruta,ncor,nt))    
    def fic(rho,z):
        Phi = []
        for i in range(0,np.shape(rho)[0]):
            Phii=[]
            for j in range(0,np.shape(z)[0]):
                point = (rho[i], z[j]) 
                if z[j]**2 + rho[i]**2 > Rf**2:
                    Phii.append(np.nan)
                elif z[j]**2 + rho[i]**2 < R0**2:
                    Phii.append(np.nan)
                else:
                    if z[j]<0:
                        if rho[i]<0:
                            point = (-rho[i], -z[j])
                            Phii.append((-u[1](point))**2 + (-u[2](point))**2)
                        else:
                            point = (rho[i], -z[j])
                            Phii.append((-u[1](point))**2 + (-u[2](point))**2)
                    else:
                        if rho[i]<0:
                            point = (-rho[i], z[j])
                            Phii.append((u[1](point))**2 + (u[2](point))**2)
                        else:
                            Phii.append((u[1](point))**2 + (u[2](point))**2)    
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.transpose(Phi)     
    def fics(rho,z, s):
        Phi = []
        for i in range(0,np.shape(rho)[0]):
            Phii=[]
            for j in range(0,np.shape(z)[0]):
                point = (rho[i], z[j]) 
                if z[j]**2 + rho[i]**2 > Rf**2:
                    Phii.append(np.nan)
                elif z[j]**2 + rho[i]**2 < R0**2:
                    Phii.append(np.nan)
                else:
                    if z[j]<0:
                        if rho[i]<0:
                            point = (-rho[i], -z[j])
                            Phii.append((-u[s](point))**2)
                        else:
                            point = (rho[i], -z[j])
                            Phii.append((-u[s](point))**2)
                    else:
                        if rho[i]<0:
                            point = (-rho[i], z[j])
                            Phii.append((u[s](point))**2)
                        else:
                            Phii.append((u[s](point))**2)    
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.transpose(Phi) 

    def psi(rho,z):    
        psi = []
        for i in range(0,np.shape(rho)[0]):
            psii=[]
            for j in range(0,np.shape(z)[0]):
                point = (rho[i], abs(z[j]))     
                if z[j]**2 + rho[i]**2 > Rf**2:
                    psii.append(np.nan)
                else:
                    psii.append(u[0](point))    
            psi.append(psii)
        psi = np.array(psi)
        return np.transpose(psi)

    from matplotlib import rcParams
    rcParams['figure.figsize'] = (6, 5)
    E0, expo0, E1, expo1 = data
##############     (Phi_1^2 + Phi_0^2) zoom   y  Phi_0^2 zoom    #########################################
#    Rf2 = Rf/2000.
#    z = np.linspace(-Rf2 + tol, Rf2 - tol, 201)
#    rho = np.linspace(-Rf2 + tol, Rf2 - tol, 201)    
#    Rho, Z = np.meshgrid(rho, z)
#    pts.densityplot(Rho,Z,fic(rho,z),r'$\hat{\mu} y$',r'$\hat{\mu} z$',
#                    r'$\Phi(0,y,z)^2$',
#                    r"$\hat{E}_0 =$ %.2f $\times 10^{%d}$, $\hat{E}_1 =$ %.2f $\times 10^{%d}$"%(E0, expo0, E1, expo1),
#                    "%s/phi_yz_zoom_%d.png"%(ruta,ncor))
#    pts.densityplot(Rho,Z,fics(rho,z,2),r'$\hat{\mu} y$',r'$\hat{\mu} z$',
#                    r'$\Phi_0(0,y,z)^2$',
#                    r"$\hat{E}_0 =$ %.2f $\times 10^{%d}$, $\hat{E}_1 =$ %.2f $\times 10^{%d}$"%(E0, expo0, E1, expo1),
#                    "%s/phib_yz_zoom_%d.png"%(ruta,ncor), show=show, aspect='1/1')
#    rho = np.linspace(R0, Rf2 - tol, 1001)    
#    points = [(a, 0.) for a in rho]
#    philine = np.array([u[2](point) for point in points])
#    pts.parametricplot(rho,philine,r'$\hat{\mu}\rho$',r'$\Phi_0(\rho,0)/c^2 $',r"$\hat{E}=%.4f \times 10^{%d}$" % (E0,expo0),
#                       "%s/phiz0_%d.png"%(ruta,ncor))
##############       Psi r grandes       #########################################
    rho = np.linspace(tol, Rf - tol, 201)
    z = np.linspace(-Rf + tol, Rf - tol, 201)
    Rho, Z = np.meshgrid(rho, z) 
    pts.densityplot(Rho,Z,psi(rho,z), r'$\hat{\mu} \rho$', r'$\hat{\mu} z$',
                    r'$\psi(\rho,z)/c^2$',
                    r"$\hat{E}_0 =$ %.2f $\times 10^{%d}$, $\hat{E}_1 =$ %.2f $\times 10^{%d}$"%(E0, expo0, E1, expo1),
                    "%s/psi_yz_%d.png"%(ruta,ncor), show = show)
##############       Psi zooom       #########################################
#    rho = np.linspace(tol, Rf2 - tol, 201)
#    Rho, Z = np.meshgrid(rho, z) 
#    pts.densityplot(Rho,Z,psi(rho,z), r'$\hat{\mu} \rho$', r'$\hat{\mu} z$',
#                    r'$\psi(\rho,z)/c^2$',
#                    r"$\hat{E}_0 =$ %.2f $\times 10^{%d}$, $\hat{E}_1 =$ %.2f $\times 10^{%d}$"%(E0, expo0, E1, expo1),
#                    "%s/psi_yz_zooom_%d.png"%(ruta,ncor))
##############       Phi_1^2 y (Phi_1^2 + Phi_0^2) log       #########################################
    z = np.linspace(-Rf + tol, Rf - tol, 201)
    rho = np.linspace(-Rf + tol, Rf - tol, 201)    
    Rho, Z = np.meshgrid(rho, z) 
    pts.densityplot(Rho,Z,fics(rho,z,1),r'$\hat{\mu} y$',r'$\hat{\mu} z$',
                    r'$\Phi_1(0,y,z)^2$', 
                    r"$\hat{E}_0 =$ %.2f $\times 10^{%d}$, $\hat{E}_1 =$ %.2f $\times 10^{%d}$"%(E0, expo0, E1, expo1),
                    "%s/phie_yz_%d.png"%(ruta,ncor), show = show, aspect='1/1')
    pts.densityplot(Rho,Z,fic(rho,z),r'$\hat{\mu} y$', r'$\hat{\mu} z$',
                    r'$\Phi^2(0,y,z)$',
                    r"$\hat{E}_0 =$ %.2f $\times 10^{%d}$, $\hat{E}_1 =$ %.2f $\times 10^{%d}$"%(data[0],data[1],data[2],data[3]),
                    "%s/phi_yz_%d.png"%(ruta,ncor),
                    show = show, aspect='1/1')
    pts.densityplot(Rho,Z,fics(rho,z,2),r'$\hat{\mu} y$',r'$\hat{\mu} z$',
                    r'$\Phi_0(0,y,z)^2$',
                    r"$\hat{E}_0 =$ %.2f $\times 10^{%d}$, $\hat{E}_1 =$ %.2f $\times 10^{%d}$"%(E0, expo0, E1, expo1),
                    "%s/phib_yz_%d.png"%(ruta,ncor), show=show, aspect='1/1')
##############           SFDM density with units           ####################
#    cons = 1.65818e12
#    rlam= 9.0e-3
#    mu = 15.655 ### 10^-25
#    mue =25
#    
##    rlam= 1.5e-3
##    mu = 156.55 ### 10^-24
##    mue =24
#    
##    rlam= 1e-4
##    mu = 156550. ### 10^-21
##    mue = 21
#
#    rlam= 1e-3
#    rlam = 1e-6
#    mu = 15655.0 ### 10^-22 1/kpc
#    mue = 22 
#
#    z = np.linspace(-Rf + tol, Rf - tol, 201) 
#       
#    X, Z = np.meshgrid(z/(rlam*mu),z/(rlam*mu))    
#    pts.densityplot(X, Z, cons*(mu/1000.)**2*rlam**2*fics(z,z,1), r"$x$ (kpc)",
#                    r"$z$ (kpc)",r'$|\Phi_{1}|^2$ $(\frac{M_\odot}{pc^3})$',
#                    r"$\hat{E}_0 =$ %.2f $\times 10^{%d}$, $\hat{E}_1 =$ %.2f $\times 10^{%d}$, $\mu = 10^{-%d}$eV/$c^2$"%(data[0],data[1],data[2],data[3],mue),
#                    "%s/phie_yz_units_%d.png"%(ruta,ncor))
#    
#    Rf2 = Rf/200.
#    z = np.linspace(-Rf2 + tol, Rf2 - tol, 201)
#    mu = 15.6550 ### 10^-21 1/pc
#    X, Z = np.meshgrid(z/(rlam*mu),z/(rlam*mu)) 
#    pts.densityplot(X, Z, cons*(mu)**2*rlam**2*fics(z,z,2), r"$x$ (pc)",
#                    r"$z$ (pc)",r'$|\Phi_{0}|^2$ $(\frac{M_\odot}{pc^3})$',
#                    r"$\hat{E}_0 =$ %.2f $\times 10^{%d}$, $\hat{E}_1 =$ %.2f $\times 10^{%d}$, $\mu = 10^{-%d}$eV/$c^2$"%(data[0],data[1],data[2],data[3],mue),
#                    "%s/phib_yz_zoom_units_%d.png"%(ruta,ncor))
                    
def plots_pyplot_3(u, Rf, ncor,En,expon, ruta='/home/jordi/KG2D'):
#################       PLOTS with pyplot         #############################  
    tol = 0.001 # avoid hitting points outside the domain
    u.set_allow_extrapolation(True)  
    def fic(x,y):
        Phi = []
        for i in range(0,np.shape(x)[0]):
            Phii=[]
            for j in range(0,np.shape(y)[0]):
                point = (np.sqrt(x[j]**2 + y[j]**2), np.arctan2(y[j],x[j])) 
                if x[j]**2 + y[i]**2 > Rf**2:
                    Phii.append(np.nan)
                else:
                    Phii.append((-u[1](point))**2 + (-u[0](point))**2) 
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.transpose(Phi)     
    x = np.linspace(-Rf + tol, Rf - tol, 201)
    y = np.linspace(-Rf + tol, Rf - tol, 201)    
    X, Y = np.meshgrid(x, y)
    from matplotlib import rcParams
    rcParams['figure.figsize'] = (6, 5)
    pts.densityplot(X,Y,fic(x,y),r'$\hat{\mu} x$',r'$\hat{\mu} y$',r'$\Phi(x,y)^2$',
                    r'$En=%f \times 10^{%d}$'%(En, expon),
                    "%s/phi_yz_%d.png"%(ruta,ncor))

def plots_pyplot_3D(u, Rf, ncor, ruta='/home/jordi/gross3D'):
#################       PLOTS with pyplot         #############################  
    tol = 0.001 # avoid hitting points outside the domain
    u.set_allow_extrapolation(True)  
    rho = np.linspace(tol, Rf - tol, 201)
#    points = [(a, 0.,0.) for a in rho]
#    psiline = np.array([u[0](point) for point in points])
#    points = [(0., a, 0.) for a in rho]
#    philine = np.array([u[1](point) for point in points])
#    pts.parametricplot(rho,psiline,r'$\hat{\mu}\rho$',r'$\psi (\rho,0,0)/c^2 $',r"$\hat{E}=$",
#                       "%s/psiz0_%d.png"%(ruta,ncor))
#    pts.parametricplot(rho,philine,r'$\hat{\mu}\rho$',r'$\Phi (0,0,z) $',r"$\hat{E}=$",
#                       "%s/phiz0_%d.png"%(ruta,ncor))
    def fic(rho,z):
        Phi = []
        for i in range(0,np.shape(rho)[0]):
            Phii=[]
            for j in range(0,np.shape(z)[0]):
                point = (rho[i], z[j], np.pi) 
                if z[j]**2 + rho[i]**2 > Rf**2:
                    Phii.append(np.nan)
                else:
                    if z[j]<0:
                        if rho[i]<0:
                            point = (-rho[i], -z[j], np.pi)
                            Phii.append((-u[1](point))**2)
                        else:
                            point = (rho[i], -z[j], np.pi)
                            Phii.append((-u[1](point))**2)
                    else:
                        if rho[i]<0:
                            point = (-rho[i], z[j], np.pi)
                            Phii.append((u[1](point))**2)
                        else:
                            Phii.append((u[1](point))**2)    
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.transpose(Phi)     

    z = np.linspace(-Rf + tol, Rf - tol, 201)
    rho = np.linspace(-Rf + tol, Rf - tol, 201)    
    Rho, Z = np.meshgrid(rho, z)
    from matplotlib import rcParams
    rcParams['figure.figsize'] = (6, 5)
    pts.densityplot(Rho,Z,fic(rho,z),r'$\hat{\mu} x$',r'$\hat{\mu} z$',r'$\Phi(x,0,z)^2$',
                    "","%s/phi_xz_%d.png"%(ruta,ncor))
    def fic2(rho,z):
        Phi = []
        for i in range(0,np.shape(rho)[0]):
            Phii=[]
            for j in range(0,np.shape(z)[0]):
                point = (rho[i], z[j], np.pi/2.) 
                if z[j]**2 + rho[i]**2 > Rf**2:
                    Phii.append(np.nan)
                else:
                    if z[j]<0:
                        if rho[i]<0:
                            point = (-rho[i], -z[j], np.pi/2.)
                            Phii.append((-u[1](point))**2)
                        else:
                            point = (rho[i], -z[j], np.pi/2.)
                            Phii.append((-u[1](point))**2)
                    else:
                        if rho[i]<0:
                            point = (-rho[i], z[j], np.pi/2.)
                            Phii.append((u[1](point))**2)
                        else:
                            Phii.append((u[1](point))**2)    
            Phi.append(Phii)
        Phi = np.array(Phi)
        return np.transpose(Phi)     

    z = np.linspace(-Rf + tol, Rf - tol, 201)
    rho = np.linspace(-Rf + tol, Rf - tol, 201)    
    Rho, Z = np.meshgrid(rho, z)
    from matplotlib import rcParams
    rcParams['figure.figsize'] = (6, 5)
    pts.densityplot(Rho,Z,fic2(rho,z),r'$\hat{\mu} y$',r'$\hat{\mu} z$',r'$\Phi(0,y,z)^2$',
                    "","%s/phi_yz_%d.png"%(ruta,ncor))
###############################################################################    
###### Funcion para crear archivos density 3d para graficar en mathematica ####################
###############################################################################       
def ficuad3D(u, x, y, z, Rf):
    Phi = []
    for i in range(0,np.shape(x)[0]):
        Phii=[]
        for j in range(0,np.shape(y)[0]):
            Phiii=[]
            for k in range(0,np.shape(z)[0]):
                if z[k]**2 + x[i]**2 + y[j]**2> Rf**2:
                    Phiii.append(np.nan)
                else:
                    point = (np.sqrt(x[i]**2 + y[j]**2), z[k])
                    if z[k]<0:
                        point = (np.sqrt(x[i]**2 + y[j]**2), -z[k])
                        Phiii.append((-u[1](point))**2)
                    else:
                        Phiii.append((u[1](point))**2)
            Phii.append(Phiii)
        Phi.append(Phii)
    Phi = np.array(Phi)
    return np.transpose(Phi)

def density3d_files(u,Rf,dire):         
    p = 80   
    x = np.linspace(-Rf , Rf , p)
    y = np.linspace(-Rf , Rf , p)
    z = np.linspace(-Rf , Rf , p)
    a= ficuad3D(u, x, y, z, Rf) 
    pp= p**3
    b = a.reshape(1,pp)
    
    np.savetxt(dire,b,delimiter=',')
