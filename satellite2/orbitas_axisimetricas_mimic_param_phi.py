#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
órbitas

"""
import scipy.integrate as spi
import numpy as np
import plots_jordi as pts

#POTENCIAL
#V(x,y)=-a*(y-pi/2)^2 - b/x - M/(d^2 + f^2 + x^2 
#    + 2*d*(f^2+x^2*cos(y)^2)^(1/2))^(1/2) 
# x es \hat{\mu} r 
# y es \theta
###### SFDM Potential Parameters
###### SFDM Potential Parameters
#M = 11590.
#a = 0.2957
#b = 1378.
#d = 4943.
#f = 1096.
b_bar=0
#####   Constantes
mu = 15.6378 # en 1/pc
c = 0.3  # pc/año
#####       condiciones iniciales
ri = 8.5*1e3        # parcecs
#
Mmw = 0.005  #pc
Mhal= 0.05
b = mu*Mhal
#print "b=",b
#M=0.
#a=0.
#d=0.
#f=0.
L = ri*mu*np.sqrt(Mhal)/np.sqrt(ri)
y0_0 = 0.01      #\hat{\mu} dr/dphi
y1_0 = ri*mu    #x(0)= \mu r(0)
y2_0 = 0.0001      #dtheta/dphi (0)
y3_0 = np.pi/2  # theta(0)
y0 = [y0_0, y1_0, y2_0, y3_0]

print "condiciones iniciales", y0, "L=", L
labcond = r"$L=$ %d , $\frac{dr}{d\phi}(0)=$ %d , $r(0)=$ %f Kpc, $\frac{d\theta}{d\phi}(0)=$ %f, $\theta(0)=\pi/2$ "
conds = (L, 0, ri, y2_0)
ncor = 1
##  u = phi
u0 = 0.
uf = 2*np.pi
du = uf/10000

u = np.arange(u0, uf, du)

def solver3_phi(u0,uf,du,y0,ncor, L, M, a, b, d, f, b_bar,titles,nomarchvs,ruta):
   
    def func(t, y, M, a, b, d, f, L):
        return [y[1]*y[2]**2+y[1]*np.sin(y[3])**2*2.*y[0]**2/y[1] 
                + 2.*y[0]*y[2]*np.cos(y[3])/np.sin(y[3]) 
                - ((y[1]*np.sin(y[3])**4/L**2)*(b/(y[1]**2) + (M*y[1]*(1+((d*np.cos(y[3])**2)/np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))))/(np.sqrt(d**2 + f**2 + y[1]**2 + 2*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3))),
                y[0],
                -2.*y[2]**2*np.cos(y[3])/np.sin(y[3]) + np.sin(y[3])*np.cos(y[3])
                -((y[1]**4*np.sin(y[3])**4/L**2)*((a*(np.pi-2*y[3])/y[1]**2) - (d*M*np.cos(y[3])*np.sin(y[3]))/(np.sqrt(f**2+y[1]**2*np.cos(y[3])**2) * np.sqrt(d**2 + f**2 + y[1]**2 + 2*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3))),
                y[2]]
    u = np.arange(u0, uf, du)
    y = spi.odeint(func, y0, u, args=(M,a,b,d,f,L), tfirst = True, tcrit=[0,np.pi,2*np.pi])
#    y = spi.odeint(func, y0, u, args=(M,a,b,d,f,L), Dfun = jacob, tfirst = True)
    pts.los3plotscoords(u,y[:, 1],y[:, 3],u,
                    "%s/r(phi)_ode_%s_%d" % (ruta,nomarchvs,ncor),
                    "%s/theta(phi)_ode_%s_%d" % (ruta,nomarchvs,ncor),
                    "%s/phi(phi)_ode_%s_%d" % (ruta,nomarchvs,ncor), titles)
    
    pts.los2plotsvels(u,y[:, 0],y[:, 2],
                    "%s/vr(phi)_ode_%s_%d" % (ruta,nomarchvs,ncor),
                    "%s/vtheta(phi)_ode_%s_%d" % (ruta,nomarchvs,ncor),
                    titles)
    
    pts.parametricplot(y[:, 1]*np.cos(u),y[:, 1]*np.sin(u),u,
                   r"$x\mu$",r"$y\mu$",
                   titles,"%s/xy_paramt_ode_%s_%d" % (ruta,nomarchvs,ncor))
    
    pts.los2plots3d(y[:, 1],y[:, 3],u, u,
                "%s/orbita3D_%s_1_ode_%d" % (ruta,nomarchvs,ncor),
                "%s/orbita3D_%s_2_ode_%d" % (ruta,nomarchvs,ncor), titles)
##########################################################################    


#solver3_phi(u0,uf,du,y0,ncor, L, M, a, b, d, f, b_bar,labcond % conds,'mim','esf/mim/paramphi')
###########################################################################
###########################################################################
###########################################################################
#max_steps = 40
#coord = np.array([[y0_0,y1_0,y2_0,y3_0]])
#fi = np.array([u0])
#
##sols = spi.LSODA(fun, u0, y0, uf, first_step=None)
#sols = spi.BDF(fun, u0, y0, uf, first_step=None)
##sols = spi.Radau(fun, u0, y0, uf, first_step=None)
#
#sols.y 
#for i in range(max_steps):
#    sols.step()
#    fi= np.append(fi,[sols.t],axis=0)
#    coord= np.append(coord,[sols.y],axis=0)
#
#print coord.shape, fi.shape
#
#los2plotscoords(fi,coord[:,1],coord[:,3],
#            "r(phi)_mim_LSODA_%d" % ncor,"theta(phi)_mim_LSODA_%d" % ncor)
#
#parametricplot(coord[:,1]*np.cos(fi),coord[:,1]*np.sin(fi),
#               r"$x\mu$",r"$y\mu$",
#              labcond % conds,"orb_proyxy_mim_LSODA_%d" % ncor)
#parametricplot(coord[:,1]*np.sin(coord[:,3]),coord[:,1]*np.cos(coord[:,3]),
#               r"$y\mu$",r"$z\mu$",
#              labcond % conds,"orb_proyyz_mim_LSODA_%d" % ncor)
#parametricplot(coord[:,1]*np.sin(coord[:,3]),coord[:,1]*np.cos(coord[:,3]),
#               r"$x\mu$",r"$z\mu$",
#              labcond % conds,"orb_proyxz_mim_LSODA_%d" % ncor)
#
#los2plots3d(fi,coord[:,1],coord[:,3],
#            "phi_mim_3D_1_LSODA_%d" % ncor,"phi_mim_3D_2_LSODA_%d" % ncor,labcond % conds)
##########################################################################
    
def solver_phi(u0,uf,y0,ncor,metodo,teval,L,M,a,b,d,f,b_bar,titles,nomarchvs,ruta):
    #metodo= 'RK45', 'Radau' o 'LSODA'
    def fun(t, y):
        return [y[1]*y[2]**2 + y[1]*np.sin(y[3])**2*2.*y[0]**2/y[1] 
                + 2.*y[0]*y[2]*np.cos(y[3])/np.sin(y[3]) 
                - ((y[1]*np.sin(y[3])**4/L**2)*(b/(y[1]**2) + (M*y[1]*(1+((d*np.cos(y[3])**2)/np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))))/(np.sqrt(d**2 + f**2 + y[1]**2 + 2*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3))),
                y[0],
                -2.*y[2]**2*np.cos(y[3])/np.sin(y[3]) + np.sin(y[3])*np.cos(y[3])
                -((y[1]**4*np.sin(y[3])**4/L**2)*((a*(np.pi-2*y[3])/y[1]**2) - (d*M*np.cos(y[3])*np.sin(y[3]))/(np.sqrt(f**2+y[1]**2*np.cos(y[3])**2) * np.sqrt(d**2 + f**2 + y[1]**2 + 2*d*np.sqrt(f**2+y[1]**2*np.cos(y[3])**2))**3))),
                y[2]]

    sol = spi.solve_ivp(fun, [u0,uf], y0, method=metodo, t_eval=teval)
    if sol.status==0:
        pts.los3plotscoords(sol.t,sol.y[1],sol.y[3],sol.t,
                        "%s/r(phi)_%s_%d" % (ruta,nomarchvs,ncor),
                        "%s/theta(phi)_%s_%d" % (ruta,nomarchvs,ncor),
                        "%s/phi(phi)_%s_%d" % (ruta,nomarchvs,ncor), titles)
        
        pts.los2plotsvels(sol.t,sol.y[0],sol.y[2],
                        "%s/vr(phi)_%s_%d" % (ruta,nomarchvs,ncor),
                        "%s/vtheta(phi)_%s_%d" % (ruta,nomarchvs,ncor), titles)
        
        pts.parametricplot(sol.y[1]*np.cos(sol.t),sol.y[1]*np.sin(sol.t),
                           sol.t,r"$x\mu$",r"$y\mu$",
                           titles,"%s/xy_paramphi_%s_%d" % (ruta,nomarchvs,ncor))
      
        pts.los2plots3d(sol.y[1],sol.y[3],sol.t,sol.t,
                    "%s/orbita3d_%s_1_%d" % (ruta,nomarchvs,ncor),
                    "%s/orbita3d_%s_2_%d" % (ruta,nomarchvs,ncor), titles)  

        print "xmax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.cos(sol.t))/15637.8, "Kpc"
        print "ymax=", np.amax(sol.y[1]*np.sin(sol.y[3])*np.sin(sol.t))/15637.8, "Kpc"
        print "zmax=", np.amax(sol.y[1]*np.cos(sol.y[3]))/15637.8, "Kpc"
        print "rmax=", np.amax(sol.y[1])/15637.8, "Kpc"

    else:
        print "status", sol.status    
    
solver_phi(u0,uf,y0,ncor,'LSODA',u,L,M,a,b,d,f,b_bar,labcond % conds,'mim','esf/mim/paramphi')   
    
#def dercruz(t,y):
#    return -(d*M*(y[1]**3*np.cos(y[3])**3*(d**2 + f**2 - 2*y[1]**2 - d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))) + f**2*y[1]*np.cos(y[3])*(2*d**2 + 2*f**2 - y[1]**2 + 4*d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))*np.sin(y[3]))/((f**2 + y[1]**2*np.cos(y[3])**2)^(3/2)*(d**2 + f**2 + y[1]**2 + 2*d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))**(5/2))
#
#def der2x(t,y):
#    return -((2*b)/y[1]**3) - (3*M*(y[1] + (d*y[1]*np.cos(y[3])**2)/(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))**2/(d**2 + f**2 + y[1]**2 + 2*d*(f**2 + y[1]**2*np.cos(y[3])**2**(1/2))**(5/2)) + (M*(2 - (2*d*y[1]**2*np.cos(y[3])**4)/(f**2 + y[1]**2*np.cos(y[3])**2)**(3/2) + (2*d*np.cos(y[3])**2)/(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))/(2*(d**2 + f**2 + y[1]**2 + 2*d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))**(3/2))))
#def der2teta(t,y):
#    return -2*a - (d*M*x**2*np.cos(y[3])**2*(4*d**2*(f**2 + x**2) + 4*(f**2 + x**2)**2 + 2**(1/2)*d*(4*f**2 + 7*x**2)*(2*f**2 + x**2 + x**2*np.cos(2*y[3]))**(1/2) - 3*2**(1/2)*d*x**2*np.cos(2*y[3])*(2*f**2 + x**2 + x**2*np.cos(2*y[3]))**(1/2)))/(4*(f**2 + x**2*np.cos(y[3])**2)**(3/2)*(d**2 + f**2 + x**2 + 2*d*(f**2 + x**2*np.cos(y[3])**2)**(1/2))**(5/2)) + (d*M*x**2*np.sin(y[3])**2)/((f**2 + x**2*np.cos(y[3])**2)**(1/2)*(d**2 + f**2 + x**2 + 2*d*(f**2 + x**2*np.cos(y[3])**2)**(1/2))**(3/2))
#
#def jacob(t, y):
#    return [[4*y[0]/y[1]+2*y[2]*np.cos(y[3])/np.sin(y[3]), 
#             y[2]**2+np.sin(y[3])**2 - 2*y[0]**2/y[1]**2 -
#             4*y[1]**3*np.sin(y[3])**4*((b/(y[1]**2) + (M*y[1]*(1+((d*np.cos(y[3])**2)/(f**2+y[1]**2*np.cos(y[3])**2)**(1/2))))/((d**2 + f**2 + y[1]**2 + 2*d*(f**2+y[1]**2*np.cos(y[3])**2)**(1/2))**(3/2))))/L**2 - y[1]**4*np.sin(y[3])**4*(-((2*b)/y[1]**3) - (3*M*(y[1] + (d*y[1]*np.cos(y[3])**2)/(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))**2/(d**2 + f**2 + y[1]**2 + 2*d*(f**2 + y[1]**2*np.cos(y[3])**2**(1/2))**(5/2)) + (M*(2 - (2*d*y[1]**2*np.cos(y[3])**4)/(f**2 + y[1]**2*np.cos(y[3])**2)**(3/2) + (2*d*np.cos(y[3])**2)/(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))/(2*(d**2 + f**2 + y[1]**2 + 2*d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))**(3/2)))))/L**2,
#             2*y[1]*y[2]+2*y[0]*np.cos(y[3])/np.sin(y[3]),
#             2*y[1]*np.cos(y[3])*np.sin(y[3])+2*y[0]*y[2]/np.sin(y[3])**2 
#             -4*y[1]**4*np.sin(y[3])**3*np.cos(y[3])*(b/(y[1]**2) + (M*y[1]*(1+((d*np.cos(y[3])**2)/(f**2+y[1]**2*np.cos(y[3])**2)**(1/2))))/((d**2 + f**2 + y[1]**2 + 2*d*(f**2+y[1]**2*np.cos(y[3])**2)**(1/2))**(3/2)))/L**2
#             -y[1]**4*np.sin(y[3])**4*(-(d*M*(y[1]**3*np.cos(y[3])**3*(d**2 + f**2 - 2*y[1]**2 - d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))) + f**2*y[1]*np.cos(y[3])*(2*d**2 + 2*f**2 - y[1]**2 + 4*d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))*np.sin(y[3]))/((f**2 + y[1]**2*np.cos(y[3])**2)**(3/2)*(d**2 + f**2 + y[1]**2 + 2*d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))**(5/2)))/L**2],
#            [1,0,0,0],
#            [0,
#             -4*y[0]**3*np.sin(y[3])**4*((a*(np.pi-2*y[3])/y[1]**2) - (d*M*np.cos(y[3])*np.sin(y[3]))/((f**2+y[1]**2*np.cos(y[3])**2)**(1/2) * (d**2 + f**2 + y[1]**2 + 2*d*(f**2+y[1]**2*np.cos(y[3])**2)**(1/2))**(3/2)))/L**2 -y[1]**4*np.sin(y[3])**3*(-(d*M*(y[1]**3*np.cos(y[3])**3*(d**2 + f**2 - 2*y[1]**2 - d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))) + f**2*y[1]*np.cos(y[3])*(2*d**2 + 2*f**2 - y[1]**2 + 4*d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))*np.sin(y[3]))/((f**2 + y[1]**2*np.cos(y[3])**2)**(3/2)*(d**2 + f**2 + y[1]**2 + 2*d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))**(5/2)))/L**2,
#             4*y[2]*np.cos(y[3])/np.sin(y[3]),
#             2*y[2]**2*np.cos(y[3])**2/np.sin(y[3])**2 + np.cos(y[3])**2 - np.sin(y[3])**2 -y[1]**4*np.sin(y[3])**4*(-2*a - (d*M*y[1]**2*np.cos(y[3])**2*(4*d**2*(f**2 + y[1]**2) + 4*(f**2 + y[1]**2)**2 + 2**(1/2)*d*(4*f**2 + 7*y[1]**2)*(2*f**2 + y[1]**2 + y[1]**2*np.cos(2*y[3]))**(1/2) - 3*2**(1/2)*d*y[1]**2*np.cos(2*y[3])*(2*f**2 + y[1]**2 + y[1]**2*np.cos(2*y[3]))**(1/2)))/(4*(f**2 + y[1]**2*np.cos(y[3])**2)**(3/2)*(d**2 + f**2 + y[1]**2 + 2*d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))**(5/2)) + (d*M*y[1]**2*np.sin(y[3])**2)/((f**2 + y[1]**2*np.cos(y[3])**2)**(1/2)*(d**2 + f**2 + y[1]**2 + 2*d*(f**2 + y[1]**2*np.cos(y[3])**2)**(1/2))**(3/2)))/L**2 - 4*y[1]**4*np.sin(y[3])**3*np.cos(y[3])*((a*(np.pi-2*y[3])/y[1]**2) - (d*M*np.cos(y[3])*np.sin(y[3]))/((f**2+y[1]**2*np.cos(y[3])**2)**(1/2) * (d**2 + f**2 + y[1]**2 + 2*d*(f**2+y[1]**2*np.cos(y[3])**2)**(1/2))**(3/2)))],
#            [0,0,1,0]]
