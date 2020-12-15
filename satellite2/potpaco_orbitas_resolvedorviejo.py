#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:56:55 2020

@author: jordi
"""

    def resolvedor(u0,uf,y0,method,teval,L,ncor, conds):
        #method= 'RK45', 'Radau' or 'LSODA'
#        def event(t,y):
#            vrho, rho, vz, z, phi = y
#            return z

        def fun(t,y):
            vrho, rho, vz, z, phi = y
            return [L**2/rho**3 + derrho(rho,z),
                    vrho,
                    derz(rho,z),
                    vz,
                    L/rho**2]   
        sol = solve_ivp(fun, [u0,uf], y0, method=method, t_eval=teval,
                        dense_output=True)#, events=event)
        if sol.status==0:
            t = sol.t
            vrho, rho, vz, z, phi = sol.y
            vphi = L/rho
            R2 = rho**2 + z**2
            if np.any(R2<rmin**2)==True:
                print(i,'r min !!!!!')
            if np.any(R2>12.**2)==True:
                print(i,'r max !!!!!')
                print('v2(0)=', np.sqrt(vrho[0]**2 + vz[0]**2+ vphi[0]**2))
                
            if plotting ==True:
#                plotscoordscyl(t, rho, z, phi, vrho, vz,
#                               uf, "%s/pot_%d/orbitas"%(direct,nsol), ncor,
#                               labcond % conds)
                los2plots3dcyl(rho, z, phi, t,
                               "%s/%spot_%d/orbitas/%d"%(direct,de,nsol,k), ncor,
                               labcond % conds, Rf, galaxia=False, MO=False)
#             tc = sol.t_events
#            print("tc=",tc[0])
#            np.save("%s/pot_%d/orbitas/tc_%d"%(direct,nsol,ncor), tc[0]) 
#            condin = np.transpose(np.array(sol.sol(sol.t_events[0])))
#            np.save("%s/pot_%d/orbitas/cond_ini_%d"%(direct,nsol,ncor), condin) 
#            print("ic=",condin)
#            print( "tiempo final=", u0-tc[0])
            return [t, rho, z, phi, vrho, vz, vphi]
        else:
            print("status", sol.status)
            
    labcond = r"$l\hat{\mu}/c=$ %.5f,+ $v_\rho(0)=$ %.5f $c$, $\hat{\mu}\rho(0)=$ %.1f, $v_z(0)=$ %.5f $c$,\
    $\hat{\mu} z(0)=$ %.2f"            
#    L = rho0*vphi0
#    conds = (L, y0a[0], rho0, y0a[2], y0a[3])
#    y0aa = [y0a[0], rho0, y0a[2], y0a[3], y0a[4]]
#    t,rho,z,phi ,vrho,vz,vphi = resolvedor(0,tiem,y0aa,'RK45',uev,L,ncor, conds)
            