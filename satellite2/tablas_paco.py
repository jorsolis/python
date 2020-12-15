#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:50:22 2020

@author: jordi
"""
nsol = 5
s = "V"
f2= open("baja_dens/%sxz_%d_1.txt"%(s,nsol),"w+")
f3= open("baja_dens/%sxz_%d_2.txt"%(s,nsol),"w+")
f4= open("baja_dens/%sxz_%d_3.txt"%(s,nsol),"w+")
#s = "rho_ex"
#f2= open("baja_dens/%s_%d_1.txt"%(s,nsol),"w+")
#f3= open("baja_dens/%s_%d_2.txt"%(s,nsol),"w+")
#f4= open("baja_dens/%s_%d_3.txt"%(s,nsol),"w+")

def main():
    f= open("baja_dens/%s_%d.xz"%(s,nsol),"r")
    f1 = f.readlines()
    for i in range(4,10305,1):
        f2.write(f1[i])
    f2.close()
    for i in range(10307,20608,1):
        f3.write(f1[i])
    f3.close()
    for i in range(20610, 30911,1):
        f4.write(f1[i])
    f4.close()
        
if __name__ == '__main__':
    main()