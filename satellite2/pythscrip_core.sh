#!/bin/bash

for (( r= 45; r<=60; r=r+1))
do
    mkdir "/home/jordi/satellite/schrodinger_poisson/potpaco/baja_dens/potcore/orbitas_random_vel_new/$r"
   (echo $r)| python pot_paco_orbitas_core_new.py
done

#for (( r= 2; r<=100; r=r+1))
#do
#for (( s = 1; s<= 999; s=s+1))
#do
#    rm /home/jordi/satellite/schrodinger_poisson/potpaco/baja_dens/pot_3/orbitas_random_vel/${r}/tiemp_${s}.npy
#done
#done

#shutdown
