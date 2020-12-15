#!/bin/bash

for (( r= 51; r<=75; r=r+1))
do
    mkdir "/home/jordi/satellite/schrodinger_poisson/potpaco/baja_dens/pot_6/orbitas_random_vel/$r"
   (echo $r)| python pot_paco_orbitas_nuevo.py
done

#for (( r= 2; r<=100; r=r+1))
#do
#for (( s = 1; s<= 999; s=s+1))
#do
#    rm /home/jordi/satellite/schrodinger_poisson/potpaco/baja_dens/pot_3/orbitas_random_vel/${r}/tiemp_${s}.npy
#done
#done

#shutdown
