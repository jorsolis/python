#!/bin/bash

for (( r= 100; r<=200; r=r+10))
do
    mkdir "/home/jordi/satellite/mix_shooting/$r"
#   (echo $r)| python pot_paco_orbitas2.py
done

#for (( r= 2; r<=100; r=r+1))
#do
#for (( s = 1; s<= 999; s=s+1))
#do
#    rm /home/jordi/satellite/schrodinger_poisson/potpaco/baja_dens/pot_3/orbitas_random_vel/${r}/tiemp_${s}.npy
#done
#done
#python SP_shooting_axysimm_l2_solver_art_pacoluis.py
#shutdown
