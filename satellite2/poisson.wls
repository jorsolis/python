#!/usr/bin/env wolframscript
(* ::Package:: *)

k= 7.16613;
\[Mu]=15.6378;
\[Phi]=0;
For[r=10, r<10000, r=r+100,For[\[Theta]=0.1, \[Theta]<3.2, \[Theta]=\[Theta]+0.5, \[Psi][r, \[Theta]]=NIntegrate[(SphericalBesselJ[1,k/\[Mu] r1]^2 LegendreP[1,Cos[\[Theta]1]]^2 r1^2 Sin[\[Theta]1])/Sqrt[r^2+r1^2-2 r r1 (Cos[\[Theta]] Cos[\[Theta]1]+Sin[\[Theta]] Sin[\[Theta]1] Cos[\[Phi]-\[Phi]1])],{r1,0,5000},{\[Theta]1,0,\[Pi]},{\[Phi]1,0,2\[Pi]},WorkingPrecision->10];Print[\[Psi][r,\[Theta]]]];Print[r*100/902]]


Export["xint3.csv",Table[x,{x,311,901,10}]]
Export["thetaint3.csv",Table[\[Theta],{\[Theta],0.1,3.2,0.5}]]
Export["psiint3.csv",Table[\[Psi][r, \[Theta]]/(4 \[Pi] \[Mu]),{r,311,901,10},{\[Theta],0.1,3.2,0.5}]]
