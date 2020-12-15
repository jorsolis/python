#!/usr/bin/env wolframscript
(* ::Package:: *)

Needs["MaTeX`"]


data = Import["/home/jordi/satellite/MW_rotation_curve_data/Fits/mixSFDM/params.txt", "Table"]


Md = data[[3]][[1]];
ad = data[[4]][[1]];
bd = data[[5]][[1]];
Mb = data[[6]][[1]];
bb = data[[7]][[1]]


\[Rho]MN[x_,y_,\[Zeta]_,M_,a_,b_]:=(b^2 M (a^3+5 a^2 Sqrt[b^2+\[Zeta]^2]+3 (b^2+\[Zeta]^2)^(3/2)+a (7 b^2+x^2+y^2+7 \[Zeta]^2)))/(4 \[Pi] (b^2+\[Zeta]^2)^(3/2) (x^2+y^2+(a+Sqrt[b^2+\[Zeta]^2])^2)^(5/2))


\[Rho]Hern[x_,y_,z_,M_, a_]:= ((M a)/(2 \[Pi] Sqrt[x^2 + y^2+ z^2] (Sqrt[x^2 + y^2+ z^2] + a))) 


a = With[{M= Md, b = bd, a = ad, rmax = 150},DensityPlot3D[\[Rho]MN[x,y,z,M,a,b], {x,-rmax,rmax}, {y,-rmax,rmax}, {z,-rmax,rmax}, AxesLabel->{"x(kpc)","y(kpc)","z(kpc)"},
 PlotLabel->{M,a, b} , ImageSize->Large, PlotLegends->Placed[BarLegend[Automatic,LegendMarkerSize->240,LegendLabel->Placed[MaTeX["\\rho (10^{10}M_\\odot/\\rm{kpc}^3)",Magnification->1.3],Up]],Right]]]
Export["/home/jordi/satellite/MW_rotation_curve_data/Fits/mixSFDM/discdens.png",a]


b2 = With[{M=Mb, a =bb, rmax = 3},DensityPlot3D[\[Rho]Hern[x,y,z,M, a], {x,-rmax,rmax}, {y,-rmax,rmax}, {z,-rmax,rmax}, AxesLabel->{"x(kpc)","y(kpc)","z(kpc)"}, 
PlotLabel->{M, a}, ImageSize->Large, PlotLegends->Placed[BarLegend[Automatic,LegendMarkerSize->240,LegendLabel->Placed[MaTeX["\\rho (10^{10}M_\\odot/\\rm{kpc}^3)",Magnification->1.3],Up]],Right]]]
Export["/home/jordi/satellite/MW_rotation_curve_data/Fits/mixSFDM/bulgedens_hern.png",b2]


c = With[{M1= Md, b1 = bd, a1 = ad, M2=Mb, b2 =bb, rmax = 150}, DensityPlot3D[\[Rho]MN[x,y,z,M1,a1,b1] +\[Rho]Hern[x,y,z,M2, b2], {x,-rmax,rmax}, {y,-rmax,rmax}, {z,-rmax,rmax},
 AxesLabel->{"x(kpc)","y(kpc)","z(kpc)"}, PlotLabel->{M1,a1, b1, M2,b2}, ImageSize->Large, PlotLegends->Placed[BarLegend[Automatic,LegendMarkerSize->240,LegendLabel->Placed[MaTeX["\\rho (10^{10}M_\\odot/\\rm{kpc}^3)",Magnification->1.3],Up]],Right]]]
Export["/home/jordi/satellite/MW_rotation_curve_data/Fits/mixSFDM/galaxydens.png",c]









