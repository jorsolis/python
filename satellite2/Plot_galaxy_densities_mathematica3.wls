#!/usr/bin/env wolframscript
(* ::Package:: *)

Needs["MaTeX`"]


data = Import["/home/jordi/satellite/MW_rotation_curve_data/Fits/Gaussian/params.txt", "Table"];


Rc = data[[1]][[1]];
Mc = data[[2]][[1]];
re = data[[3]][[1]];
rs = data[[4]][[1]];
Md = data[[5]][[1]];
ad = data[[6]][[1]];
Mb = data[[7]][[1]];
bb = data[[8]][[1]];
MBH = data[[9]][[1]];
R99 = 2.38167*Rc
\[Rho]h[x_,y_,\[Zeta]_, Mc_, Rc_, re_, rs_]:=HeavisideTheta[re-Sqrt[x^2+y^2+\[Zeta]^2]] Mc/(Rc^3 Sqrt[\[Pi]]^3) Exp[-(x^2+y^2+\[Zeta]^2)/Rc^2]+ (Mc re)/(rs (Rc^3) (Sqrt[\[Pi]]^3) )  E^(-(re^2/Rc^2)) (1+re/rs)^2 HeavisideTheta[Sqrt[x^2+y^2+\[Zeta]^2]-re] rs/(Sqrt[x^2+y^2+\[Zeta]^2](1+Sqrt[x^2+y^2+\[Zeta]^2]/rs)^2)
\[Rho]d[x_,y_,\[Zeta]_,M_,a_]:=M DiracDelta[\[Zeta]]Exp[-Sqrt[x^2+y^2]/a]/(2 \[Pi] a^2) 
\[Rho]b[x_,y_,\[Zeta]_,M_,a_]:= M Exp[-Sqrt[x^2+y^2+\[Zeta]^2]/a]/(8 \[Pi] a^3) 


a = With[{M2= Md, a2 = ad, rmax = 20 ad},DensityPlot3D[\[Rho]d[x,y,z,M2,a2], {x,-rmax,rmax}, {y,-rmax,rmax}, {z,-rmax,rmax}, AxesLabel->{"x(kpc)","y(kpc)","z(kpc)"},
ImageSize->Large, PlotLegends->Placed[BarLegend[Automatic,LegendMarkerSize->240,LegendLabel->Placed[MaTeX["\\rho (10^{10}M_\\odot/\\rm{kpc}^3)",Magnification->1.3],Up]],Right]]]
Export["/home/jordi/satellite/MW_rotation_curve_data/Fits/Gaussian/discdens.png",a]


b = With[{M=Mb, b =bb, rmax = R99},DensityPlot3D[\[Rho]b[x,y,z,M,b], {x,-rmax,rmax}, {y,-rmax,rmax}, {z,-rmax,rmax}, AxesLabel->{"x(kpc)","y(kpc)","z(kpc)"}, 
PlotLabel->{M, b}, ImageSize->Large, PlotLegends->Placed[BarLegend[Automatic,LegendMarkerSize->240,LegendLabel->Placed[MaTeX["\\rho (10^{10}M_\\odot/\\rm{kpc}^3)",Magnification->1.3],Up]],Right]]]
Export["/home/jordi/satellite/MW_rotation_curve_data/Fits/Gaussian/bulgedens.png",b]


c = With[{MC=Mc, RC =Rc, RE= re, RS= rs, rmax = R99},DensityPlot3D[\[Rho]h[x,y,z, MC, RC, RE, RS], {x,-rmax,rmax}, {y,-rmax,rmax}, {z,-rmax,rmax}, AxesLabel->{"x(kpc)","y(kpc)","z(kpc)"}, 
ImageSize->Large, PlotLegends->Placed[BarLegend[Automatic,LegendMarkerSize->240,LegendLabel->Placed[MaTeX["\\rho (10^{10}M_\\odot/\\rm{kpc}^3)",Magnification->1.3],Up]],Right]]]
Export["/home/jordi/satellite/MW_rotation_curve_data/Fits/Gaussian/DMdens.png",c]


d = With[{M2= Md, a2 = ad,M=Mb, b =bb, MC=Mc, RC =Rc, RE= re, RS= rs, rmax = R99}, DensityPlot3D[\[Rho]h[x,y,z, MC, RC, RE, RS] + \[Rho]d[x,y,z,M2,a2] + \[Rho]b[x,y,z,M,b], {x,-rmax,rmax}, {y,-rmax,rmax}, {z,-rmax,rmax},
 AxesLabel->{"x(kpc)","y(kpc)","z(kpc)"}, ImageSize->Large, PlotLegends->Placed[BarLegend[Automatic,LegendMarkerSize->240,LegendLabel->Placed[MaTeX["\\rho (10^{10}M_\\odot/\\rm{kpc}^3)",Magnification->1.3],Up]],Right]]]
Export["/home/jordi/satellite/MW_rotation_curve_data/Fits/Gaussian/galaxydens.png",d]


\[Rho]d2d[x_,y_,M_,a_]:=M Exp[-Sqrt[x^2+y^2]/a]/(2 \[Pi] a^2) 


a = With[{M2= Md, a2 = ad, rmax = R99},DensityPlot[\[Rho]d2d[x,y,M2,a2], {x,-rmax,rmax}, {y,-rmax,rmax},  AxesLabel->{"x(kpc)","y(kpc)"},
ImageSize->Large, PlotLegends->Placed[BarLegend[Automatic,LegendMarkerSize->240,LegendLabel->Placed[MaTeX["\\rho (10^{10}M_\\odot/\\rm{kpc}^3)",Magnification->1.3],Up]],Right]]]
Export["/home/jordi/satellite/MW_rotation_curve_data/Fits/Gaussian/discdens.png",a]



