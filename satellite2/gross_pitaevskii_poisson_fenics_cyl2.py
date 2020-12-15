#!/usr/bin/env python
# coding: utf-8
##############################################################################
#################       se usa gross_pitaevskii_poisson_fenics_cyl   ########
##############################################################################
import numpy as np
################################################################################
di2R100 = {1 :{"Rf" : 100., "En" : 1./2., "expon": -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)}, 
      2 :{"Rf" : 100., "En": 2./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},##con fif = 1e-7 tambien funciona
      3 :{"Rf" : 100., "En": 8./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},##con fif = 1e-7 tambien funciona
      4 :{"Rf" : 100., "En": 9./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},##con fif = 1e-7 tambien funciona
      5 :{"Rf" : 100., "En" : 12./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      6 :{"Rf" : 100., "En": 16./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      7 :{"Rf" : 100., "En": 17./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      8 :{"Rf" : 100., "En" : 18./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      9 :{"Rf" : 100., "En" : 40./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      10 :{"Rf" : 100., "En": 49./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      11 :{"Rf" : 100., "En": 50./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      12 :{"Rf" : 100., "En" : 55./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)},
      13 :{"Rf" : 100., "En" : 83./2., "expon" : -4, "Vf" : -1e-2/2., "fif" : 1e-4/np.sqrt(2.)}}
#################################################################################