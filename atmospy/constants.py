# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:29:28 2015

@author: mcgibbon
"""
import numpy as np

g0 = 9.81  # Gravitational acceleration (m/s)
stefan = 5.67e-8  # Stefan-boltzmann constant (W/m^2/K^4)
r_earth = 6370000.  # Radius of Earth (m)
Omega = 2*np.pi/86400.  # Angular velocity of Earth (Rad/s)

Rd = 287.04  # R for dry air (J/kg/K)
Rv = 461.50  # R for water vapor
Cpd = 1005.7  # Specific heat of dry air at constant pressure (J/kg/K)
Cl = 4186.  # Specific heat of liquid water (J/kg/K)
Gammad = g0/Cpd  # Dry adabatic lapse rate (K/m)

# Cpv = 1870. # Specific heat of water vapor (J/kg/K)
# Cw = 4190. # Specific heat of liquid water
Lv0 = 2.501e6  # Latent heat of vaporization for water at 0 Celsius (J/kg)
# delta = 0.608
