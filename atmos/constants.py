# -*- coding: utf-8 -*-
"""
constants.py: Scientific constants in SI units.

Included constants:

* **g0** : standard acceleration of gravity
* **r_earth** : mean radius of Earth
* **Omega** : angular velocity of Earth
* **Rd** : specific gas constant for dry air
* **Rv** : specific gas constant for water vapor
* **Cpd** : specific heat capacity of dry air at constant pressure at 300K
* **Cl** : specific heat capacity of liquid water
* **Gammad** : dry adiabatic lapse rate
* **Lv0** : latent heat of vaporization for water at 0C
"""
from numpy import pi

g0 = 9.80665  # standard gravitational acceleration (m/s)
stefan = 5.67e-8  # Stefan-boltzmann constant (W/m^2/K^4)
r_earth = 6370000.  # Radius of Earth (m)
Omega = 7.2921159e-5  # Angular velocity of Earth (Rad/s)
Rd = 287.04  # R for dry air (J/kg/K)
Rv = 461.50  # R for water vapor
Cpd = 1005.  # Specific heat of dry air at constant pressure (J/kg/K)
Cl = 4186.  # Specific heat of liquid water (J/kg/K)
Gammad = g0/Cpd  # Dry adabatic lapse rate (K/m)
Lv0 = 2.501e6  # Latent heat of vaporization for water at 0 Celsius (J/kg)
