# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/disk/p/mcgibbon/.spyder2/.temp.py
"""
# Note that in this module, if multiple equations are defined that compute
# the same output quantity given the same input quantities, they must take
# their arguments in the same order. This is to simplify overriding default
# equations with optional ones.

# Design considerations:
#
# Consider e = rho_v*Rv*T for water vapor and water vapor density...
#   Isn't water vapor density the same as absolute humidity? Yes it is.
#
# Do we want to use underscores in our variable names, like u_geo instead of
# ugeo, and dp_dx instead of dpdx?
#
# How do we handle latent heat of condensation for water? Constant? Poly fit?
#
# Users may make use of the same computation repeatedly. Should we cache how
# to calculate an output quantity from a set of inputs and methods?
# Precomputing every possible option is probably too memory-intensive because
# of the large number of options, but caching could be viable.
#
# Vertical ordering... What are the consequences, and how to handle it?
#

# To implement:
#
# Look at dimensions present in the data in order to:
#   Calculate pressure from surface pressure and hydrostatic balance
#   Similarly with thermal wind and surface winds
#
#
# Equivalent potential temperature... What even.
#
# Check whether certain inputs are valid (0 < RH < 100, 0 < T, etc.)
from util import ddx
import numpy as np
import re

g = 9.81  # Gravitational acceleration (m/s)
stefan = 5.67e-8  # Stefan-boltzmann constant (W/m^2/K^4)
r_earth = 6370000.  # Radius of Earth (m)
Omega = 2*np.pi/86400.  # Angular velocity of Earth (Rad/s)

Rd = 287.04  # R for dry air (J/kg/K)
Rv = 461.50  # R for water vapor
Cp = 1005.7  # Specific heat of dry air (J/kg/K)
Gamma_d = g/Cp  # Dry adabatic lapse rate (K/m)

# Cpv = 1870. # Specific heat of water vapor (J/kg/K)
# Cw = 4190. # Specific heat of liquid water
Lv = 2.501e6  # Latent heat of vaporization for water at 0 Celsius (J/kg)
# delta = 0.608


def AH_from_q_rho(q, rho):
    '''
    Calculates absolute humidity (kg/m^3) from specific humidity (kg/kg) and
    air density (kg/m^3).

    AH = q*rho
    '''
    return q*rho


def dpdz_from_rho_hydrostatic(rho):
    '''
    Calculates vertical derivative of pressure (Pa/m) from density (kg/m^3) and
    hydrostatic balance.

    dp/dz = -rho*g
    '''
    return -rho*g


def DSE_from_T_z(T, z):
    '''
    Calculates dry static energy (J) from temperature (K) and height (m).

    DSE = Cp*T + g*z
    '''
    return Cp*T + g*z


def DSE_from_T_Phi(T, Phi):
    '''
    Calculates dry static energy (J) from temperature (K) and geopotential
    height (m^2/s^2).

    DSE = Cp*T + Phi
    '''
    return Cp*T + Phi


def MSE_from_DSE_q(DSE, q):
    '''
    Calculates moist static energy (J) from dry static energy (J) and specific
    humidity (kg/kg) assuming constant latent heat of vaporization of water.

    MSE = DSE + Lv*q
    '''
    return DSE + Lv*q


def es_from_T_Goff_Gratch(T):
    '''
    Calculate the equilibrium water vapor pressure over a plane surface
    of ice according to http://en.wikipedia.org/wiki/Goff-Gratch_equation
    The formula (T in K, es in hPa):

    The original Goff–Gratch (1946) equation reads as follows:

    Log10(es) = -7.90298 (Tst/T-1)
                + 5.02808 Log10(Tst/T)
                - 1.3816*10-7 (10^(11.344 (1-T/Tst)) - 1)
                + 8.1328*10-3 (10^(-3.49149 (Tst/T-1)) - 1)
                + Log10(es_st)
    where:
    Log10 refers to the logarithm in base 10
    es is the saturation water vapor pressure (hPa)
    T is the absolute air temperature in kelvins
    Tst is the steam-point (i.e. boiling point at 1 atm.) temperature (373.16K)
    es_st is es at the steam-point pressure (1 atm = 1013.25 hPa)

    Note:
        This formula is accurate but computationally intensive. For most
    purposes, a more approximate formula is appropriate.

    References
    ----------
        http://en.wikipedia.org/wiki/Goff-Gratch_equation
        Goff, J. A., and S. Gratch (1946) Low-pressure properties of water
    from −160 to 212 °F, in Transactions of the American Society of Heating and
    Ventilating Engineers, pp 95–122, presented at the 52nd annual meeting of
    the American Society of Heating and Ventilating Engineers, New York, 1946.
        Goff, J. A. (1957) Saturation pressure of water on the new Kelvin
    temperature scale, Transactions of the American Society of Heating and
    Ventilating Engineers, pp 347–354, presented at the semi-annual meeting of
    the American Society of Heating and Ventilating Engineers, Murray Bay, Que.
    Canada.
        World Meteorological Organization (1988) General meteorological
    standards and recommended practices, Appendix A, WMO Technical Regulations,
    WMO-No. 49.
        World Meteorological Organization (2000) General meteorological
    standards and recommended practices, Appendix A, WMO Technical Regulations,
    WMO-No. 49, corrigendum.
        Murphy, D. M. and Koop, T. (2005): Review of the vapour pressures of
    ice and supercooled water for atmospheric applications, Quarterly Journal
    of the Royal Meteorological Society 131(608): 1539–1565.
    doi:10.1256/qj.04.94
    '''
    ratio = 373.16/T
    return 101325.*10.**(-1.90298*((ratio-1.) + 5.02808*np.log10(ratio)
                         - 1.3816e-7 * (10.**(11.344*(1.-1./ratio))-1.)
                         + 8.1328e-3*(10.**(-3.49149*(ratio-1.))-1.)))


def es_from_T_Bolton(T):
    '''
    Calculates saturation vapor pressure using Bolton's fit to Wexler's
    formula. Fits Wexler's formula to an accuracy of 0.1% for temperatures
    between -35C and 35C.

    es(T) = 611.2*exp(17.67*(T-273.15)/(T-29.65))  [Pa]

    References
    ----------
    David Bolton, 1980: The Computation of Equivalent Potential Temperature.
        Mon. Wea. Rev., 108, 1046–1053.
        doi: http://dx.doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    Wexler, A. (1976): Vapor pressure formulation for water in range 0 to
        100 C. A revision. J. Res. Natl. Bur. Stand. A, 80, 775-785.
    '''
    return 611.2*np.exp(17.67*(T-273.15)/(T-29.65))


def f_from_lat(lat):
    '''
    Calculates the Coriolis parameter (Rad/s) from latitude (degrees N).

    f = 2.*Omega*sin(pi/180.*lat)
    '''
    return 2.*Omega*np.sin(np.pi/180.*lat)


def omega_from_w_rho_hydrostatic(w, rho):
    '''
    Calculates pressure tendency (Pa/s) from vertical velocity (m/s) and
    density (kg/m^3) using the hydrostatic assumption.

    omega = -rho*g*w
    '''
    return -rho*g*w


def p_ideal_gas(rho, Tv):
    '''
    Calculates pressure (Pa) from density (kg/m^3) and virtual temperature (K).

    p = rho*Rd*Tv
    '''
    return rho*Rd*Tv


def plcl_from_p_T_Td(p, T, Td):
    '''
    Calculates LCL pressure level (Pa) from pressure (Pa), temperature (K), and
    dew point temperature (K).

    Calculates the pressure of the lifting condensation level computed by an
    iterative procedure described by equations 8-12 (pp 13-14) of:

    Stipanuk, G.S., (1973) original version.
    "Algorithms for generating a skew-t, log p diagram and computing selected
    meteorological quantities."

    Atmospheric sciences laboratory
    U.S. Army Electronics Command
    White Sands Missile Range, New Mexico 88002
    '''
    raise NotImplementedError()


def Phi_from_z(z):
    '''
    Calculates geopotential height (m^2/s^2) from height (m) assuming constant
    g.

    Phi = g*z
    '''
    return g*z


def q_from_AH_rho(AH, rho):
    '''
    Calculates specific humidity (kg/kg) from absolute humidity (kg/m^3) and
    air density (kg/m^3).

    q = AH/rho
    '''
    return AH/rho


def q_from_wvap(wvap):
    '''
    Calculates specific humidity (kg/kg) from water vapor mixing ratio (kg/kg).

    q = wvap/(1+wvap)
    '''
    return wvap/(1+wvap)


def qs_from_wvaps(wvaps):
    '''
    Calculates saturation specific humidity (kg/kg) from saturated water vapor
    mixing ratio (kg/kg).

    q = wvap/(1+wvap)
    '''
    return wvaps/(1+wvaps)


def q_from_p_e(p, e):
    '''
    Calculates specific humidity (kg/kg) from air pressure (Pa) and water vapor
    partial pressure (Pa).
    '''
    return 0.622*e/(p-e)


def Gamma_m_from_wvaps_T(wvaps, T):
    '''
    Calculates saturation adiabatic lapse rate (K/m) from pressure (Pa) and
    temperature (K).

    Gamma_m = g*(1+(Lv*wvaps)/(Rd*T))/(Cp+(Lv**2*wvaps)/(Rv*T**2))

    From the American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Saturation-adiabatic_lapse_rate
    Retrieved March 25, 2015
    '''
    return g*(1+(Lv*wvaps)/(Rd*T))/(Cp+(Lv**2*wvaps)/(Rv*T**2))


def e_from_p_q(p, q):
    '''
    Calculates water vapor partial pressure (Pa) from air pressure (Pa) and
    specific humidity (kg/kg).
    '''
    return p*q/(0.622+q)


def e_from_p_T_Tw_Goff_Gratch(p, T, Tw):
    '''
    Calculates water vapor partial pressure (Pa) from air pressure (Pa),
    temperature (K), and wet bulb temperature (K). Approximates saturation
    vapor pressure at the wet bulb temperature using the Goff-Gratch equation.
    Uses an approximation from the Royal Observatory outlined in the referenced
    document.

    e = es(Tw) - 0.799e-3*p*(T-Tw)

    Reference:
    Wong, W.T. 1989: Comparison of Algorithms for the Computation of the
        Thermodynamic Properties of Moist Air, Technical Note (Local) No. 51,
        Royal Observatory, Hong Kong. Retrieved March 25, 2015 from
        http://www.weather.gov.hk/publica/tnl/tnl051.pdf
    '''
    return es_from_T_Goff_Gratch(Tw) - 0.799e-3*p*(T-Tw)


def e_from_p_T_Tw_Bolton(p, T, Tw):
    '''
    Calculates water vapor partial pressure (Pa) from air pressure (Pa),
    temperature (K), and wet bulb temperature (K). Approximates saturation
    vapor pressure at the wet bulb temperature using Bolton's approximation to
    Wexler's formula.

    Uses an approximation from the Royal Observatory outlined in the referenced
    document.

    e = es(Tw) - 0.799e-3*p*(T-Tw)

    Reference:
    Wong, W.T. 1989: Comparison of Algorithms for the Computation of the
        Thermodynamic Properties of Moist Air, Technical Note (Local) No. 51,
        Royal Observatory, Hong Kong. Retrieved March 25, 2015 from
        http://www.weather.gov.hk/publica/tnl/tnl051.pdf
    '''
    return es_from_T_Bolton(Tw) - 0.799e-3*p*(T-Tw)


def e_from_Td_Goff_Gratch(Td):
    '''
    Calculates water vapor partial pressure (Pa) from dewpoint temperature (K).
    Uses the Goff-Gratch equation for saturation water vapor partial pressure.

    e = es(Td)
    '''
    return es_from_T_Goff_Gratch(Td)


def e_from_Td_Bolton(Td):
    '''
    Calculates water vapor partial pressure (Pa) from dewpoint temperature (K).
    Uses Bolton's approximation to Wexler's formula for saturation water vapor
    partial pressure.

    e = es(Td)
    '''
    return es_from_T_Bolton(Td)


def e_from_p_es_T_Tw(p, es, T, Tw):
    '''
    Calculates water vapor partial pressure (Pa) from pressure (Pa), saturation
    water vapor partial pressure (Pa), temperature (K), and wet bulb
    temperature (K). Assumes the wet bulb is not frozen.

    e = es - 6.60e-4*(1+0.00115(Tw-273.15)*(T-Tw))*p

    References
    ----------
    Petty, G.W. 1958: A First Course in Atmospheric Thermodynamics. 1st Ed.
        Sundog Publishing. p.216
    '''
    return es-(0.000452679+7.59e-7*Tw)*(T-Tw)*p


def e_from_p_es_T_Tw_frozen_bulb(p, es, T, Tw):
    '''
    Calculates water vapor partial pressure (Pa) from pressure (Pa), saturation
    water vapor partial pressure (Pa), temperature (K), and wet bulb
    temperature (K). Assumes the wet bulb is not frozen.

    e = es - 5.82e-4*(1+0.00115(Tw-273.15)*(T-Tw))*p

    References
    ----------
    Petty, G.W. 1958: A First Course in Atmospheric Thermodynamics. 1st Ed.
        Sundog Publishing. p.216
    '''
    return es-(0.000399181+6.693e-7*Tw)*(T-Tw)*p


def RH_from_q_qs(q, qs):
    '''
    Calculates relative humidity (%) from specific humidity (kg/kg) and
    saturation specific humidity (kg/kg).

    RH = q/qs*100.
    '''
    return q/qs*100.


def RH_from_wvap_wvaps(wvap, wvaps):
    '''
    Calculates relative humidity (%) from mixing ratio (kg/kg) and
    saturation mixing ratio (kg/kg)

    RH = wvap/wvaps*100.
    '''
    return wvap/wvaps*100.


def rho_from_q_AH(q, AH):
    '''
    Calculates density (kg/m^3) from specific humidity (kg/kg) and absolute
    humidity (kg/m^3).

    rho = AH/q
    '''
    return AH/q


def rho_from_dpdz_hydrostatic(dpdz):
    '''
    Calculates density (kg/m^3) from vertical derivative of pressure (Pa/m) and
    hydrostatic balance.

    rho = -dpdz/g
    '''
    return -dpdz/g


def rho_ideal_gas(p, Tv):
    '''
    Calculates density (kg/m^3) from pressure (Pa) and virtual temperature (K).

    rho = p/(Rd*Tv)
    '''
    return p/(Rd*Tv)


def T_from_es_Bolton(es):
    '''
    Calculates temperature (K) from saturation water vapor pressure (Pa) using
    Bolton's fit to Wexler's formula, converted to K. Fits Wexler's formula to
    an accuracy of 0.1% for temperatures between -35C and 35C.

    T = (29.65*log(es)-4880.16)/(log(es)-19.48)

    References
    ----------
    David Bolton, 1980: The Computation of Equivalent Potential Temperature.
        Mon. Wea. Rev., 108, 1046–1053.
        doi: http://dx.doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    Wexler, A. (1976): Vapor pressure formulation for water in range 0 to
        100 C. A revision. J. Res. Natl. Bur. Stand. A, 80, 775-785.
    '''
    return (29.65*np.log(es)-4880.16)/(np.log(es)-19.48)


def TL_from_T_RH(T, RH):
    '''
    Calculates temperature at LCL (K) from temperature (K) and relative
    humidity (%) using Bolton (1980) equation 22.

    TL = 1./((1./T-55.)-(log(RH/100.)/2840.)) + 55.

    References
    ----------
    David Bolton, 1980: The Computation of Equivalent Potential Temperature.
        Mon. Wea. Rev., 108, 1046–1053.
        doi: http://dx.doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    '''
    return 1./((1./(T-55.))-(np.log(RH/100.)/2840.)) + 55.


def TL_from_T_Td(T, Td):
    '''
    Calculates temperature at LCL (K) from temperature (K) and dewpoint
    temperature (K) using Bolton (1980) equation 15.

    TL = 1./((1./(Td-56.))-(log(T/Td)/800.)) + 56.

    References
    ----------
    David Bolton, 1980: The Computation of Equivalent Potential Temperature.
        Mon. Wea. Rev., 108, 1046–1053.
        doi: http://dx.doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    '''
    return 1./((1./(Td-56.))-(np.log(T/Td)/800.)) + 56.


def TL_from_T_e(T, e):
    '''
    Calculates temperature at LCL (K) from temperature (K) and water vapor
    partial pressure (Pa) using Bolton (1980) equation 21.

    TL = 2840./(3.5*log(T)-log(e)-4.805) + 55.

    References
    ----------
    David Bolton, 1980: The Computation of Equivalent Potential Temperature.
        Mon. Wea. Rev., 108, 1046–1053.
        doi: http://dx.doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    '''
    return 2840./(3.5*np.log(T)-np.log(e)-4.805) + 55.


def Tae_from_T_wvap(T, wvap):
    '''
    Calculates adiabatic equivalent temperature (aka pseudoequivalent
    temperature) (K) from temperature (K) and water vapor mixing ratio (kg/kg)

    Tae = T*exp(Lv*wvap/(Cp*T))

    From the American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Equivalent_temperature
    Retrieved March 25, 2015
    '''
    return T*np.exp(Lv*wvap/(Cp*T))


def Tie_from_T_wvap(T, wvap):
    '''
    Calculates isobaric equivalent temperature (K) from temperature (K) and
    water vapor mixing ratio (kg/kg).

    Tie = T*(1.+Lv*wvap/(Cp*T))

    From the American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Equivalent_temperature
    Retrived March 25, 2015
    '''
    return T*(1.+Lv*wvap/(Cp*T))


def Tv_from_T_p_q(T, q):
    '''
    Calculates virtual temperature from temperature (K) and specific
    humidity (kg/kg).
    '''
    raise NotImplementedError()


def Tv_from_T_assuming_dry(T):
    '''
    Calculates virtual temperature from temperature assuming no moisture.
    That is to say, it returns the input back.

    This function exists to allow using temperature as virtual temperature with
    a "dry" assumption.

    Tv = T
    '''
    return 1.*T


def Tv_ideal_gas(p, rho):
    '''
    Calculates virtual temperature (K) from density (kg/m^3) and pressure (Pa).

    Tv = p/(rho*Rd)
    '''
    return p/(rho*Rd)


def Tw_from_T_RH_Stull(T, RH):
    '''
    Calculates wet bulb temperature (K) from temperature (K) and relative
    humidity (%) using Stull's empirical inverse solution.

    References
    ----------
    Roland Stull, 2011: Wet-Bulb Temperature from Relative Humidity and Air
        Temperature. J. Appl. Meteor. Climatol., 50, 2267–2269.
        doi: http://dx.doi.org/10.1175/JAMC-D-11-0143.1
    '''
    return ((T-273.15)*np.arctan(0.151977*(RH + 8.313659)**0.5)
            + np.arctan(T-273.15+RH) + np.arctan(RH - 1.676331)
            + 0.00391838*RH**1.5*np.arctan(0.023101*RH) - 4.686035 + 273.15)


def T_from_Tv_assuming_dry(Tv):
    '''
    Calculates temperature from virtual temperature assuming no moisture.
    That is to say, it returns the input back.

    This function exists to allow using temperature as virtual temperature with
    a "dry" assumption.

    T = Tv
    '''
    return 1.*Tv


def theta_from_p_T(p, T):
    '''
    Calculates potential temperature (K) from pressure (Pa) and temperature
    (K). Assumes Cp does not vary with pressure or temperature.

    theta = T*(1e5/p)**(Rd/Cp)
    '''
    return T*(1e5/p)**(Rd/Cp)


def thetae_from_theta_TL_wvap_Bolton(theta, TL, wvap):
    '''
    Calculates equivalent potential temperature (K) from potential
    temperature (K), temperature at LCL (K), and water vapor mixing ratio
    (kg/kg) using Bolton's formula (1980).

    This is one of the most accurate ways of computing thetae, with an
    error of less than 0.2K due mainly to assuming Cp does not vary with
    temperature or pressure.

    References
    ----------
    David Bolton, 1980: The Computation of Equivalent Potential Temperature.
        Mon. Wea. Rev., 108, 1046–1053.
        doi: http://dx.doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    Robert Davies-Jones, 2009: On Formulas for Equivalent Potential
        Temperature. Mon. Wea. Rev., 137, 3137–3148.
        doi: http://dx.doi.org/10.1175/2009MWR2774.1
    '''
    return theta*np.exp((3.376/TL-0.00254)*wvap*1e3*(1+0.81*wvap))


def thetaie_from_T_theta_wvap(T, theta, wvap):
    '''
    Calculates isobaric equivalent potential temperature (K) from temperature
    (K), potential temperature (K), and water vapor mixing ratio (kg/kg).

    thetaie = theta*(1+Lv*wvap/(Cp*T))

    References
    ----------
    Petty, G.W. 2008: A First Course in Atmospheric Thermodynamics,
        Sundog Publishing, pg. 203
    '''
    return theta*(1+Lv*wvap/(Cp*T))


def thetaie_from_p_Tie_wvap(p, Tie, wvap):
    '''
    Calculates isobaric equivalent potential temperature (K) from isobaric
    equivalent temperature (K) and water vapor mixing ratio (kg/kg).

    thetaie = Tie*(1e5/p)**(Rd/Cp)

    References
    ----------
    Petty, G.W. 2008: A First Course in Atmospheric Thermodynamics,
        Sundog Publishing, pg. 203
    '''
    return Tie*(1e5/p)**(Rd/Cp)


def thetaae_from_p_Tae_wvap(p, Tae, wvap):
    '''
    Calculates adiabatic equivalent potential temperature (K) from adiabatic
    equivalent temperature (K) and water vapor mixing ratio (kg/kg).

    thetaae = Tae*(1e5/p)**(Rd/Cp)
    '''
    return Tae*(1e5/p)**(Rd/Cp)


def w_from_omega_rho_hydrostatic(omega, rho):
    '''
    Calculates vertical velocity (m/s) from vertical pressure tendency (Pa/s)
    and density (kg/m^3) using the hydrostatic assumption.

    w = -omega/(rho*g)
    '''
    return -omega/(rho*g)


def wvap_from_q(q):
    '''
    Calculates water vapor mixing ratio (kg/kg) from specific humidity (kg/kg).

    wvap = q/(1-q)
    '''
    return q/(1-q)


def wvaps_from_p_es(p, es):
    '''
    Calculates saturation mixing ratio from pressure (Pa) and saturation
    vapor pressure (Pa).

    wvaps = Rd/Rv*es/(p-es)
    '''
    return Rd/Rv*es/(p-es)


def wvaps_from_qs(qs):
    '''
    Calculates saturation water vapor mixing ratio (kg/kg) from saturation
    specific humidity (kg/kg).

    wvap = q/(1-q)
    '''
    return qs/(1-qs)


def z_from_Phi(Phi):
    '''
    Calculates height (m) from geopotential height (m^2/s^2) assuming constant
    g.

    z = Phi/g
    '''
    return Phi/g


class _BaseDeriver(object):
    '''
    '''

    default_methods = {}
    optional_methods = {}
    derivative_prog = re.compile(r'd(.+)d(p|x|y|theta|z|sigma|t|lat|lon)')
    coord_types = {
        'x': 'x',
        'lon': 'x',
        'y': 'y',
        'lat': 'y',
        'theta': 'z',
        'z': 'z',
        'p': 'z',
        'sigma': 'z',
        't': 't',
    }

    def __new__(cls, *args, **kwargs):
        if cls is _BaseDeriver:
            raise TypeError('BaseDeriver may not be instantiated. Use a '
                            'subclass.')
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, methods=(), axis_coords=None, override_coord_axes=None,
                 coords_own_axis=None, **kwargs):
        '''
        Initializes with the given methods enabled, and variables passed as
        keyword arguments stored.
        '''
        if any([coord not in self.coord_types.keys()
                for coord in axis_coords]):
            raise ValueError('Invalid value given in axis_coords')
        self.methods = self._get_methods(methods)
        self.vars = kwargs
        if axis_coords is not None:
            coord_axes = {}
            for i in range(len(axis_coords)):
                coord_axes[self.coord_types[axis_coords[i]]] = i
            self.coord_axes = coord_axes
        else:
            self.coord_axes = {}
        if override_coord_axes is not None:
            self.override_coord_axes = override_coord_axes
        else:
            self.override_coords = {}
        self.coords_own_axis = {'t': 0, 'z': 0, 'y': 0, 'x': 0}
        if coords_own_axis is not None:
            self.coords_own_axis.update(coords_own_axis)

    def calculate(self, quantity_out):
        '''
        Calculates and returns a requested quantity from quantities passed in
        as keyword arguments.

        Parameters
        ----------
        varname_out : string
            Name of quantity to be calculated.
        methods : tuple, optional
            Names of methods that can be used for calculation, as strings.

        Quantity Parameters
        -------------------
        All quantity parameters are optional, and must be of the same type,
        either ndarray or iris Cube. If ndarrays are used, then units must
        match the units specified below. If iris Cube is used, any units may
        be used for input and the output will be given in the units specified
        below.


        Returns
        -------
        quantity : ndarray or iris Cube
            Calculated quantity.
            Return type is the same as quantity parameter types.

        Raises
        ------
        ValueError:
            If the output quantity cannot be determined from the input
            quantities.

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>>
        '''
        funcs, func_args, extra_values = \
            self._calculate(quantity_out, self.methods,
                            (), (), (), self.vars, ())
        # Above method completed successfully if no ValueError has been raised
        # Calculate each quantity we need to calculate in order
        for i, func in enumerate(funcs):
            # Compute this quantity
            value = func(*[self.vars[varname] for varname in func_args[i]])
            # Add it to our dictionary of quantities for successive functions
            self.vars[extra_values[i]] = value
        # The last quantity calculated will be the one for this function call
        assert extra_values[-1] == quantity_out
        return self.vars[quantity_out]

    def _calculate(self, out_name, methods, exclude, funcs, func_args,
                   in_values, extra_values):
        '''
        Tries to calculate out_name using the given methods, the variables in
        the dictionary in_values, and the calculated variables listed in
        extra_values, without using any of the variables listed in exclude.

        Returns
        -------
        Returns a tuple of functions, function arguments, and the variables
        they calculate, such that if you execute those functions in order and
        use their calculated variables in successive functions, the last one
        (if any) will successfully calculate the variable denoted by out_name.

        funcs: tuple
            The functions that need to be called in order to calculate the
            variable referred to by out_name
        func_args: tuple
            A tuple of tuples of strings denoting the variables that must be
            passed as arguments to each function in funcs
        extra_values: tuple
            A tuple of strings denoting the variables that are output by
            each function in funcs

        Raises
        ------
        ValueError:
            If the variable denoted by out_name cannot be calculated with the
            given methods and in_values
        '''
        # For debugging purposes, make sure nothing in extra_values is given
        # in in_values
        assert not any([value in in_values.keys() for value in extra_values])
        # Check if we already have the value desired
        if out_name in in_values.keys() or out_name in extra_values:
            return (), (), ()
        # Make sure we have a method for this quantity
        if out_name not in methods.keys():  # No method to get this quantity
            raise ValueError('Could not determine {} as no available method '
                             'determines it'.format(out_name))
        # Check whether we already have calculated everything to complete
        # one of the available methods
        for args, func in methods[out_name].items():
            # See if all arguments are either provided or already calculated
            if all([(arg in in_values.keys() or arg in extra_values) for arg
                    in args]):
                # If this is the case, use this method
                return (funcs + (func,), func_args + (args,),
                        extra_values + (out_name,))
        # See if we can calculate from a method by calculating the missing
        # arguments
        for args, func in methods[out_name].items():
            # make sure none of the method arguments are excluded (we've
            # already tried to get it and are nested deeper in our search)
            if any([arg in exclude for arg in args]):
                continue  # Go to the next method
            temp_funcs, temp_func_args = funcs, func_args
            temp_extra_values = extra_values
            for arg in args:  # See if we have all our arguments
                # See if this argument was input or we've already determined
                # it can be derived
                if arg in in_values.keys() or arg in temp_extra_values:
                    # We already have this, check the next one
                    continue
                try:
                    result = self._get_derivative(arg, methods, exclude +
                                                  (out_name,), temp_funcs,
                                                  temp_func_args, in_values,
                                                  temp_extra_values)
                except ValueError:
                    pass
                # We'd need to calculate this argument. See if we can
                try:
                    result = self._calculate(arg, methods, exclude +
                                             (out_name,), temp_funcs,
                                             temp_func_args, in_values,
                                             temp_extra_values)
                except ValueError:
                    break  # We can't get this argument
                # We can get this argument, so add it to our list
                temp_funcs, temp_func_args, temp_extra_values = result
            else:  # We have all our arguments, since we didn't break
                return (temp_funcs + (func,), temp_func_args + (args,),
                        temp_extra_values + (out_name,))
        # We could not calculate this with any method available
        raise ValueError('Could not determine {} from given variables'
                         ' {}'.format(out_name,
                                      ', '.join(tuple(in_values.keys()) +
                                                extra_values)))

    def _get_derivative(self, out_name, methods, exclude, funcs, func_args,
                        in_values, extra_values):
        '''
        Tries to determine how to calculate varname, assuming varname is of
        the form d(var)d(coord), where var is a quantity and coord is a
        coordinate. Raises ValueError if this cannot be done or if varname
        is of the wrong form.

        Returns
        -------
        Returns a tuple of functions, function arguments, and the variables
        they calculate, such that if you execute those functions in order and
        use their calculated variables in successive functions, the last one
        will successfully calculate the variable denoted by out_name.

        funcs: tuple
            The functions that need to be called in order to calculate the
            variable referred to by out_name
        func_args: tuple
            A tuple of tuples of strings denoting the variables that must be
            passed as arguments to each function in funcs
        extra_values: tuple
            A tuple of strings denoting the variables that are output by
            each function in funcs
        '''
        match = self.derivative_prog.match(out_name)
        if match is None:
            raise ValueError('out_name is not in the form of a derivative')
        varname = match.group(1)
        coordname = match.group(2)
        temp_funcs, temp_func_args = funcs, func_args
        temp_extra_values = extra_values
        # Make sure we actually have our variable and coordinate
        for arg in (varname, coordname):
            if arg in exclude:
                raise ValueError('require an excluded variable to '
                                 'calculate the derivative')
            # See if this argument was input or we've already determined
            # it can be derived
            if arg in in_values.keys() or arg in temp_extra_values:
                # We already have this, check the next one
                continue
            try:
                # Check if this is a nested derivative
                result = self._get_derivative(arg, methods, exclude +
                                              (out_name,), temp_funcs,
                                              temp_func_args, in_values,
                                              temp_extra_values)
            except ValueError:
                pass
            # We'd need to calculate this argument. See if we can
            try:
                result = self._calculate(arg, methods, exclude +
                                         (out_name,), temp_funcs,
                                         temp_func_args, in_values,
                                         temp_extra_values)
            except ValueError:
                break  # We can't get this argument
            # We can get this argument, so add it to our list
            temp_funcs, temp_func_args, temp_extra_values = result
        else:  # We have all our arguments, since we didn't break
            # Now we have to construct a function...
            func, args = self._construct_derivative(varname, coordname)
            return (temp_funcs + (func,), temp_func_args + (args,),
                    temp_extra_values + (out_name,))
        raise ValueError('cannot calculate {} from available'
                         ' quantities'.format(out_name))

    def _construct_derivative(self, varname, coordname):
        '''
        Constructs a function that computes the derivative of the variable
        specified by varname with respect to the coordinate specified by
        coordname.

        Returns
        -------
        func: function
            Function which takes in the arguments returned by this function and
            returns the derivative of the variable specified by varname with
            respect to the coordinate specified by coordname
        args: iterable
            An iterable of strings corresponding to the variables that must be
            passed as arguments to the function returned by this function
        '''
        self.coord_axes
        self.override_coord_axes
        self.coords_own_axis
        self.coord_types
        axis = self.coord_axes[self.coord_types[coordname]]
        func = lambda data, coord: \
            ddx(data, axis, x=coord,
                axis_x=self.coords_own_axis[self.coord_types[coordname]])
        return (func, (varname, coordname))

    def _get_methods(self, method_options):
        '''
        Returns a dictionary of methods including the default methods of the
        class and specified optional methods. Will override a default method
        if an optional method is given that takes the same inputs and produces
        the same output.

        Parameters
        ----------
        methods: iterable
            Strings referring to optional methods in self.optional_methods.

        Raises
        ------
        ValueError
            If a method given is not present in self.optional_methods
            If multiple optional methods are selected which calculate the same
                output quantity from the same input quantities
        '''
        methods = {}
        for methodname in method_options:
            # Check that this is actually a valid method
            if methodname not in self.optional_methods.keys():
                raise ValueError('method {} is not defined'.format(methodname))
            # Iterate through each quantity this method lets you calculate
            for varname in self.optional_methods[methodname].keys():
                # Make sure we have a dictionary defined for this quantity
                if varname not in methods.keys():
                    methods[varname] = {}
                equations = self.optional_methods[methodname][varname]
                # Make sure another method is not already selected that does
                # this same calculation
                for vars_input in equations.keys():
                    if vars_input in methods[varname].keys():
                        raise ValueError(('Multiple methods for {} --> {} chos'
                                          'en').format(', '.join(vars_input),
                                                       varname))
                methods[varname].update(equations)
        output_methods = self.default_methods.copy()
        output_methods.update(methods)
        return output_methods


class FluidDeriver(_BaseDeriver):
    default_methods = {
        'AH': {('q', 'rho',): AH_from_q_rho},
        'p': {('rho', 'Tv',): p_ideal_gas},
        'q': {('wvap',): q_from_wvap,
              ('AH', 'rho',): q_from_AH_rho},
        'wvaps': {('p', 'es',): wvaps_from_p_es},
        'RH': {('q', 'qs',): RH_from_q_qs,
               ('wvap', 'wvaps',): RH_from_wvap_wvaps},
        'rho': {('p', 'Tv',): rho_ideal_gas,
                ('q', 'AH',): rho_from_q_AH},
        'Tv': {('p', 'rho',): Tv_ideal_gas},
        'theta': {('p', 'T',): theta_from_p_T},
        'wvap': {('q',): wvap_from_q},
    }
    optional_methods = {
        'Goff-Gratch': {
            'es': {('T'): es_from_T_Goff_Gratch},
        },
        'Bolton': {
            'es': {('T'): es_from_T_Bolton},
        },
        'hydrostatic': {
            'dpdz': {('rho'): dpdz_from_rho_hydrostatic},
            'rho': {('dpdz'): rho_from_dpdz_hydrostatic},
        },
        'dry': {
            'Tv': {('T',): Tv_from_T_assuming_dry},
            'T': {('Tv',): T_from_Tv_assuming_dry},
        }
    }

    def calculate(self, quantity_out, **kwargs):
        '''
        Calculates and returns a requested quantity from quantities passed in
        as keyword arguments.

        Parameters
        ----------
        varname_out : string
            Name of quantity to be calculated.
        methods : tuple, optional
            Names of methods that can be used for calculation, as strings.

        Quantity Parameters
        -------------------
        All quantity parameters are optional, and must be of the same type,
        either ndarray or iris Cube. If ndarrays are used, then units must
        match the units specified below. If iris Cube is used, any units may
        be used for input and the output will be given in the units specified
        below.

        q: Specific humidity (kg/kg)
        T: Temperature (K)
        p: Pressure (Pa)
        RH: Relative humidity (%)
        Tw: Wet bulb temperature (K)
        Tv: Virtual temperature (K)
        rho: Air density (kg/m^3)

        Returns
        -------
        quantity : ndarray or iris Cube
            Calculated quantity.
            Return type is the same as quantity parameter types.

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>>
        '''
        return super(FluidDeriver, self).calculate(quantity_out, **kwargs)


def calculate(output, methods=(), axis_coords=None, override_coord_axes=None,
              coords_own_axis=None, **kwargs):
    '''
    Calculates and returns a requested quantity from quantities passed in as
    keyword arguments.

    Parameters
    ----------
    output : string or iterable
        Name of quantity to be calculated. If iterable, should contain strings
        which are names of quantities to be calculated.
    methods : tuple, optional
        Names of methods that can be used for calculation, as strings.

    Quantity Parameters
    -------------------
    All quantity parameters are optional, and must be of the same type,
    either ndarray or iris Cube. If ndarrays are used, then units must match
    the units specified below. If iris Cube is used, any units may be used
    for input and the output will be given in the units specified below.

    T: Temperature (K)
    p: Pressure (Pa)
    q: Specific humidity (kg/kg)
    RH: Relative humidity (%)
    AH: Absolute humidity (kg/m^3)
    wvap: Water vapor mixing ratio (kg/kg)
    Tw: Wet bulb temperature (K)
    Tv: Virtual temperature (K)
    rho: Air density (kg/m^3)

    Returns
    -------
    quantity : ndarray or iris Cube
        Calculated quantity.
        Return type is the same as quantity parameter types.
        If multiple quantities are requested, returns a tuple containing the
        quantities.

    See Also
    --------

    Notes
    -----

    Examples
    --------
    >>>
    '''
    deriver = FluidDeriver(methods, axis_coords, override_coord_axes,
                           coords_own_axis, **kwargs)
    try:
        result = [deriver.calculate(var) for var in output]
    except TypeError:  # raised if output is not iterable
        result = deriver.calculate(output)
    if len(result) == 1:
        return result[0]
    else:
        return result
