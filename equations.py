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
import numpy as np
from constants import g0, stefan, r_earth, Omega, Rd, Rv, Cpd, Gammad, Lv0


# Define some decorators for our equations
def _assumes(*args):
    '''Stores a function's assumptions as an attribute.'''
    args = tuple(args)

    def decorator(func):
        func.assumptions = args
        return func
    return decorator


def _inputs(*args):
    '''Stores a function's input quantities as an attribute.'''
    args = tuple(args)

    def decorator(func):
        func.inputs = args
        return func
    return decorator


@_assumes()
def AH_from_q_rho(q, rho):
    '''
    Calculates absolute humidity (kg/m^3) from specific humidity (kg/kg) and
    air density (kg/m^3).

    AH = q*rho
    '''
    return q*rho


@_assumes('hydrostatic')
def dpdz_from_rho_hydrostatic(rho):
    '''
    Calculates vertical derivative of pressure (Pa/m) from density (kg/m^3),
    assuming hydrostatic balance.

    dp/dz = -rho*g0
    '''
    return -rho*g0


@_assumes('constant g')
def DSE_from_T_z(T, z):
    '''
    Calculates dry static energy (J) from temperature (K) and height (m)
    assuming constant g.

    DSE = Cp*T + g0*z
    '''
    return Cpd*T + g0*z


@_assumes()
def DSE_from_T_Phi(T, Phi):
    '''
    Calculates dry static energy (J) from temperature (K) and geopotential
    height (m^2/s^2).

    DSE = Cp*T + Phi
    '''
    return Cpd*T + Phi


@_assumes()
def e_from_p_q(p, q):
    '''
    Calculates water vapor partial pressure (Pa) from air pressure (Pa) and
    specific humidity (kg/kg).
    '''
    return p*q/(0.622+q)


@_assumes('goff-gratch')
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


@_assumes('bolton')
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


@_assumes('goff-gratch')
def e_from_Td_Goff_Gratch(Td):
    '''
    Calculates water vapor partial pressure (Pa) from dewpoint temperature (K).
    Uses the Goff-Gratch equation for saturation water vapor partial pressure.

    e = es(Td)
    '''
    return es_from_T_Goff_Gratch(Td)


@_assumes('bolton')
def e_from_Td_Bolton(Td):
    '''
    Calculates water vapor partial pressure (Pa) from dewpoint temperature (K).
    Uses Bolton's approximation to Wexler's formula for saturation water vapor
    partial pressure.

    e = es(Td)
    '''
    return es_from_T_Bolton(Td)


@_assumes('unfrozen bulb')
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


@_assumes('frozen bulb')
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


@_assumes('goff-gratch')
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


@_assumes('bolton')
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


@_assumes()
def f_from_lat(lat):
    '''
    Calculates the Coriolis parameter (Rad/s) from latitude (degrees N).

    f = 2.*Omega*sin(pi/180.*lat)
    '''
    return 2.*Omega*np.sin(np.pi/180.*lat)


@_assumes('constant g', 'constant Lv')
def Gammam_from_wvaps_T(wvaps, T):
    '''
    Calculates saturation adiabatic lapse rate (K/m) from pressure (Pa) and
    temperature (K), assuming constant g and latent heat of vaporization of
    water.

    Gammam = g0*(1+(Lv0*wvaps)/(Rd*T))/(Cpd+(Lv0**2*wvaps)/(Rv*T**2))

    From the American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Saturation-adiabatic_lapse_rate
    Retrieved March 25, 2015
    '''
    return g0*(1+(Lv0*wvaps)/(Rd*T))/(Cpd+(Lv0**2*wvaps)/(Rv*T**2))


@_assumes('constant Lv')
def MSE_from_DSE_q(DSE, q):
    '''
    Calculates moist static energy (J) from dry static energy (J) and specific
    humidity (kg/kg) assuming constant latent heat of vaporization of water.

    MSE = DSE + Lv0*q
    '''
    return DSE + Lv0*q


@_assumes('hydrostatic')
def omega_from_w_rho_hydrostatic(w, rho):
    '''
    Calculates pressure tendency (Pa/s) from vertical velocity (m/s) and
    density (kg/m^3) using the hydrostatic assumption.

    omega = -rho*g0*w
    '''
    return -rho*g0*w


@_assumes('ideal gas')
def p_from_rho_Tv_ideal_gas(rho, Tv):
    '''
    Calculates pressure (Pa) from density (kg/m^3) and virtual temperature (K)
    assuming an ideal gas.

    p = rho*Rd*Tv
    '''
    return rho*Rd*Tv


@_assumes('stipanuk')
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


@_assumes('constant g')
def Phi_from_z(z):
    '''
    Calculates geopotential height (m^2/s^2) from height (m) assuming constant
    g.

    Phi = g0*z
    '''
    return g0*z


@_assumes()
def q_from_AH_rho(AH, rho):
    '''
    Calculates specific humidity (kg/kg) from absolute humidity (kg/m^3) and
    air density (kg/m^3).

    q = AH/rho
    '''
    return AH/rho


@_assumes()
def q_from_wvap(wvap):
    '''
    Calculates specific humidity (kg/kg) from water vapor mixing ratio (kg/kg).

    q = wvap/(1+wvap)
    '''
    return wvap/(1+wvap)


@_assumes()
def q_from_p_e(p, e):
    '''
    Calculates specific humidity (kg/kg) from air pressure (Pa) and water vapor
    partial pressure (Pa).
    '''
    return 0.622*e/(p-e)


@_assumes()
def qs_from_wvaps(wvaps):
    '''
    Calculates saturation specific humidity (kg/kg) from saturated water vapor
    mixing ratio (kg/kg).

    q = wvap/(1+wvap)
    '''
    return wvaps/(1+wvaps)


@_assumes()
def RH_from_q_qs(q, qs):
    '''
    Calculates relative humidity (%) from specific humidity (kg/kg) and
    saturation specific humidity (kg/kg).

    RH = q/qs*100.
    '''
    return q/qs*100.


@_assumes()
def RH_from_wvap_wvaps(wvap, wvaps):
    '''
    Calculates relative humidity (%) from mixing ratio (kg/kg) and
    saturation mixing ratio (kg/kg)

    RH = wvap/wvaps*100.
    '''
    return wvap/wvaps*100.


@_assumes()
def rho_from_q_AH(q, AH):
    '''
    Calculates density (kg/m^3) from specific humidity (kg/kg) and absolute
    humidity (kg/m^3).

    rho = AH/q
    '''
    return AH/q


@_assumes('hydrostatic', 'constant g')
def rho_from_dpdz_hydrostatic(dpdz):
    '''
    Calculates density (kg/m^3) from vertical derivative of pressure (Pa/m)
    assuming hydrostatic balance and constant g.

    rho = -dpdz/g0
    '''
    return -dpdz/g0


@_assumes('ideal gas')
def rho_from_p_Tv_ideal_gas(p, Tv):
    '''
    Calculates density (kg/m^3) from pressure (Pa) and virtual temperature (K)
    assuming an ideal gas.

    rho = p/(Rd*Tv)
    '''
    return p/(Rd*Tv)


@_assumes('bolton')
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


@_assumes('bolton')
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


@_assumes('bolton')
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


@_assumes('bolton')
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


@_assumes('constant Lv')
def Tae_from_T_wvap(T, wvap):
    '''
    Calculates adiabatic equivalent temperature (aka pseudoequivalent
    temperature) (K) from temperature (K) and water vapor mixing ratio (kg/kg)

    Tae = T*exp(Lv0*wvap/(Cpd*T))

    From the American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Equivalent_temperature
    Retrieved March 25, 2015
    '''
    return T*np.exp(Lv0*wvap/(Cpd*T))


@_assumes('constant Lv')
def Tie_from_T_wvap(T, wvap):
    '''
    Calculates isobaric equivalent temperature (K) from temperature (K) and
    water vapor mixing ratio (kg/kg).

    Tie = T*(1.+Lv0*wvap/(Cpd*T))

    From the American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Equivalent_temperature
    Retrived March 25, 2015
    '''
    return T*(1.+Lv0*wvap/(Cpd*T))


@_assumes()
def Tv_from_T_p_q(T, q):
    '''
    Calculates virtual temperature from temperature (K) and specific
    humidity (kg/kg).
    '''
    raise NotImplementedError()


@_assumes('Tv equals T')
def Tv_from_T_assuming_Tv_equals_T(T):
    '''
    Calculates virtual temperature from temperature assuming no moisture.
    That is to say, it returns the input back.

    This function exists to allow using temperature as virtual temperature with
    a "dry" assumption.

    Tv = T
    '''
    return 1.*T


@_assumes('ideal gas')
def Tv_ideal_gas(p, rho):
    '''
    Calculates virtual temperature (K) from density (kg/m^3) and pressure (Pa).

    Tv = p/(rho*Rd)
    '''
    return p/(rho*Rd)


@_assumes('stull')
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


@_assumes('Tv equals T')
def T_from_Tv_assuming_Tv_equals_T(Tv):
    '''
    Calculates temperature from virtual temperature assuming no moisture.
    That is to say, it returns the input back.

    This function exists to allow using temperature as virtual temperature with
    a "dry" assumption.

    T = Tv
    '''
    return 1.*Tv


@_assumes('constant Cp')
def theta_from_p_T(p, T):
    '''
    Calculates potential temperature (K) from pressure (Pa) and temperature
    (K). Assumes Cp does not vary with pressure or temperature.

    theta = T*(1e5/p)**(Rd/Cpd)
    '''
    return T*(1e5/p)**(Rd/Cpd)


@_assumes('bolton', 'constant Cp')
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


@_assumes('constant Lv')
def thetaie_from_T_theta_wvap(T, theta, wvap):
    '''
    Calculates isobaric equivalent potential temperature (K) from temperature
    (K), potential temperature (K), and water vapor mixing ratio (kg/kg).

    thetaie = theta*(1+Lv0*wvap/(Cpd*T))

    References
    ----------
    Petty, G.W. 2008: A First Course in Atmospheric Thermodynamics,
        Sundog Publishing, pg. 203
    '''
    return theta*(1+Lv0*wvap/(Cpd*T))


@_assumes('constant Cp')
def thetaie_from_p_Tie_wvap(p, Tie, wvap):
    '''
    Calculates isobaric equivalent potential temperature (K) from isobaric
    equivalent temperature (K) and water vapor mixing ratio (kg/kg).

    thetaie = Tie*(1e5/p)**(Rd/Cpd)

    References
    ----------
    Petty, G.W. 2008: A First Course in Atmospheric Thermodynamics,
        Sundog Publishing, pg. 203
    '''
    return Tie*(1e5/p)**(Rd/Cpd)


@_assumes('constant Cp')
def thetaae_from_p_Tae_wvap(p, Tae, wvap):
    '''
    Calculates adiabatic equivalent potential temperature (K) from adiabatic
    equivalent temperature (K) and water vapor mixing ratio (kg/kg).

    thetaae = Tae*(1e5/p)**(Rd/Cpd)
    '''
    return Tae*(1e5/p)**(Rd/Cpd)


@_assumes('constant g')
def w_from_omega_rho_hydrostatic(omega, rho):
    '''
    Calculates vertical velocity (m/s) from vertical pressure tendency (Pa/s)
    and density (kg/m^3) using the hydrostatic assumption.

    w = -omega/(rho*g0)
    '''
    return -omega/(rho*g0)


@_assumes()
def wvap_from_q(q):
    '''
    Calculates water vapor mixing ratio (kg/kg) from specific humidity (kg/kg).

    wvap = q/(1-q)
    '''
    return q/(1-q)


@_assumes()
def wvaps_from_p_es(p, es):
    '''
    Calculates saturation mixing ratio from pressure (Pa) and saturation
    vapor pressure (Pa).

    wvaps = Rd/Rv*es/(p-es)
    '''
    return Rd/Rv*es/(p-es)


@_assumes()
def wvaps_from_qs(qs):
    '''
    Calculates saturation water vapor mixing ratio (kg/kg) from saturation
    specific humidity (kg/kg).

    wvap = q/(1-q)
    '''
    return qs/(1-qs)


@_assumes('constant g')
def z_from_Phi(Phi):
    '''
    Calculates height (m) from geopotential height (m^2/s^2) assuming constant
    g.

    z = Phi/g0
    '''
    return Phi/g0
