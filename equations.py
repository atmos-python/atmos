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
# Need an equation or two for Td
#
# Need some more "shortcut" equations
#
# Equivalent potential temperature... What even.
#
# Check whether certain inputs are valid (0 < RH < 100, 0 < T, etc.)
import numpy as np
from constants import g0, Omega, Rd, Rv, Cpd, Lv0
from decorators import assumes, equation_docstring

quantities = {
    'AH': {
        'name': 'absolute humidity',
        'units': 'kg/m^3',
    },
    'DSE': {
        'name': 'dry static energy',
        'units': 'J',
    },
    'e': {
        'name': 'water vapor partial pressure',
        'units': 'Pa',
    },
    'es': {
        'name': 'saturation water vapor partial pressure',
        'units': 'Pa',
    },
    'f': {
        'name': 'Coriolis parameter',
        'units': 'Hz',
    },
    'Gammam': {
        'name': 'moist adiabatic lapse rate',
        'units': 'K/m',
    },
    'lat': {
        'name': 'latitude',
        'units': 'degrees N',
    },
    'lon': {
        'name': 'longitude',
        'units': 'degrees E',
    },
    'MSE': {
        'name': 'moist static energy',
        'units': 'J',
    },
    'N2': {
        'name': 'squared Brunt-Väisälä frequency',
        'units': 'Hz^2',
    },
    'omega': {
        'name': 'vertical velocity expressed as tendency of pressure',
        'units': 'Pa/s',
    },
    'p': {
        'name': 'pressure',
        'units': 'Pa',
    },
    'plcl': {
        'name': 'pressure at lifting condensation level',
        'units': 'Pa',
    },
    'Phi': {
        'name': 'geopotential',
        'units': 'm^2/s^2',
    },
    'qv': {
        'name': 'specific humidity',
        'units': 'kg/kg',
    },
    'qvs': {
        'name': 'saturation specific humidity',
        'units': 'kg/kg',
    },
    'RB': {
        'name': 'bulk Richardson number',
        'units': 'unitless',
    },
    'RH': {
        'name': 'relative humidity',
        'units': '%',
    },
    'rho': {
        'name': 'density',
        'units': 'kg/m^3',
    },
    'rv': {
        'name': 'water vapor mixing ratio',
        'units': 'kg/kg',
    },
    'rvs': {
        'name': 'saturation water vapor mixing ratio',
        'units': 'kg/kg',
    },
    'T': {
        'name': 'temperature',
        'units': 'K',
    },
    'Td': {
        'name': 'dewpoint temperature',
        'units': 'K',
    },
    'Tlcl': {
        'name': 'temperature at lifting condensation level',
        'units': 'K',
    },
    'Tv': {
        'name': 'virtual temperature',
        'units': 'K',
    },
    'Tae': {
        'name': 'adiabatic equivalent temperature',
        'units': 'K',
    },
    'Tie': {
        'name': 'isobaric equivalent temperature',
        'units': 'K',
    },
    'Tw': {
        'name': 'wet bulb temperature',
        'units': 'K',
    },
    'theta': {
        'name': 'potential temperature',
        'units': 'K',
    },
    'thetae': {
        'name': 'equivalent temperature',
        'units': 'K',
    },
    'thetaae': {
        'name': 'adiabatic equivalent temperature',
        'units': 'K',
    },
    'thetaie': {
        'name': 'isentropic equivalent temperature',
        'units': 'K',
    },
    'u': {
        'name': 'eastward zonal wind velocity',
        'units': 'm/s',
    },
    'v': {
        'name': 'northward meridional wind velocity',
        'units': 'm/s',
    },
    'w': {
        'name': 'vertical velocity',
        'units': 'm/s',
    },
    'x': {
        'name': 'x',
        'units': 'm',
    },
    'y': {
        'name': 'y',
        'units': 'm',
    },
    'z': {
        'name': 'height',
        'units': 'm',
    },
    'Z': {
        'name': 'geopotential height',
        'units': 'm',
    }
}
# all_methods = ('ideal gas', 'hydrostatic', 'constant g', 'constant Lv',
#                'bolton', 'goff-gratch', 'frozen bulb', 'unfrozen bulb',
#                'stipanuk', 'dry', 'Tv equals T')

assumptions = {
    'hydrostatic': 'hydrostatic balance',
    'constant g': 'g is constant',
    'constant Lv': 'latent heat of vaporization of water is constant',
    'ideal gas': 'the ideal gas law holds',
    'bolton': 'the assumptions in Bolton (1980) hold',
    'goff-gratch': 'the Goff-Gratch equation for es',
    'frozen bulb': 'the bulb is frozen',
    'unfrozen bulb': 'the bulb is not frozen',
    'Tv equals T': 'the virtual temperature correction can be neglected',
    'constant Cp': 'Cp is constant and equal to Cp for dry air at 0C',
    'no liquid water': 'liquid water can be neglected',
    'no solid water': 'ice can be neglected',
}


def autodoc(**kwargs):
    return equation_docstring(quantities, assumptions, **kwargs)


@autodoc(equation='AH = qv*rho')
@assumes()
def AH_from_qv_rho(qv, rho):
    return qv*rho


@autodoc(equation='DSE = Cpd*T + g0*z')
@assumes('constant g')
def DSE_from_T_z(T, z):
    return Cpd*T + g0*z


@autodoc(equation='DSE = Cpd*T + Phi')
@assumes()
def DSE_from_T_Phi(T, Phi):
    return Cpd*T + Phi


@autodoc(equation=' p*qv/(0.622+qv)')
@assumes()
def e_from_p_qv(p, qv):
    return p*qv/(0.622+qv)


@assumes('goff-gratch')
def e_from_p_T_Tw_Goff_Gratch(p, T, Tw):
    '''
    Calculates water vapor partial pressure (Pa) from air pressure (Pa),
    temperature (K), and wet bulb temperature (K). Approximates saturation
    vapor pressure at the wet bulb temperature using the Goff-Gratch equation.
    Uses an approximation from the Royal Observatory outlined in the referenced
    document.

    e = es(Tw) - 0.799e-3*p*(T-Tw)

    References
    ----------
    Wong, W.T. 1989: Comparison of Algorithms for the Computation of the
        Thermodynamic Properties of Moist Air, Technical Note (Local) No. 51,
        Royal Observatory, Hong Kong. Retrieved March 25, 2015 from
        http://www.weather.gov.hk/publica/tnl/tnl051.pdf
    '''
    return es_from_T_Goff_Gratch(Tw) - 0.799e-3*p*(T-Tw)


@assumes('bolton')
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


@autodoc(equation='e = es(Td)')
@assumes('goff-gratch')
def e_from_Td_Goff_Gratch(Td):
    return es_from_T_Goff_Gratch(Td)


@autodoc(equation='e = es(Td)')
@assumes('bolton')
def e_from_Td_Bolton(Td):
    return es_from_T_Bolton(Td)


@autodoc(equation='e = es - 6.60e-4*(1+0.00115(Tw-273.15)*(T-Tw))*p',
         references='''
Petty, G.W. 1958: A First Course in Atmospheric Thermodynamics. 1st Ed.
    Sundog Publishing. p.216
''')
@assumes('unfrozen bulb')
def e_from_p_es_T_Tw(p, es, T, Tw):
    return es-(0.000452679+7.59e-7*Tw)*(T-Tw)*p


@autodoc(equation='e = es - 5.82e-4*(1+0.00115(Tw-273.15)*(T-Tw))*p',
         references='''
Petty, G.W. 1958: A First Course in Atmospheric Thermodynamics. 1st Ed.
    Sundog Publishing. p.216
''')
@assumes('frozen bulb')
def e_from_p_es_T_Tw_frozen_bulb(p, es, T, Tw):
    return es-(0.000399181+6.693e-7*Tw)*(T-Tw)*p


@assumes('goff-gratch')
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


@assumes('bolton')
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


@autodoc(equation='f = 2.*Omega*sin(pi/180.*lat)')
@assumes()
def f_from_lat(lat):
    return 2.*Omega*np.sin(np.pi/180.*lat)


@autodoc(
    equation='Gammam = g0*(1+(Lv0*rvs)/(Rd*T))/(Cpd+(Lv0**2*rvs)/(Rv*T**2))',
    references='''
American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Saturation-adiabatic_lapse_rate
    Retrieved March 25, 2015
''')
@assumes('constant g', 'constant Lv')
def Gammam_from_rvs_T(rvs, T):
    '''
    Calculates saturation adiabatic lapse rate (K/m) from water vapor mixing
    ratio (kg/kg) and temperature (K), assuming constant g and latent heat of
    vaporization of water.

    Gammam = g0*(1+(Lv0*rvs)/(Rd*T))/(Cpd+(Lv0**2*rvs)/(Rv*T**2))

    From the American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Saturation-adiabatic_lapse_rate
    Retrieved March 25, 2015
    '''
    return g0*(1+(Lv0*rvs)/(Rd*T))/(Cpd+(Lv0**2*rvs)/(Rv*T**2))


@autodoc(equation='MSE = DSE + Lv0*qv')
@assumes('constant Lv')
def MSE_from_DSE_qv(DSE, qv):
    return DSE + Lv0*qv


@autodoc(equation='omega = -rho*g0*w')
@assumes('hydrostatic')
def omega_from_w_rho_hydrostatic(w, rho):
    return -rho*g0*w


@autodoc(equation='p = rho*Rd*Tv')
@assumes('ideal gas')
def p_from_rho_Tv_ideal_gas(rho, Tv):
    return rho*Rd*Tv


@autodoc(equation='plcl = p*(Tlcl/T)**(Cpd/Rd)')
@assumes('constant Cp')
def plcl_from_p_T_Tlcl(p, T, Tlcl):
    return p*(Tlcl/T)**(Cpd/Rd)


# =============================================================================
# @assumes('stipanuk')
# def plcl_from_p_T_Td(p, T, Td):
#     '''
#     Calculates LCL pressure level (Pa) from pressure (Pa), temperature (K),
#     and dew point temperature (K).
#
#     Calculates the pressure of the lifting condensation level computed by an
#     iterative procedure described by equations 8-12 (pp 13-14) of:
#
#     Stipanuk, G.S., (1973) original version.
#     "Algorithms for generating a skew-t, log p diagram and computing selected
#     meteorological quantities."
#
#     Atmospheric sciences laboratory
#     U.S. Army Electronics Command
#     White Sands Missile Range, New Mexico 88002
#     '''
#     raise NotImplementedError()
# =============================================================================


@autodoc(equation='Phi = g0*z')
@assumes('constant g')
def Phi_from_z(z):
    return g0*z


@autodoc(equation='qv = AH/rho')
@assumes()
def qv_from_AH_rho(AH, rho):
    return AH/rho


@autodoc(equation='qv = rv/(1+rv)')
@assumes()
def qv_from_rv(rv):
    return rv/(1.+rv)


@autodoc(equation='qv = (Rd/Rv)*e/(p-(1-Rd/Rv)*e)')
@assumes()
def qv_from_p_e(p, e):
    return 0.622*e/(p-0.378*e)


@autodoc(equation='qvs = rvs/(1+rvs)')
@assumes()
def qvs_from_rvs(rvs):
    return rvs/(1+rvs)


@autodoc(equation='qv = qv_from_p_e(p, es)')
@assumes()
def qvs_from_p_es(p, es):
    return qv_from_p_e(p, es)


@autodoc(equation='RH = qv/qvs*100.')
@assumes()
def RH_from_qv_qvs(qv, qvs):
    return qv/qvs*100.


@autodoc(equation='RH = rv/rvs*100.')
@assumes()
def RH_from_rv_rvs(rv, rvs):
    return rv/rvs*100.


@autodoc(equation='rho = AH/qv')
@assumes()
def rho_from_qv_AH(qv, AH):
    return AH/qv


@autodoc(equation='rho = p/(Rd*Tv)')
@assumes('ideal gas')
def rho_from_p_Tv_ideal_gas(p, Tv):
    return p/(Rd*Tv)


@autodoc(equation='rv = qv/(1-qv)')
@assumes()
def rv_from_qv(qv):
    return qv/(1-qv)


@autodoc(equation='(Rd/Rv)*e/(p-e)')
@assumes()
def rv_from_p_e(p, e):
    return 0.622*e/(p-e)


@autodoc(equation='rvs = rv_from_p_e(p, es)')
@assumes()
def rvs_from_p_es(p, es):
    return rv_from_p_e(p, es)


@autodoc(equation='rv = rv_from_qv(qvs)')
@assumes()
def rvs_from_qvs(qvs):
    return rv_from_qv(qvs)


@assumes('bolton')
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


@assumes('bolton')
def Tlcl_from_T_RH(T, RH):
    '''
    Calculates temperature at LCL (K) from temperature (K) and relative
    humidity (%) using Bolton (1980) equation 22.

    Tlcl = 1./((1./T-55.)-(log(RH/100.)/2840.)) + 55.

    References
    ----------
    David Bolton, 1980: The Computation of Equivalent Potential Temperature.
        Mon. Wea. Rev., 108, 1046–1053.
        doi: http://dx.doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    '''
    return 1./((1./(T-55.))-(np.log(RH/100.)/2840.)) + 55.


@assumes('bolton')
def Tlcl_from_T_Td(T, Td):
    '''
    Calculates temperature at LCL (K) from temperature (K) and dewpoint
    temperature (K) using Bolton (1980) equation 15.

    Tlcl = 1./((1./(Td-56.))-(log(T/Td)/800.)) + 56.

    References
    ----------
    David Bolton, 1980: The Computation of Equivalent Potential Temperature.
        Mon. Wea. Rev., 108, 1046–1053.
        doi: http://dx.doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    '''
    return 1./((1./(Td-56.))+(np.log(T/Td)/800.)) + 56.


@assumes('bolton')
def Tlcl_from_T_e(T, e):
    '''
    Calculates temperature at LCL (K) from temperature (K) and water vapor
    partial pressure (Pa) using Bolton (1980) equation 21.

    Tlcl = 2840./(3.5*log(T)-log(e)-4.805) + 55.

    References
    ----------
    David Bolton, 1980: The Computation of Equivalent Potential Temperature.
        Mon. Wea. Rev., 108, 1046–1053.
        doi: http://dx.doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    '''
    return 2840./(3.5*np.log(T)-np.log(e)-4.805) + 55.


@autodoc(
    equation='Tae = T*exp(Lv0*rv/(Cpd*T))',
    references='''
American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Equivalent_temperature
    Retrieved March 25, 2015
''')
@assumes('constant Lv')
def Tae_from_T_rv(T, rv):
    return T*np.exp(Lv0*rv/(Cpd*T))


@autodoc(
    equation='Tie = T*(1.+Lv0*rv/(Cpd*T))',
    references='''
American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Equivalent_temperature
    Retrieved March 25, 2015
''')
@assumes('constant Lv')
def Tie_from_T_rv(T, rv):
    return T*(1.+Lv0*rv/(Cpd*T))


@autodoc(equation='Tv/(1+0.608*qv)')
@assumes('no liquid water', 'no solid water')
def T_from_Tv_qv(Tv, qv):
    return Tv/(1+0.608*qv)


@autodoc(equation='T*(1+0.608*qv)')
@assumes('no liquid water', 'no solid water')
def Tv_from_T_qv(T, qv):
    return T*(1+0.608*qv)


@assumes('Tv equals T')
def Tv_from_T_assuming_Tv_equals_T(T):
    '''
    Calculates virtual temperature from temperature assuming no moisture.
    That is to say, it returns the input back.

    This function exists to allow using temperature as virtual temperature with
    a "dry" assumption.

    Tv = T
    '''
    return 1.*T


@autodoc(equation='Tv = p/(rho*Rd)')
@assumes('ideal gas')
def Tv_from_p_rho_ideal_gas(p, rho):
    return p/(rho*Rd)


@assumes('stull')
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
            + np.arctan(T-273.15+RH) - np.arctan(RH - 1.676331)
            + 0.00391838*RH**1.5*np.arctan(0.023101*RH) - 4.686035 + 273.15)


@assumes('Tv equals T')
def T_from_Tv_assuming_Tv_equals_T(Tv):
    '''
    Calculates temperature from virtual temperature assuming no moisture.
    That is to say, it returns the input back.

    This function exists to allow using temperature as virtual temperature with
    a "dry" assumption.

    T = Tv
    '''
    return 1.*Tv


@autodoc(equation='theta = T*(1e5/p)**(Rd/Cpd)')
@assumes('constant Cp')
def theta_from_p_T(p, T):
    return T*(1e5/p)**(Rd/Cpd)


@assumes('bolton', 'constant Cp')
def thetae_from_theta_Tlcl_rv_Bolton(theta, Tlcl, rv):
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
    return theta*np.exp((3.376/Tlcl-0.00254)*rv*1e3*(1+0.81*rv))


@autodoc(
    equation='thetaie = theta*(1+Lv0*rv/(Cpd*T))',
    references='''
Petty, G.W. 2008: A First Course in Atmospheric Thermodynamics,
    Sundog Publishing, pg. 203
''')
@assumes('constant Lv')
def thetaie_from_T_theta_rv(T, theta, rv):
    return theta*(1+Lv0*rv/(Cpd*T))


@autodoc(
    equation='thetaie = Tie*(1e5/p)**(Rd/Cpd)',
    references='''
Petty, G.W. 2008: A First Course in Atmospheric Thermodynamics,
    Sundog Publishing, pg. 203
''')
@assumes('constant Cp')
def thetaie_from_p_Tie_rv(p, Tie, rv):
    return Tie*(1e5/p)**(Rd/Cpd)


@autodoc(equation='thetaae = Tae*(1e5/p)**(Rd/Cpd)')
@assumes('constant Cp')
def thetaae_from_p_Tae_rv(p, Tae, rv):
    return Tae*(1e5/p)**(Rd/Cpd)


@autodoc(equation='w = -omega/(rho*g0)')
@assumes('constant g')
def w_from_omega_rho_hydrostatic(omega, rho):
    return -omega/(rho*g0)


@autodoc(equation='z = Phi/g0')
@assumes('constant g')
def z_from_Phi(Phi):
    return Phi/g0
