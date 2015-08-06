# -*- coding: utf-8 -*-
"""
equations.py: Fluid dynamics equations for atmospheric sciences.
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
# Need an equation or two for Td
#
# Need some more "shortcut" equations
#
# Check whether certain inputs are valid (0 < RH < 100, 0 < T, etc.)
# can't use numexpr with unicode_literals because of bugs in older versions
# of numexpr
from __future__ import division, absolute_import
import numpy as np
import numexpr as ne
from numpy import pi
from atmos.constants import g0, Omega, Rd, Rv, Cpd, Lv0, Cl
from atmos.decorators import assumes, overridden_by_assumptions
from atmos.decorators import equation_docstring

ref = {'AMS Glossary Gammam': '''
American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Saturation-adiabatic_lapse_rate
    Retrieved March 25, 2015''',
       'AMS Glossary thetae': '''
American Meteorological Society Glossary of Meteorology
    http://glossary.ametsoc.org/wiki/Equivalent_potential_temperature
    Retrieved April 23, 2015''',
       'Petty 2008': '''
Petty, G.W. 2008: A First Course in Atmospheric Thermodynamics. 1st Ed.
    Sundog Publishing.''',
       'Goff-Gratch': '''
Goff, J. A., and Gratch, S. 1946: Low-pressure properties of water
    from -160 to 212 F, in Transactions of the American Society of
    Heating and Ventilating Engineers, pp 95-122, presented at the
    52nd annual meeting of the American Society of Heating and
    Ventilating Engineers, New York, 1946.''',
       'Wexler 1976': '''
Wexler, A. (1976): Vapor pressure formulation for water in range 0 to
    100 C. A revision. J. Res. Natl. Bur. Stand. A, 80, 775-785.''',
       'Bolton 1980': '''
Bolton, D. 1980: The Computation of Equivalent Potential Temperature.
    Mon. Wea. Rev., 108, 1046-1053.
    doi: http://dx.doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2''',
       'Stull 2011': '''
Stull, R. 2011: Wet-Bulb Temperature from Relative Humidity and Air
    Temperature. J. Appl. Meteor. Climatol., 50, 2267-2269.
    doi: http://dx.doi.org/10.1175/JAMC-D-11-0143.1''',
       'Davies-Jones 2009': '''
Davies-Jones, R. 2009: On Formulas for Equivalent Potential
    Temperature. Mon. Wea. Rev., 137, 3137-3148.
    doi: http://dx.doi.org/10.1175/2009MWR2774.1'''
       }

# A dictionary describing the quantities used for and computed by the equations
# in this module. This makes it possible to automatically list these in
# documentation.
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
        'name': 'saturation water vapor partial pressure over water',
        'units': 'Pa',
    },
    'esi': {
        'name': 'saturation water vapor partial pressure over ice',
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
        'units': 'degrees',
    },
    'lon': {
        'name': 'longitude',
        'units': 'degrees',
    },
    'MSE': {
        'name': 'moist static energy',
        'units': 'J',
    },
    'N2': {
        'name': 'squared Brunt-Vaisala frequency',
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
        'name': 'saturation specific humidity with respect to liquid water',
        'units': 'kg/kg',
    },
    'qvsi': {
        'name': 'saturation specific humidity with respect to ice',
        'units': 'kg/kg',
    },
    'qi': {
        'name': 'specific humidity with respect to ice',
        'units': 'kg/kg',
    },
    'ql': {
        'name': 'specific humidity with respect to liquid water',
        'units': 'kg/kg',
    },
    'qt': {
        'name': 'specific humidity with respect to total water',
        'units': 'kg/kg',
    },
    'RB': {
        'name': 'bulk Richardson number',
        'units': 'dimensionless',
    },
    'RH': {
        'name': 'relative humidity with respect to liquid water',
        'units': 'percent',
    },
    'RHi': {
        'name': 'relative humidity with respect to ice',
        'units': 'percent',
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
        'name': 'saturation water vapor mixing ratio with respect to liquid '
                'water',
        'units': 'kg/kg',
    },
    'rvsi': {
        'name': 'saturation water vapor mixing ratio with respect to ice',
        'units': 'kg/kg',
    },
    'ri': {
        'name': 'ice mixing ratio',
        'units': 'kg/kg',
    },
    'rl': {
        'name': 'liquid water mixing ratio',
        'units': 'kg/kg',
    },
    'rt': {
        'name': 'total water mixing ratio',
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
    'Tw': {
        'name': 'wet bulb temperature',
        'units': 'K',
    },
    'theta': {
        'name': 'potential temperature',
        'units': 'K',
    },
    'thetae': {
        'name': 'equivalent potential temperature',
        'units': 'K',
    },
    'thetaes': {
        'name': 'saturation equivalent potential temperature',
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


# A dict of assumptions used by equations in this module. This helps allow
# automatic docstring generation.
assumptions = {
    'hydrostatic': 'hydrostatic balance',
    'constant g': 'g is constant',
    'constant Lv': 'latent heat of vaporization of water is constant',
    'ideal gas': 'the ideal gas law holds',
    'bolton': 'the assumptions in Bolton (1980) hold',
    'goff-gratch': 'the Goff-Gratch equation for es and esi',
    'frozen bulb': 'the bulb is frozen',
    'unfrozen bulb': 'the bulb is not frozen',
    'Tv equals T': 'the virtual temperature correction can be neglected',
    'constant Cp': 'Cp is constant and equal to Cp for dry air at 0C',
    'no liquid water': 'liquid water can be neglected',
    'no ice': 'ice can be neglected',
    'low water vapor': ('terms that are second-order in moisture quantities '
                        'can be neglected (eg. qv == rv)'),
    'cimo': 'the CIMO guide equation for esi',
}


# particularize the docstring decorator for this module's quantities and
# assumptions.
def autodoc(**kwargs):
    return equation_docstring(quantities, assumptions, **kwargs)


# Note that autodoc() must always be placed *above* assumes(), so that it
# has information about the assumptions (each decorator decorates the result
# of what is below it).
@autodoc(equation=r'AH = q_v \rho')
@assumes()
def AH_from_qv_rho(qv, rho):
    return ne.evaluate('qv*rho')


@autodoc(equation=r'DSE = C_{pd} T + g_0 z')
@assumes('constant g')
def DSE_from_T_z(T, z):
    return ne.evaluate('Cpd*T + g0*z')


@autodoc(equation=r'DSE = C_{pd} T + \Phi')
@assumes()
def DSE_from_T_Phi(T, Phi):
    return ne.evaluate('Cpd*T + Phi')


@autodoc(equation=r'e = p \frac{q_v}{0.622+q_v}')
@assumes()
def e_from_p_qv(p, qv):
    return ne.evaluate('p*qv/(0.622+qv)')


@autodoc(equation=r'e = es(Td)', references=ref['Goff-Gratch'])
@assumes('goff-gratch')
def e_from_Td_Goff_Gratch(Td):
    return es_from_T_Goff_Gratch(Td)


@autodoc(equation=r'e = es(Td)')
@assumes('bolton')
def e_from_Td_Bolton(Td):
    return es_from_T_Bolton(Td)


@autodoc(equation=r'e = es(T_w) - (6.60 \times 10^{-4}) '
         '(1 + 0.00115 (T_w-273.15) (T-T_w)) p',
         references=ref['Petty 2008'])
@assumes('unfrozen bulb', 'goff-gratch')
def e_from_p_T_Tw_Goff_Gratch(p, T, Tw):
    es = es_from_T_Goff_Gratch(Tw)
    return ne.evaluate('es-(0.000452679+7.59e-7*Tw)*(T-Tw)*p')


@autodoc(equation=r'e = es(T_w) - (5.82 \times 10^{-4}) '
         r'(1 + 0.00115 (T_w-273.15) '
         r' (T-T_w)) p',
         references=ref['Petty 2008'])
@assumes('frozen bulb', 'goff-gratch')
def e_from_p_T_Tw_frozen_bulb_Goff_Gratch(p, T, Tw):
    es = es_from_T_Goff_Gratch(Tw)
    return ne.evaluate('es-(0.000399181+6.693e-7*Tw)*(T-Tw)*p')


@autodoc(equation=r'e = es(T_w) - (6.60 \times 10^{-4}) '
         r'(1 + 0.00115 (T_w-273.15) (T-T_w)) p',
         references=ref['Petty 2008'])
@assumes('unfrozen bulb', 'bolton')
def e_from_p_T_Tw_Bolton(p, T, Tw):
    es = es_from_T_Bolton(Tw)
    return ne.evaluate('es-(0.000452679+7.59e-7*Tw)*(T-Tw)*p')


@autodoc(equation=r'e = es(T_w) - (5.82 \times 10^{-4}) '
         r'(1 + 0.00115 (T_w-273.15) '
         r' (T-T_w)) p',
         references=ref['Petty 2008'])
@assumes('frozen bulb', 'bolton')
def e_from_p_T_Tw_frozen_bulb_Bolton(p, T, Tw):
    es = es_from_T_Bolton(Tw)
    return ne.evaluate('es-(0.000399181+6.693e-7*Tw)*(T-Tw)*p')


@autodoc(references=ref['Goff-Gratch'] + '''
Goff, J. A. (1957) Saturation pressure of water on the new Kelvin
    temperature scale, Transactions of the American Society of Heating and
    Ventilating Engineers, pp 347-354, presented at the semi-annual meeting of
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
    of the Royal Meteorological Society 131(608): 1539-1565.
    doi:10.1256/qj.04.94''',
         notes='''
The original Goff-Gratch (1946) equation reads as follows:

| Log10(es) = -7.90298 (Tst/T-1)
|             + 5.02808 Log10(Tst/T)
|             - 1.3816*10-7 (10^(11.344 (1-T/Tst)) - 1)
|             + 8.1328*10-3 (10^(-3.49149 (Tst/T-1)) - 1)
|             + Log10(es_st)

where:
* Log10 refers to the logarithm in base 10
* es is the saturation water vapor pressure (hPa)
* T is the absolute air temperature in kelvins
* Tst is the steam-point (i.e. boiling point at 1 atm.) temperature (373.16K)
* es_st is es at the steam-point pressure (1 atm = 1013.25 hPa)

This formula is accurate but computationally intensive. For most purposes,
a more approximate formula is appropriate.''')
@assumes('goff-gratch')
def es_from_T_Goff_Gratch(T):
    return ne.evaluate(
        '''101325.*10.**(-7.90298*(373.16/T-1.) + 5.02808*log10(373.16/T)
        - 1.3816e-7 * (10.**(11.344*(1.-T/373.16))-1.)
        + 8.1328e-3*(10.**(-3.49149*(373.16/T-1.))-1.))''')


@autodoc(equation=r'es(T) = 611.2 exp(17.67 '
         '\frac{T-273.15}{T-29.65})',
         references=ref['Bolton 1980'] + ref['Wexler 1976'],
         notes='''
Fits Wexler's formula to an accuracy of 0.1% for temperatures between
-35C and 35C.''')
@assumes('bolton')
def es_from_T_Bolton(T):
    return ne.evaluate('611.2*exp(17.67*(T-273.15)/(T-29.65))')


@autodoc(equation=r'T_d = \frac{17.67*273.15 - 29.65 ln(\frac{e}{611.2})}'
                  r'{17.67-ln(\frac{e}{611.2})}',
         references=ref['Bolton 1980'],
         notes='''
Obtained by inverting Bolton's formula, es(Td) = T.''')
@assumes('bolton')
def Td_from_e_Bolton(e):
    return ne.evaluate('(17.67*273.15 - 29.65*log(e/611.2))/'
                       '(17.67-log(e/611.2))')


@autodoc(
    equation=r'esi(T) = 610.71 * 10^{9.09718 (273.16/T - 1) - 3.56654 '
             r'log_{10}(273.16/T) + 0.876793 (1 - T/273.16)}',
    notes='''
Valid between -100C and 0C.''')
@assumes('goff-gratch')
def esi_from_T_Goff_Gratch(T):
    return ne.evaluate(
        '''610.71 * 10**(-9.09718*(273.16/T - 1) - 3.56654*log10(273.16/T)
                      + 0.876793*(1 - T/273.16))''')


@autodoc(
    equation=r'esi = 6.112*e^{22.46*\frac{T - 273.15}{T - 0.53}}',
    notes='''
Matches Goff-Gratch within 0.2% from -70C to 0C, 2.5% from -100C to -70C.''')
@assumes('cimo')
def esi_from_T_CIMO(T):
    return ne.evaluate('611.2*exp(22.46*(T - 273.15)/(T - 0.53))')


@autodoc(equation=r'f = 2 \Omega sin(\frac{\pi}{180.} lat)')
@assumes()
def f_from_lat(lat):
    return ne.evaluate('2.*Omega*sin(pi/180.*lat)')


@autodoc(
    equation=r'Gammam = g_0 \frac{1+\frac{L_{v0}*r_{vs}}{R_d T}}{C_{pd}+'
    r'\frac{L_{v0}^2*r_{vs}}{R_v T^2}}',
    references=ref['AMS Glossary Gammam'])
@assumes('constant g', 'constant Lv')
def Gammam_from_rvs_T(rvs, T):
    return ne.evaluate('g0*(1+(Lv0*rvs)/(Rd*T))/(Cpd+(Lv0**2*rvs*0.622)/'
                       '(Rd*T**2))')


@autodoc(equation=r'MSE = DSE + L_{v0} q_v')
@assumes('constant Lv')
def MSE_from_DSE_qv(DSE, qv):
    return ne.evaluate('DSE + Lv0*qv')


@autodoc(equation=r'\omega = - \rho g_0 w')
@assumes('hydrostatic', 'constant g')
def omega_from_w_rho_hydrostatic(w, rho):
    return ne.evaluate('-rho*g0*w')


@autodoc(equation=r'p = \rho R_d T_v')
@assumes('ideal gas')
def p_from_rho_Tv_ideal_gas(rho, Tv):
    return ne.evaluate('rho*Rd*Tv')


@autodoc(equation=r'p_{lcl} = p (\frac{T_{lcl}}{T})^(\frac{C_{pd}}{R_d})')
@assumes('constant Cp')
def plcl_from_p_T_Tlcl(p, T, Tlcl):
    return ne.evaluate('p*(Tlcl/T)**(Cpd/Rd)')


@autodoc(equation=r'Phi = g_0 z')
@assumes('constant g')
def Phi_from_z(z):
    return ne.evaluate('g0*z')


@autodoc(equation=r'q_v = \frac{AH}{\rho}')
@assumes()
def qv_from_AH_rho(AH, rho):
    return ne.evaluate('AH/rho')


@autodoc(equation=r'q_v = \frac{r_v}{1+r_v}')
@assumes()
@overridden_by_assumptions('low water vapor')
def qv_from_rv(rv):
    return ne.evaluate('rv/(1.+rv)')


@autodoc(equation=r'q_v = r_v')
@assumes('low water vapor')
def qv_from_rv_lwv(rv):
    return 1.*rv


@autodoc(equation=r'q_v = \frac{R_d}{R_v} \frac{e}{p-(1-\frac{R_d}{R_v}) e}')
@assumes()
@overridden_by_assumptions('low water vapor')
def qv_from_p_e(p, e):
    return ne.evaluate('0.622*e/(p-0.378*e)')


@autodoc(equation=r'qv = (\frac{R_d}{R_v}) \frac{e}{p}')
@assumes('low water vapor')
def qv_from_p_e_lwv(p, e):
    return ne.evaluate('0.622*e/p')


@autodoc(equation=r'q_{vs} = \frac{r_{vs}}{1+r_{vs}}')
@assumes()
@overridden_by_assumptions('low water vapor')
def qvs_from_rvs(rvs):
    return qv_from_rv(rvs)


@autodoc(equation=r'q_v = r_v')
@assumes('low water vapor')
def qvs_from_rvs_lwv(rvs):
    return 1.*rvs


@autodoc(equation=r'q_{vs} = qv\_from\_p\_e(p, e_s)')
@assumes()
@overridden_by_assumptions('low water vapor')
def qvs_from_p_es(p, es):
    return qv_from_p_e(p, es)


@autodoc(equation=r'q_{vs} = qv\_from\_p\_e\_lwv(p, e_s)')
@assumes('low water vapor')
def qvs_from_p_es_lwv(p, es):
    return qv_from_p_e_lwv(p, es)


@autodoc(equation=r'q_{vsi} = qv\_from\_p\_e(p, e_{si})')
@assumes()
@overridden_by_assumptions('low water vapor')
def qvsi_from_p_esi(p, esi):
    return qv_from_p_e(p, esi)


@autodoc(equation=r'q_{vsi} = qv\_from\_p\_e\_lwv(p, e_{si})')
@assumes('low water vapor')
def qvsi_from_p_esi_lwv(p, esi):
    return qv_from_p_e_lwv(p, esi)


@autodoc(equation=r'q_t = q_i+q_v+q_l')
@assumes()
@overridden_by_assumptions('no liquid water', 'no ice')
def qt_from_qi_qv_ql(qi, qv, ql):
    return ne.evaluate('qi+qv+ql')


@autodoc(equation=r'q_t = q_v+q_l')
@assumes('no ice')
@overridden_by_assumptions('no liquid water')
def qt_from_qv_ql(qv, ql):
    return ne.evaluate('qv+ql')


@autodoc(equation='q_t = q_v')
@assumes('no liquid water', 'no ice')
def qt_from_qv(qv):
    return 1.*qv


@autodoc(equation='q_t = q_v+q_l')
@assumes('no liquid water')
@overridden_by_assumptions('no ice')
def qt_from_qv_qi(qv, qi):
    return ne.evaluate('qv+qi')


@autodoc(equation='q_v = q_t')
@assumes('no liquid water', 'no ice')
def qv_from_qt(qt):
    return 1.*qt


@autodoc(equation='q_v = q_t-q_l-q_i')
@assumes()
@overridden_by_assumptions('no liquid water', 'no ice')
def qv_from_qt_ql_qi(qt, ql, qi):
    return ne.evaluate('qt-ql-qi')


@autodoc(equation='q_v = q_t-q_l')
@assumes('no ice')
@overridden_by_assumptions('no liquid water')
def qv_from_qt_ql(qt, ql):
    return ne.evaluate('qt-ql')


@autodoc(equation='q_v = q_t - q_i')
@assumes('no liquid water')
@overridden_by_assumptions('no ice')
def qv_from_qt_qi(qt, qi):
    return ne.evaluate('qt-qi')


@autodoc(equation='q_i = q_t-q_v-q_l')
@assumes()
@overridden_by_assumptions('no liquid water', 'no ice')
def qi_from_qt_qv_ql(qt, qv, ql):
    return ne.evaluate('qt-qv-ql')


@autodoc(equation='q_i = q_t-q_v')
@assumes('no liquid water')
@overridden_by_assumptions('no ice')
def qi_from_qt_qv(qt, qv):
    return ne.evaluate('qt-qv')


@autodoc(equation='q_l = q_t-q_v-q_i')
@assumes()
@overridden_by_assumptions('no liquid water', 'no ice')
def ql_from_qt_qv_qi(qt, qv, qi):
    return ne.evaluate('qt-qv-qi')


@autodoc(equation='q_l = q_t-q_v')
@assumes('no ice')
@overridden_by_assumptions('no liquid water')
def ql_from_qt_qv(qt, qv):
    return ne.evaluate('qt-qv')


@autodoc(equation=r'RH = \frac{r_v}{r_{vs}} \times 100')
@assumes()
def RH_from_rv_rvs(rv, rvs):
    return ne.evaluate('rv/rvs*100.')


@autodoc(equation=r'RH_i = \frac{r_v}{r_{vsi}} \times 100')
@assumes()
def RHi_from_rv_rvsi(rv, rvsi):
    return ne.evaluate('rv/rvsi*100.')


@autodoc(equation=r'RH = \frac{q_{v}}{q_{vs}} \times 100')
@assumes('low water vapor')
def RH_from_qv_qvs_lwv(qv, qvs):
    return ne.evaluate('qv/qvs*100.')


@autodoc(equation=r'RH_i = \frac{q_{v}}{q_{vsi}} \times 100')
@assumes('low water vapor')
def RHi_from_qv_qvsi_lwv(qv, qvsi):
    return ne.evaluate('qv/qvsi*100.')


@autodoc(equation=r'\rho = \frac{AH}{q_v}')
@assumes()
def rho_from_qv_AH(qv, AH):
    return ne.evaluate('AH/qv')


@autodoc(equation=r'\rho = \frac{p}{R_d T_v}')
@assumes('ideal gas')
def rho_from_p_Tv_ideal_gas(p, Tv):
    return ne.evaluate('p/(Rd*Tv)')


@autodoc(equation=r'r_v = \frac{q_v}{1-q_v}')
@assumes()
@overridden_by_assumptions('low water vapor')
def rv_from_qv(qv):
    return ne.evaluate('qv/(1-qv)')


@autodoc(equation='r_v = q_v')
@assumes('low water vapor')
def rv_from_qv_lwv(qv):
    return 1.*qv


@autodoc(equation='r_v = \frac{-311 (T-T_v)}{500 T - 311 T_v}')
@assumes()
@overridden_by_assumptions('low water vapor')
def rv_from_Tv_T(Tv, T):
    return ne.evaluate('-311*(T-Tv)/(500*T-311*Tv)')


@autodoc(equation='r_v = (\frac{T_v}{T} - 1)\frac{0.622}{1-0.622}')
@assumes('low water vapor')
def rv_from_Tv_T_lwv(Tv, T):
    return ne.evaluate('(Tv/T - 1)*(0.622/(1-0.622))')


@autodoc(equation=r'r_v = \frac{RH}{100} r_{vs}')
@assumes()
def rv_from_RH_rvs(RH, rvs):
    return ne.evaluate('RH/100.*rvs')


@autodoc(equation=r'r_v = \frac{RH_i}{100} r_{vsi}')
@assumes()
def rv_from_RHi_rvsi(RHi, rvsi):
    return ne.evaluate('RHi/100.*rvsi')


@autodoc(equation=r'rv = (\frac{Rd}{Rv}) \frac{e}{p-e}')
@assumes()
def rv_from_p_e(p, e):
    return ne.evaluate('0.622*e/(p-e)')


@autodoc(equation=r'r_t = r_i+r_v+r_l')
@assumes()
@overridden_by_assumptions('no liquid water', 'no ice')
def rt_from_ri_rv_rl(ri, rv, rl):
    return ne.evaluate('ri+rv+rl')


@autodoc(equation=r'r_t = r_v+r_l')
@assumes('no ice')
@overridden_by_assumptions('no liquid water')
def rt_from_rv_rl(rv, rl):
    return ne.evaluate('rv+rl')


@autodoc(equation=r'r_t = r_v')
@assumes('no liquid water', 'no ice')
def rt_from_rv(rv):
    return 1.*rv


@autodoc(equation=r'r_t = r_v+r_l')
@assumes('no liquid water')
@overridden_by_assumptions('no ice')
def rt_from_rv_ri(rv, ri):
    return ne.evaluate('rv+ri')


@autodoc(equation=r'r_v = r_t')
@assumes('no liquid water', 'no ice')
def rv_from_rt(rt):
    return 1.*rt


@autodoc(equation=r'r_v = r_t-r_l-r_i')
@assumes()
@overridden_by_assumptions('no liquid water', 'no ice')
def rv_from_rt_rl_ri(rt, rl, ri):
    return ne.evaluate('rt-rl-ri')


@autodoc(equation=r'r_v = r_t-r_l')
@assumes('no ice')
@overridden_by_assumptions('no liquid water')
def rv_from_rt_rl(rt, rl):
    return ne.evaluate('rt-rl')


@autodoc(equation=r'r_v = r_t - r_i')
@assumes('no liquid water')
@overridden_by_assumptions('no ice')
def rv_from_rt_ri(rt, ri):
    return ne.evaluate('rt-ri')


@autodoc(equation=r'r_i = r_t-r_v-r_l')
@assumes()
@overridden_by_assumptions('no liquid water', 'no ice')
def ri_from_rt_rv_rl(rt, rv, rl):
    return ne.evaluate('rt-rv-rl')


@autodoc(equation=r'r_i = r_t-r_v')
@assumes('no liquid water')
@overridden_by_assumptions('no ice')
def ri_from_rt_rv(rt, rv):
    return ne.evaluate('rt-rv')


@autodoc(equation=r'r_l = r_t-r_v-r_i')
@assumes()
@overridden_by_assumptions('no liquid water', 'no ice')
def rl_from_rt_rv_ri(rt, rv, ri):
    return ne.evaluate('rt-rv-ri')


@autodoc(equation=r'r_l = r_t-r_v')
@assumes('no ice')
@overridden_by_assumptions('no liquid water')
def rl_from_rt_rv(rt, rv):
    return ne.evaluate('rt-rv')


@autodoc(equation=r'r_{vs} = rv\_from\_p\_e(p, e_s)')
@assumes()
def rvs_from_p_es(p, es):
    return rv_from_p_e(p, es)


@autodoc(equation=r'r_{vsi} = rv\_from\_p\_e(p, e_{si})')
@assumes()
def rvsi_from_p_esi(p, esi):
    return rv_from_p_e(p, esi)


@autodoc(equation=r'r_{vs} = rv\_from\_qv(q_{vs})')
@assumes()
@overridden_by_assumptions('low water vapor')
def rvs_from_qvs(qvs):
    return rv_from_qv(qvs)


@autodoc(equation=r'r_{vsi} = rv\_from\_qv(q_{vsi})')
@assumes()
@overridden_by_assumptions('low water vapor')
def rvsi_from_qvsi(qvsi):
    return rv_from_qv(qvsi)


@autodoc(equation=r'r_{vsi} = rv\_from\_qv(q_{vsi})')
@assumes('low water vapor')
def rvsi_from_qvsi_lwv(qvsi):
    return rv_from_qv_lwv(qvsi)


@autodoc(equation=r'r_v = rv\_from\_qv(q_{vs})')
@assumes('low water vapor')
def rvs_from_qvs_lwv(qvs):
    return rv_from_qv_lwv(qvs)


@autodoc(equation=r'T = \frac{29.65 log(es)-4880.16}{log(es)-19.48}',
         references=ref['Bolton 1980'] + ref['Wexler 1976'],
         notes='''
Fits Wexler's formula to an accuracy of 0.1% for temperatures between
-35C and 35C.''')
@assumes('bolton')
def T_from_es_Bolton(es):
    return ne.evaluate('(59300*log(5*es/3056)-9653121)/(2000*log(5*es/3056)-'
                       '35340)')


@autodoc(equation=r'T_{lcl} = ((\frac{1}{T-55}-(\frac{log(\frac{RH}{100})}'
         '{2840}))^{-1} + 55',
         references=ref['Bolton 1980'],
         notes='Uses Bolton (1980) equation 22.')
@assumes('bolton')
def Tlcl_from_T_RH(T, RH):
    return ne.evaluate('1./((1./(T-55.))-(log(RH/100.)/2840.)) + 55.')


@autodoc(equation=r'T_{lcl} = ((1./(Td-56.))-(log(T/Td)/800.))^{-1} + 56.',
         references=ref['Bolton 1980'],
         notes='Uses Bolton (1980) equation 15.')
@assumes('bolton')
def Tlcl_from_T_Td(T, Td):
    return ne.evaluate('1./((1./(Td-56.))+(log(T/Td)/800.)) + 56.')


@autodoc(equation=r'T_{lcl} = \frac{2840}{3.5 log(T) - log(e) - 4.805} + 55',
         references=ref['Bolton 1980'],
         notes='Uses Bolton(1980) equation 21.')
@assumes('bolton')
def Tlcl_from_T_e(T, e):
    return ne.evaluate('2840./(3.5*log(T)-log(e)-4.805) + 55.')


@autodoc(equation=r'T = \theta (\frac{10^5}{p})^{-\frac{R_d}{C_{pd}}}')
@assumes('constant Cp')
def T_from_p_theta(p, theta):
    return ne.evaluate('theta*(1e5/p)**(-Rd/Cpd)')


@autodoc(
    equation=r'T_v = \frac{T}{1-\frac{e}{p}(1-0.622)}',
    notes="""
Neglects density effects of liquid and solid water""")
@assumes()
@overridden_by_assumptions('Tv equals T')
def Tv_from_p_e_T(p, e, T):
    return ne.evaluate('T/(1-e/p*(1-0.622))')


@autodoc(
    equation=r'T = T_v (1-\frac{e}{p}(1-0.622))',
    notes="""
Neglects density effects of liquid and solid water""")
@assumes()
@overridden_by_assumptions('Tv equals T')
def T_from_p_e_Tv(p, e, Tv):
    return ne.evaluate('Tv*(1-e/p*(1-0.622))')


@autodoc(
    equation=r'T_v = T \frac{1 + \frac{r_v}{0.622}}{1+r_v}',
    notes="""
Neglects density effects of liquid and solid water""")
@assumes()
@overridden_by_assumptions('low water vapor', 'Tv equals T')
def Tv_from_T_rv(T, rv):
    return ne.evaluate('T*(1+rv/0.622)/(1+rv)')


@autodoc(
    equation=r'T = T_v \frac{1 + r_v}{1+ \frac{r_v}{0.622}',
    notes="""
Neglects density effects of liquid and solid water""")
@assumes()
@overridden_by_assumptions('low water vapor', 'Tv equals T')
def T_from_Tv_rv(Tv, rv):
    return ne.evaluate('Tv/(1+rv/0.622)*(1+rv)')


@autodoc(
    equation=r'T_v = T (1 + (\frac{1}{0.622} - 1) r_v)',
    notes="""
Neglects density effects of liquid and solid water""")
@assumes('low water vapor')
@overridden_by_assumptions('Tv equals T')
def Tv_from_T_rv_lwv(T, rv):
    return ne.evaluate('T*(1+(1/0.622-1)*rv)')


@autodoc(
    equation=r'T = \frac{T_v}{1 + (\frac{1}{0.622} - 1) r_v}',
    notes="""
Neglects density effects of liquid and solid water""")
@assumes('low water vapor')
@overridden_by_assumptions('Tv equals T')
def T_from_Tv_rv_lwv(Tv, rv):
    return ne.evaluate('Tv/(1+(1/0.622-1)*rv)')


@autodoc(equation=r'T_v = T',
         notes='''
This function exists to allow using temperature as virtual temperature.''')
@assumes('Tv equals T')
def Tv_from_T_assuming_Tv_equals_T(T):
    return 1.*T


@autodoc(equation=r'T_v = \frac{p}{\rho R_d}')
@assumes('ideal gas')
def Tv_from_p_rho_ideal_gas(p, rho):
    return ne.evaluate('p/(rho*Rd)')


@autodoc(references=ref['Stull 2011'],
         notes='''
Uses the empirical inverse solution from Stull (2011). Only valid at 101.3kPa.
''')
@assumes()
def Tw_from_T_RH_Stull(T, RH):
    return ne.evaluate('''((T-273.15)*arctan(0.151977*(RH + 8.313659)**0.5)
        + arctan(T-273.15+RH) - arctan(RH - 1.676331)
        + 0.00391838*RH**1.5*arctan(0.023101*RH) - 4.686035 + 273.15)''')


@autodoc(equation=r'T = T_v',
         notes='''
This function exists to allow using temperature as virtual temperature.''')
@assumes('Tv equals T')
def T_from_Tv_assuming_Tv_equals_T(Tv):
    return 1.*Tv


@autodoc(equation=r'\theta = T (\frac{10^5}{p})^(\frac{R_d}{C_{pd}})')
@assumes('constant Cp')
def theta_from_p_T(p, T):
    return ne.evaluate('T*(1e5/p)**(Rd/Cpd)')


@autodoc(
    equation=r'\theta_e = T (\frac{10^5}{p})^(\frac{R_d}{C_{pd}})'
    r'(1-0.28 r_v)) exp((\frac{3.376}{T_{lcl}}-0.00254) (r_v \times 10^3) '
    r'(1+0.81 r_v))',
    references=ref['Bolton 1980'] + ref['Davies-Jones 2009'],
    notes='''
This is one of the most accurate ways of computing thetae, with an
error of less than 0.2K due mainly to assuming Cp does not vary with
temperature or pressure.''')
@assumes('bolton', 'constant Cp', 'no liquid water')
def thetae_from_p_T_Tlcl_rv_Bolton(p, T, Tlcl, rv):
    return ne.evaluate('T*(1e5/p)**((Rd/Cpd)*(1-0.28*rv))*exp((3.376/Tlcl-'
                       '0.00254)*rv*1e3*(1+0.81*rv))')


@autodoc(equation=r'\theta_e = T (\frac{10^5}{p})^(\frac{R_d}{C_{pd}) + '
         r'r_t C_l}) RH^{-r_v \frac{R_v}{C_{pd} +'
         r'r_t C_l}} exp(L_v \frac{r_v}{C_{pd}+r_t C_l})',
         references=ref['AMS Glossary thetae'])
@assumes()
@overridden_by_assumptions('low water vapor')
def thetae_from_p_e_T_RH_rv_rt(p, e, T, RH, rv, rt):
    return ne.evaluate('T*(1e5/(p-e))**(Rd/(Cpd + rt*Cl))*RH**(-rv*Rv/(Cpd + '
                       'rt*Cl))*exp(Lv0*rv/((Cpd+rt*Cl)*T))')


@autodoc(equation=r'\theta_e = T*(\frac{10^5}{p})^(\frac{R_d}{C_{pd}}) '
         r'RH^{-r_v \frac{Rv}{C_{pd}}} exp(L_v \frac{Rv}{C_{pd}})')
@assumes('low water vapor')
def thetae_from_T_RH_rv_lwv(T, RH, rv):
    return ne.evaluate('T*(1e5/p)**(Rd/Cpd)*RH**(-rv*Rv/Cpd)*'
                       'exp(Lv0*rv/(Cpd*T)')


@autodoc(equation=r'\theta_{es} = thetae\_from\_p\_T\_Tlcl\_rv\_Bolton(p, T, '
         r'T, r_{vs})',
         references=ref['Bolton 1980'] + ref['Davies-Jones 2009'],
         notes='''
See thetae_from_theta_Tlcl_rv_Bolton for more information.''')
@assumes('bolton', 'constant Cp')
def thetaes_from_p_T_rvs_Bolton(p, T, rvs):
    return thetae_from_p_T_Tlcl_rv_Bolton(p, T, T, rvs)


@autodoc(equation=r'w = - \frac{\omega}{\rho g_0}')
@assumes('constant g', 'hydrostatic')
def w_from_omega_rho_hydrostatic(omega, rho):
    return ne.evaluate('-omega/(rho*g0)')


@autodoc(equation=r'z = \frac{\Phi}{g_0}')
@assumes('constant g')
def z_from_Phi(Phi):
    return ne.evaluate('Phi/g0')
