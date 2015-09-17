# -*- coding: utf-8 -*-
"""
test.py: Testing suite for other modules.
"""
from __future__ import division, unicode_literals
import unittest
import nose
import numpy as np
from atmos import equations
from atmos import util
from atmos import decorators
from nose.tools import raises
from atmos.constants import Rd
from atmos.solve import BaseSolver, FluidSolver, calculate, \
    _get_module_methods, _get_calculatable_methods_dict,\
    _get_shortest_solution
from atmos.util import quantity_string, assumption_list_string, \
    quantity_spec_string, doc_paragraph, \
    strings_to_list_string
try:
    from inspect import getfullargspec
except ImportError:
    from inspect import getargspec as getfullargspec


def test_quantities_dict_complete():
    names = [item['output'] for item in _get_module_methods(equations)]
    for name in names:
        if name not in equations.quantities.keys():
            try:
                util.parse_derivative_string(name)
            except ValueError:
                raise AssertionError('{0} not in quantities dict'.format(name))


def test_get_module_methods_nonempty():
    result = _get_module_methods(equations)
    assert len(result) > 0


def test_default_assumptions_exist():
    for m in FluidSolver.default_assumptions:
        if m not in FluidSolver.all_assumptions:
            raise AssertionError('{0} not a valid assumption'.format(m))


class ddxTests(unittest.TestCase):

    def setUp(self):
        self.data = np.zeros((2, 3))
        self.data[:] = np.array([0., 5., 10.])[None, :]
        self.axis1 = np.array([5., 6., 7.])
        self.axis2 = np.array([[5., 6., 7.], [8., 10., 12.]])
        self.deriv1 = 5.*np.ones((2, 3))
        self.deriv2 = 5.*np.ones((2, 3))
        self.deriv2[1, :] *= 0.5

    @raises(ValueError)
    def test_invalid_data_axis(self):
        util.ddx(self.data, axis=2)

    @raises(ValueError)
    def test_invalid_axis_x(self):
        util.ddx(self.data, axis=2, x=self.axis1, axis_x=1)


class ClosestValTests(unittest.TestCase):

    def setUp(self):
        self.list = [1., 5., 10.]
        self.array = np.array([1., 5., 10.])

    def tearDown(self):
        self.list = None
        self.array = None

    @raises(ValueError)
    def testValueErrorOnEmptyList(self):
        util.closest_val(1., [])

    @raises(ValueError)
    def testValueErrorOnEmptyArray(self):
        util.closest_val(1., np.array([]))

    def testClosestValSingle(self):
        val = util.closest_val(1., np.array([50.]))
        assert val == 0

    def testClosestValInternal(self):
        val = util.closest_val(3.5, self.array)
        assert val == 1

    def testClosestValBelow(self):
        val = util.closest_val(-5., self.array)
        assert val == 0

    def testClosestValAbove(self):
        val = util.closest_val(20., self.array)
        assert val == 2

    def testClosestValInternalNegative(self):
        val = util.closest_val(-3.5, -1*self.array)
        assert val == 1

    def testClosestValInternalList(self):
        val = util.closest_val(3.5, self.list)
        assert val == 1

    def testClosestValBelowList(self):
        val = util.closest_val(-5., self.list)
        assert val == 0

    def testClosestValAboveList(self):
        val = util.closest_val(20., self.list)
        assert val == 2


class AreaPolySphereTests(unittest.TestCase):

    def setUp(self):
        self.lat1 = np.array([0.0, 90.0, 0.0])
        self.lon1 = np.array([0.0, 0.0, 90.0])
        self.area1 = 1.5708
        self.tol1 = 0.0001

    @raises(ValueError)
    def test_area_poly_sphere_insufficient_vertices(self):
        util.area_poly_sphere(self.lat1[:2], self.lon1[:2], 1.)

    def test_area_poly_sphere(self):
        area_calc = util.area_poly_sphere(self.lat1, self.lon1, 1.)
        assert abs(area_calc - self.area1) < self.tol1

    def test_area_poly_sphere_different_radius(self):
        area_calc = util.area_poly_sphere(self.lat1, self.lon1, 2.)
        assert abs(area_calc - 4*self.area1) < 4*self.tol1


class DecoratorTests(unittest.TestCase):

    def setUp(self):
        def dummyfunction(x):
            '''Dummy docstring'''
            return x
        self.func = dummyfunction
        self.func_name = dummyfunction.__name__
        self.func_argspec = getfullargspec(dummyfunction)
        self.quantity_dict = {
            'T': {'name': 'air temperature', 'units': 'K'},
            'qv': {'name': 'specific humidity', 'units': 'kg/kg'},
            'p': {'name': 'air pressure', 'units': 'Pa'},
        }
        self.assumption_dict = {
            'a1': 'a1_long',
            'a2': 'a2_long',
            'a3': 'a3_long',
        }
        self.func_argspec = getfullargspec(self.func)

    def tearDown(self):
        self.func = None
        self.quantity_dict = None
        self.assumption_dict = None

    def test_assumes_empty(self, **kwargs):
        func = decorators.assumes()(self.func)
        assert func.assumptions == ()
        assert func.__name__ == self.func_name
        assert getfullargspec(func) == self.func_argspec

    def test_assumes_single(self, **kwargs):
        func = decorators.assumes('a1')(self.func)
        assert func.assumptions == ('a1',)
        assert func.__name__ == self.func_name
        assert getfullargspec(func) == self.func_argspec

    def test_assumes_triple(self, **kwargs):
        func = decorators.assumes('a1', 'a2', 'a3')(self.func)
        assert func.assumptions == ('a1', 'a2', 'a3',)
        assert func.__name__ == self.func_name
        assert getfullargspec(func) == self.func_argspec

    def test_overridden_by_assumptions_empty(self, **kwargs):
        func = decorators.overridden_by_assumptions()(self.func)
        assert func.overridden_by_assumptions == ()
        assert func.__name__ == self.func_name
        assert getfullargspec(func) == self.func_argspec

    def test_overridden_by_assumptions_single(self, **kwargs):
        func = decorators.overridden_by_assumptions('a1')(self.func)
        assert func.overridden_by_assumptions == ('a1',)
        assert func.__name__ == self.func_name
        assert getfullargspec(func) == self.func_argspec

    def test_overridden_by_assumptions_triple(self, **kwargs):
        func = decorators.overridden_by_assumptions(
            'a1', 'a2', 'a3')(self.func)
        assert func.overridden_by_assumptions == ('a1', 'a2', 'a3',)
        assert func.__name__ == self.func_name
        assert getfullargspec(func) == self.func_argspec

    @raises(ValueError)
    def test_autodoc_invalid_no_extra_args(self):
        def invalid_function(T, p):
            return T
        decorators.equation_docstring(
            self.quantity_dict, self.assumption_dict)(invalid_function)

    @raises(ValueError)
    def test_autodoc_invalid_args_function(self):
        def T_from_x_y(x, y):
            return x
        decorators.equation_docstring(
            self.quantity_dict, self.assumption_dict)(T_from_x_y)

    @raises(ValueError)
    def test_autodoc_invalid_extra_args(self):
        def invalid_function(T, p):
            return T
        decorators.equation_docstring(
            self.quantity_dict, self.assumption_dict, equation='x=y',
            references='reference here', notes='c sharp')(invalid_function)


class StringUtilityTests(unittest.TestCase):

    def setUp(self):
        self.quantity_dict = {
            'T': {'name': 'air temperature', 'units': 'K'},
            'qv': {'name': 'specific humidity', 'units': 'kg/kg'},
            'p': {'name': 'air pressure', 'units': 'Pa'},
        }
        self.assumption_dict = {
            'a1': 'a1_long',
            'a2': 'a2_long',
            'a3': 'a3_long',
        }

    def tearDown(self):
        self.quantity_dict = None
        self.assumption_dict = None

    def test_quantity_string(self):
        string = quantity_string('T', self.quantity_dict)
        assert string == 'air temperature (K)'

    @raises(ValueError)
    def test_quantity_string_invalid_quantity(self):
        quantity_string('rhombus', self.quantity_dict)

    @raises(ValueError)
    def test_strings_to_list_string_empty(self):
        string = strings_to_list_string(())
        assert string == ''

    @raises(TypeError)
    def test_strings_to_list_string_input_string(self):
        strings_to_list_string('hello')

    def test_strings_to_list_string_single(self):
        string = strings_to_list_string(('string1',))
        assert string == 'string1'

    def test_strings_to_list_string_double(self):
        string = strings_to_list_string(('string1', 'string2'))
        assert string == 'string1 and string2'

    def test_strings_to_list_string_triple(self):
        string = strings_to_list_string(('string1', 'string2', 'string3'))
        assert string == 'string1, string2, and string3'

    @raises(ValueError)
    def test_assumption_list_string_empty(self):
        assumption_list_string((), self.assumption_dict)

    @raises(TypeError)
    def test_assumption_list_string_input_string(self):
        assumption_list_string('hello', self.assumption_dict)

    @raises(ValueError)
    def test_assumption_list_string_invalid(self):
        assumption_list_string(('ooglymoogly',), self.assumption_dict)

    def test_assumption_list_string_single(self):
        string = assumption_list_string(('a1',), self.assumption_dict)
        assert string == 'a1_long'

    def test_assumption_list_string_double(self):
        string = assumption_list_string(('a1', 'a2'), self.assumption_dict)
        assert string == 'a1_long and a2_long'

    def test_assumption_list_string_triple(self):
        string = assumption_list_string(('a1', 'a2', 'a3'),
                                        self.assumption_dict)
        assert string == 'a1_long, a2_long, and a3_long'

    @raises(ValueError)
    def test_quantity_spec_string_invalid(self):
        quantity_spec_string('rv', self.quantity_dict)

    def test_quantity_spec_string_valid(self):
        string = quantity_spec_string('T', self.quantity_dict)
        assert string == ('T : float or ndarray\n'
                          '    Data for air temperature (K).')

    def test_doc_paragraph_no_wrap_for_short_string(self):
        string = doc_paragraph('The quick brown fox jumped over the yellow '
                               'doge. The quick brown fox jumped over')
        assert '\n' not in string

    def test_doc_paragraph_wraps_on_word(self):
        string = doc_paragraph('The quick brown fox jumped over the yellow '
                               'doge. The quick brown fox jumped over.')
        if (string != 'The quick brown fox jumped over the yellow doge. '
                'The quick brown fox jumped\nover.'):
            raise AssertionError('incorrect string "{0}"'.format(string))

    def test_doc_paragraph_indent_wrap(self):
        string = doc_paragraph('The quick brown fox jumped over the yellow '
                               'doge. The quick brown fox jumped over',
                               indent=1)
        if (string != ' The quick brown fox jumped over the yellow doge. '
                'The quick brown fox jumped\n over'):
            raise AssertionError('incorrect string "{0}"'.format(string))

    def test_doc_paragraph_zero_indent(self):
        string = doc_paragraph('The quick brown fox jumped over the yellow '
                               'doge. The quick brown fox jumped over.',
                               indent=0)
        if (string != 'The quick brown fox jumped over the yellow doge. '
                'The quick brown fox jumped\nover.'):
            raise AssertionError('incorrect string "{0}"'.format(string))


class ParseDerivativeStringTests(unittest.TestCase):

    def setUp(self):
        self.quantity_dict = {
            'T': {'name': 'air temperature', 'units': 'K'},
            'qv': {'name': 'specific humidity', 'units': 'kg/kg'},
            'p': {'name': 'air pressure', 'units': 'Pa'},
        }

    def tearDown(self):
        self.quantity_dict = None

    @raises(ValueError)
    def test_invalid_format(self):
        util.parse_derivative_string('ooglymoogly', self.quantity_dict)

    @raises(ValueError)
    def test_invalid_variable(self):
        util.parse_derivative_string('dpdz', self.quantity_dict)

    def test_dTdp(self):
        var1, var2 = util.parse_derivative_string('dTdp', self.quantity_dict)
        assert var1 == 'T'
        assert var2 == 'p'

    @raises(ValueError)
    def test_dpdT(self):
        util.parse_derivative_string('dpdT', self.quantity_dict)


class OverriddenByAssumptionsTests(unittest.TestCase):
    def test_overridden_by_assumptions_empty(self):
        @decorators.overridden_by_assumptions()
        def foo():
            return None
        assert foo.overridden_by_assumptions == ()

    def test_overridden_by_assumptions_single(self):
        @decorators.overridden_by_assumptions('test assumption')
        def foo():
            return None
        assert foo.overridden_by_assumptions == ('test assumption',)

    def test_overridden_by_assumptions_multiple(self):
        @decorators.overridden_by_assumptions('test assumption', 'a2')
        def foo():
            return None
        assert foo.overridden_by_assumptions == ('test assumption', 'a2')


class GetCalculatableMethodsDictTests(unittest.TestCase):

    def test_get_calculatable_methods_dict_empty(self):
        methods = {}
        out_methods = _get_calculatable_methods_dict((), methods)
        assert isinstance(out_methods, dict)
        assert len(out_methods) == 0

    def test_get_calculatable_methods_dict_returns_correct_type(self):
        methods = {'a': {('b',): lambda x: x}}
        out_methods = _get_calculatable_methods_dict(('b',), methods)
        assert isinstance(out_methods, dict)

    def test_get_calculatable_methods_dict_gets_single_method(self):
        methods = {'a': {('b',): lambda x: x}}
        out_methods = _get_calculatable_methods_dict(('b',), methods)
        assert 'a' in out_methods.keys()

    def test_get_calculatable_methods_dict_removes_correct_second_method(
            self):
        methods = {'a': {('b',): lambda x: x,
                         ('c', 'b'): lambda x: x}}
        out_methods = _get_calculatable_methods_dict(('b', 'c'), methods)
        assert 'a' in out_methods.keys()
        assert ('b',) in out_methods['a'].keys()
        assert len(out_methods['a']) == 1

    def test_get_calculatable_methods_dict_removes_irrelevant_second_method(
            self):
        methods = {'a': {('b', 'd'): lambda x, y: x,
                         ('c',): lambda x: x}}
        out_methods = _get_calculatable_methods_dict(('b', 'd'), methods)
        assert 'a' in out_methods.keys()
        assert ('b', 'd') in out_methods['a'].keys()
        assert len(out_methods['a']) == 1

    def test_get_calculatable_methods_dict_gets_no_methods(self):
        methods = {'a': {('b',): lambda x: x},
                   'x': {('y', 'z'): lambda y, z: y*z}
                   }
        out_methods = _get_calculatable_methods_dict(('q',), methods)
        assert isinstance(out_methods, dict)
        assert len(out_methods) == 0

    def test_get_calculatable_methods_dict_doesnt_calculate_input(self):
        methods = {'a': {('b',): lambda x: x},
                   'x': {('y', 'z'): lambda y, z: y*z}
                   }
        out_methods = _get_calculatable_methods_dict(('b', 'a'), methods)
        assert isinstance(out_methods, dict)
        assert len(out_methods) == 0


class BaseSolverTests(unittest.TestCase):

    @raises(NotImplementedError)
    def test_cannot_instantiate(self):
        BaseSolver()


class FluidSolverTests(unittest.TestCase):

    def setUp(self):
        shape = (3, 4, 2, 2)
        self.vars1 = {'Tv': np.ones(shape),
                      'p': np.ones(shape)}
        self.vars2 = {'T': np.ones(shape),
                      'p': np.ones(shape)}

    def tearDown(self):
        self.vars1 = None
        self.vars2 = None

    def test_creation_no_arguments(self):
        FluidSolver()

    def test_is_instance_of_BaseSolver(self):
        deriv = FluidSolver()
        assert isinstance(deriv, BaseSolver)

    def test_creation_one_method(self):
        FluidSolver(assumptions=('hydrostatic',))

    def test_creation_compatible_methods(self):
        FluidSolver(assumptions=('hydrostatic', 'Tv equals T',))

    @raises(ValueError)
    def test_creation_incompatible_methods(self):
        FluidSolver(assumptions=('Goff-Gratch', 'Wexler',))

    @raises(ValueError)
    def test_creation_undefined_method(self):
        FluidSolver(assumptions=('moocow',))

    @raises(ValueError)
    def test_creation_undefined_method_with_defined_method(self):
        FluidSolver(assumptions=('hydrostatic', 'moocow',))

    def test_creation_with_vars(self):
        FluidSolver(**self.vars1)

    def test_creation_with_vars_and_method(self):
        FluidSolver(assumptions=('Tv equals T',), **self.vars1)

    def test_simple_calculation(self):
        deriver = FluidSolver(**self.vars1)
        rho = deriver.calculate('rho')
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)

    def test_depth_2_calculation(self):
        deriver = FluidSolver(assumptions=FluidSolver.default_assumptions +
                              ('Tv equals T',), **self.vars2)
        rho = deriver.calculate('rho')
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)

    def test_double_calculation(self):
        deriver = FluidSolver(add_assumptions=('Tv equals T',), **self.vars2)
        Tv = deriver.calculate('Tv')
        rho = deriver.calculate('rho')
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)
        assert isinstance(Tv, np.ndarray)

    def test_double_reverse_calculation(self):
        deriver = FluidSolver(add_assumptions=('Tv equals T',), **self.vars2)
        rho = deriver.calculate('rho')
        print('now Tv')
        Tv = deriver.calculate('Tv')
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)
        assert isinstance(Tv, np.ndarray)


class calculateTests(unittest.TestCase):

    def setUp(self):
        self.shape = (3, 4, 2, 2)
        self.vars1 = {'Tv': np.ones(self.shape),
                      'p': np.ones(self.shape)}
        self.vars2 = {'T': np.ones(self.shape),
                      'p': np.ones(self.shape)}

    def tearDown(self):
        self.vars1 = None
        self.vars2 = None

    def test_simple_calculation(self):
        rho = calculate('rho', **self.vars1)
        assert (rho.shape == self.shape)
        assert (rho == 1/Rd).all()
        self.assertTrue(isinstance(rho, np.ndarray),
                        'returned rho should be ndarray')

    def test_returns_float(self):
        rho = calculate('rho', Tv=1., p=1.)
        self.assertTrue(isinstance(rho, float),
                        'returned rho should be float')

    def test_depth_2_calculation(self):
        rho = calculate('rho', add_assumptions=('Tv equals T',), **self.vars2)
        assert rho.shape == self.shape
        assert (rho == 1/Rd).all()
        self.assertTrue(isinstance(rho, np.ndarray),
                        'returned rho should be ndarray')

    def test_double_calculation(self):
        Tv, rho = calculate('Tv', 'rho', add_assumptions=('Tv equals T',),
                            **self.vars2)
        assert Tv.shape == self.shape
        assert rho.shape == self.shape
        assert (rho == 1/Rd).all()
        self.assertTrue(isinstance(rho, np.ndarray),
                        'returned rho should be ndarray')
        self.assertTrue(isinstance(Tv, np.ndarray),
                        'returned Tv should be ndarray')

    def test_double_reverse_calculation(self):
        rho, Tv = calculate('rho', 'Tv', add_assumptions=('Tv equals T',),
                            **self.vars2)
        assert (rho == 1/Rd).all()
        self.assertTrue(isinstance(rho, np.ndarray),
                        'returned rho should be ndarray')
        self.assertTrue(isinstance(Tv, np.ndarray),
                        'returned Tv should be ndarray')

    def test_T_from_Tv(self):
        assert calculate('T', Tv=1., add_assumptions=('Tv equals T',)) == 1.
        assert calculate('T', Tv=5., add_assumptions=('Tv equals T',)) == 5.

    def test_rv_from_qv(self):
        self.assertAlmostEqual(calculate('rv', qv=0.005), 0.005025125628140704)

    def test_qv_from_rv(self):
        self.assertAlmostEqual(calculate('qv', rv=0.005), 0.004975124378109453)


class CalculateWithUnitsTests(unittest.TestCase):

    def setUp(self):
        self.units_dict = {
            'T': ('K', 'degC', 'degF'),
            'p': ('hPa', 'Pa'),
            'Tv': ('K', 'degC', 'degF'),
            'rv': ('g/kg', 'kg/kg'),
            'qv': ('g/kg', 'kg/kg'),
            'rvs': ('g/kg', 'kg/kg'),
            'RH': ('percent',),
        }

    def tearDown(self):
        self.units_dict = None

    def test_returns_same_value(self):
        for quantity in self.units_dict.keys():
            for unit in self.units_dict[quantity]:
                kwargs = {}
                kwargs[quantity + '_unit'] = unit
                kwargs[quantity] = 1.5
                result = calculate(quantity, **kwargs)
                print(result, quantity, unit)
                self.assertAlmostEqual(
                    result, 1.5, msg='calculate should return the same value '
                    'when given a value as input')

    def test_input_unit(self):
        rho = calculate('rho', Tv=1., p=0.01, p_unit='hPa')
        self.assertEqual(rho, 1./Rd)

    def test_output_unit(self):
        p = calculate('p', Tv=1., rho=1./Rd, p_unit='millibar')
        self.assertEqual(p, 0.01)


class TestSolveValuesNearSkewT(unittest.TestCase):

    def setUp(self):
        self.quantities = {'p': 9e4, 'theta': 14.+273.15,
                           'rv': 1e-3, 'Tlcl': -22.+273.15,
                           'thetae': 17.+273.15, 'Tw': -4.+273.15,
                           'Td': -18.+273.15, 'plcl': 65000.,
                           }
        self.quantities['T'] = calculate('T', **self.quantities)
        self.quantities['Tv'] = calculate('Tv', **self.quantities)
        self.quantities['rho'] = calculate('rho', **self.quantities)
        self.add_assumptions = ('bolton', 'unfrozen bulb')

    def _generator(self, quantity, tolerance):
        skew_T_value = self.quantities.pop(quantity)
        calculated_value, funcs = calculate(
            quantity, add_assumptions=self.add_assumptions,
            debug=True, **self.quantities)
        diff = abs(skew_T_value - calculated_value)
        if diff > tolerance:
            err_msg = ('Value {0:.4f} is too far away from '
                       '{1:.4f} for {2}.'.format(
                           calculated_value, skew_T_value, quantity))
            err_msg += '\nfunctions used:\n'
            err_msg += '\n'.join([f.__name__ for f in funcs])
            raise AssertionError(err_msg)

    def tearDown(self):
        self.quantities = None

    def test_calculate_precursors(self):
        pass

    def test_calculate_p(self):
        self._generator('p', 10000.)

    def test_calculate_Tv(self):
        self._generator('Tv', 1.)

    def test_calculate_Td(self):
        self._generator('Td', 1.)

    def test_calculate_theta(self):
        self._generator('theta', 1.)

    def test_calculate_rv(self):
        self._generator('rv', 1e-3)

    def test_calculate_Tlcl(self):
        self._generator('Tlcl', 1.)

    def test_calculate_thetae(self):
        quantity = 'thetae'
        skew_T_value = self.quantities.pop(quantity)
        self.quantities.pop('Tlcl')  # let us calculate this ourselves
        calculated_value, funcs = calculate(
            quantity, add_assumptions=('bolton', 'unfrozen bulb'),
            debug=True,
            **self.quantities)
        diff = abs(skew_T_value - calculated_value)
        if diff > 2.:
            err_msg = ('Value {:.2f} is too far away from '
                       '{:.2f} for {}.'.format(
                           calculated_value, skew_T_value, quantity))
            err_msg += '\nfunctions used:\n'
            err_msg += '\n'.join([f.__name__ for f in funcs])
            raise AssertionError(err_msg)

    def test_calculate_plcl(self):
        self._generator('plcl', 10000.)


class TestSolveValuesNearSkewTAlternateUnits(unittest.TestCase):

    def setUp(self):
        self.quantities = {'p': 8.9e2, 'theta': 14.+273.15,
                           'rv': 1., 'Tlcl': -22.5+273.15,
                           'thetae': 17.+273.15, 'Tw': -2.5,
                           'Td': -18.5+273.15, 'plcl': 62500.,
                           }
        self.units = {'p_unit': 'hPa', 'Tv_units': 'degC', 'Tw_unit': 'degC',
                      'rv_unit': 'g/kg'}
        kwargs = {}
        kwargs.update(self.quantities)
        kwargs.update(self.units)
        self.quantities['T'] = calculate('T', **kwargs)
        self.quantities['Tv'] = calculate('Tv', **kwargs)
        self.quantities['rho'] = calculate('rho', **kwargs)
        self.add_assumptions = ('bolton', 'unfrozen bulb')

    def _generator(self, quantity, tolerance):
        skew_T_value = self.quantities.pop(quantity)
        kwargs = {}
        kwargs.update(self.quantities)
        kwargs.update(self.units)
        calculated_value, funcs = calculate(
            quantity, add_assumptions=self.add_assumptions,
            debug=True, **kwargs)
        diff = abs(skew_T_value - calculated_value)
        if diff > tolerance:
            err_msg = ('Value {:.2f} is too far away from '
                       '{:.2f} for {}.'.format(
                           calculated_value, skew_T_value, quantity))
            err_msg += '\nfunctions used:\n'
            err_msg += '\n'.join([f.__name__ for f in funcs])
            raise AssertionError(err_msg)

    def tearDown(self):
        self.quantities = None

    def test_calculate_precursors(self):
        pass

    def test_calculate_p(self):
        self._generator('p', 100.)

    def test_calculate_Tv(self):
        self._generator('Tv', 1.)

    def test_calculate_theta(self):
        self._generator('theta', 1.)

    def test_calculate_rv(self):
        self._generator('rv', 1e-3)

    def test_calculate_Tlcl(self):
        self._generator('Tlcl', 1.)

    def test_calculate_thetae(self):
        quantity = 'thetae'
        skew_T_value = self.quantities.pop(quantity)
        self.quantities.pop('Tlcl')  # let us calculate this ourselves
        kwargs = {}
        kwargs.update(self.quantities)
        kwargs.update(self.units)
        calculated_value, funcs = calculate(
            quantity, add_assumptions=('bolton', 'unfrozen bulb'),
            debug=True,
            **kwargs)
        diff = abs(skew_T_value - calculated_value)
        if diff > 1.:
            err_msg = ('Value {0:.2f} is too far away from '
                       '{1:.2f} for {2}.'.format(
                           calculated_value, skew_T_value, quantity))
            err_msg += '\nfunctions used:\n'
            err_msg += '\n'.join([f.__name__ for f in funcs])
            raise AssertionError(err_msg)

    def test_calculate_Tw(self):
        quantity = 'Tw'
        skew_T_value = self.quantities.pop(quantity)
        kwargs = {}
        kwargs.update(self.quantities)
        kwargs.update(self.units)
        calculated_value, funcs = calculate(
            quantity, add_assumptions=('bolton', 'unfrozen bulb'),
            debug=True,
            **kwargs)
        diff = abs(skew_T_value - calculated_value)
        if diff > 1.:
            err_msg = ('Value {:.2f} is too far away from '
                       '{:.2f} for {}.'.format(
                           calculated_value, skew_T_value, quantity))
            err_msg += '\nfunctions used:\n'
            err_msg += '\n'.join([f.__name__ for f in funcs])
            raise AssertionError(err_msg)

#    def test_calculate_Td(self):
#        self._generator('Td', 1.)

    def test_calculate_plcl(self):
        self._generator('plcl', 10000.)


class TestSolveValuesNearSkewTAssumingLowMoisture(TestSolveValuesNearSkewT):

    def setUp(self):
        super(TestSolveValuesNearSkewTAssumingLowMoisture, self).setUp()
        self.add_assumptions = ('bolton', 'unfrozen bulb', 'low water vapor')


class TestSolveValuesNearSkewTVeryMoist(TestSolveValuesNearSkewT):

    def setUp(self):
        self.quantities = {'p': 8.9e4, 'theta': 18.4+273.15,
                           'rv': 6e-3, 'Tlcl': 3.8+273.15,
                           'thetae': 36.5+273.15,
                           'Tw': 6.5+273.15, 'Td': 4.8+273.15, 'plcl': 83500.,
                           }
        self.quantities['T'] = calculate('T', **self.quantities)
        self.quantities['Tv'] = calculate('Tv', **self.quantities)
        self.quantities['rho'] = calculate('rho', **self.quantities)
        self.add_assumptions = ('bolton', 'unfrozen bulb')


class GetShortestSolutionTests(unittest.TestCase):

    def setUp(self):
        self.methods = {
            'x': {('a', 'b'): lambda a, b: a},
            'y': {('x',): lambda x: x},
            'z': {('x', 'y'): lambda x, y: x*y},
            'w': {('z', 'y'): lambda x, y: x*y},
        }

    def tearDown(self):
        self.methods = None

    def test_no_calculation(self):
        sol = _get_shortest_solution(('x',), ('x',), (), self.methods)
        assert isinstance(sol, tuple)
        assert len(sol) == 3
        assert len(sol[0]) == 0
        assert len(sol[1]) == 0
        assert len(sol[2]) == 0

    def test_simplest_calculation(self):
        sol = _get_shortest_solution(('y',), ('x',), (), self.methods)
        assert len(sol[0]) == 1

    def test_depth_2_calculation(self):
        _get_shortest_solution(('z',), ('a', 'b'), (), self.methods)

    def test_depth_3_calculation(self):
        _get_shortest_solution(('w',), ('a', 'b'), (), self.methods)


class EquationTests(unittest.TestCase):

    def _assert_accurate_values(self, func, in_values, out_values, tols):
        for i, args in enumerate(in_values):
            out_calc = func(*args)
            if abs(out_calc - out_values[i]) > tols[i]:
                raise AssertionError('Calculated value '
                '{0} from inputs {1} is more than {2} '
                'away from {3}'.format(out_calc, args, tols[i], out_values[i]))

    def test_e_from_Td_Bolton(self):
        func = equations.e_from_Td_Bolton
        in_values = [(273.15,), (273.15+20,), (273.15+40,), (273.15+50,)]
        out_values = [603, 2310, 7297, 12210]
        tols = [603*0.02, 2310*0.02, 7297*0.02, 12210*0.02]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_e_from_Td_Goff_Gratch(self):
        func = equations.e_from_Td_Goff_Gratch
        in_values = [(273.15,), (273.15+20,), (273.15+40,), (273.15+50,)]
        out_values = [603, 2310, 7297, 12210]
        tols = [603*0.02, 2310*0.02, 7297*0.02, 12210*0.02]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_e_from_p_T_Tw_Bolton(self):
        func = equations.e_from_p_T_Tw_Bolton
        in_values = [(1e5, 273.15+10, 273.15+5), (8e4, 273.15+15, 273.15+5.4)]
        out_values = [549.7, 382.8]
        tols = [549.7*0.1/3.45, 0.05*382.8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_e_from_p_T_Tw_Goff_Gratch(self):
        func = equations.e_from_p_T_Tw_Goff_Gratch
        in_values = [(1e5, 273.15+10, 273.15+5), (8e4, 273.15+15, 273.15+5.4)]
        out_values = [549.7, 382.8]
        tols = [549.7*0.1/3.45, 0.05*382.8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_es_from_T_Bolton(self):
        func = equations.es_from_T_Bolton
        in_values = [(273.15,), (273.15+20,), (273.15+40,), (273.15+50,)]
        out_values = [603, 2310, 7297, 12210]
        tols = [603*0.02, 2310*0.02, 7297*0.02, 12210*0.02]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_es_from_T_Goff_Gratch(self):
        func = equations.es_from_T_Goff_Gratch
        in_values = [(273.15,), (273.15+20,), (273.15+40,), (273.15+50,)]
        out_values = [603, 2310, 7297, 12210]
        tols = [603*0.02, 2310*0.02, 7297*0.02, 12210*0.02]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_f_from_lat(self):
        func = equations.f_from_lat
        in_values = [(0.,), (45.,), (90.,)]
        out_values = [0., 1.031e-4, 1.458e-4]
        tols = [0.001e-4, 0.001e-4, 0.001e-4]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_Gammam_from_rvs_T(self):
        func = equations.Gammam_from_rvs_T
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_MSE_from_DSE_qv(self):
        func = equations.MSE_from_DSE_qv
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_omega_from_w_rho_hydrostatic(self):
        func = equations.omega_from_w_rho_hydrostatic
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_p_from_rho_Tv_ideal_gas(self):
        func = equations.p_from_rho_Tv_ideal_gas
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_plcl_from_p_T_Tlcl(self):
        func = equations.plcl_from_p_T_Tlcl
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_Phi_from_z(self):
        func = equations.Phi_from_z
        in_values = [(0.,), (30.,), (100.,)]
        out_values = [0., 294.3, 981.]
        tols = [0.01, 0.2, 1.]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qv_from_AH_rho(self):
        func = equations.qv_from_AH_rho
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qv_from_rv(self):
        func = equations.qv_from_rv
        in_values = [(0.,), (0.005,), (0.1,)]
        out_values = [0., 0.00498, 0.091]
        tols = [0.01, 0.00001, 0.001]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qv_from_rv_lwv(self):
        func = equations.qv_from_rv_lwv
        in_values = [(0.,), (0.005,), (0.1,)]
        out_values = [0., 0.005, 0.1]
        tols = [1e-8, 1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qv_from_p_e(self):
        func = equations.qv_from_p_e
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qv_from_p_e_lwv(self):
        func = equations.qv_from_p_e_lwv
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qvs_from_rvs(self):
        func = equations.qvs_from_rvs
        in_values = [(0.,), (0.005,), (0.1,)]
        out_values = [0., 0.00498, 0.091]
        tols = [0.01, 0.00001, 0.001]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qvs_from_rvs_lwv(self):
        func = equations.qvs_from_rvs_lwv
        in_values = [(0.,), (0.005,), (0.1,)]
        out_values = [0., 0.005, 0.1]
        tols = [1e-8, 1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qvs_from_p_es(self):
        func = equations.qvs_from_p_es
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qvs_from_p_es_lwv(self):
        func = equations.qvs_from_p_es_lwv
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qt_from_qi_qv_ql(self):
        func = equations.qt_from_qi_qv_ql
        in_values = [(0., 0., 0.), (0.005, 0.001, 0.004)]
        out_values = [0., 0.01]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qt_from_qv_ql(self):
        func = equations.qt_from_qv_ql
        in_values = [(0., 0.), (0.003, 0.002)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qt_from_qv(self):
        func = equations.qt_from_qv
        in_values = [(0.,), (0.005,), (0.1,)]
        out_values = [0., 0.005, 0.1]
        tols = [1e-8, 1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qt_from_qv_qi(self):
        func = equations.qt_from_qv_qi
        in_values = [(0., 0.), (0.003, 0.002)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qv_from_qt(self):
        func = equations.qv_from_qt
        in_values = [(0.,), (0.005,), (0.1,)]
        out_values = [0., 0.005, 0.1]
        tols = [1e-8, 1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qv_from_qt_ql_qi(self):
        func = equations.qv_from_qt_ql_qi
        in_values = [(0., 0., 0.), (0.01, 0.003, 0.002)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qv_from_qt_ql(self):
        func = equations.qv_from_qt_ql
        in_values = [(0., 0.), (0.01, 0.005)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qv_from_qt_qi(self):
        func = equations.qv_from_qt_qi
        in_values = [(0., 0.), (0.01, 0.005)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qi_from_qt_qv_ql(self):
        func = equations.qv_from_qt_ql_qi
        in_values = [(0., 0., 0.), (0.01, 0.003, 0.002)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_qi_from_qt_qv(self):
        func = equations.qi_from_qt_qv
        in_values = [(0., 0.), (0.01, 0.005)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_ql_from_qt_qv_qi(self):
        func = equations.qv_from_qt_ql_qi
        in_values = [(0., 0., 0.), (0.01, 0.003, 0.002)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_ql_from_qt_qv(self):
        func = equations.ql_from_qt_qv
        in_values = [(0., 0.), (0.01, 0.005)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_RH_from_rv_rvs(self):
        func = equations.RH_from_rv_rvs
        in_values = [(5., 100.), (1e-3, 2e-3)]
        out_values = [5., 50.]
        tols = [0.01, 0.01]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_RH_from_qv_qvs_lwv(self):
        func = equations.RH_from_qv_qvs_lwv
        in_values = [(5., 100.), (1e-3, 2e-3)]
        out_values = [5., 50.]
        tols = [0.01, 0.01]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rho_from_qv_AH(self):
        func = equations.rho_from_qv_AH
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rho_from_p_Tv_ideal_gas(self):
        func = equations.rho_from_p_Tv_ideal_gas
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rv_from_qv(self):
        func = equations.rv_from_qv
        in_values = [(0.,), (0.05,), (0.1,)]
        out_values = [0., 0.05263, 0.11111]
        tols = [1e-8, 0.0001, 0.0001]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rv_from_qv_lwv(self):
        func = equations.rv_from_qv_lwv
        in_values = [(0.,), (0.005,), (0.1,)]
        out_values = [0., 0.005, 0.1]
        tols = [1e-8, 1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rv_from_p_e(self):
        func = equations.rv_from_p_e
        in_values = [(1e5, 500.), (9e4, 1000.), (8e4, 0.)]
        out_values = [0.003125, 0.006988, 0.]
        tols = [1e-6, 1e-6, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rt_from_ri_rv_rl(self):
        func = equations.rt_from_ri_rv_rl
        in_values = [(0., 0., 0.), (0.005, 0.001, 0.004)]
        out_values = [0., 0.01]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rt_from_rv_rl(self):
        func = equations.rt_from_rv_rl
        in_values = [(0., 0.), (0.003, 0.002)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rt_from_rv(self):
        func = equations.rt_from_rv
        in_values = [(0.,), (0.005,), (0.1,)]
        out_values = [0., 0.005, 0.1]
        tols = [1e-8, 1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rt_from_rv_ri(self):
        func = equations.rt_from_rv_ri
        in_values = [(0., 0.), (0.003, 0.002)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rv_from_rt(self):
        func = equations.rv_from_rt
        in_values = [(0.,), (0.005,), (0.1,)]
        out_values = [0., 0.005, 0.1]
        tols = [1e-8, 1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rv_from_rt_rl_ri(self):
        func = equations.rv_from_rt_rl_ri
        in_values = [(0., 0., 0.), (0.01, 0.003, 0.002)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rv_from_rt_rl(self):
        func = equations.rv_from_rt_rl
        in_values = [(0., 0.), (0.01, 0.005)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rv_from_rt_ri(self):
        func = equations.rv_from_rt_ri
        in_values = [(0., 0.), (0.01, 0.005)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_ri_from_rt_rv_rl(self):
        func = equations.rv_from_rt_rl_ri
        in_values = [(0., 0., 0.), (0.01, 0.003, 0.002)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_ri_from_rt_rv(self):
        func = equations.ri_from_rt_rv
        in_values = [(0., 0.), (0.01, 0.005)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rl_from_rt_rv_ri(self):
        func = equations.rv_from_rt_rl_ri
        in_values = [(0., 0., 0.), (0.01, 0.003, 0.002)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rl_from_rt_rv(self):
        func = equations.rl_from_rt_rv
        in_values = [(0., 0.), (0.01, 0.005)]
        out_values = [0., 0.005]
        tols = [1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rvs_from_p_es(self):
        func = equations.rvs_from_p_es
        in_values = [(1e5, 500.), (9e4, 1000.), (8e4, 0.)]
        out_values = [0.003125, 0.006988, 0.]
        tols = [1e-6, 1e-6, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rvs_from_qvs(self):
        func = equations.rvs_from_qvs
        in_values = [(0.,), (0.05,), (0.1,)]
        out_values = [0., 0.05263, 0.11111]
        tols = [1e-8, 0.0001, 0.0001]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_rvs_from_qvs_lwv(self):
        func = equations.rvs_from_qvs_lwv
        in_values = [(0.,), (0.005,), (0.1,)]
        out_values = [0., 0.005, 0.1]
        tols = [1e-8, 1e-8, 1e-8]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_T_from_es_Bolton(self):
        func = equations.T_from_es_Bolton
        in_values = [(603,), (2310,), (7297,), (12210,)]
        out_values = [273.15, 273.15+20, 273.15+40, 273.15+50]
        tols = [1, 1, 1, 1]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_Tlcl_from_T_RH(self):
        func = equations.Tlcl_from_T_RH
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_Tlcl_from_T_Td(self):
        func = equations.Tlcl_from_T_Td
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_Tlcl_from_T_e(self):
        func = equations.Tlcl_from_T_e
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_Tv_from_T_assuming_Tv_equals_T(self):
        func = equations.Tv_from_T_assuming_Tv_equals_T
        in_values = [(273.15,), (100.,), (300.,)]
        out_values = [273.15, 100., 300.]
        tols = [0.001, 0.001, 0.001]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_Tv_from_p_rho_ideal_gas(self):
        func = equations.Tv_from_p_rho_ideal_gas
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_Tw_from_T_RH_Stull(self):
        func = equations.Tw_from_T_RH_Stull
        in_values = [(20+273.15, 50)]
        out_values = [13.7+273.15, ]
        tols = [0.1]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_T_from_Tv_assuming_Tv_equals_T(self):
        func = equations.T_from_Tv_assuming_Tv_equals_T
        in_values = [(273.15,), (100.,), (300.,)]
        out_values = [273.15, 100., 300.]
        tols = [0.001, 0.001, 0.001]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_theta_from_p_T(self):
        func = equations.theta_from_p_T
        in_values = [(75000., 273.15), (1e5, 253.15), (10000., 253.15)]
        out_values = [296.57, 253.15, 489.07]
        tols = [0.1, 0.01, 0.5]
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_thetae_from_p_T_Tlcl_rv_Bolton(self):
        func = equations.thetae_from_p_T_Tlcl_rv_Bolton
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_thetae_from_p_e_T_RH_rv_rt(self):
        func = equations.thetae_from_p_e_T_RH_rv_rt
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_thetae_from_T_RH_rv_lwv(self):
        func = equations.thetae_from_T_RH_rv_lwv
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_thetaes_from_p_T_rvs_Bolton(self):
        func = equations.thetaes_from_p_T_rvs_Bolton
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_w_from_omega_rho_hydrostatic(self):
        func = equations.w_from_omega_rho_hydrostatic
        in_values = []
        out_values = []
        tols = []
        self._assert_accurate_values(func, in_values, out_values, tols)

    def test_z_from_Phi(self):
        func = equations.z_from_Phi
        in_values = [(0.,), (294.3,), (981.,)]
        out_values = [0., 30., 100.]
        tols = [1e-8, 0.1, 0.1]
        self._assert_accurate_values(func, in_values, out_values, tols)


if __name__ == '__main__':
    nose.run()
