# -*- coding: utf-8 -*-
"""
test.py: Testing suite for other modules.
"""
import unittest
import nose
import numpy as np
import equations
import util
import decorators
from nose.tools import raises
from constants import Rd
from solve import BaseSolver, FluidSolver, calculate, \
    _get_module_methods, _get_calculatable_methods_dict,\
    _get_shortest_solution


def test_quantities_dict_complete():
    names = [item['output'] for item in _get_module_methods(equations)]
    for name in names:
        if name not in equations.quantities.keys():
            try:
                util.parse_derivative_string(name)
            except ValueError:
                raise AssertionError('{} not in quantities dict'.format(name))


def test_get_module_methods_nonempty():
    result = _get_module_methods(equations)
    assert len(result) > 0


def test_default_assumptions_exist():
    for m in FluidSolver.default_assumptions:
        if m not in FluidSolver.all_assumptions:
            raise AssertionError('{} not a valid assumption'.format(m))


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

    @raises(TypeError)
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
        assert isinstance(rho, np.ndarray)

    def test_returns_float(self):
        rho = calculate('rho', Tv=1., p=1.)
        assert isinstance(rho, float)

    def test_depth_2_calculation(self):
        rho = calculate('rho', add_assumptions=('Tv equals T',), **self.vars2)
        assert rho.shape == self.shape
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)

    def test_double_calculation(self):
        Tv, rho = calculate('Tv', 'rho', add_assumptions=('Tv equals T',),
                            **self.vars2)
        assert Tv.shape == self.shape
        assert rho.shape == self.shape
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)
        assert isinstance(Tv, np.ndarray)

    def test_double_reverse_calculation(self):
        rho, Tv = calculate('rho', 'Tv', add_assumptions=('Tv equals T',),
                            **self.vars2)
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)
        assert isinstance(Tv, np.ndarray)


class TestSolveValuesNearSkewT(unittest.TestCase):

    def setUp(self):
        self.quantities = {'p': 8.9e4, 'Tv': 4.5+273.15, 'theta': 14.+273.15,
                           'rv': 1e-3, 'Tlcl': -22.5+273.15,
                           'thetae': 17.+273.15, 'Tw': -2.5+273.15,
                           'Td': -18.5+273.15, 'plcl': 62500.,
                           }
        self.quantities['T'] = calculate('T', **self.quantities)
        self.quantities['rho'] = calculate('rho', **self.quantities)
        self.add_assumptions = ('bolton', 'unfrozen bulb')

    def _generator(self, quantity, tolerance):
        skew_T_value = self.quantities.pop(quantity)
        calculated_value, funcs = calculate(
            quantity, add_assumptions=self.add_assumptions,
            debug=True, **self.quantities)
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
        self._generator('p', 10000.)

    def test_calculate_Tv(self):
        self._generator('Tv', 1.)

    def test_calculate_theta(self):
        self._generator('theta', 1.)

    def test_calculate_rv(self):
        self._generator('rv', 1e-3)

    def test_calculate_Tlcl(self):
        self._generator('Tlcl', 1.)

    def test_calculate_thetae(self):
        self._generator('thetae', 1.)

    def test_calculate_Tw(self):
        quantity = 'Tw'
        skew_T_value = self.quantities.pop(quantity)
        calculated_value, funcs = calculate(
            quantity, add_assumptions=('bolton', 'unfrozen bulb'),
            debug=True,
            **self.quantities)
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
        self.quantities = {'p': 8.9e4, 'Tv': 9.+273.15, 'theta': 18.4+273.15,
                           'rv': 6e-3, 'Tlcl': 3.8+273.15,
                           'thetae': 36.5+273.15,
                           'Tw': 6.5+273.15, 'Td': 4.8+273.15, 'plcl': 83500.,
                           }
        self.quantities['T'] = calculate('T', **self.quantities)
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


class EquationTest(unittest.TestCase):

    in_values = []
    out_values = []
    tols = []
    func = None

    def test_accurate_values(self):
        for i, args in enumerate(self.in_values):
            out_calc = self.func(*args)
            if abs(out_calc - self.out_values[i]) > self.tols[i]:
                raise AssertionError(
                    'Calculated value {} from inputs {} is more than {}'
                    'away from {}'.format(out_calc, args, self.tols[i],
                                          self.out_values[i]))


if __name__ == '__main__':
    nose.run()
