# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:44:56 2015

@author: mcgibbon
"""
import unittest
import nose
import numpy as np
import equations
from nose.tools import raises
from constants import Rd
from solve import _BaseSolver, FluidSolver, calculate, _get_best_result, \
    _get_methods, default_methods, all_methods, _get_relevant_methods


def test_get_methods_nonempty():
    result = _get_methods(equations)
    assert len(result) > 0


def test_default_methods_exist():
    for m in default_methods:
        if m not in all_methods:
            raise AssertionError('{} not a valid method'.format(m))


def test_get_relevant_methods_empty():
    methods = {}
    out_methods = _get_relevant_methods((), methods)
    assert isinstance(out_methods, dict)
    assert len(out_methods) == 0


def test_get_relevant_methods_returns_correct_type():
    methods = {'a': {('b',): lambda x: x}}
    out_methods = _get_relevant_methods(('b',), methods)
    assert isinstance(out_methods, dict)


def test_get_relevant_methods_gets_single_method():
    methods = {'a': {('b',): lambda x: x}}
    out_methods = _get_relevant_methods(('b',), methods)
    assert 'a' in out_methods.keys()


def test_get_relevant_methods_removes_correct_second_method():
    methods = {'a': {('b',): lambda x: x,
                     ('c', 'b'): lambda x: x}}
    out_methods = _get_relevant_methods(('b', 'c'), methods)
    assert 'a' in out_methods.keys()
    assert ('b',) in out_methods['a'].keys()
    assert len(out_methods['a']) == 1


def test_get_relevant_methods_removes_irrelevant_second_method():
    methods = {'a': {('b', 'd'): lambda x, y: x,
                     ('c',): lambda x: x}}
    out_methods = _get_relevant_methods(('b', 'd'), methods)
    assert 'a' in out_methods.keys()
    assert ('b', 'd') in out_methods['a'].keys()
    assert len(out_methods['a']) == 1


def test_get_relevant_methods_gets_no_methods():
    methods = {'a': {('b',): lambda x: x},
               'x': {('y', 'z'): lambda y, z: y*z}
               }
    out_methods = _get_relevant_methods(('q',), methods)
    assert isinstance(out_methods, dict)
    assert len(out_methods) == 0


def test_get_relevant_methods_doesnt_calculate_input():
    methods = {'a': {('b',): lambda x: x},
               'x': {('y', 'z'): lambda y, z: y*z}
               }
    out_methods = _get_relevant_methods(('b', 'a'), methods)
    assert isinstance(out_methods, dict)
    assert len(out_methods) == 0


@raises(ValueError)
def test_get_best_result_empty():
    '''With empty input should raise ValueError.'''
    _get_best_result(())


def test_get_best_result_single():
    '''With single input should return that input'''
    f = lambda x: x
    tup = ((f,), ('p',), ('p',),)
    result = _get_best_result((tup,))
    assert result == tup


def test_get_best_result_multiple():
    '''With single input should return input with fewest function calls.'''
    f = lambda x: x
    tup1 = ((f,), ('p',), ('r',),)
    tup2 = ((f, f), ('p', 'q'), ('r', 'h'),)
    tup3 = ((f, f, f), ('p', 'q', 'a'), ('r', 'h', 'm'))
    result = _get_best_result((tup1, tup2, tup3))
    assert result == tup1


def test_get_best_result_multiple_repeats():
    f = lambda x: x
    tup1 = ((f,), ('p',), ('r',),)
    tup2 = ((f, f), ('p', 'q'), ('r', 'h'),)
    tup3 = ((f, f, f), ('p', 'q', 'a'), ('r', 'h', 'm'))
    result = _get_best_result((tup3, tup2, tup1, tup2, tup1, tup3))
    assert result == tup1


class BaseSolverTests(unittest.TestCase):

    @raises(TypeError)
    def test_cannot_instantiate(self):
        _BaseSolver()


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

    def test_is_instance_of_BaseDeriver(self):
        deriv = FluidSolver()
        assert isinstance(deriv, _BaseSolver)

    def test_creation_one_method(self):
        FluidSolver(methods=('hydrostatic',))

    def test_creation_compatible_methods(self):
        FluidSolver(methods=('hydrostatic', 'Tv equals T',))

    @raises(ValueError)
    def test_creation_incompatible_methods(self):
        FluidSolver(methods=('Goff-Gratch', 'Wexler',))

    @raises(ValueError)
    def test_creation_undefined_method(self):
        FluidSolver(methods=('moocow',))

    @raises(ValueError)
    def test_creation_undefined_method_with_defined_method(self):
        FluidSolver(methods=('hydrostatic', 'moocow',))

    def test_creation_with_vars(self):
        FluidSolver(**self.vars1)

    def test_creation_with_vars_and_method(self):
        FluidSolver(methods=('Tv equals T',), **self.vars1)

    def test_simple_calculation(self):
        deriver = FluidSolver(methods=default_methods, **self.vars1)
        rho = deriver.calculate('rho')
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)

    def test_depth_2_calculation(self):
        deriver = FluidSolver(methods=default_methods + ('Tv equals T',),
                              **self.vars2)
        rho = deriver.calculate('rho')
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)


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

    def test_depth_2_calculation(self):
        rho = calculate('rho', methods=default_methods +
                        ('Tv equals T',), **self.vars2)
        assert rho.shape == self.shape
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)

    def test_double_calculation(self):
        Tv, rho = calculate('Tv', 'rho', methods=default_methods +
                            ('Tv equals T',), **self.vars2)
        assert Tv.shape == self.shape
        assert rho.shape == self.shape
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)
        assert isinstance(Tv, np.ndarray)

    def test_double_reverse_calculation(self):
        rho, Tv = calculate('rho', 'Tv', methods=default_methods +
                            ('Tv equals T',), **self.vars2)
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)
        assert isinstance(Tv, np.ndarray)


if __name__ == '__main__':
    nose.run()
