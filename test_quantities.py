# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:44:56 2015

@author: mcgibbon
"""
import unittest
import nose
import numpy as np
from nose.tools import raises
from quantities import *
from quantities import _BaseDeriver


class BaseDeriverTests(unittest.TestCase):

    @raises(TypeError)
    def test_cannot_instantiate(self):
        _BaseDeriver()


class FluidDeriverTests(unittest.TestCase):

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
        FluidDeriver()

    def test_is_instance_of_BaseDeriver(self):
        deriv = FluidDeriver()
        assert isinstance(deriv, _BaseDeriver)

    def test_creation_one_method(self):
        FluidDeriver(methods=('hydrostatic',))

    def test_creation_compatible_methods(self):
        FluidDeriver(methods=('hydrostatic', 'dry',))

    @raises(ValueError)
    def test_creation_incompatible_methods(self):
        FluidDeriver(methods=('Goff-Gratch', 'Wexler',))

    @raises(ValueError)
    def test_creation_undefined_method(self):
        FluidDeriver(methods=('moocow',))

    @raises(ValueError)
    def test_creation_undefined_method_with_defined_method(self):
        FluidDeriver(methods=('hydrostatic', 'moocow',))

    def test_creation_with_vars(self):
        FluidDeriver(**self.vars1)

    def test_creation_with_vars_and_method(self):
        FluidDeriver(methods=('dry',), **self.vars1)

    def test_simple_calculation(self):
        deriver = FluidDeriver(**self.vars1)
        rho = deriver.calculate('rho')
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)

    def test_depth_2_calculation(self):
        deriver = FluidDeriver(methods=('dry',), **self.vars2)
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
        rho = calculate('rho', methods=('dry',), **self.vars2)
        assert rho.shape == self.shape
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)

    def test_double_calculation(self):
        Tv, rho = calculate('Tv', 'rho', methods=('dry',), **self.vars2)
        assert Tv.shape == self.shape
        assert rho.shape == self.shape
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)
        assert isinstance(Tv, np.ndarray)

    def test_double_reverse_calculation(self):
        rho, Tv = calculate('rho', 'Tv', methods=('dry',), **self.vars2)
        assert (rho == 1/Rd).all()
        assert isinstance(rho, np.ndarray)
        assert isinstance(Tv, np.ndarray)


if __name__ == '__main__':
    nose.run()
