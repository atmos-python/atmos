# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:33:01 2015

@author: mcgibbon
"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('_cython_quantities.pyx')
)
