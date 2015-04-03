# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:59:55 2015

@author: mcgibbon
"""
import inspect


# Define some decorators for our equations
def assumes(*args):
    '''Stores a function's assumptions as an attribute.'''
    args = tuple(args)

    def decorator(func):
        func.assumptions = args
        return func
    return decorator


def equation_docstring(quantities):
    def decorator(func):
        out_name_end_index = func__name__.find('_from_')
        if out_name_end_index == -1:
            raise ValueError('equation_docstring decorator must be applied to '
                             'function whose name contains "_from_"')
        out_quantity = func.__name__[:out_name_end_index]
        in_quantities = inspect.getargspec(func).args
        docstring = ''

    raise NotImplementedError
    return decorator
