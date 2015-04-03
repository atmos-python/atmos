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


def equation_docstring(quantity_dict, assumption_dict):
    def quantity_string(name):
        return '{} ({})'.format(quantity_dict[name]['name'],
                                quantity_dict[name]['units'])

    def strings_to_list_string(strings):
        if len(strings) == 1:
            return strings[0]
        elif len(strings) == 2:
            return ' and '.join(strings)
        else:
            return '{}, and {}'.format(', '.join(strings[:-1]),
                                       strings[-1])

    def quantity_list_string(names):
        assert len(names) > 0
        q_strings = [quantity_string(name) for name in names]
        return strings_to_list_string(q_strings)

    def assumption_list_string(assumptions):
        assumption_strings = [assumption_dict[a] for a in assumptions]
        return strings_to_list_string(assumption_strings)

    def decorator(func):
        out_name_end_index = func.__name__.find('_from_')
        if out_name_end_index == -1:
            raise ValueError('equation_docstring decorator must be applied to '
                             'function whose name contains "_from_"')
        out_quantity = func.__name__[:out_name_end_index]
        in_quantities = inspect.getargspec(func).args
        docstring = 'Calculates {} from {}'.format(
            quantity_string(out_quantity),
            quantity_list_string(in_quantities))
        try:
            if len(func.assumptions) > 0:
                docstring += ', assuming {}'.format(
                    assumption_list_string(func.assumptions))
        except AttributeError:
            pass
        docstring += '.'
        func.func_doc = docstring
        return func

    raise NotImplementedError
    return decorator
