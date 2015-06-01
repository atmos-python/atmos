# -*- coding: utf-8 -*-
"""
decorators.py: Function decorators used by the rest of this module.
"""
from __future__ import division, absolute_import, unicode_literals
import inspect
from atmos.util import quantity_string, assumption_list_string, \
    quantity_spec_string, doc_paragraph


# Define some decorators for our equations
def assumes(*args):
    '''Stores a function's assumptions as an attribute.'''
    args = tuple(args)

    def decorator(func):
        func.assumptions = args
        return func
    return decorator


def overridden_by_assumptions(*args):
    '''Stores what assumptions a function is overridden by as an attribute.'''
    args = tuple(args)

    def decorator(func):
        func.overridden_by_assumptions = args
        return func
    return decorator


def equation_docstring(quantity_dict, assumption_dict,
                       equation=None, references=None, notes=None):
    '''
Creates a decorator that adds a docstring to an equation function.

Parameters
----------

quantity_dict : dict
    A dictionary describing the quantities used in the equations. Its keys
    should be abbreviations for the quantities, and its values should be a
    dictionary of the form {'name': string, 'units': string}.
assumption_dict : dict
    A dictionary describing the assumptions used by the equations. Its keys
    should be short forms of the assumptions, and its values should be long
    forms of the assumptions, as you would insert into the sentence
    'Calculates (quantity) assuming (assumption 1), (assumption 2), and
    (assumption 3).'
equation : string, optional
    A string describing the equation the function uses. Should be wrapped
    to be no more than 80 characters in length.
references : string, optional
    A string providing references for the function. Should be wrapped to be
    no more than 80 characters in length.

Raises
------

ValueError:
    If the function name does not follow (varname)_from_(any text here), or
    if an argument of the function or the varname (as above) is not present
    in quantity_dict, or if an assumption in func.assumptions is not present
    in the assumption_dict.
    '''
    # Now we have our utility functions, let's define the decorator itself
    def decorator(func):
        out_name_end_index = func.__name__.find('_from_')
        if out_name_end_index == -1:
            raise ValueError('equation_docstring decorator must be applied to '
                             'function whose name contains "_from_"')
        out_quantity = func.__name__[:out_name_end_index]
        in_quantities = inspect.getargspec(func).args
        docstring = 'Calculates {0}'.format(
            quantity_string(out_quantity, quantity_dict))
        try:
            if len(func.assumptions) > 0:
                docstring += ' assuming {0}'.format(
                    assumption_list_string(func.assumptions, assumption_dict))
        except AttributeError:
            pass
        docstring += '.'
        docstring = doc_paragraph(docstring)
        docstring += '\n\n'
        if equation is not None:
            func.equation = equation
            docstring += doc_paragraph(':math:`' + equation.strip() + '`')
            docstring += '\n\n'
        docstring += 'Parameters\n'
        docstring += '----------\n\n'
        docstring += '\n'.join([quantity_spec_string(q, quantity_dict)
                                for q in in_quantities])
        docstring += '\n\n'
        docstring += 'Returns\n'
        docstring += '-------\n\n'
        docstring += quantity_spec_string(out_quantity, quantity_dict)
        if notes is not None:
            docstring += '\n\n'
            docstring += 'Notes\n'
            docstring += '-----\n\n'
            docstring += notes.strip()
        if references is not None:
            if notes is None:  # still need notes header for references
                docstring += '\n\n'
                docstring += 'Notes\n'
                docstring += '-----\n\n'
            func.references = references
            docstring += '\n\n'
            docstring += '**References**\n\n'
            docstring += references.strip()
        docstring += '\n'
        func.__doc__ = docstring
        return func

    return decorator
