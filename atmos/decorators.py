# -*- coding: utf-8 -*-
"""
decorators.py: Function decorators used by the rest of this module.
"""
import inspect
from textwrap import wrap


# Define some decorators for our equations
def assumes(*args):
    '''Stores a function's assumptions as an attribute.'''
    args = tuple(args)

    def decorator(func):
        func.assumptions = tuple(args)
        return func
    return decorator


def overridden_by_assumptions(*args):
    '''Stores what assumptions a function is overridden by as an attribute.'''
    args = tuple(args)

    def decorator(func):
        func.overridden_by_assumptions = tuple(args)
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
    '''
    def quantity_string(name):
        '''Takes in an abbreviation for a quantity, and returns a more
           descriptive string of the quantity as "name (units)"
        '''
        return '{} ({})'.format(quantity_dict[name]['name'],
                                quantity_dict[name]['units'])

    def strings_to_list_string(strings):
        '''Takes a list of strings presumably containing words and phrases,
           and returns a "list" form of those strings, like:

           >>> strings_to_list_string(('cats', 'dogs'))
           >>> 'cats and dogs'

           or

           >>> strings_to_list_string(('pizza', 'pop', 'chips'))
           >>> 'pizza, pop, and chips'
        '''
        if len(strings) == 1:
            return strings[0]
        elif len(strings) == 2:
            return ' and '.join(strings)
        else:
            return '{}, and {}'.format(', '.join(strings[:-1]),
                                       strings[-1])

    def quantity_list_string(names):
        '''Takes in a list of quantity abbreviations, and returns a "list"
           form of those quantities expanded descriptively as name (units).
           See quantity_string(name) and strings_to_list_string(strings).
        '''
        assert len(names) > 0
        q_strings = [quantity_string(name) for name in names]
        return strings_to_list_string(q_strings)

    def assumption_list_string(assumptions):
        '''Takes in a list of short forms of assumptions, and returns a "list"
           form of the long form of the assumptions.
        '''
        assumption_strings = [assumption_dict[a] for a in assumptions]
        return strings_to_list_string(assumption_strings)

    def quantity_spec_string(name):
        '''Returns a quantity specification for docstrings. Example:
           >>> quantity_spec_string('Tv')
           >>> 'Tv : ndarray\n    Data for virtual temperature.'
        '''
        s = '{} : ndarray\n'.format(name)
        s += doc_paragraph('Data for {}.'.format(
            quantity_string(name)), indent=4)
        return s

    def doc_paragraph(s, indent=0):
        '''Takes in a string without wrapping corresponding to a paragraph,
           and returns a version of that string wrapped to be at most 80
           characters in length on each line.
           If indent is given, ensures each line is indented to that number
           of spaces.
        '''

        return '\n'.join([' '*indent + l for l in wrap(s, width=80-indent)])

    # Now we have our utility functions, let's define the decorator itself
    def decorator(func):
        out_name_end_index = func.__name__.find('_from_')
        if out_name_end_index == -1:
            raise ValueError('equation_docstring decorator must be applied to '
                             'function whose name contains "_from_"')
        out_quantity = func.__name__[:out_name_end_index]
        in_quantities = inspect.getargspec(func).args
        docstring = 'Calculates {}'.format(
            quantity_string(out_quantity))
        try:
            if len(func.assumptions) > 0:
                docstring += ' assuming {}'.format(
                    assumption_list_string(func.assumptions))
        except AttributeError:
            pass
        docstring += '.'
        docstring = doc_paragraph(docstring)
        docstring += '\n\n'
        if equation is not None:
            func.func_dict['equation'] = equation
            docstring += equation.strip() + '\n\n'
        docstring += 'Parameters\n'
        docstring += '----------\n'
        docstring += '\n'.join([quantity_spec_string(q)
                                for q in in_quantities])
        docstring += '\n\n'
        docstring += 'Returns\n'
        docstring += '-------\n'
        docstring += quantity_spec_string(out_quantity)
        if notes is not None:
            docstring += '\n\n'
            docstring += 'Notes\n'
            docstring += '-----\n'
            docstring += notes.strip()
        if references is not None:
            func.func_dict['references'] = references
            docstring += '\n\n'
            docstring += 'References\n'
            docstring += '----------\n'
            docstring += references.strip()
        docstring += '\n'
        func.func_doc = docstring
        return func

    return decorator
