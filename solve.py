# -*- coding: utf-8 -*-
"""
solve.py: Utilities that use equations to solve for quantities, given other
    quantities and a set of assumptions.

@author: Jeremy McGibbon
"""
import inspect
import equations
from types import MethodType
from six import add_metaclass
from textwrap import wrap


class ExcludeError(Exception):
    pass


def _get_relevant_methods(inputs, methods):
    '''
    Given an iterable of input variable names and a methods dictionary,
    returns the subset of that methods dictionary that can be calculated and
    which doesn't calculate something we already have. Additionally it may
    only contain one method for any given output variable, which is the one
    with the fewest possible arguments.
    '''
    # Initialize a return dictionary
    relevant_methods = {}
    # Iterate through each potential method output
    for var in methods.keys():
        # See if we already have this output
        if var in inputs:
            pass  # if we have it, we don't need to calculate it!
        else:
            # Initialize a dict for this output variable
            var_dict = {}
            for args, func in methods[var].items():
                # See if we have what we need to solve this equation
                if all([arg in inputs for arg in args]):
                    # If we do, add it to the var_dict
                    var_dict[args] = func
            if len(var_dict) == 0:
                # No methods for this variable, keep going
                pass
            elif len(var_dict) == 1:
                # Exactly one method, perfect.
                relevant_methods[var] = var_dict
            else:
                # More than one method, find the one with the least arguments
                min_args = min(var_dict.keys(), key=lambda x: len(x))
                relevant_methods[var] = {min_args: var_dict[min_args]}
    return relevant_methods


def _get_shortest_solution(outputs, inputs, exclude, methods):
    '''
    Parameters
    ----------
    outputs: tuple
        Strings corresponding to the final variables we want output
    inputs: tuple
        Strings corresponding to the variables we have so far
    exclude: tuple
        Strings corresponding to variables that won't help calculate the
        outputs.
    methods: dict
        A methods dictionary

    Returns (funcs, func_args, extra_values).
    '''
    relevant_methods = _get_relevant_methods(inputs, methods)
    # Check if we can already directly compute the outputs
    if len(relevant_methods) == 0:
        raise ValueError('cannot calculate outputs from inputs')
    if all([(o in relevant_methods.keys()) or (o in inputs) for o in outputs]):
        funcs = []
        args = []
        extra_values = []
        for o in outputs:
            if o not in inputs:
                o_args, o_func = relevant_methods[o].items()[0]
                funcs.append(o_func)
                args.append(o_args)
                extra_values.append(o)
        return tuple(funcs), tuple(args), tuple(extra_values)
    next_variables = [key for key in relevant_methods.keys()
                      if key not in exclude]
    if len(next_variables) == 0:
        raise ExcludeError
    results = []
    intermediates = []
    for i in range(len(next_variables)):
        try:
            results.append(_get_shortest_solution(
                outputs, inputs + (next_variables[i],), exclude, methods))
        except ExcludeError:
            continue
        intermediates.append(next_variables[i])
        exclude = exclude + (next_variables[i],)
    if len(results) == 0:
        # all subresults raised ExcludeError
        raise ExcludeError

    def option_key(a):
        return len(a[0]) + 0.001*len(a[1])
    best_result = min(results, key=option_key)
    best_index = results.index(best_result)
    best_intermediate = intermediates[best_index]
    args, func = relevant_methods[best_intermediate].items()[0]
    best_option = ((func,) + best_result[0], (args,) + best_result[1],
                   (best_intermediate,) + best_result[2])
    return best_option


def _get_module_methods(module):
    '''
    Returns a methods dictionary corresponding to the equations in the given
    module.

    The methods dictionary is in this form:
    {'output': [(args, assumptions, function),
                (args, assumptions, function), ...]
     'another output': [...]
     ...
    }

    Where args is a list of arguments that get passed into the function,
    assumptions are strings indicating the assumptions of the equation on
    which the function is based, and function is the function itself that
    computes the output.
    '''
    # Set up the methods dict we will eventually return
    methods = {}
    # Get functions and their outputs from our equation module
    funcs = []
    outputs = []
    for item in inspect.getmembers(equations):
        if (item[0][0] != '_' and '_from_' in item[0]):
            funcs.append(item[1])
            outputs.append(item[0][:item[0].find('_from_')])
    # Store our funcs in methods
    for i in range(len(funcs)):
        if outputs[i] not in methods.keys():
            methods[outputs[i]] = []
        args = tuple(inspect.getargspec(funcs[i]).args)
        try:
            assumptions = tuple(funcs[i].assumptions)
        except AttributeError:
            raise NotImplementedError('function {} in equations module has no '
                                      'assumption '
                                      'definition'.format(funcs[i].__name__))
        methods[outputs[i]].append((args, assumptions, funcs[i]))
    return methods


def _fill_doc(s, module, default_assumptions):
        assumptions = module.assumptions
        s = s.replace(
            '<assumptions list goes here>',
            '\n'.join(sorted(
                ["'{}': {}".format(a, desc) for a, desc in
                 assumptions.items()],
                key=lambda x: x.lower())))
        s = s.replace(
            '<default assumptions list goes here>',
            '\n'.join(
                wrap('Default assumptions are ' +
                     ', '.join(["'{}'".format(a) for a in
                                default_assumptions]) + '.', width=80)))
        s = s.replace(
            '<quantity parameter list goes here>',
            '\n'.join(sorted([
                '{} {} ({})'.format(
                    (q + ' :').ljust(9), info['name'], info['units'])
                for q, info in
                module.quantities.items()
                ], key=lambda x: x.lower())))
        return s


class SolverMeta(type):

    def __new__(cls, name, parents, dct):
        # create a class_id if it's not specified
        calculate_method = None
        if dct['_equation_module'] is not None:
            def calculate_func(self, *args):
                return parents[0].calculate(self, *args)
            calculate_func.__doc__ = _fill_doc(
                parents[0].calculate.__doc__, dct['_equation_module'],
                dct['default_assumptions'])
            if '__doc__' in dct.keys():
                dct['__doc__'] = _fill_doc(
                    dct['__doc__'], dct['_equation_module'],
                    dct['default_assumptions'])
            dct['all_assumptions'] = tuple(
                set([]).union(*[f[1].func_dict['assumptions']
                                for f in inspect.getmembers(equations)
                                if hasattr(f[1], 'func_dict') and
                                'assumptions' in
                                f[1].func_dict.keys()]))

        # we need to call type.__new__ to complete the initialization
        instance = super(SolverMeta, cls).__new__(cls, name, parents, dct)
        if calculate_method is not None:
            instance.calculate = MethodType(calculate_func, instance, cls)
        return instance


@add_metaclass(SolverMeta)
class BaseSolver(object):
    '''
Base class for solving systems of equations. Should not be instantiated,
as it is not associated with any equations.
    '''

    _equation_module = None

    def __new__(cls, *args, **kwargs):
        if cls is BaseSolver:
            raise TypeError('BaseDeriver may not be instantiated. Use a '
                            'subclass.')
        if cls._equation_module is None:
            raise NotImplementedError('Class must have _equation_module '
                                      'defined')
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, **kwargs):
        if 'debug' in kwargs.keys():
            self._debug = kwargs['debug']
        else:
            self._debug = False
        if 'assumptions' in kwargs.keys():
            if ('add_assumptions' in kwargs.keys() or
                    'remove_assumptions' in kwargs.keys()):
                raise ValueError('cannot give kwarg assumptions with '
                                 'add_assumptions or remove_assumptions')
            assumptions = kwargs['assumptions']
        else:
            assumptions = self.default_assumptions
            if 'add_assumptions' in kwargs.keys():
                if 'remove_assumptions' in kwargs.keys():
                    if any([a in kwargs['remove_assumptions']
                            for a in kwargs['add_assumptions']]):
                        raise ValueError('assumption may not be present in '
                                         'both add_assumptions and '
                                         'remove_assumptions')
                assumptions = assumptions + tuple(
                    [a for a in kwargs['add_assumptions'] if a not in
                     assumptions])
            if 'remove_assumptions' in kwargs.keys():
                assumptions = tuple([a for a in assumptions if a not in
                                     kwargs['remove_assumptions']])
        self.methods = self._get_methods(assumptions)
        self.vars = kwargs

    def calculate(self, *args):
        '''
Calculates and returns a requested quantity from quantities passed in
as keyword arguments.

Parameters
----------
varname_out : string
    Name of quantity to be calculated.

Quantity Parameters
-------------------
<quantity parameter list goes here>

Returns
-------
quantity : ndarray
    Calculated quantity, in units listed under quantity parameters.

Raises
------
ValueError:
    If the output quantity cannot be determined from the input
    quantities.
        '''
        funcs, func_args, extra_values = \
            _get_shortest_solution(tuple(args), tuple(self.vars.keys()), (),
                                   self.methods)
        # Above method completed successfully if no ValueError has been raised
        # Calculate each quantity we need to calculate in order
        for i, func in enumerate(funcs):
            # Compute this quantity
            value = func(*[self.vars[varname] for varname in func_args[i]])
            # Add it to our dictionary of quantities for successive functions
            self.vars[extra_values[i]] = value
        if self._debug:
            if len(args) == 1:
                return self.vars[args[0]], funcs
            else:
                return [self.vars[arg] for arg in args] + [funcs, ]
        else:
            if len(args) == 1:
                return self.vars[args[0]]
            else:
                return [self.vars[arg] for arg in args]

    def _get_quantity_parameter_list(self):
        return '\n'.join([
            '        {}: {} ({})'.format(q, info['name'], info['units'])
            for q, info in self._equation_module.quantities.items()
        ])

    def _get_methods(self, method_options):
        '''
Returns a dictionary of methods including the default methods of the
class and specified optional methods. Will override a default method
if an optional method is given that takes the same inputs and produces
the same output.

Parameters
----------
methods: iterable
    Strings referring to optional methods in self.optional_methods.

Returns
-------
methods : dict
    A dictionary whose keys are strings indicating output variables,
    and values are dictionaries indicating equations for that output. The
    equation dictionary's keys are strings indicating variables to use
    as function arguments, and its values are the functions themselves.

Raises
------
ValueError
    If a method given is not present in self.optional_methods.
    If multiple optional methods are selected which calculate the same
    output quantity from the same input quantities.
        '''
        for m in method_options:
            if m not in self.all_assumptions:
                raise ValueError('method {} matches no equations'.format(m))
        methods = {}
        module_methods = _get_module_methods(self._equation_module)
        # Go through each output variable
        for output, L in module_methods.items():
            # Go through each potential equation
            for args, assumptions, func in L:
                # See if we're using the equation's assumptions
                if all(item in method_options for item in assumptions):
                    # At this point, we want to add the equation
                    # Make sure we have a dict to add it to
                    if output not in methods.keys():
                        methods[output] = {}
                    # Check if this is a duplicate equation
                    if args in methods[output].keys():
                        raise ValueError('methods given define duplicate '
                                         'equations')
                    methods[output][args] = func
        return methods


class FluidSolver(BaseSolver):
    '''
Initializes with the given assumptions enabled, and variables passed as
keyword arguments stored.

Parameters
----------
assumptions : tuple, optional
    Strings specifying which assumptions to enable. See below for options.
<default assumptions list goes here>

Returns
-------
out : FluidSolver
    A FluidSolver object with the specified assumptions and variables.

Assumptions
-----------
<assumptions list goes here>

Examples
--------
>>> solver = FluidSolver(rho=array1, p=array2)

Non-default assumptions:

>>> solver = FluidSolver(assumptions=('Tv equals T'), rho=array1, p=array2)
>>> T = solver.calculate('T')
    '''

    _equation_module = equations
    default_assumptions = (
        'ideal gas', 'hydrostatic', 'constant g', 'constant Lv', 'constant Cp',
        'no liquid water', 'no solid water', 'bolton',)


def calculate(*args, **kwargs):
    '''
Calculates and returns a requested quantity from quantities passed in as
keyword arguments.

Parameters
----------
args : string
    Names of quantities to be calculated.
assumptions : tuple, optional
    Names of assumptions that can be used for calculation, as strings.
<default assumptions list goes here>

Assumptions
-----------
<assumptions list goes here>

Quantity Parameters
-------------------
All quantity parameters are optional, and must be of the same type,
either ndarray or iris Cube. If ndarrays are used, then units must match
the units specified below. If iris Cube is used, any units may be used
for input and the output will be given in the units specified below.

<quantity parameter list goes here>

Returns
-------
quantity : ndarray or iris Cube
    Calculated quantity.
    Return type is the same as quantity parameter types.
    If multiple quantities are requested, returns a tuple containing the
    quantities.

Notes
-----
Calculating multiple quantities at once can avoid re-computing intermediate
quantities, but requires more memory.

Examples
--------
>>>
    '''
    def get_arg(name, kwargs):
        try:
            arg = kwargs.pop(name)
        except:
            arg = None
        return arg
    solver = FluidSolver(**kwargs)
    result = [solver.calculate(var) for var in args]
    if len(result) == 1:
        return result[0]
    else:
        return result

calculate.__doc__ = _fill_doc(calculate.__doc__, equations,
                              FluidSolver.default_assumptions)
