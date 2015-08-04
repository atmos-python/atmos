# -*- coding: utf-8 -*-
"""
solve.py: Utilities that use equations to solve for quantities, given other
    quantities and a set of assumptions.
"""
from __future__ import division, absolute_import, unicode_literals
import inspect
from atmos import equations
import numpy as np
from six import add_metaclass, string_types
from textwrap import wrap
import re
import pint
try:
    from inspect import getfullargspec
except ImportError:
    from inspect import getargspec as getfullargspec

# initialize a unit registry for converting units
_ureg = pint.UnitRegistry()
_ureg.define('fraction = [fraction] = frac = ratio')
_ureg.define('percent = 0.01*fraction = %')
# regex program for identifying unit-specifying keyword arguments for
# solvers
_unit_kwarg_prog = re.compile(r'^(.+)_unit$|^(.+)_units$')


class ExcludeError(Exception):
    """
    Used in calculating shortest solutions to indicate solutions where all
    remaining variables to calculated are excluded.
    """
    pass


def get_calculatable_quantities(inputs, methods):
    '''
    Given an interable of input quantity names and a methods dictionary,
    returns a list of output quantities that can be calculated.
    '''
    output_quantities = []
    updated = True
    while updated:
        updated = False
        for output in methods.keys():
            if output in output_quantities or output in inputs:
                # we already know we can calculate this
                continue
            for args, func in methods[output].items():
                if all([arg in inputs or arg in output_quantities
                        for arg in args]):
                    output_quantities.append(output)
                    updated = True
                    break
    return tuple(output_quantities) + tuple(inputs)


def _get_methods_that_calculate_outputs(inputs, outputs, methods):
    '''
    Given iterables of input variable names, output variable names,
    and a methods dictionary, returns the subset of the methods dictionary
    that can be calculated, doesn't calculate something we already have,
    and only contains equations that might help calculate the outputs from
    the inputs.
    '''
    # Get a list of everything that we can possibly calculate
    # This is useful in figuring out whether we can calculate arguments
    intermediates = get_calculatable_quantities(inputs, methods)
    # Initialize our return dictionary
    return_methods = {}
    # list so that we can append arguments that need to be output for
    # some of the paths
    outputs = list(outputs)
    # keep track of when to exit the while loop
    keep_going = True
    while keep_going:
        # If there are no updates in a pass, the loop will exit
        keep_going = False
        for output in outputs:
            try:
                output_dict = return_methods[output]
            except:
                output_dict = {}
            for args, func in methods[output].items():
                # only check the method if we're not already returning it
                if args not in output_dict.keys():
                    # Initialize a list of intermediates needed to use
                    # this method, to add to outputs if we find we can
                    # use it.
                    needed = []
                    for arg in args:
                        if arg in inputs:
                            # we have this argument
                            pass
                        elif arg in outputs:
                            # we may need to calculate one output using
                            # another output
                            pass
                        elif arg in intermediates:
                            if arg not in outputs:
                                # don't need to add to needed if it's already
                                # been put in outputs
                                needed.append(arg)
                        else:
                            # Can't do this func
                            break
                    else:  # did not break, can calculate this
                        output_dict[args] = func
                        if len(needed) > 0:
                            # We added an output, so need another loop
                            outputs.extend(needed)
                            keep_going = True
            if len(output_dict) > 0:
                return_methods[output] = output_dict
    return return_methods


def _get_calculatable_methods_dict(inputs, methods):
    '''
    Given an iterable of input variable names and a methods dictionary,
    returns the subset of that methods dictionary that can be calculated and
    which doesn't calculate something we already have. Additionally it may
    only contain one method for any given output variable, which is the one
    with the fewest possible arguments.
    '''
    # Initialize a return dictionary
    calculatable_methods = {}
    # Iterate through each potential method output
    for var in methods.keys():
        # See if we already have this output
        if var in inputs:
            continue  # if we have it, we don't need to calculate it!
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
                continue
            elif len(var_dict) == 1:
                # Exactly one method, perfect.
                calculatable_methods[var] = var_dict
            else:
                # More than one method, find the one with the least arguments
                min_args = min(var_dict.keys(), key=lambda x: len(x))
                calculatable_methods[var] = {min_args: var_dict[min_args]}
    return calculatable_methods


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
    methods = _get_methods_that_calculate_outputs(inputs, outputs,
                                                  methods)
    calculatable_methods = _get_calculatable_methods_dict(inputs, methods)
    # Check if we can already directly compute the outputs
    if all([(o in calculatable_methods.keys()) or (o in inputs)
            for o in outputs]):
        funcs = []
        args = []
        extra_values = []
        for o in outputs:
            if o not in inputs:
                o_args, o_func = list(calculatable_methods[o].items())[0]
                funcs.append(o_func)
                args.append(o_args)
                extra_values.append(o)
        return tuple(funcs), tuple(args), tuple(extra_values)
    # Check if there's nothing left to try to calculate
    if len(calculatable_methods) == 0:
        raise ValueError('cannot calculate outputs from inputs')
    next_variables = [key for key in calculatable_methods.keys()
                      if key not in exclude]
    if len(next_variables) == 0:
        raise ExcludeError
    # We're not in a base case, so recurse this function
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

    # sort based on shortest solution, with tiebreaker based on total args
    # passed
    def option_key(a):
        return len(a[0]) + 0.001*sum(len(b) for b in a[1])
    best_result = min(results, key=option_key)
    best_index = results.index(best_result)
    best_intermediate = intermediates[best_index]
    args, func = list(calculatable_methods[best_intermediate].items())[0]
    best_option = ((func,) + best_result[0], (args,) + best_result[1],
                   (best_intermediate,) + best_result[2])
    return best_option


def _get_module_methods(module):
    '''
    Returns a methods list corresponding to the equations in the given
    module. Each entry is a dictionary with keys 'output', 'args', and
    'func' corresponding to the output, arguments, and function of the
    method. The entries may optionally include 'assumptions' and
    'overridden_by_assumptions' as keys, stating which assumptions are
    required to use the method, and which assumptions mean the method
    should not be used because it is overridden.
    '''
    # Set up the methods dict we will eventually return
    methods = []
    funcs = []
    for item in inspect.getmembers(equations):
        if (item[0][0] != '_' and '_from_' in item[0]):
            func = item[1]
            output = item[0][:item[0].find('_from_')]
        # avoid returning duplicates
        if func in funcs:
            continue
        else:
            funcs.append(func)
        args = tuple(getfullargspec(func).args)
        try:
            assumptions = tuple(func.assumptions)
        except AttributeError:
            raise NotImplementedError('function {0} in equations module has no'
                                      ' assumption '
                                      'definition'.format(func.__name__))
        try:
            overridden_by_assumptions = func.overridden_by_assumptions
        except AttributeError:
            overridden_by_assumptions = ()
        methods.append({
            'func': func,
            'args': args,
            'output': output,
            'assumptions': assumptions,
            'overridden_by_assumptions': overridden_by_assumptions,
        })
    return methods


def _fill_doc(s, module, default_assumptions):
        assumptions = module.assumptions
        s = s.replace(
            '<assumptions list goes here>',
            '\n'.join(sorted(
                ["* **{0}** -- {1}".format(a, desc) for a, desc in
                 assumptions.items()],
                key=lambda x: x.lower())))
        s = s.replace(
            '<default assumptions list goes here>',
            '\n'.join(
                wrap('Default assumptions are ' +
                     ', '.join(["'{0}'".format(a) for a in
                                default_assumptions]) + '.', width=80)))
        s = s.replace(
            '<quantity parameter list goes here>',
            '\n'.join(sorted([
                '* **{0}** -- {1} ({2})'.format(
                    q, info['name'], info['units'])
                for q, info in
                module.quantities.items()
                ], key=lambda x: x.lower())))
        return s


def _check_scalar(value):
    '''If value is a 0-dimensional array, returns the contents of value.
       Otherwise, returns value.
    '''
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            # We have a 0-dimensional array
            return value[None][0]
    return value


# We need to define a MetaClass in order to have dynamic docstrings for our
# Solver objects, generated based on the equations module
class SolverMeta(type):
    '''
Metaclass for BaseSolver to automatically generate docstrings and assumption
lists for subclasses of BaseSolver.
    '''

    def __new__(cls, name, parents, dct):
        if dct['_equation_module'] is not None:
            # Update the class docstring
            if '__doc__' in dct.keys():
                dct['__doc__'] = _fill_doc(
                    dct['__doc__'], dct['_equation_module'],
                    dct['default_assumptions'])

            assumptions = set([])
            for f in inspect.getmembers(equations):
                try:
                    assumptions.update(f[1].assumptions)
                except AttributeError:
                    pass
            dct['all_assumptions'] = tuple(assumptions)

        # we need to call type.__new__ to complete the initialization
        instance = super(SolverMeta, cls).__new__(cls, name, parents, dct)
        return instance


@add_metaclass(SolverMeta)
class BaseSolver(object):
    '''
Base class for solving systems of equations. Should not be instantiated,
as it is not associated with any equations.

Initializes with the given assumptions enabled, and variables passed as
keyword arguments stored.

Parameters
----------

assumptions : tuple, optional
    Strings specifying which assumptions to enable. Overrides the default
    assumptions. See below for a list of default assumptions.
add_assumptions : tuple, optional
    Strings specifying assumptions to use in addition to the default
    assumptions. May not be given in combination with the assumptions kwarg.
remove_assumptions : tuple, optional
    Strings specifying assumptions not to use from the default assumptions.
    May not be given in combination with the assumptions kwarg. May not
    contain strings that are contained in add_assumptions, if given.
**kwargs : ndarray, optional
    Keyword arguments used to pass in arrays of data that correspond to
    quantities used for calculations, or unit specifications for quantities.
    For a complete list of kwargs that may be used, see the Quantity Parameters
    section below.

Returns
-------
out : BaseSolver
    A BaseSolver object with the specified assumptions and variables.

Notes
-----

**Quantity kwargs**

<quantity parameter list goes here>

In addition to the quantities above, kwargs of the form <quantity>_unit or
<quantity>_units can be used with a string specifying a unit for the quantity.
This will cause input data for that quantity to be assumed to be in that
unit, and output data for that quantity to be given in that unit. Note this
must be specified separately for *each* quantity. Acceptable units are the
units available in the Pint package, with the exception that RH can be in
units of "fraction" or "percent".

**Assumptions**

<default assumptions list goes here>

**Assumption descriptions**

<assumptions list goes here>
    '''

    _equation_module = None
    _solutions = None

    def __init__(self, **kwargs):
        if self._equation_module is None:
            raise NotImplementedError('Class needs _equation_module '
                                      'defined')
        if 'debug' in kwargs.keys():
            self._debug = kwargs.pop('debug')
        else:
            self._debug = False
        # get reference units from equation module
        self._ref_units = {}
        # this could be made a class variable for optimization
        for quantity in self._equation_module.quantities.keys():
            self._ref_units[quantity] = \
                _ureg(self._equation_module.quantities[
                      quantity]['units'])
        # make sure add and remove assumptions are tuples, not strings
        if ('add_assumptions' in kwargs.keys() and
                isinstance(kwargs['add_assumptions'], string_types)):
            kwargs['add_assumptions'] = (kwargs['add_assumptions'],)
        if ('remove_assumptions' in kwargs.keys() and
                isinstance(kwargs['remove_assumptions'], string_types)):
            kwargs['remove_assumptions'] = (kwargs['remove_assumptions'],)
        # See if an assumption set was given
        if 'assumptions' in kwargs.keys():
            # If it was, make sure it wasn't given with other ways of
            # setting assumptions (by modifying the default assumptions)
            if ('add_assumptions' in kwargs.keys() or
                    'remove_assumptions' in kwargs.keys()):
                raise ValueError('cannot give kwarg assumptions with '
                                 'add_assumptions or remove_assumptions')
            assumptions = kwargs.pop('assumptions')
        else:
            # if it wasn't, modify the default assumptions
            assumptions = self.default_assumptions
            if 'add_assumptions' in kwargs.keys():
                if 'remove_assumptions' in kwargs.keys():
                    # make sure there is no overlap
                    if any([a in kwargs['remove_assumptions']
                            for a in kwargs['add_assumptions']]):
                        raise ValueError('assumption may not be present in '
                                         'both add_assumptions and '
                                         'remove_assumptions')
                # add assumptions, avoiding duplicates
                assumptions = assumptions + tuple(
                    [a for a in kwargs.pop('add_assumptions') if a not in
                     assumptions])
            if 'remove_assumptions' in kwargs.keys():
                # remove assumptions if present
                remove_assumptions = kwargs.pop('remove_assumptions')
                self._ensure_assumptions(*assumptions)
                assumptions = tuple([a for a in assumptions if a not in
                                     remove_assumptions])
        # Make sure all set assumptions are valid (not misspelt, for instance)
        self._ensure_assumptions(*assumptions)
        # now that we have our assumptions, use them to set the methods
        self.methods = self._get_methods(assumptions)
        self.assumptions = assumptions
        # take out any unit designations
        self.units = {}
        remove_kwargs = []
        for kwarg in kwargs:
            m = _unit_kwarg_prog.match(kwarg)
            if m is not None:
                # select whichever group is not None
                var = m.group(1) or m.group(2)
                self._ensure_quantities(var)
                if var in self.units:
                    raise ValueError(
                        'units for {} specified multiple times'.format(var))
                unit_str = kwargs[kwarg]
                remove_kwargs.append(kwarg)
                if not isinstance(unit_str, string_types):
                    raise TypeError('units must be strings')
                units = _ureg(unit_str)
                self.units[var] = units
        for kwarg in remove_kwargs:
            kwargs.pop(kwarg)
        # make sure the remaining variables are quantities
        self._ensure_quantities(*kwargs.keys())
        # convert quantities to reference units
        for kwarg in kwargs:
            if (kwarg in self.units and
                    self.units[kwarg] != self._ref_units[kwarg]):
                # special unit defined
                # convert to reference unit for calculations
                kwargs[kwarg] = _ureg.Quantity(kwargs[kwarg],
                                               self.units[kwarg]).to(
                    self._ref_units[kwarg]).magnitude
        # also store the quantities
        self.vars = kwargs

    def _ensure_assumptions(self, *args):
        '''Raises ValueError if any of the args are not strings corresponding
           to short forms of assumptions for this Solver.
        '''
        for arg in args:
            if arg not in self.all_assumptions:
                raise ValueError('{0} does not correspond to a valid '
                                 'assumption.'.format(arg))

    def _ensure_quantities(self, *args):
        '''Raises ValueError if any of the args are not strings corresponding
           to quantity abbreviations for this Solver.
        '''
        for arg in args:
            if arg not in self._equation_module.quantities.keys():
                raise ValueError('{0} does not correspond to a valid '
                                 'quantity.'.format(arg))

    def calculate(self, *args):
        '''
Calculates and returns a requested quantity from quantities stored in this
object at initialization.

Parameters
----------
*args : string
    Name of quantity to be calculated.

Returns
-------
quantity : ndarray
    Calculated quantity, in units listed under quantity parameters.

Notes
-----
See the documentation for this object for a complete list of quantities
that may be calculated, in the "Quantity Parameters" section.

Raises
------
ValueError:
    If the output quantity cannot be determined from the input
    quantities.

Examples
--------

Calculating pressure from virtual temperature and density:

>>> solver = FluidSolver(Tv=273., rho=1.27)
>>> solver.calculate('p')
99519.638400000011

Same calculation, but also returning a list of functions used:

>>> solver = FluidSolver(Tv=273., rho=1.27, debug=True)
>>> p, funcs = solver.calculate('p')
>>> funcs
(<function atmos.equations.p_from_rho_Tv_ideal_gas>,)

Same calculation with temperature instead, ignoring virtual temperature
correction:

>>> solver = FluidSolver(T=273., rho=1.27, add_assumptions=('Tv equals T',))
>>> solver.calculate('p',)
99519.638400000011
        '''
        self._ensure_quantities(*args)
        possible_quantities = get_calculatable_quantities(self.vars.keys(),
                                                          self.methods)
        for arg in args:
            if arg not in possible_quantities:
                raise ValueError('cannot calculate {0} from inputs'.format(
                    arg))
        # prepare a signature for this solution request
        sig = (args, tuple(self.vars.keys()), tuple(self.assumptions))
        # check if we've already tried to solve this
        if self.__class__._solutions is not None:
            if sig in self.__class__._solutions:
                # if we have, use the cached solution
                funcs, func_args, extra_values = self.__class__._solutions[sig]
            else:
                # solve and cache the solution
                funcs, func_args, extra_values = \
                    _get_shortest_solution(args, tuple(self.vars.keys()), (),
                                           self.methods)
                self.__class__._solutions[sig] = (funcs, func_args,
                                                  extra_values)
        else:  # no solution caching for this class
            funcs, func_args, extra_values = \
                _get_shortest_solution(args, tuple(self.vars.keys()), (),
                                       self.methods)
        # Above method completed successfully if no ValueError has been raised
        # Calculate each quantity we need to calculate in order
        for i, func in enumerate(funcs):
            # Compute this quantity
            value = func(*[self.vars[varname] for varname in func_args[i]])
            # Add it to our dictionary of quantities for successive functions
            self.vars[extra_values[i]] = value
        return_list = []
        # do corrections for non-standard units
        for arg in args:
            if arg in self.units and self.units[arg] != self._ref_units[arg]:
                return_list.append((self._ref_units[arg] * self.vars[arg]).to(
                    self.units[arg]).magnitude)
            else:
                return_list.append(self.vars[arg])
        if self._debug:
            # We should return a list of funcs as the last item returned
            if len(return_list) == 1:
                return _check_scalar(return_list[0]), funcs
            else:
                return ([_check_scalar(val) for val in return_list] +
                        [funcs, ])
        else:
            # no function debugging, just return the quantities
            if len(args) == 1:
                return _check_scalar(return_list[0])
            else:
                return [_check_scalar(val) for val in return_list]

    def _get_methods(self, assumptions):
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
        # make sure all assumptions actually apply to equations
        # this will warn the user of typos
        for a in assumptions:
            if a not in self.all_assumptions:
                raise ValueError('assumption {0} matches no '
                                 'equations'.format(a))
        # create a dictionary to which we will add methods
        methods = {}
        # get a set of all the methods in the module
        module_methods = _get_module_methods(self._equation_module)
        # Go through each output variable
        for dct in module_methods:
            # Make sure this method is not overridden
            if any(item in assumptions for item in
                    dct['overridden_by_assumptions']):
                continue
            # Make sure all assumptions of the method are satisfied
            elif all(item in assumptions for item in dct['assumptions']):
                # Make sure we have a dict entry for this output quantity
                if dct['output'] not in methods.keys():
                    methods[dct['output']] = {}
                # Make sure we aren't defining methods with same signature
                if dct['args'] in methods[dct['output']].keys():
                    raise ValueError(
                        'assumptions given define duplicate '
                        'equations {0} and {1}'.format(
                            str(dct['func']),
                            str(methods[dct['output']][dct['args']])))
                # Add the method to the methods dict
                methods[dct['output']][dct['args']] = dct['func']
        return methods


class FluidSolver(BaseSolver):
    '''
Initializes with the given assumptions enabled, and variables passed as
keyword arguments stored.

Parameters
----------

assumptions : tuple, optional
    Strings specifying which assumptions to enable. Overrides the default
    assumptions. See below for a list of default assumptions.
add_assumptions : tuple, optional
    Strings specifying assumptions to use in addition to the default
    assumptions. May not be given in combination with the assumptions kwarg.
remove_assumptions : tuple, optional
    Strings specifying assumptions not to use from the default assumptions.
    May not be given in combination with the assumptions kwarg. May not
    contain strings that are contained in add_assumptions, if given.
**kwargs : ndarray, optional
    Keyword arguments used to pass in arrays of data that correspond to
    quantities used for calculations, or unit specifications for quantities.
    For a complete list of kwargs that may be used, see the Quantity Parameters
    section below.

Returns
-------
out : FluidSolver
    A FluidSolver object with the specified assumptions and variables.

Notes
-----

**Quantity kwargs**

<quantity parameter list goes here>

In addition to the quantities above, kwargs of the form <quantity>_unit or
<quantity>_units can be used with a string specifying a unit for the quantity.
This will cause input data for that quantity to be assumed to be in that
unit, and output data for that quantity to be given in that unit. Note this
must be specified separately for *each* quantity. Acceptable units are the
units available in the Pint package, with the exception that RH can be in
units of "fraction" or "percent".

**Assumptions**

<default assumptions list goes here>

**Assumption descriptions**

<assumptions list goes here>

Examples
--------

Calculating pressure from virtual temperature and density:

>>> solver = FluidSolver(Tv=273., rho=1.27)
>>> solver.calculate('p')
99519.638400000011

Same calculation, but also returning a list of functions used:

>>> solver = FluidSolver(Tv=273., rho=1.27, debug=True)
>>> p, funcs = solver.calculate('p')
>>> funcs
(<function atmos.equations.p_from_rho_Tv_ideal_gas>,)

Same calculation with temperature instead, ignoring virtual temperature
correction:

>>> solver = FluidSolver(T=273., rho=1.27, add_assumptions=('Tv equals T',))
>>> solver.calculate('p',)
99519.638400000011
    '''

    # module containing fluid dynamics equations
    _equation_module = equations
    # which assumptions to use by default
    default_assumptions = (
        'ideal gas', 'hydrostatic', 'constant g', 'constant Lv', 'constant Cp',
        'no liquid water', 'no ice', 'bolton', 'cimo')
    _solutions = {}


def calculate(*args, **kwargs):
    '''
Calculates and returns a requested quantity from quantities passed in as
keyword arguments.

Parameters
----------

\*args : string
    Names of quantities to be calculated.
assumptions : tuple, optional
    Strings specifying which assumptions to enable. Overrides the default
    assumptions. See below for a list of default assumptions.
add_assumptions : tuple, optional
    Strings specifying assumptions to use in addition to the default
    assumptions. May not be given in combination with the assumptions kwarg.
remove_assumptions : tuple, optional
    Strings specifying assumptions not to use from the default assumptions.
    May not be given in combination with the assumptions kwarg. May not
    contain strings that are contained in add_assumptions, if given.
\*\*kwargs : ndarray, optional
    Keyword arguments used to pass in arrays of data that correspond to
    quantities used for calculations, or unit specifications for quantities.
    For a complete list of kwargs that may be used, see the Quantity Parameters
    section below.

Returns
-------

quantity : ndarray
    Calculated quantity.
    Return type is the same as quantity parameter types.
    If multiple quantities are requested, returns a tuple containing the
    quantities.

Notes
-----

Calculating multiple quantities at once can avoid re-computing intermediate
quantities, but requires more memory.

**Quantity kwargs**

<quantity parameter list goes here>

In addition to the quantities above, kwargs of the form <quantity>_unit or
<quantity>_units can be used with a string specifying a unit for the quantity.
This will cause input data for that quantity to be assumed to be in that
unit, and output data for that quantity to be given in that unit. Note this
must be specified separately for *each* quantity. Acceptable units are the
units available in the Pint package, with the exception that RH can be in
units of "fraction" or "percent".

**Assumptions**

<default assumptions list goes here>

**Assumption descriptions**

<assumptions list goes here>

Examples
--------

Calculating pressure from virtual temperature and density:

>>> calculate('p', Tv=273., rho=1.27)
99519.638400000011

Same calculation, but also returning a list of functions used:

>>> p, funcs = calculate('p', Tv=273., rho=1.27, debug=True)
>>> funcs
(<function atmos.equations.p_from_rho_Tv_ideal_gas>,)

Same calculation with temperature instead, ignoring virtual temperature
correction:

>>> calculate('p', T=273., rho=1.27, add_assumptions=('Tv equals T',))
99519.638400000011
'''
    if len(args) == 0:
        raise ValueError('must specify quantities to calculate')
    # initialize a solver to do the work
    solver = FluidSolver(**kwargs)
    # get the output
    return solver.calculate(*args)

# autocomplete some sections of the docstring for calculate
calculate.__doc__ = _fill_doc(calculate.__doc__, equations,
                              FluidSolver.default_assumptions)
