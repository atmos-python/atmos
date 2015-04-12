# -*- coding: utf-8 -*-
"""
solve.py: Utilities that use equations to solve for quantities, given other
    quantities and a set of assumptions.
"""
import inspect
import equations
from six import add_metaclass
from textwrap import wrap


class ExcludeError(Exception):
    pass


def _get_first_pass_methods(inputs, outputs, methods):
    '''
    Given iterables of input variable names, output variable names,
    and a methods dictionary, returns the subset of the methods dictionary
    that can be calculated, doesn't calculate something we already have,
    and only contains equations that might help calculate the outputs from
    the inputs.
    '''
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
            # set a dynamic list of the assumptions in the equations module
            dct['all_assumptions'] = tuple(
                set([]).union(*[f[1].func_dict['assumptions']
                                for f in inspect.getmembers(equations)
                                if hasattr(f[1], 'func_dict') and
                                'assumptions' in
                                f[1].func_dict.keys()]))

        # we need to call type.__new__ to complete the initialization
        instance = super(SolverMeta, cls).__new__(cls, name, parents, dct)
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
            self._debug = kwargs.pop('debug')
        else:
            self._debug = False
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
        # make sure the remaining variables are quantities
        self._ensure_quantities(*kwargs.keys())
        # also store the quantities
        self.vars = kwargs

    def _ensure_assumptions(self, *args):
        '''Raises ValueError if any of the args are not strings corresponding
           to short forms of assumptions for this Solver.
        '''
        for arg in args:
            if arg not in self.all_assumptions:
                raise ValueError('{} does not correspond to a valid '
                                 'assumption.'.format(arg))

    def _ensure_quantities(self, *args):
        '''Raises ValueError if any of the args are not strings corresponding
           to quantity abbreviations for this Solver.
        '''
        for arg in args:
            if arg not in self._equation_module.quantities.keys():
                raise ValueError('{} does not correspond to a valid '
                                 'quantity.'.format(arg))

    def calculate(self, *args):
        '''
Calculates and returns a requested quantity from quantities stored in this
object at initialization.

Parameters
----------
varname_out : string
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
        '''
        self._ensure_quantities(*args)
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
            # We should return a list of funcs as the last item returned
            if len(args) == 1:
                return self.vars[args[0]], funcs
            else:
                return [self.vars[arg] for arg in args] + [funcs, ]
        else:
            # no function debugging, just return the quantities
            if len(args) == 1:
                return self.vars[args[0]]
            else:
                return [self.vars[arg] for arg in args]

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
                raise ValueError('assumption {} matches no '
                                 'equations'.format(a))
        # create a dictionary to which we will add methods
        methods = {}
        # get a set of all the methods in the module
        module_methods = _get_module_methods(self._equation_module)
        # Go through each output variable
        for output, L in module_methods.items():
            # Go through each potential equation
            for args, func_assumptions, func in L:
                # See if we're using the equation's assumptions
                if all(item in assumptions for item in func_assumptions):
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
    Strings specifying which assumptions to enable. Overrides the default
    assumptions. See below for a list of default assumptions.
add_assumptions : tuple, optional
    Strings specifying assumptions to use in addition to the default
    assumptions. May not be given in combination with the assumptions kwarg.
remove_assumptions : tuple, optional
    Strings specifying assumptions not to use from the default assumptions.
    May not be given in combination with the assumptions kwarg. May not
    contain strings that are contained in add_assumptions, if given.
quantity : ndarray, optional
    Keyword arguments used to pass in arrays of data that correspond to
    quantities used for calculations. For a complete list of kwargs that
    may be used, see the Quantity Parameters section below.

Quantity Parameters
-------------------
<quantity parameter list goes here>

Returns
-------
out : FluidSolver
    A FluidSolver object with the specified assumptions and variables.

Assumptions
-----------
<default assumptions list goes here>

Assumption descriptions:
<assumptions list goes here>

Examples
--------
>>> solver = FluidSolver(rho=array1, p=array2)

Non-default assumptions:

>>> solver = FluidSolver(add_assumptions=('Tv equals T'), rho=array1, p=array2)
    '''

    # module containing fluid dynamics equations
    _equation_module = equations
    # which assumptions to use by default
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
    Strings specifying which assumptions to enable. Overrides the default
    assumptions. See below for a list of default assumptions.
add_assumptions : tuple, optional
    Strings specifying assumptions to use in addition to the default
    assumptions. May not be given in combination with the assumptions kwarg.
remove_assumptions : tuple, optional
    Strings specifying assumptions not to use from the default assumptions.
    May not be given in combination with the assumptions kwarg. May not
    contain strings that are contained in add_assumptions, if given.
quantity : ndarray, optional
    Keyword arguments used to pass in arrays of data that correspond to
    quantities used for calculations. For a complete list of kwargs that
    may be used, see the Quantity Parameters section below.

Assumptions
-----------
<default assumptions list goes here>

Assumption descriptions:
<assumptions list goes here>

Quantity Parameters
-------------------
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
    # initialize a solver to do the work
    solver = FluidSolver(**kwargs)
    # get the output
    result = [solver.calculate(var) for var in args]
    if len(result) == 1:
        # return a single value if there is only one value
        return result[0]
    else:
        # otherwise return a list of values
        return result

# autocomplete some sections of the docstring for calculate
calculate.__doc__ = _fill_doc(calculate.__doc__, equations,
                              FluidSolver.default_assumptions)
