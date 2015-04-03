# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:31:40 2015

@author: mcgibbon
"""
from util import ddx
import re
import inspect
import equations

default_methods = ('ideal gas', 'hydrostatic', 'constant g', 'constant Lv')
all_methods = tuple(set([]).union(*[f[1].func_dict['assumptions']
                                    for f in inspect.getmembers(equations)
                                    if hasattr(f[1], 'func_dict') and
                                    'assumptions' in
                                    f[1].func_dict.keys()]))
# all_methods = ('ideal gas', 'hydrostatic', 'constant g', 'constant Lv',
#                'bolton', 'goff-gratch', 'frozen bulb', 'unfrozen bulb',
#                'stipanuk', 'dry', 'Tv equals T')


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
    result = _get_shortest_solution(outputs, inputs + (next_variables[0],),
                                    exclude, methods)
    args, func = relevant_methods[next_variables[0]].items()[0]
    best_option = ((func,) + result[0], (args,) + result[1],
                   (next_variables[0],) + result[2])
    exclude = exclude + (next_variables[0],)
    for intermediate in next_variables[1:]:
        if intermediate in exclude:
            continue
        result = _get_shortest_solution(outputs, inputs + (intermediate,),
                                        exclude, methods)
        args, func = relevant_methods[intermediate].items()[0]
        next_option = ((func,) + result[0], (args,) + result[1],
                       (next_variables[0],) + result[2])
        # Compare number of calculations
        if len(next_option[0]) < len(best_option[0]):
            # update the new best option
            best_option = next_option
        elif (len(next_option[0]) == len(best_option[0]) and
              sum([len(tup) for tup in next_option[1]]) <
              sum([len(tup) for tup in next_option[1]])):
                # update the new best option
                best_option = next_option
        exclude = exclude + (intermediate,)
    return best_option


def _get_methods(module):
    '''
    Returns a methods dictionary corresponding to the equations in the given
    module.
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
            raise NotImplementedError('function {} in equations module as no '
                                      'assumption '
                                      'definition'.format(funcs[i].__name__))
        methods[outputs[i]].append((args, assumptions, funcs[i]))
    return methods


def _get_best_result(results):
    '''
    Takes in an iterable of (funcs, func_args, outputs) tuples, and returns the
    one that corresponds to the "best" solution. This means the fewest
    function calls, with ties broken by the fewest total function arguments.

    Note the definition of "best" solution is subject to change.
    '''
    if len(results) == 0:
        raise ValueError('results must be nonempty')
    min_length = min([len(r[0]) for r in results])
    first_pass = [r for r in results if len(r[0]) == min_length]
    if len(first_pass) == 1:
        return first_pass[0]
    else:
        min_args = min([sum([len(args) for args in r[1]]) for r in results])
        second_pass = [r for r in results if
                       sum([len(args) for args in r[1]]) == min_args]
        return second_pass[0]


class _BaseSolver(object):
    '''
    '''

    _methods = {}
    derivative_prog = re.compile(r'd(.+)d(p|x|y|theta|z|sigma|t|lat|lon)')
    coord_types = {
        'x': 'x',
        'lon': 'x',
        'y': 'y',
        'lat': 'y',
        'theta': 'z',
        'z': 'z',
        'p': 'z',
        'sigma': 'z',
        't': 't',
    }

    def __new__(cls, *args, **kwargs):
        if cls is _BaseSolver:
            raise TypeError('BaseDeriver may not be instantiated. Use a '
                            'subclass.')
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, methods=(), derivative=None,
                 axis_coords=None, override_coord_axes=None,
                 coords_own_axis=None, **kwargs):
        '''
        Initializes with the given methods enabled, and variables passed as
        keyword arguments stored.

        Parameters
        ----------
        methods : tuple, optional
            Strings specifying which methods to enable.
        derivative : str, optional
            Which spatial derivative calculation to use. Set to 'centered' for
            second-order centered finite difference with first-order
            forward and backward differencing at boundaries, or None to disable
            derivatives.
        axis_coords : iterable, optional
            Defines the default coordinate to assume for each axis. Should
            contain 't' to denote a time-like coordinate, 'z' to denote a
            vertical-like coordinate, 'y' to denote a meridional-like
            component, and 'x' to denote a zonal-like component.
            Only required for calculations that require this knowledge.
        override_coord_axes : mapping, optional
            A mapping of quantity strings to tuples defining the axes of those
            quantities, overriding the default of axis_coords.
        coords_own_axis : mapping, optional
            A mapping of quantity strings for coordinate-like quantities to
            their index which corresponds to their own axis. For example, if
            lon is given as an array in [lat, lon], its value would be 1,
            whereas if it is a 1-D array its value would be 0. This is assumed
            to be 0 by default.

        Notes
        -----
        y and x defined in axis_coords need not be meridional and longitudinal,
        so long as it is understood that any zonal or meridional quantities
        calculated by this object (such as u and v) were done so under this
        assumption.
        '''
        if axis_coords is not None and any([coord not in
                                            self.coord_types.keys()
                                            for coord in axis_coords]):
            raise ValueError('Invalid value given in axis_coords')
        self.methods = self._get_methods(methods)
        self.vars = kwargs
        if axis_coords is not None:
            coord_axes = {}
            for i in range(len(axis_coords)):
                coord_axes[self.coord_types[axis_coords[i]]] = i
            self.coord_axes = coord_axes
        else:
            self.coord_axes = {}
        if override_coord_axes is not None:
            self.override_coord_axes = override_coord_axes
        else:
            self.override_coords = {}
        self.coords_own_axis = {'t': 0, 'z': 0, 'y': 0, 'x': 0}
        if coords_own_axis is not None:
            self.coords_own_axis.update(coords_own_axis)
        if derivative not in ('centered', None):
            raise ValueError('invalid value given for derivative')
        self._derivative = derivative

    def calculate(self, *args):
        '''
        Calculates and returns a requested quantity from quantities passed in
        as keyword arguments.

        Parameters
        ----------
        varname_out : string
            Name of quantity to be calculated.
        methods : tuple, optional
            Names of methods that can be used for calculation, as strings.

        Quantity Parameters
        -------------------
        All quantity parameters are optional, and must be of the same type,
        either ndarray or iris Cube. If ndarrays are used, then units must
        match the units specified below. If iris Cube is used, any units may
        be used for input and the output will be given in the units specified
        below.


        Returns
        -------
        quantity : ndarray or iris Cube
            Calculated quantity.
            Return type is the same as quantity parameter types.

        Raises
        ------
        ValueError:
            If the output quantity cannot be determined from the input
            quantities.

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>>
        '''
        funcs, func_args, extra_values = \
            _get_shortest_solution(tuple(args), tuple(self.vars.keys()), (),
                                   self.methods)
        print(funcs, func_args, extra_values)
        # Above method completed successfully if no ValueError has been raised
        # Calculate each quantity we need to calculate in order
        print(funcs, func_args, extra_values)
        for i, func in enumerate(funcs):
            # Compute this quantity
            value = func(*[self.vars[varname] for varname in func_args[i]])
            # Add it to our dictionary of quantities for successive functions
            self.vars[extra_values[i]] = value
        if len(args) == 1:
            return self.vars[args[0]]
        else:
            return [self.vars[arg] for arg in args]

    def _calculate(self, out_name, methods, exclude, funcs, func_args,
                   in_values, extra_values):
        '''
        Tries to calculate out_name using the given methods, the variables in
        the dictionary in_values, and the calculated variables listed in
        extra_values, without using any of the variables listed in exclude.

        Returns
        -------
        Returns a tuple of functions, function arguments, and the variables
        they calculate, such that if you execute those functions in order and
        use their calculated variables in successive functions, the last one
        (if any) will successfully calculate the variable denoted by out_name.

        funcs: tuple
            The functions that need to be called in order to calculate the
            variable referred to by out_name
        func_args: tuple
            A tuple of tuples of strings denoting the variables that must be
            passed as arguments to each function in funcs
        extra_values: tuple
            A tuple of strings denoting the variables that are output by
            each function in funcs

        Raises
        ------
        ValueError:
            If the variable denoted by out_name cannot be calculated with the
            given methods and in_values
        '''
        # For debugging purposes, make sure nothing in extra_values is given
        # in in_values
        assert not any([value in in_values.keys() for value in extra_values])
        # Check if we already have the value desired
        if out_name in in_values.keys() or out_name in extra_values:
            return (), (), ()
        # Make sure we have a method for this quantity
        if out_name not in methods.keys():  # No method to get this quantity
            raise ValueError('Could not determine {} as no available method '
                             'determines it'.format(out_name))
        # Check whether we already have calculated everything to complete
        # one of the available methods
        results = []
        for args, func in methods[out_name].items():
            # See if all arguments are either provided or already calculated
            if all([(arg in in_values.keys() or arg in extra_values) for arg
                    in args]):
                # If this is the case, use this method
                results.append(((func,), (args,), (out_name,)))
        if len(results) > 0:
            result = _get_best_result(results)
            return (funcs + result[0], func_args + result[1],
                    extra_values + result[2])
        # See if we can calculate from a method by calculating the missing
        # arguments
        for args, func in methods[out_name].items():
            # make sure none of the method arguments are excluded (we've
            # already tried to get it and are nested deeper in our search)
            if any([arg in exclude for arg in args]):
                continue  # Go to the next method
            temp_funcs, temp_func_args = funcs, func_args
            temp_extra_values = extra_values
            for arg in args:  # See if we have all our arguments
                # See if this argument was input or we've already determined
                # it can be derived
                if arg in in_values.keys() or arg in temp_extra_values:
                    # We already have this, check the next one
                    continue
                # See if this is a derivative and we can get it
                try:
                    result = self._get_derivative(arg, methods, exclude +
                                                  (out_name,), temp_funcs,
                                                  temp_func_args, in_values,
                                                  temp_extra_values)
                except ValueError:
                    # either it's not a derivative or we can't calculate it
                    pass
                # We'd need to calculate this argument. See if we can
                try:
                    result = self._calculate(arg, methods, exclude +
                                             (out_name,), temp_funcs,
                                             temp_func_args, in_values,
                                             temp_extra_values)
                except ValueError:
                    break  # We can't get this argument
                # We can get this argument, so add it to our list
                temp_funcs, temp_func_args, temp_extra_values = result
            else:  # We have all our arguments, since we didn't break
                # Add this possible solution to our list
                results.append((temp_funcs + (func,), temp_func_args + (args,),
                                temp_extra_values + (out_name,)))
        if len(results) == 0:
            # We could not calculate this with any method available
            raise ValueError('Could not determine {} from given variables'
                             ' {}'.format(out_name,
                                          ', '.join(tuple(in_values.keys()) +
                                                    extra_values)))
        else:
            result = _get_best_result(results)
            return (funcs + result[0], func_args + result[1],
                    extra_values + result[2])

    def _get_derivative(self, out_name, methods, exclude, funcs, func_args,
                        in_values, extra_values):
        '''
        Tries to determine how to calculate varname, assuming varname is of
        the form d(var)d(coord), where var is a quantity and coord is a
        coordinate. Raises ValueError if this cannot be done or if varname
        is of the wrong form.

        Returns
        -------
        Returns a tuple of functions, function arguments, and the variables
        they calculate, such that if you execute those functions in order and
        use their calculated variables in successive functions, the last one
        will successfully calculate the variable denoted by out_name.

        funcs: tuple
            The functions that need to be called in order to calculate the
            variable referred to by out_name
        func_args: tuple
            A tuple of tuples of strings denoting the variables that must be
            passed as arguments to each function in funcs
        extra_values: tuple
            A tuple of strings denoting the variables that are output by
            each function in funcs
        '''
        # Check if derivatives are disabled
        if self._derivative is None:
            return None
        match = self.derivative_prog.match(out_name)
        if match is None:
            raise ValueError('out_name is not in the form of a derivative')
        varname = match.group(1)
        coordname = match.group(2)
        temp_funcs, temp_func_args = funcs, func_args
        temp_extra_values = extra_values
        # Make sure we actually have our variable and coordinate
        for arg in (varname, coordname):
            if arg in exclude:
                raise ValueError('require an excluded variable to '
                                 'calculate the derivative')
            # See if this argument was input or we've already determined
            # it can be derived
            if arg in in_values.keys() or arg in temp_extra_values:
                # We already have this, check the next one
                continue
            try:
                # Check if this is a nested derivative
                result = self._get_derivative(arg, methods, exclude +
                                              (out_name,), temp_funcs,
                                              temp_func_args, in_values,
                                              temp_extra_values)
            except ValueError:
                pass
            # We'd need to calculate this argument. See if we can
            try:
                result = self._calculate(arg, methods, exclude +
                                         (out_name,), temp_funcs,
                                         temp_func_args, in_values,
                                         temp_extra_values)
            except ValueError:
                break  # We can't get this argument
            # We can get this argument, so add it to our list
            temp_funcs, temp_func_args, temp_extra_values = result
        else:  # We have all our arguments, since we didn't break
            # Now we have to construct a function...
            func, args = self._construct_derivative(varname, coordname)
            return (temp_funcs + (func,), temp_func_args + (args,),
                    temp_extra_values + (out_name,))
        raise ValueError('cannot calculate {} from available'
                         ' quantities'.format(out_name))

    def _construct_derivative(self, varname, coordname):
        '''
        Constructs a function that computes the derivative of the variable
        specified by varname with respect to the coordinate specified by
        coordname.

        Returns
        -------
        func: function
            Function which takes in the arguments returned by this function and
            returns the derivative of the variable specified by varname with
            respect to the coordinate specified by coordname
        args: iterable
            An iterable of strings corresponding to the variables that must be
            passed as arguments to the function returned by this function
        '''
        axis = self.coord_axes[self.coord_types[coordname]]
        if self._derivative == 'centered':
            func = lambda data, coord: \
                ddx(data, axis, x=coord,
                    axis_x=self.coords_own_axis[self.coord_types[coordname]])
        else:
            raise ValueError('invalid derivative type used to construct '
                             'Deriver')
        return (func, (varname, coordname))

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

        Raises
        ------
        ValueError
            If a method given is not present in self.optional_methods
            If multiple optional methods are selected which calculate the same
                output quantity from the same input quantities
        '''
        for m in method_options:
            if m not in all_methods:
                raise ValueError('method {} matches no equations'.format(m))
        methods = {}
        # Go through each output variable
        for output, L in self._methods.items():
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


class FluidSolver(_BaseSolver):

    _methods = _get_methods(equations)

    def calculate(self, quantity_out, **kwargs):
        '''
        Calculates and returns a requested quantity from quantities passed in
        as keyword arguments.

        Parameters
        ----------
        varname_out : string
            Name of quantity to be calculated.
        methods : tuple, optional
            Names of methods that can be used for calculation, as strings.

        Quantity Parameters
        -------------------
        All quantity parameters are optional, and must be of the same type,
        either ndarray or iris Cube. If ndarrays are used, then units must
        match the units specified below. If iris Cube is used, any units may
        be used for input and the output will be given in the units specified
        below.

        q: Specific humidity (kg/kg)
        T: Temperature (K)
        p: Pressure (Pa)
        RH: Relative humidity (%)
        Tw: Wet bulb temperature (K)
        Tv: Virtual temperature (K)
        rho: Air density (kg/m^3)

        Returns
        -------
        quantity : ndarray or iris Cube
            Calculated quantity.
            Return type is the same as quantity parameter types.

        Raises
        ------

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>>
        '''
        return super(FluidSolver, self).calculate(quantity_out, **kwargs)


def calculate(*args, **kwargs):
    '''
    Calculates and returns a requested quantity from quantities passed in as
    keyword arguments.

    Parameters
    ----------
    args : string
        Names of quantities to be calculated.
    methods : tuple, optional
        Names of methods that can be used for calculation, as strings.
    derivative : str, optional
        Which spatial derivative calculation to use. Set to 'centered' for
        second-order centered finite difference, or None to disable
        derivatives.
    periodic : bool, optional
        Whether the domain is periodic in space. Used when calculating
        spatial derivatives.
    axis_coords : iterable, optional
        Defines the default coordinate to assume for each axis. Should
        contain 't' to denote a time-like coordinate, 'z' to denote a
        vertical-like coordinate, 'y' to denote a meridional-like
        component, and 'x' to denote a zonal-like component.
        Only required for calculations that require this knowledge.
    override_coord_axes : mapping, optional
        A mapping of quantity strings to tuples defining the axes of those
        quantities, overriding the default of axis_coords.
    coords_own_axis : mapping, optional
        A mapping of quantity strings for coordinate-like quantities to
        their index which corresponds to their own axis. For example, if
        lon is given as an array in [lat, lon], its value would be 1,
        whereas if it is a 1-D array its value would be 0. This is assumed
        to be 0 by default.

    Quantity Parameters
    -------------------
    All quantity parameters are optional, and must be of the same type,
    either ndarray or iris Cube. If ndarrays are used, then units must match
    the units specified below. If iris Cube is used, any units may be used
    for input and the output will be given in the units specified below.

    T: Temperature (K)
    p: Pressure (Pa)
    q: Specific humidity (kg/kg)
    RH: Relative humidity (%)
    AH: Absolute humidity (kg/m^3)
    wvap: Water vapor mixing ratio (kg/kg)
    Tw: Wet bulb temperature (K)
    Tv: Virtual temperature (K)
    rho: Air density (kg/m^3)

    Returns
    -------
    quantity : ndarray or iris Cube
        Calculated quantity.
        Return type is the same as quantity parameter types.
        If multiple quantities are requested, returns a tuple containing the
        quantities.

    See Also
    --------

    Notes
    -----
    y and x defined in axis_coords need not be meridional and longitudinal,
    so long as it is understood that any zonal or meridional quantities
    calculated by this object (such as u and v) were done so under this
    assumption.

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
    try:
        methods = kwargs.pop('methods')
    except KeyError:
        methods = default_methods
    derivative = get_arg('derivative', kwargs)
    axis_coords = get_arg('axis_coords', kwargs)
    override_coord_axes = get_arg('override_coord_axes', kwargs)
    coords_own_axis = get_arg('coords_own_axis', kwargs)
    solver = FluidSolver(methods, derivative,
                         axis_coords, override_coord_axes,
                         coords_own_axis, **kwargs)
    result = [solver.calculate(var) for var in args]
    if len(result) == 1:
        return result[0]
    else:
        return result
