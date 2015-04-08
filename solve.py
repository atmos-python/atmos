# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:31:40 2015

@author: mcgibbon
"""
import inspect
import equations

default_assumptions = ('ideal gas', 'hydrostatic', 'constant g', 'constant Lv',
                       'constant Cp', 'no liquid water', 'no solid water',
                       'bolton',)
all_assumptions = tuple(set([]).union(*[f[1].func_dict['assumptions']
                                        for f in inspect.getmembers(equations)
                                        if hasattr(f[1], 'func_dict') and
                                        'assumptions' in
                                        f[1].func_dict.keys()]))
# all_assumptions = ('ideal gas', 'hydrostatic', 'constant g', 'constant Lv',
#                'bolton', 'goff-gratch', 'frozen bulb', 'unfrozen bulb',
#                'stipanuk', 'dry', 'Tv equals T')


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
            raise NotImplementedError('function {} in equations module has no '
                                      'assumption '
                                      'definition'.format(funcs[i].__name__))
        methods[outputs[i]].append((args, assumptions, funcs[i]))
    return methods


class _BaseSolver(object):
    '''
    '''

    _methods = {}

    def __new__(cls, *args, **kwargs):
        if cls is _BaseSolver:
            raise TypeError('BaseDeriver may not be instantiated. Use a '
                            'subclass.')
        return object.__new__(cls, *args, **kwargs)

    def __init__(self, assumptions=(), derivative=None,
                 axis_coords=None, override_coord_axes=None,
                 coords_own_axis=None, **kwargs):
        '''
        Initializes with the given assumptions enabled, and variables passed as
        keyword arguments stored.

        Parameters
        ----------
        assumptions : tuple, optional
            Strings specifying which assumptions to enable.
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
        if 'debug' in kwargs.keys():
            self._debug = kwargs['debug']
        else:
            self._debug = False
        if axis_coords is not None and any([coord not in
                                            self.coord_types.keys()
                                            for coord in axis_coords]):
            raise ValueError('Invalid value given in axis_coords')
        self.methods = self._get_methods(assumptions)
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
            if m not in all_assumptions:
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
        assumptions = kwargs.pop('assumptions')
    except KeyError:
        assumptions = default_assumptions
    derivative = get_arg('derivative', kwargs)
    axis_coords = get_arg('axis_coords', kwargs)
    override_coord_axes = get_arg('override_coord_axes', kwargs)
    coords_own_axis = get_arg('coords_own_axis', kwargs)
    solver = FluidSolver(assumptions, derivative,
                         axis_coords, override_coord_axes,
                         coords_own_axis, **kwargs)
    result = [solver.calculate(var) for var in args]
    if len(result) == 1:
        return result[0]
    else:
        return result
