# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:31:40 2015

@author: mcgibbon
"""
from util import ddx
import re


class _BaseSolver(object):
    '''
    '''

    default_methods = {}
    optional_methods = {}
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
        if any([coord not in self.coord_types.keys()
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

    def calculate(self, quantity_out):
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
            self._calculate(quantity_out, self.methods,
                            (), (), (), self.vars, ())
        # Above method completed successfully if no ValueError has been raised
        # Calculate each quantity we need to calculate in order
        for i, func in enumerate(funcs):
            # Compute this quantity
            value = func(*[self.vars[varname] for varname in func_args[i]])
            # Add it to our dictionary of quantities for successive functions
            self.vars[extra_values[i]] = value
        # The last quantity calculated will be the one for this function call
        assert extra_values[-1] == quantity_out
        return self.vars[quantity_out]

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
        for args, func in methods[out_name].items():
            # See if all arguments are either provided or already calculated
            if all([(arg in in_values.keys() or arg in extra_values) for arg
                    in args]):
                # If this is the case, use this method
                return (funcs + (func,), func_args + (args,),
                        extra_values + (out_name,))
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
                try:
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
                return (temp_funcs + (func,), temp_func_args + (args,),
                        temp_extra_values + (out_name,))
        # We could not calculate this with any method available
        raise ValueError('Could not determine {} from given variables'
                         ' {}'.format(out_name,
                                      ', '.join(tuple(in_values.keys()) +
                                                extra_values)))

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
        if self.derivative is None:
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
        methods = {}
        for methodname in method_options:
            # Check that this is actually a valid method
            if methodname not in self.optional_methods.keys():
                raise ValueError('method {} is not defined'.format(methodname))
            # Iterate through each quantity this method lets you calculate
            for varname in self.optional_methods[methodname].keys():
                # Make sure we have a dictionary defined for this quantity
                if varname not in methods.keys():
                    methods[varname] = {}
                equations = self.optional_methods[methodname][varname]
                # Make sure another method is not already selected that does
                # this same calculation
                for vars_input in equations.keys():
                    if vars_input in methods[varname].keys():
                        raise ValueError(('Multiple methods for {} --> {} chos'
                                          'en').format(', '.join(vars_input),
                                                       varname))
                methods[varname].update(equations)
        output_methods = self.default_methods.copy()
        output_methods.update(methods)
        return output_methods


class FluidSolver(_BaseSolver):

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


def calculate(output, methods=(), derivative=None,
              axis_coords=None, override_coord_axes=None,
              coords_own_axis=None, **kwargs):
    '''
    Calculates and returns a requested quantity from quantities passed in as
    keyword arguments.

    Parameters
    ----------
    output : string or iterable
        Name of quantity to be calculated. If iterable, should contain strings
        which are names of quantities to be calculated.
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
    deriver = FluidSolver(methods, derivative,
                          axis_coords, override_coord_axes,
                          coords_own_axis, **kwargs)
    try:
        result = [deriver.calculate(var) for var in output]
    except TypeError:  # raised if output is not iterable
        result = deriver.calculate(output)
    if len(result) == 1:
        return result[0]
    else:
        return result
