# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:11:26 2015

@author: mcgibbon
"""
from __future__ import division, absolute_import, unicode_literals
import numpy as np
import re
import six
from scipy.ndimage.filters import convolve1d

derivative_prog = re.compile(r'd(.+)d(p|x|y|theta|z|sigma|t|lat|lon)')
from textwrap import wrap


def sma(array, window_size, axis=-1, mode='reflect', **kwargs):
    """
Computes a 1D simple moving average along the given axis.

Parameters
----------
array : ndarray
    Array on which to perform the convolution.
window_size: int
    Width of the simple moving average window in indices.
axis : int, optional
    Axis along which to perform the moving average
mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
    The mode parameter determines how the array borders are handled, where
    cval is the value when mode is equal to ‘constant’. Default is ‘reflect’.
kwargs : optional
    Other arguments to pass to `scipy.ndimage.filters.convolve1d`

Returns
-------
sma : ndarray
    Simple moving average of the given array with the specified window size
    along the requested axis.

Raises
------
TypeError:
    If window_size or axis are not integers.
    """
    kwargs['axis'] = axis
    kwargs['mode'] = mode
    if not isinstance(window_size, int):
        raise TypeError('window_size must be an integer')
    if not isinstance(kwargs['axis'], int):
        raise TypeError('axis must be an integer')
    return convolve1d(array, np.repeat(1.0, window_size)/window_size, **kwargs)


def quantity_string(name, quantity_dict):
    '''Takes in an abbreviation for a quantity and a quantity dictionary,
       and returns a more descriptive string of the quantity as "name (units)."
       Raises ValueError if the name is not in quantity_dict
    '''
    if name not in quantity_dict.keys():
        raise ValueError('{0} is not present in quantity_dict'.format(name))
    return '{0} ({1})'.format(quantity_dict[name]['name'],
                              quantity_dict[name]['units'])


def strings_to_list_string(strings):
    '''Takes a list of strings presumably containing words and phrases,
       and returns a "list" form of those strings, like:

       >>> strings_to_list_string(('cats', 'dogs'))
       >>> 'cats and dogs'

       or

       >>> strings_to_list_string(('pizza', 'pop', 'chips'))
       >>> 'pizza, pop, and chips'

       Raises ValueError if strings is empty.
    '''
    if isinstance(strings, six.string_types):
        raise TypeError('strings must be an iterable of strings, not a string '
                        'itself')
    if len(strings) == 0:
        raise ValueError('strings may not be empty')
    elif len(strings) == 1:
        return strings[0]
    elif len(strings) == 2:
        return ' and '.join(strings)
    else:
        return '{0}, and {1}'.format(', '.join(strings[:-1]),
                                     strings[-1])


def assumption_list_string(assumptions, assumption_dict):
    '''
Takes in a list of short forms of assumptions and an assumption
dictionary, and returns a "list" form of the long form of the
assumptions.

Raises
------
ValueError
    if one of the assumptions is not in assumption_dict.
    '''
    if isinstance(assumptions, six.string_types):
        raise TypeError('assumptions must be an iterable of strings, not a '
                        'string itself')
    for a in assumptions:
        if a not in assumption_dict.keys():
            raise ValueError('{} not present in assumption_dict'.format(a))
    assumption_strings = [assumption_dict[a] for a in assumptions]
    return strings_to_list_string(assumption_strings)


def quantity_spec_string(name, quantity_dict):
    '''
Returns a quantity specification for docstrings.

Example
-------

>>> quantity_spec_string('Tv')
>>> 'Tv : float or ndarray\n    Data for virtual temperature.'
    '''
    if name not in quantity_dict.keys():
        raise ValueError('{0} not present in quantity_dict'.format(name))
    s = '{0} : float or ndarray\n'.format(name)
    s += doc_paragraph('Data for {0}.'.format(
        quantity_string(name, quantity_dict)), indent=4)
    return s


def doc_paragraph(s, indent=0):
    '''Takes in a string without wrapping corresponding to a paragraph,
       and returns a version of that string wrapped to be at most 80
       characters in length on each line.
       If indent is given, ensures each line is indented to that number
       of spaces.
    '''
    return '\n'.join([' '*indent + l for l in wrap(s, width=80-indent)])


def parse_derivative_string(string, quantity_dict):
    '''
    Assuming the string is of the form d(var1)d(var2), returns var1, var2.
    Raises ValueError if the string is not of this form, or if the vars
    are not keys in the quantity_dict, or if var2 is not a coordinate-like
    variable.
    '''
    match = derivative_prog.match(string)
    if match is None:
        raise ValueError('string is not in the form of a derivative')
    varname = match.group(1)
    coordname = match.group(2)
    if (varname not in quantity_dict.keys() or
            coordname not in quantity_dict.keys()):
        raise ValueError('variable in string not a valid quantity')
    return varname, coordname


def closest_val(x, L):
    '''
    Finds the index value in an iterable closest to a desired value.

    Parameters
    ----------
    x : object
        The desired value.
    L : iterable
        The iterable in which to search for the desired value.

    Returns
    -------
    index : int
        The index of the closest value to x in L.

    Notes
    -----
    Assumes x and the entries of L are of comparable types.

    Raises
    ------
    ValueError:
        if L is empty
    '''
    # Make sure the iterable is nonempty
    if len(L) == 0:
        raise ValueError('L must not be empty')
    if isinstance(L, np.ndarray):
        # use faster numpy methods if using a numpy array
        return (np.abs(L-x)).argmin()
    # for a general iterable (like a list) we need general Python
    # start by assuming the first item is closest
    min_index = 0
    min_diff = abs(L[0] - x)
    i = 1
    while i < len(L):
        # check if each other item is closer than our current closest
        diff = abs(L[i] - x)
        if diff < min_diff:
            # if it is, set it as the new closest
            min_index = i
            min_diff = diff
        i += 1
    return min_index


def area_poly_sphere(lat, lon, r_sphere):
    '''
Calculates the area enclosed by an arbitrary polygon on the sphere.

Parameters
----------

lat : iterable
    The latitudes, in degrees, of the vertex locations of the polygon, in
    clockwise order.
lon : iterable
    The longitudes, in degrees, of the vertex locations of the polygon, in
    clockwise order.

Returns
-------

area : float
    The desired spherical area in the same units as r_sphere.

Notes
-----

This function assumes the vertices form a valid polygon (edges do not
intersect each other).

**References**

Computing the Area of a Spherical Polygon of Arbitrary Shape
Bevis and Cambareri (1987)
Mathematical Geology, vol.19, Issue 4, pp 335-346
    '''
    dtr = np.pi/180.

    def _tranlon(plat, plon, qlat, qlon):
        t = np.sin((qlon-plon)*dtr)*np.cos(qlat*dtr)
        b = (np.sin(qlat*dtr)*np.cos(plat*dtr) -
             np.cos(qlat*dtr)*np.sin(plat*dtr)*np.cos((qlon-plon)*dtr))
        return np.arctan2(t, b)

    if len(lat) < 3:
        raise ValueError('lat must have at least 3 vertices')
    if len(lat) != len(lon):
        raise ValueError('lat and lon must have the same length')
    total = 0.
    for i in range(-1, len(lat)-1):
        fang = _tranlon(lat[i], lon[i], lat[i+1], lon[i+1])
        bang = _tranlon(lat[i], lon[i], lat[i-1], lon[i-1])
        fvb = bang - fang
        if fvb < 0:
            fvb += 2.*np.pi
        total += fvb
    return (total - np.pi*float(len(lat)-2))*r_sphere**2


def ddx(data, axis=0, dx=None, x=None, axis_x=0, boundary='forward-backward'):
    '''
Calculates a second-order centered finite difference derivative of data
along the specified axis.

Parameters
----------

data : ndarray
    Data on which we are taking a derivative.
axis : int
    Index of the data array on which to take the derivative.
dx : float, optional
    Constant grid spacing value. Will assume constant grid spacing if
    given. May not be used with argument x. Default value is 1 unless
    x is given.
x : ndarray, optional
    Values of the axis along which we are taking a derivative to allow
    variable grid spacing. May not be given with argument dx.
axis_x : int, optional
    Index of the x array on which to take the derivative. Does nothing if
    x is not given as an argument.
boundary : string, optional
    Boundary condition. If 'periodic', assume periodic boundary condition
    for centered difference. If 'forward-backward', take first-order
    forward or backward derivatives at boundary.

Returns
-------

derivative : ndarray
    Derivative of the data along the specified axis.

Raises
------

ValueError:
    If an invalid boundary condition choice is given, if both dx and x are
    specified, if axis is out of the valid range for the shape of the data,
    or if x is specified and axis_x is out of the valid range for the shape
    of x.
    '''
    if abs(axis) >= len(data.shape):
        raise ValueError('axis is out of bounds for the shape of data')
    if x is not None and abs(axis_x) > len(x.shape):
        raise ValueError('axis_x is out of bounds for the shape of x')
    if dx is not None and x is not None:
        raise ValueError('may not give both x and dx as keyword arguments')
    if boundary == 'periodic':
        deriv = (np.roll(data, -1, axis) - np.roll(data, 1, axis)
                 )/(np.roll(x, -1, axis_x) - np.roll(x, 1, axis_x))
    elif boundary == 'forward-backward':
        # We will take forward-backward differencing at edges
        # need some fancy indexing to handle arbitrary derivative axis
        # Initialize our index lists
        front = [slice(s) for s in data.shape]
        back = [slice(s) for s in data.shape]
        target = [slice(s) for s in data.shape]
        # Set our index values for the derivative axis
        # front is the +1 index for derivative
        front[axis] = np.array([1, -1])
        # back is the -1 index for derivative
        back[axis] = np.array([0, -2])
        # target is the position where the derivative is being calculated
        target[axis] = np.array([0, -1])
        if dx is not None:  # grid spacing is constant
            deriv = (np.roll(data, -1, axis) - np.roll(data, 1, axis))/(2.*dx)
            deriv[target] = (data[front]-data[back])/dx
        else:  # grid spacing is arbitrary
            # Need to calculate differences for our grid positions, too!
            # first take care of the centered differences
            dx = (np.roll(x, -1, axis_x) - np.roll(x, 1, axis_x))
            # now handle those annoying edge points, like with the data above
            front_x = [slice(s) for s in x.shape]
            back_x = [slice(s) for s in x.shape]
            target_x = [slice(s) for s in x.shape]
            front_x[axis_x] = np.array([1, -1])
            back_x[axis_x] = np.array([0, -2])
            target_x[axis] = np.array([0, -1])
            dx[target_x] = (x[front_x] - x[back_x])
            # Here dx spans two grid indices, no need for *2
            deriv = (np.roll(data, -1, axis) - np.roll(data, 1, axis))/dx
            deriv[target] = (data[front] - data[back])/dx
    else:  # invalid boundary condition was given
        raise ValueError('Invalid option {} for boundary '
                         'condition.'.format(boundary))
    return deriv


def d_x(data, axis, boundary='forward-backward'):
    '''
Calculates a second-order centered finite difference of data along the
specified axis.

Parameters
----------

data : ndarray
    Data on which we are taking a derivative.
axis : int
    Index of the data array on which to take the difference.
boundary : string, optional
    Boundary condition. If 'periodic', assume periodic boundary condition
    for centered difference. If 'forward-backward', take first-order
    forward or backward derivatives at boundary.

Returns
-------

derivative : ndarray
    Derivative of the data along the specified axis.

Raises
------

ValueError:
    If an invalid boundary condition choice is given, if both dx and x are
    specified, if axis is out of the valid range for the shape of the data,
    or if x is specified and axis_x is out of the valid range for the shape
    of x.
    '''
    if abs(axis) > len(data.shape):
        raise ValueError('axis is out of bounds for the shape of data')
    if boundary == 'periodic':
        diff = np.roll(data, -1, axis) - np.roll(data, 1, axis)
    elif boundary == 'forward-backward':
        # We will take forward-backward differencing at edges
        # need some fancy indexing to handle arbitrary derivative axis
        # Initialize our index lists
        front = [slice(s) for s in data.shape]
        back = [slice(s) for s in data.shape]
        target = [slice(s) for s in data.shape]
        # Set our index values for the derivative axis
        # front is the +1 index for derivative
        front[axis] = np.array([1, -1])
        # back is the -1 index for derivative
        back[axis] = np.array([0, -2])
        # target is the position where the derivative is being calculated
        target[axis] = np.array([0, -1])
        diff = (np.roll(data, -1, axis) - np.roll(data, 1, axis))/(2.)
        diff[target] = (data[front]-data[back])
    else:  # invalid boundary condition was given
        raise ValueError('Invalid option {} for boundary '
                         'condition.'.format(boundary))
    return diff
