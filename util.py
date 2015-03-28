# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:11:26 2015

@author: mcgibbon
"""
import numpy as np


def ddx(data, axis, dx=None, x=None, axis_x=0, boundary='forward-backward'):
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
        given. May not be used with argument x.
    x : ndarray, optional
        Position in the coordinate system of the data along the axis on which
        we are taking a derivative. If x is not given, returns a derivative
        with respect to the coordinate index (i.e. assumes a grid spacing of
        1). May not be used with argument dx.
    axis_x : int, optional
        Index of the x array on which to take the derivative. Does nothing if
        x is not given as an argument.
    boundary: string, optional
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
        If an invalid boundary condition choice is given.
        If both dx and x are specified.
        If axis is out of the valid range for the shape of the data
        If x is specified and axis_x is out of the valid range for the shape
            of x
    '''
    if abs(axis) > len(data.shape):
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
            deriv = (np.roll(data, -1, axis) - np.roll(data, 1, axis))/(2*dx)
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
