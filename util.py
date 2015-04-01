# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:11:26 2015

@author: mcgibbon
"""
import numpy as np


def advect_var(dvardx, dvardy, u, v):
    '''
    Calculates the horizontal advective tendency of a variable in units/s
    from the horizontal gradients of that variable in units/m and the
    horizontal winds in m/s.
    '''
    return u*dvardx + v*dvardy


def landsea_mask(lat, lon, basemap=None, basemap_lat=None, basemap_lon=None):
    '''
    Calculates a land sea mask for a given latitude and longitude array.

    Parameters
    ----------
    lat : ndarray
        Latitudes in degrees N.
    lon : ndarray
        Longitudes in degrees E.
    basemap: ndarray, optional
        A 2D array of type byte or int that contains the baseline land sea
        mask from which the new land sea mask is generated. By default uses
        the grid distributed with NCL, available at
        https://www.ncl.ucar.edu/Applications/Data/cdf/landsea.nc
    basemap_lat: ndarray, optional
        The latitudes of basemap. If not specified, assumes a regularly spaced
        grid from -90 to 90 degrees.
    basemap_lon: ndarray, optional
        The longitudes of basemap. If not specified, assumes a regularly spaced
        grid from -180 to 180 degrees.
    '''
    raise NotImplementedError


def gaussian_latitudes(nlat):
    '''
    Generates gaussian latitudes.

    Parameters
    ----------
    nlat : int
        The number of latitudes desired

    Returns
    -------
    lat : ndarray
        A one-dimensional array containing the gaussian latitudes.
    '''
    raise NotImplementedError


def gaussian_latitude_weights(nlat):
    '''
    Generates gaussian weights.

    Parameters
    ----------
    nlat : int
        The number of latitudes desired

    Returns
    -------
    weights : ndarray
        A one-dimensional array containing the gaussian weights.
    '''
    raise NotImplementedError


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
    if len(L) == 0:
        raise ValueError('L must not be empty')
    i = 0
    min_index = 0
    min_diff = abs(L[0] - x)
    while i < len(L):
        diff = abs(L[i] - x)
        if diff < min_diff:
            min_index = i
            min_diff = diff
        i += 1
    return min_index


def dpres_isobaric(p_lev, p_sfc, p_top, vertical_axis=None, fill_value=np.NaN):
    '''
    Calculates the pressure layer thicknesses of a constant pressure level
    coordinate system.

    Parameters
    ----------
    p_lev : ndarray
        A one dimensional array containing the constant pressure levels. May
        be in ascending or descending order.
    p_sfc : float or ndarray
        A scalar or an array containing the surface pressure data. Must have
        the same units as p_lev.
    p_top : float or ndarray
        A scalar or an array specifying the top of the column. Must have the
        same units as p_lev. If an array is given, must have the same shape
        as p_sfc.
    vertical_axis : int, optional
        The index of the returned array that should correspond to the vertical.
        Must be between 0 and the number of axes in p_sfc (inclusive).

    Returns
    -------
    dpres : ndarray
        An array specifying the pressure layer thicknesses. If p_sfc is a
        float, will be one-dimensional. Otherwise, will have one more dimension
        than p_sfc. Which axis corresponds to the vertical can be given by the
        keyword argument vertical_axis. If it is not given, the vertical axis
        will be 0 if p_sfc has 1 or 2 dimensions, or 1 if p_sfc has more
        dimensions. Note that this replicates NCL behavior for 2- and
        3-dimensional arrays. The size of the vertical dimension will be the
        same as the size of p_lev.

    See Also
    --------
    dpres_hybrid : Pressure layer thicknesses of a hybrid coordinate system

    Notes
    -----
    Calculates the layer pressure thickness of a constant pressure level
    system. At each grid point the sum of the pressure thicknesses equates to
    [p_sfc-p_top]. At each grid point, the returned values above ptop and below
    psfc will be set to fill_value. If p_top or p_sfc is between p_lev levels
    then the layer thickness is modifed accordingly.

    Raises
    ------
    ValueError
        If vertical_axis is given and is not between 0 and the number of
        axes in p_sfc.
    '''
    raise NotImplementedError


def dpres_hybrid(p_sfc, hybrid_a, hybrid_b, p0=1e5, vertical_axis=None):
    '''
    Calculates the pressure layer thicknesses of a hybrid coordinate system.

    Parameters
    ----------
    p_sfc : ndarray
        An array with surface pressure data.
    hybrid_a : ndarray
        A one-dimensional array equal to the hybrid A coefficients. Usually,
        the "interface" coefficients are input.
    hybrid_b : ndarray
        A one-dimensional array equal to the hybrid B coefficients. Usually,
        the "interface" coefficients are input.
    p0 : float, optional
        A scalar value equal to the surface reference pressure. Must have the
        same units as ps. By default, 10^5 Pa is used.
    vertical_axis : int, optional
        The index of the returned array that should correspond to the vertical.
        Must be between 0 and the number of axes in p_sfc (inclusive).

    Returns
    -------
    dpres : ndarray
        An array specifying the pressure layer thicknesses. If p_sfc is a
        float, will be one-dimensional. Otherwise, will have one more dimension
        than p_sfc. Which axis corresponds to the vertical can be given by the
        keyword argument vertical_axis. If it is not given, the vertical axis
        will be 0 if p_sfc has 1 or 2 dimensions, or 1 if p_sfc has more
        dimensions. Note that this replicates NCL behavior for 2- and
        3-dimensional arrays. The size of the vertical dimension will be the
        one less than the size of hybrid_a.

    Notes
    -----
    Calculates the layer pressure thickness of a hybrid coordinate system. At
    each grid point the sum of the pressure thicknesses equates to [psfc-ptop].
    At each grid point, the returned values above ptop and below psfc will be
    set to fill_value. If ptop or psfc is between plev levels then the layer
    thickness is modifed accordingly.

    Raises
    ------
    ValueError
        If vertical_axis is given and is not between 0 and the number of
        axes in p_sfc.
    '''
    raise NotImplementedError


def geopotential_height_hybrid(psfc, Phisfc, Tv, hyam, hybm, hyai, hybi,
                               p0=1e5, vertical_axis=None):
    '''
    Computes geopotential height in hybrid coordinates.

    Parameters
    ----------
    psfc : ndarray
        Surface pressure in Pa.
    Phisfc : ndarray
        Surface geopotential in m^2/s^2. If it is not the same shape as ps,
        then it must correspond to the rightmost dimensions of ps. May not have
        more dimensions than ps.
    Tv : ndarray
        Virtual temperature in K, ordered top-to-bottom.
    hyam: ndarray
        One-dimensional array of hybrid A coefficients (layer midpoints),
        ordered bottom-to-top.
    hybm: ndarray
        One-dimensional array of hybrid B coefficients (layer midpoints),
        ordered bottom-to-top.
    hyai: ndarray
        One-dimensional array of hybrid A coefficients (layer interfaces),
        ordered bottom-to-top.
    hybi: ndarray
        One-dimensional array of hybrid B coefficients (layer interfaces),
        ordered bottom-to-top.
    vertical_axis : int, optional
        The index of Tv that corresponds to the vertical. By default, is 0 if
        Tv has 3 or fewer axes, and 1 if Tv has more axes.

    Returns
    -------
    Phi : ndarray
        Geopotential height values. Array has the same shape as Tv.

    Notes
    -----
    Assumes no missing values in input.
    '''
    raise NotImplementedError


def hybrid_interpolate(data, ps, hybrid_a_in, hybrid_b_in, hybrid_a_out,
                       hybrid_b_out, p0=1e5, vertical_axis=None,
                       extrapolate='missing'):
    '''
    Interpolates from data on one set of hybrid levels to another set of hybrid
    levels.

    Parameters
    ----------
    data: ndarray
        Data to be interpolated.
    ps: ndarray
        Surface pressure. If given in units other than Pa, p0 should be
        specified. Its rightmost axes must correspond to the rightmost axes
        of data, not including the vertical axis of data.
    hybrid_a_in: ndarray
        Hybrid A coefficients associated with the input data.
    hybrid_b_in: ndarray
        Hybrid B coefficients associated with the input data.
    hybrid_a_out: ndarray
        Hybrid A coefficients of the returned data.
    hybrid_b_out: ndarray
        Hybrid B coefficients of the returned data.
    p0: float
        Surface reference pressure. Must be in the same units as ps.
    vertical_axis : int, optional
        The index of data that corresponds to the vertical. By default, is 0 if
        data has 3 or fewer axes, and 1 if data has more axes.
    extrapolate : str, optional
        Determines how output values outside of the range of the input axis
        should be handled. If 'missing', they are set to NaN. If 'nearest',
        they are set to the nearest input value.

    Returns
    -------
    data_out : ndarray
        data interpolated to the new hybrid vertical axis.
    '''
    raise NotImplementedError


def isobaric_to_hybrid(data, p, ps, hybrid_a, hybrid_b, p0=1e5,
                       vertical_axis=None, extrapolate='missing'):
    '''
    Interpolates data on constant pressure levels to hybrid levels.

    Parameters
    ----------
    data: ndarray
        Data to be interpolated.
    p : ndarray
        A one-dimensional array with the pressure levels of data. Must have
        the same units as ps and p0.
    ps: ndarray
        Surface pressure. If given in units other than Pa, p0 should be
        specified. Its rightmost axes must correspond to the rightmost axes
        of data, not including the vertical axis of data.
    hybrid_a: ndarray
        Hybrid A coefficients of the returned data.
    hybrid_b: ndarray
        Hybrid B coefficients of the returned data.
    p0: float
        Surface reference pressure. Must be in the same units as ps.
    vertical_axis : int, optional
        The index of data that corresponds to the vertical. By default, is 0 if
        data has 3 or fewer axes, and 1 if Tv has more axes.
    extrapolate : str, optional
        Determines how output values outside of the range of the input axis
        should be handled. If 'missing', they are set to NaN. If 'nearest',
        they are set to the nearest input value.

    Returns
    -------
    data_out : ndarray
        data interpolated to the hybrid vertical axis.
    '''
    raise NotImplementedError


def Phi_from_p_Tv_Phisfc_hydrostatic(p, Tv, Phisfc, vertical_axis=None,
                                     vertical_order='up'):
    '''
    Calculates geopotential height (m^2/s^2) from pressure (Pa), virtual
    temperature (K), and surface geopotential height (m^2/s^2).

    Parameters
    ----------
    p : ndarray
        Pressure in Pa. If the array is not one-dimensional, it must have
        the same shape as Tv and each axis must correspond to the same axis
        as Tv.
    Tv : ndarray
        Virtual temperature in K.
    Phisfc : ndarray
        Surface geopotential height in m^2/s^2.
    vertical_axis : int, optional
        Index of Tv corresponding to the vertical. By default is 0 if Tv has 3
        or fewer axes, and 1 otherwise.
    vertical_order : str, optional
        Order of the vertical axes of p and Tv. Must be 'up' or 'down' if
        given.

    Returns
    -------
    Phi: ndarray
        Geopotential height values in m^2/s^2. Array has the same shape as Tv.
    '''
    raise NotImplementedError


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

    Reference
    ---------
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
    for i in range(-1, len(lat)):
        fang = _tranlon(lat[i], lon[i], lat[i+1], lon[i+1])
        bang = _tranlon(lat[i], lon[i], lat[i-1], lon[i-1])
        fvb = bang - fang
        if fvb < 0:
            fvb += 2.*np.pi
        total += fvb
    return (total - np.pi*float(len(lat)-2))*r_sphere**2


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
