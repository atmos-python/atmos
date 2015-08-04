"""
plot.py: Utilities for plotting meteorological data. Importing this package
gives access to the "skewT" projection.

This file was originally edited from code in MetPy. The MetPy copyright
disclamer is at the bottom of the source code of this file.
"""

import numpy as np
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.projections import register_projection
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from atmos import calculate
from atmos.constants import g0
from scipy.integrate import odeint
from atmos.util import closest_val
from pkg_resources import resource_filename


# The sole purpose of this class is to look at the upper, lower, or total
# interval as appropriate and see what parts of the tick to draw, if any.
class SkewXTick(maxis.XTick):
    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group(self.__name__)

        lower_interval = self.axes.xaxis.lower_interval
        upper_interval = self.axes.xaxis.upper_interval

        if self.gridOn and transforms.interval_contains(
                self.axes.xaxis.get_view_interval(), self.get_loc()):
            self.gridline.draw(renderer)

        if transforms.interval_contains(lower_interval, self.get_loc()):
            if self.tick1On:
                self.tick1line.draw(renderer)
            if self.label1On:
                self.label1.draw(renderer)

        if transforms.interval_contains(upper_interval, self.get_loc()):
            if self.tick2On:
                self.tick2line.draw(renderer)
            if self.label2On:
                self.label2.draw(renderer)

        renderer.close_group(self.__name__)


# This class exists to provide two separate sets of intervals to the tick,
# as well as create instances of the custom tick
class SkewXAxis(maxis.XAxis):
    def __init__(self, *args, **kwargs):
        maxis.XAxis.__init__(self, *args, **kwargs)
        self.upper_interval = 0.0, 1.0

    def _get_tick(self, major):
        return SkewXTick(self.axes, 0, '', major=major)

    @property
    def lower_interval(self):
        return self.axes.viewLim.intervalx

    def get_view_interval(self):
        return self.upper_interval[0], self.axes.viewLim.intervalx[1]

class SkewYAxis(maxis.YAxis):
    pass


# This class exists to calculate the separate data range of the
# upper X-axis and draw the spine there. It also provides this range
# to the X-axis artist for ticking and gridlines
class SkewSpine(mspines.Spine):
    def _adjust_location(self):
        trans = self.axes.transDataToAxes.inverted()
        if self.spine_type == 'top':
            yloc = 1.0
        else:
            yloc = 0.0
        left = trans.transform_point((0.0, yloc))[0]
        right = trans.transform_point((1.0, yloc))[0]

        pts = self._path.vertices
        pts[0, 0] = left
        pts[1, 0] = right
        self.axis.upper_interval = (left, right)


# This class handles registration of the skew-xaxes as a projection as well
# as setting up the appropriate transformations. It also overrides standard
# spines and axes instances as appropriate.
class SkewTAxes(Axes):
    # The projection must specify a name.  This will be used be the
    # user to select the projection, i.e. ``subplot(111,
    # projection='skewx')``.
    name = 'skewT'
    default_xlim = (-40, 50)
    default_ylim = (1050, 100)

    def __init__(self, *args, **kwargs):
        # This needs to be popped and set before moving on
        self.rot = kwargs.pop('rotation', 45)
        Axes.__init__(self, *args, **kwargs)

    def _init_axis(self):
        # Taken from Axes and modified to use our modified X-axis
        self.xaxis = SkewXAxis(self)
        self.spines['top'].register_axis(self.xaxis)
        self.spines['bottom'].register_axis(self.xaxis)
        self.yaxis = maxis.YAxis(self)
        self.yaxis.set_major_formatter(ScalarFormatter())
        self.yaxis.set_major_locator(MultipleLocator(100))
        self.spines['left'].register_axis(self.yaxis)
        self.spines['right'].register_axis(self.yaxis)

    def _gen_axes_spines(self, locations=None, offset=0.0, units='inches'):
        # pylint: disable=unused-argument
        spines = {'top': SkewSpine.linear_spine(self, 'top'),
                  'bottom': mspines.Spine.linear_spine(self, 'bottom'),
                  'left': mspines.Spine.linear_spine(self, 'left'),
                  'right': mspines.Spine.linear_spine(self, 'right')}
        return spines

    def _set_lim_and_transforms(self):
        """
        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.
        """
        # Get the standard transform setup from the Axes base class
        Axes._set_lim_and_transforms(self)

        # Need to put the skew in the middle, after the scale and limits,
        # but before the transAxes. This way, the skew is done in Axes
        # coordinates thus performing the transform around the proper origin
        # We keep the pre-transAxes transform around for other users, like the
        # spines for finding bounds
        self.transDataToAxes = (self.transScale +
                                (self.transLimits +
                                 transforms.Affine2D().skew_deg(self.rot, 0)))

        # Create the full transform from Data to Pixels
        self.transData = self.transDataToAxes + self.transAxes

        # Blended transforms like this need to have the skewing applied using
        # both axes, in axes coords like before.
        self._xaxis_transform = (transforms.blended_transform_factory(
            self.transScale + self.transLimits,
            transforms.IdentityTransform()) +
            transforms.Affine2D().skew_deg(self.rot, 0)) + self.transAxes

    def cla(self):
        Axes.cla(self)
        # Disables the log-formatting that comes with semilogy
        self.yaxis.set_major_formatter(ScalarFormatter())
        self.yaxis.set_major_locator(MultipleLocator(100))
        if not self.yaxis_inverted():
            self.invert_yaxis()

        # Try to make sane default temperature plotting
        self.xaxis.set_major_locator(MultipleLocator(5))
        self.xaxis.set_major_formatter(ScalarFormatter())
        self.set_xlim(*self.default_xlim)
        self.set_ylim(*self.default_ylim)

    def semilogy(self, p, T, *args, **kwargs):
        """
        """
        # We need to replace the overridden plot with the original Axis plot
        # since it is called within Axes.semilogy
        no_plot = SkewTAxes.plot
        SkewTAxes.plot = Axes.plot
        Axes.semilogy(self, T, p, *args, **kwargs)
        # Be sure to put back the overridden plot method
        SkewTAxes.plot = no_plot
        self.yaxis.set_major_formatter(ScalarFormatter())
        self.yaxis.set_major_locator(MultipleLocator(100))
        labels = self.xaxis.get_ticklabels()
        for label in labels:
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_fontsize(8)
            label.set_color('#B31515')
        self.grid(True)
        self.grid(axis='top', color='#B31515', linestyle='-', linewidth=1,
                  alpha=0.5, zorder=1.1)
        self.grid(axis='x', color='#B31515', linestyle='-', linewidth=1,
                  alpha=0.5, zorder=1.1)
        self.grid(axis='y', color='k', linestyle='-', linewidth=0.5, alpha=0.5,
                  zorder=1.1)
        self.set_xlabel(r'Temperature ($^{\circ} C$)', color='#B31515')
        self.set_ylabel('Pressure ($hPa$)')
        self.plot_mixing_lines()
        self.plot_dry_adiabats()
        self.plot_moist_adiabats()

    def plot(self, *args, **kwargs):
        """
        """
        self.semilogy(*args, **kwargs)

    def semilogx(self, *args, **kwargs):
        """
        """
        raise NotImplementedError(
            'Skew-T is not logarithmic in T, use semilogy')

    def skew_plot(self, p, T, *args, **kwargs):
        r'''Plot data.

        Simple wrapper around plot so that pressure is the first (independent)
        input. This is essentially a wrapper around `semilogy`.

        Parameters
        ----------
        p : array_like
            pressure values
        T : array_like
            temperature values, can also be used for things like dew point
        args
            Other positional arguments to pass to `semilogy`
        kwargs
            Other keyword arguments to pass to `semilogy`

        See Also
        --------
        `matplotlib.Axes.semilogy`
        '''

        # Skew-T logP plotting
        self.semilogy(T, p, *args, **kwargs)

    def plot_barbs(self, p, u, v, xloc=1.0, x_clip_radius=0.08,
                   y_clip_radius=0.08, **kwargs):
        r'''Plot wind barbs.

        Adds wind barbs to the skew-T plot. This is a wrapper around the
        `barbs` command that adds to appropriate transform to place the
        barbs in a vertical line, located as a function of pressure.

        Parameters
        ----------
        p : array_like
            pressure values
        u : array_like
            U (East-West) component of wind
        v : array_like
            V (North-South) component of wind
        xloc : float, optional
            Position for the barbs, in normalized axes coordinates, where 0.0
            denotes far left and 1.0 denotes far right. Defaults to far right.
        x_clip_radius : float, optional
            Space, in normalized axes coordinates, to leave before clipping
            wind barbs in the x-direction. Defaults to 0.08.
        y_clip_radius : float, optional
            Space, in normalized axes coordinates, to leave above/below plot
            before clipping wind barbs in the y-direction. Defaults to 0.08.
        kwargs
            Other keyword arguments to pass to `barbs`

        See Also
        --------
        `matplotlib.Axes.barbs`
        '''

        # Assemble array of x-locations in axes space
        x = np.empty_like(p)
        x.fill(xloc)

        # Do barbs plot at this location
        b = self.barbs(x, p, u, v,
                       transform=self.get_yaxis_transform(which='tick2'),
                       clip_on=True, **kwargs)

        # Override the default clip box, which is the axes rectangle, so we can
        # have barbs that extend outside.
        ax_bbox = transforms.Bbox([[xloc-x_clip_radius, -y_clip_radius],
                                   [xloc+x_clip_radius, 1.0 + y_clip_radius]])
        b.set_clip_box(transforms.TransformedBbox(ax_bbox, self.transAxes))

    def plot_dry_adiabats(self, t0=None, p=None, **kwargs):
        r'''Plot dry adiabats.

        Adds dry adiabats (lines of constant potential temperature) to the
        plot. The default style of these lines is dashed red lines with an
        alpha value of 0.5. These can be overridden using keyword arguments.

        Parameters
        ----------
        t0 : array_like, optional
            Starting temperature values in Kelvin. If none are given, they will be
            generated using the current temperature range at the bottom of
            the plot.
        p : array_like, optional
            Pressure values to be included in the dry adiabats. If not
            specified, they will be linearly distributed across the current
            plotted pressure range.0.5
        kwargs
            Other keyword arguments to pass to `matplotlib.collections.LineCollection`

        See Also#B85C00
        --------
        plot_moist_adiabats
        `matplotlib.collections.LineCollection`
        `metpy.calc.dry_lapse`
        '''

        # Determine set of starting temps if necessary
        if t0 is None:
            xmin, xmax = self.get_xlim()
            t0 = np.arange(xmin, xmax + 201, 10)

        # Get pressure levels based on ylims if necessary
        if p is None:
            p = np.linspace(*self.get_ylim())

        # Assemble into data for plotting
        t = calculate('T', theta=t0[:, None], p=p, p_units='hPa',
                      T_units='degC', theta_units='degC')
        linedata = [np.vstack((ti, p)).T for ti in t]

        # Add to plot
        kwargs.setdefault('colors', '#A65300')
        kwargs.setdefault('linestyles', '-')
        kwargs.setdefault('alpha', 1)
        kwargs.setdefault('linewidth', 0.5)
        kwargs.setdefault('zorder', 1.1)
        self.add_collection(LineCollection(linedata, **kwargs))
        t0 = t0.flatten()
        T_label = calculate('T', p=140, p_units='hPa', theta=t0,
                             T_units='degC', theta_units='degC')
        for i in range(len(t0)):
            format_string = '{:.0f}'
            self.text(T_label[i], 140, format_string.format(t0[i]),
                      fontsize=8, ha='left', va='center', rotation=-60,
                      color='#A65300', bbox={
                          'facecolor': 'w', 'edgecolor': 'w', 'alpha': 0,
                      }, zorder=1.2).set_clip_on(True)

    def plot_moist_adiabats(self, t0=None, p=None, **kwargs):
        r'''Plot moist adiabats.

        Adds saturated pseudo-adiabats (lines of constant equivalent potential
        temperature) to the plot. The default style of these lines is dashed
        blue lines with an alpha value of 0.5. These can be overridden using
        keyword arguments.

        Parameters
        ----------
        t0 : array_like, optional
            Starting temperature values in Kelvin. If none are given, they will be
            generated using the current temperature range at the bottom of
            the plot.
        p : array_like, optional
            Pressure values to be included in the moist adiabats. If not
            specified, they will be linearly distributed across the current
            plotted pressure range.
        kwargs
            Other keyword arguments to pass to `matplotlib.collections.LineCollection`

        See Also
        --------
        plot_dry_adiabats
        `matplotlib.collections.LineCollection`
        `metpy.calc.moist_lapse`
        '''
        def dT_dp(y, p0):
            return calculate('Gammam', T=y, p=p0, RH=100., p_units='hPa',
                             T_units='degC')/(
                g0*calculate('rho', T=y, p=p0, p_units='hPa', T_units='degC',
                             RH=100.))*100.

        if t0 is None and p is None:
            if (self.get_xlim() == self.default_xlim and
                    self.get_ylim() == self.default_ylim):
                data = np.load(resource_filename(
                    __name__, 'data/default_moist_adiabat_data.npz'))
                p = data['p']
                t0 = data['t0']
                t = data['t']
        else:
            # Determine set of starting temps if necessary
            if t0 is None:
                xmin, xmax = self.get_xlim()
                t0 = np.concatenate((np.arange(xmin, 0, 5),
                                     np.arange(0, xmax + 51, 5)))
            # Get pressure levels based on ylims if necessary
            if p is None:
                p = np.linspace(*self.get_ylim())
            t0_base = odeint(dT_dp, t0, np.array([1e3, p[0]],
                                                 dtype=np.float64))[-1, :]
    
            # Assemble into data for plotting
            result = odeint(dT_dp, t0_base, p)
            t = result.T
        linedata = [np.vstack((ti, p)).T for ti in t]

        # Add to plot
        kwargs.setdefault('colors', '#166916')
        kwargs.setdefault('linestyles', '-')
        kwargs.setdefault('alpha', 1)
        kwargs.setdefault('linewidth', 0.5)
        kwargs.setdefault('zorder', 1.1)
        self.add_collection(LineCollection(linedata, **kwargs))
        label_index = closest_val(240., p)
        T_label = t[:,label_index].flatten()
        for i in range(len(t0)):
            format_string = '{:.0f}'
            self.text(T_label[i], p[label_index], format_string.format(t0[i]),
                      fontsize=8, ha='left', va='center', rotation=-65,
                      color='#166916', bbox={
                          'facecolor': 'w', 'edgecolor': 'w', 'alpha': 0,
                      }, zorder=1.2).set_clip_on(True)

    def plot_mixing_lines(self, rv=None, p=None, **kwargs):
        r'''Plot lines of constant mixing ratio.

        Adds lines of constant mixing ratio (isohumes) to the
        plot. The default style of these lines is dashed green lines with an
        alpha value of 0.8. These can be overridden using keyword arguments.

        Parameters
        ----------
        rv : array_like, optional
            Unitless mixing ratio values to plot. If none are given, default
            values are used.
        p : array_like, optional
            Pressure values to be included in the isohumes. If not
            specified, they will be linearly distributed across the current
            plotted pressure range up to 600 mb.
        kwargs
            Other keyword arguments to pass to `matplotlib.collections.LineCollection`

        See Also
        --------
        `matplotlib.collections.LineCollection`
        '''

        # Default mixing level values if necessary
        if rv is None:
            rv = np.array([
                0.1e-3, 0.2e-3, 0.5e-3, 1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 6e-3,
                8e-3, 10e-3, 12e-3, 15e-3, 20e-3, 30e-3, 40e-3,
                50e-3]).reshape(-1, 1)
        else:
            rv = np.asarray(rv).reshape(-1, 1)

        # Set pressure range if necessary
        if p is None:
            p = np.linspace(min(self.get_ylim()), max(self.get_ylim()))
        else:
            p = np.asarray(p)

        # Assemble data for plotting
        Td = calculate(
            'Td', p=p, rv=rv, p_units='hPa', rv_units='kg/kg',
            Td_units='degC')
        Td_label = calculate('Td', p=550, p_units='hPa', rv=rv,
                             Td_units='degC')
        linedata = [np.vstack((t, p)).T for t in Td]

        # Add to plot
        kwargs.setdefault('colors', '#166916')
        kwargs.setdefault('linestyles', '--')
        kwargs.setdefault('alpha', 1)
        kwargs.setdefault('linewidth', 0.5)
        kwargs.setdefault('zorder', 1.1)
        self.add_collection(LineCollection(linedata, **kwargs))
        rv = rv.flatten() * 1000
        for i in range(len(rv)):
            if rv[i] < 1:
                format_string = '{:.1f}'
            else:
                format_string = '{:.0f}'
            self.text(Td_label[i], 550, format_string.format(rv[i]),
                      fontsize=8, ha='right', va='center', rotation=60,
                      color='#166916', bbox={
                          'facecolor': 'w', 'edgecolor': 'w', 'alpha': 0,
                      }, zorder=1.2).set_clip_on(True)


# Now register the projection with matplotlib so the user can select
# it.
register_projection(SkewTAxes)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
#    fig = plt.figure(figsize=(6, 6))
#    ax = fig.add_subplot(1, 1, 1, projection='skewT')
    fig, ax = plt.subplots(1, 1, figsize=(8,8),
                           subplot_kw={'projection': 'skewT'})
#    ax.skew_plot(np.linspace(1e3, 100, 100), np.linspace(0,-50, 100))
    ax.plot(np.linspace(1e3, 100, 100), np.linspace(0,-50, 100))
    plt.show()

# Copyright (c) 2008-2014, MetPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#    * Neither the name of the MetPy Developers nor the names of any
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
