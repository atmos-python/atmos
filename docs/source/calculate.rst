Using calculate()
=================

What does calculate do?
-----------------------

:func:`atmos.calculate` takes input variables (like
pressure, virtual temperature, water vapor mixing ratio, etc.) and information
about what assumptions you're willing to make (hydrostatic? low water vapor?
ignore virtual temperature correction? use an empirical formula for
equivalent potential temperature?), and from that calculates any desired
output variables that you request and can be calculated.

This function is essentially a wrapper for :class:`atmos.FluidSolver`, so
much or all of its functionality will be the same, and the documentation for
the two is very similar.

What can it calculate?
----------------------

Anything that can be calculated by equations in :mod:`atmos.equations`.
If you find that calculate() can't do a calculation you might expect it
to, check the equations it has available and make sure you're using the right
variables, or enabling the right assumptions. A common problem is using *T*
instead of *Tv* and expecting the ideal gas law to work.

A simple example
----------------

By default, a certain set of assumptions are used, such as that we are
considering an ideal gas, and so can use ideal gas law. This allows us to do
simple calculations that use the default assumptions. For example, to
calculate pressure from virtual temperature and density::

    >>> import atmos
    >>> atmos.calculate('p', Tv=273., rho=1.27)
    99519.638400000011

Or to calculate relative humidity from water vapor mixing ratio and
saturation water vapor mixing ratio (which needs no assumptions)::

    >>> import atmos
    >>> atmos.calculate('RH', rv=0.001, rvs=0.002)
    50.0

For a full list of default assumptions, see :func:`atmos.calculate`.

Specifying Units
----------------

By default, SI units are assumed. These can be overridden with keyword
arguments of the form {quantity name}_unit or {quantity name}_units.
Specifying units makes it so that both inputs and outputs of the quantity
will be in the specified units.

To get pressure in hPa::

    >>> import atmos
    >>> atmos.calculate('p', p_units='hPa', Tv=273., rho=1.27)
    995.19638400000008

To specify mixing ratio in g/kg::

    >>> import atmos
    >>> atmos.calculate('RH', rv=1, rvs=0.002, rv_unit='g/kg')
    50.0

Note that either "_unit" or "_units" can be used, and that units must be
specified for each quantity independently.

Unit handling is performed by the cfunits_ package, and so any units available
in that package (notably any units recognized by UDUNITS_) should be recognized.

Viewing equation functions used
-------------------------------

Calculating pressure from virtual temperature and density, also returning a
list of functions used::

    >>> import atmos
    >>> p, funcs = atmos.calculate('p', Tv=273., rho=1.27, debug=True)
    >>> funcs
    (<function atmos.equations.p_from_rho_Tv_ideal_gas>,)

Adding and removing assumptions
-------------------------------

If you want to use assumptions that are not enabled by default (such as
ignoring the virtual temperature correction), you can use the add_assumptions
keyword argument, which takes a tuple of strings specifying assumptions.
The exact string to enter for each assumption is detailed in
:func:`atmos.calculate`. For example, to calculate T instead of Tv, neglecting
the virtual temperature correction::

  >>> import atmos
  >>> atmos.calculate('p', T=273., rho=1.27, add_assumptions=('Tv equals T',))
  99519.638400000011

Overriding assumptions
----------------------

If you want to ignore the default assumptions entirely, you could specify
your own assumptions::

    >>> import atmos
    >>> assumptions = ('ideal gas', 'bolton')
    >>> atmos.calculate('p', Tv=273., rho=1.27, assumptions=assumptions)
    99519.638400000011

Specifying quantities with a dictionary
---------------------------------------

If you are repeatedly calculating different quantities, you may want to use
a dictionary to more easily pass in quantities as keyword arguments. Adding
\*\* to the beginning of a dictionary variable as an argument passes in
each of the (key, value) pairs in that dictionary as a separate keyword
argument. For example::

    >>> import atmos
    >>> data = {'Tv': 273., 'rho': 1.27}
    >>> data['p'] = atmos.calculate('p', **data)
    >>> data['p']
    99519.638400000011

Function reference
------------------

.. autofunction:: atmos.calculate

.. _cfunits: https://pypi.python.org/pypi/cfunits/1.0.1
.. _UDUNITS: http://www.unidata.ucar.edu/software/udunits/


