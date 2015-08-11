Introduction to atmos
=====================

What this package does
----------------------

**atmos** is meant to be a library of utility code for use in atmospheric
sciences. Its main functionality is currently to take input variables (like
pressure, virtual temperature, water vapor mixing ratio, etc.) and information
about what assumptions you're willing to make (hydrostatic? low water vapor?
ignore virtual temperature correction? use an empirical formula for
equivalent potential temperature?), and from that calculate any desired
output variables that you request and can be calculated.

Variable names
--------------

To make coding simpler by avoiding long names for quantities, a set of fairly
reasonable short-forms for different quantities are used by this package.
For example, air density is represented by "rho", and air temperature by "T".
For a complete list of quantities and their abbreviations, see the
documentation for :func:`atmos.calculate` or :class:`atmos.FluidSolver`.

Units
-----

By default, all quantities are input and output in SI units. Notably, pressure
quantities are input and output in Pascals, and temperature quantities are in
degrees Kelvin. A full list of units for different variables is available
in the documentation for :func:`atmos.calculate` or
:class:`atmos.FluidSolver`. These units can be overridden with keyword
arguments of the form {quantity name}_unit or {quantity name}_units.
Specifying units makes it so that both inputs and outputs of the quantity
will be in the specified units.

To get pressure in hPa::

    >>> import atmos
    >>> atmos.calculate('p', p_units='hPa', Tv=273., rho=1.27)
    99519.638400000011

To specify mixing ratio in g/kg::

    >>> import atmos
    >>> atmos.calculate('RH', rv=1, rvs=0.002, rv_unit='g/kg')
    50.0

Note that either "_unit" or "_units" can be used, and that units must be
specified for each quantity independently.

Unit handling is performed by the cfunits_ package, and so any units available
in that package (notably any units recognized by UDUNITS_) should be recognized.

Assumptions
-----------

By default, a set of (what are hopefully) fairly reasonable assumptions are
used by :class:`atmos.FluidSolver` and :func:`atmos.calculate`. These can be
added to or removed from
by tuples of string options supplied as keyword arguments *add_assumptions*
and *remove_assumptions*, respectively, or completely overridden by supplying
a tuple for the keyword argument *assumptions*. For information on what
default assumptions are used and all assumptions available, see the
documentation for :func:`atmos.calculate` or :class:`atmos.FluidSolver`.

Requests and Feedback
---------------------

This module is in ongoing development, and feedback is appreciated. In
particular, if there is functionality you would like to see or equations
that should be added (or corrected), please e-mail mcgibbon (at) uw {dot} edu.

.. _cfunits: https://pypi.python.org/pypi/cfunits/1.0.1
.. _UDUNITS: http://www.unidata.ucar.edu/software/udunits/

