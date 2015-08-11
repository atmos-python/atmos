Using FluidSolver
=================

What does FluidSolver do?
-------------------------

:class:`atmos.FluidSolver` takes input variables (like
pressure, virtual temperature, water vapor mixing ratio, etc.) and information
about what assumptions you're willing to make (hydrostatic? low water vapor?
ignore virtual temperature correction? use an empirical formula for
equivalent potential temperature?), and from that calculates any desired
output variables that you request and can be calculated.

The main benefit of using :class:`atmos.FluidSolver` instead of
:func:`atmos.calculate` is that the FluidSolver object has memory. It can keep
track of what assumptions you enabled, as well as what quantities you've given
it and it has calculated.

What can it calculate?
----------------------

Anything that can be calculated by equations in :mod:`atmos.equations`.
If you find that the FluidSolver can't do a calculation you might expect it
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
    >>> solver = atmos.FluidSolver(Tv=273., rho=1.27)
    >>> solver.calculate('p')
    99519.638400000011

Or to calculate relative humidity from water vapor mixing ratio and
saturation water vapor mixing ratio (which needs no assumptions)::

    >>> import atmos
    >>> solver = atmos.FluidSolver(rv=0.001, rvs=0.002)
    >>> solver.calculate('RH')
    50.0

For a full list of default assumptions, see :class:`atmos.FluidSolver`.


Specifying Units
----------------

By default, SI units are assumed. These can be overridden with keyword
arguments of the form {quantity name}_unit or {quantity name}_units.
Specifying units makes it so that both inputs and outputs of the quantity
will be in the specified units.

To get pressure in hPa::

    >>> import atmos
    >>> solver = atmos.FluidSolver(Tv=273., rho=1.27, p_units='hPa')
    >>> atmos.calculate('p')
    995.19638400000008

To specify mixing ratio in g/kg::

    >>> import atmos
    >>> solver = atmos.FluidSolver(rv=1, rvs=0.002, rv_unit='g/kg')
    >>> atmos.calculate('RH')
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
    >>> solver = atmos.FluidSolver(Tv=273., rho=1.27, debug=True)
    >>> p, funcs = solver.calculate('p')
    >>> funcs
    (<function atmos.equations.p_from_rho_Tv_ideal_gas>,)

Adding and removing assumptions
-------------------------------

If you want to use assumptions that are not enabled by default (such as
ignoring the virtual temperature correction), you can use the add_assumptions
keyword argument, which takes a tuple of strings specifying assumptions.
The exact string to enter for each assumption is detailed in
:class:`atmos.FluidSolver`. For example, to calculate T instead of Tv,
neglecting the virtual temperature correction::

  >>> import atmos
  >>> solver = atmos.FluidSolver(T=273., rho=1.27, add_assumptions=('Tv equals T',))
  >>> solver.calculate('p')
  99519.638400000011

Overriding assumptions
----------------------

If you want to ignore the default assumptions entirely, you could specify
your own assumptions::

    >>> import atmos
    >>> solver = atmos.FluidSolver(Tv=273., rho=1.27, assumptions=('ideal gas', 'bolton'))
    >>> solver.calculate('p')
    99519.638400000011

Class reference
---------------

.. autoclass:: atmos.FluidSolver

.. _cfunits: https://pypi.python.org/pypi/cfunits/1.0.1
.. _UDUNITS: http://www.unidata.ucar.edu/software/udunits/

