# GIAPY: Glacial Isostatic Adjustment in PYthon

This is an opensource python package in development for copmuting
the Glacial Isostatic Adjustment of the surface of the earth in response to
shifting surface loads and comparing it to geophysical data. 

This release provides a scaled, normalized relaxation method for computing elastic love numbers (Kachuck, 2017).

In future releases, I will include methods for the viscoelastic relaxation and solving the sea level equation. If you are interested in having a look, these can currently be found in the development branch.

## Dependencies
[numpy](http://www.numpy.org) : Numerical computing package.

[numba](https://numba.pydata.org) : (optional) Provides a just in time compiler for faster computation. The easiest way to get this package is to use the [Continuum Analytics Anaconda python distribution](http://www.anaconda.com), with which numba is installed by default.

# Licensing
This work is distributed under the MIT license ('LICENSE').

## References
Cathles, L.M. (1975). *The Viscosity of the Earth's Mantle.* Princeton University Press. Princeton, NJ. 

Kachuck, S.B. and Cathles, L.M. (2017, September). *Normalized relaxation method for efficient computation of elastic Love numbers.* Oral presentation at the 1st circular Workshop on Glacial Isostatic Adjustment and Elastic Deformation, Reykjavik, Iceland.
