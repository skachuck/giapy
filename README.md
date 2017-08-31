# GIAPY: Glacial Isostatic Adjustment in PYthon

**The current version is not yet ready for public use. If you are chomping at the bit, go for it and let me know. If you are looking for the relaxation method, you can find it in giapy/numTools/solvde.py and giapy/earth_tools/elasticlove.py**

This is an opensource python package in development for copmuting
the Glacial Isostatic Adjustment of the surface of the earth in response to
shifting surface loads and comparing it to geophysical data. 
The method used here is based principally on that of Cathles (1975). 

The primary computation occus in ```giapy.giasim```, which is responsible for 
computing the Sea Level Equation (Farrell and Clark 1976). The method is spherically-symmetric,
gravitationally-consistent, with a redistributing ocean load that obeys shifting coastlines 
and floating ice (see e.g., Milne et al. 1999). As yet, rotation has not been implemented.

There are modules for manipulating ice models (```giapy.ice```), for 
computing the response to harmonic loads (```giapy.earth```), and for data 
integration (```giapy.data```). 

## Dependencies
basemap : this package does the majority of the geographic plotting and provides 
            some useful tools for dealing with geographic data.
numba   : Provides a just in time compiler for faster computation.
emcee   : (optional) The MCMC Hammer. Affine invariant MCMC algorithm for ensemble sampling.

# Licensing
This work is distributed under the MIT license (`LICENSE').

## References
Cathles, L.M. "The Viscosity of the Earth's Mantle." Princeton University
    Press. Princeton, NJ. 1975.
Farrell, W.E. and Clark J.A. (1976) "On Postglacial Sea Level" Geophys. J. Int. 46.
Milne, G.A., Mitrovica, J.X., and Davis, J.L. (1999) "Near-field hydro-isostasy: 
    the implementation of a revised sea-level equation." Geophys. J. Int. 139.
