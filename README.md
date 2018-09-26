# giapy: Glacial Isostatic Adjustment in PYthon

This is an opensource python package in development for copmuting
the Glacial Isostatic Adjustment of the surface of the earth in response to
shifting surface loads and comparing it to geophysical data. 

This release provides a scaled, normalized relaxation method for computing elastic love numbers (Kachuck, 2017).

In future releases, I will include methods for the viscoelastic relaxation and solving the sea level equation. If you are interested in having a look, these can currently be found in the development branch.

## Installation
Run ```$ python setup.py build && python setup.py install``` to register the command line tools ```giapy-ellove``` and ```giapy-velove```.

## Use
Try ```$ giapy-ellove 100```, which will output the first 100 elastic love numbers h', l', k', suitable for input into, e.g., [REAR](https://github.com/danielemelini/rear) (Melini et al., 2014). See ```$ giapy-ellove -h``` to customize use.

Similarly, you can run ```$ giapy-velove 10``` for the decay spectra of the three love numbers for the first 10 order numbers.

To perform the Sea Level Equation benchmarks from Martinec et al. (2018), run ```$ python tests/sle-test.py```.

## Dependencies
[numpy](http://www.numpy.org) : Numerical computing package.

[numba](https://numba.pydata.org) : (optional) Provides a just in time compiler for faster computation. The easiest way to get this package is to use the [Continuum Analytics Anaconda python distribution](http://www.anaconda.com), with which numba is installed by default.

## Citation
Please cite this code as "Samuel B. Kachuck (2017) *giapy: Glacial Isostatic Adjustment in PYthon* (1.0.0) [Source code] [https://github.com/skachuck/giapy](https://github.com/skachuck/giapy)."

# Licensing
This work is distributed under the MIT license ('LICENSE').

## References
Cathles, L.M. (1975). *The Viscosity of the Earth's Mantle.* Princeton University Press. Princeton, NJ. 

Kachuck, S.B. and Cathles, L.M. (2017, September). *Normalized relaxation method for efficient computation of elastic Love numbers.* Oral presentation at the 1st circular Workshop on Glacial Isostatic Adjustment and Elastic Deformation, Reykjavik, Iceland.

Martinec, Z, Klemann, V, van der Wal, W, Riva, REM, Spada, G, Sun, Y, Melini, D, Kachuck, SB, Barletta, V, Simon, K, A, G & James, T (2018), 'A benchmark study of numerical implementations of the sea-level equation in GIA modelling' Geophysical Journal International, vol 215, no. 1, pp. 389-414. DOI: 10.1093/gji/ggy280

Melini D., Gegout P., Spada G, King M. (2014) REAR - a regional ElAstic Rebound calculator.
User manual for version 1.0, available on–line at: [http://hpc.rm.ingv.it/rear](http://hpc.rm.ingv.it/rear).

&copy; 2017 S.B. Kachuck
