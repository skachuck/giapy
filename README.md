# giapy: Glacial Isostatic Adjustment in PYthon

This is an opensource python package in development for copmuting
the Glacial Isostatic Adjustment of the surface of the earth in response to
shifting surface loads and comparing it to geophysical data. 

This release provides a scaled, normalized relaxation method for computing the elastic and viscoelastic Love numers (Kachuck & Cathles, 2019) and the sea level equation (Martinec et al., 2018).

## Installation
The current version is built on python 2.7, as a key dependency (the spherical harmonics library) has not yet been released on python 3, although I am looking into alternatives. I highly recommend the [Continuum Analytics Anaconda python distribution](http://www.anaconda.com). Once this has been installed, install the necessary packages with
```$ conda config --add channels conda-forge```

```$ conda install numpy scipy matplotlib numba pyspharm basemap```

Now giapy's command line tools can be be installed ```$ python setup.py build && python setup.py install``` to register the command line tools ```giapy-ellove``` and ```giapy-velove```.

## Use
Try ```$ giapy-ellove 100```, which will output the first 100 elastic love numbers h', l', k', suitable for input into, e.g., [REAR](https://github.com/danielemelini/rear) (Melini et al., 2014). See ```$ giapy-ellove -h``` to customize use.

Similarly, you can run ```$ giapy-velove 10``` for the decay spectra of the three love numbers for the first 10 order numbers.

To perform the Sea Level Equation benchmarks from Martinec et al. (2018), run ```$ python tests/sle-test.py```.

## Dependencies
[numpy](http://www.numpy.org), scipy, matplotlib : Numerical computing and plotting packages.

[numba](https://numba.pydata.org) : Provides a just in time compiler for faster computation. The easiest way to get this package is to use the [Continuum Analytics Anaconda python distribution](http://www.anaconda.com), with which numba is installed by default.

[pyspharm](https://github.com/jswhit/pyspharm) : Python interface to the NCAR SPHEREPACK library.

[basemap](https://matplotlib.org/basemap) : A matplotlib toolkit for plotting and manipulating data on 2D projections.

## Citation
Please cite this code as "Samuel B. Kachuck (2017) *giapy: Glacial Isostatic Adjustment in PYthon* (1.0.0) [Source code] [https://github.com/skachuck/giapy](https://github.com/skachuck/giapy)."

# Licensing
This work is distributed under the MIT license ('LICENSE').

## References
Cathles, L.M. (1975). *The Viscosity of the Earth's Mantle.* Princeton University Press. Princeton, NJ. 

Kachuck, S.B. and Cathles, L.M. (2019). 'Benchmarked computation of time-domain viscoelastic Love numbers for adiabatic mantles.' Geophysical Journal International. DOI: [https://doi.org/10.1093/gji/ggz276](https://doi.org/10.1093/gji/ggz276)

Martinec, Z., Klemann, V., van der Wal, W., Riva, R.E.M., Spada, G., Melini, D., Simon, K.M., Blank, B., Sun, Y. A, G., & James, T., Barletta, V.R., Kachuck, S.B. (2018), 'A benchmark study of numerical implementations of the sea-level equation in GIA modelling' Geophysical Journal International, vol 215, no. 1, pp. 389-414. DOI: 10.1093/gji/ggy280

Melini D., Gegout P., Spada G, King M. (2014) REAR - a regional ElAstic Rebound calculator.
User manual for version 1.0, available on–line at: [http://hpc.rm.ingv.it/rear](http://hpc.rm.ingv.it/rear).

&copy; 2017 S.B. Kachuck
