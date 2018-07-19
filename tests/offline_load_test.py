"""
sle_test.py
Author: Samuel B. Kachuck
Date: Sep 19, 2017

Benchmark the sea level equation.

Test cases from https://geofjv.troja.mff.cuni.cz/GIABenchmark.

"""
import os, sys, shutil

import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import giapy
import giapy.earth_tools.earthSphericalLap
from giapy.map_tools import haversine
from giapy.icehistory import PersistentIceHistory

# Each case is a tuple of (theta_0, phi_0, h_0)
#                         (cent colat, lon, height)
ice_spatial_cases = {'L1': (0, 0, 1500), 
                     'L2': (25, 75, 1500),
                     'L3': (25, 75, 500)}

# Each case is a tuple of (theta_b, phi_b, bmax, b0)
#                         (cent colat, lon, basin params)
topo_spatial_cases = {'B0': (0, 0, 0, 0),
                      'B1': (100, 320, 760, 1200),
                      'B2': (35, 25, 760, 1200),
                      'B3': (35, 25, 3800, 6000)}

LATFAC = 2
NLON, NLAT = 720, 360*LATFAC+1
#NLON, NLAT = 512, 1025

def sphericalload(Lons, Lats, lonc, latc, h0, alpha=10):
    """
    Return a spherical cap at lonc, latc on grid Lons, Lats.

    Parameters
    ----------
    Lons, Lats - 2D meshgrids of longitudes and latitudes
    lonc, latc - the longitude and latitude of spherical cap center
    h0 - the maximum height of the spherical cap
    alpha - the angular size of the cap
    """

    #theta0, phi0, h0 = ice_spatial_cases[spatial]
    
    alpha = np.radians(alpha)
    if alpha <1e-2: return np.zeros_like(Lons)

    # delta is angular distance from center of load
    delta = haversine(Lats, latc, Lons, lonc, r=1, radians=True)

    load = h0*np.sqrt((np.cos(delta) - np.cos(alpha)) /
                      (1. - np.cos(alpha))* (delta <= alpha)) 

    return load

def gen_offline_icehistory(spatial='L1', evolution='T1', tstep=0.02, **kwargs):
    """
    Generate a benchmark ice history, given spatial and evolution codes.

    Parameters
    ----------
    spatial - code for spatial distribution (see ice_spatial_cases)
        May be 'L1', 'L2', or 'L3'
    evolution - code for time evolution
        'T1' is a heaviside load initiated at 10 kyr
        'T2' is a linear increase of height and extent over 10 kyr (or t2)
        'T3' is a linear decrease of height and extent over 10 kyr (or t2)
    tstep - the time step of the ice load in kyr (default 0.02 kyr)
    h0    - the maximum height of the ice load (overrides ice_spatial_case)
    t2    - the time of final removal or growth (default 5)

    Returns
    -------
    icehistory - <giapy.icehistory.PersistentIceHistory>
    """

    drctry = kwargs.get('drctry', './')
    fbase = kwargs.get('fbase', 'icefile_t')
    fext = kwargs.get('fext', '.txt')
    def fname(t):
        return drctry+'icefiles/'+fbase+str(t)+fext
    os.mkdir(drctry+'icefiles')


    Lons, Lats = np.meshgrid(np.linspace(-np.pi, np.pi, NLON),
                             np.linspace(-np.pi/2, np.pi/2, NLAT))

    colatc, lonc, h0 = ice_spatial_cases[spatial]
    h0 = kwargs.get('h0', h0)
    t2 = kwargs.get('t2', 5)
    # t2 must be a valid timestep given tstep
    t2 = tstep*np.round(t2/tstep)

    # convert colat to lat
    latc = 90 - colatc
    # convert degrees to radians
    latc, lonc = np.radians([latc, lonc])

    if evolution=='T1':
        load = np.zeros((2, NLAT, NLON))
        load[1,:,:] = sphericalload(Lons, Lats, lonc, latc, h0)
        times = np.arange(0, 10+2*tstep, tstep)[::-1]

        metadata = {'Lon'               : Lons,
                    'Lat'               : Lats,
                    'nlat'              : NLAT,
                    'shape'             : Lons.shape,
                    '_alterationMask'   : np.zeros_like(Lats),
                    'areaProps'         : {},
                    'areaVerts'         : {},
                    'times'             : times,
                    'stageOrder'        : [0]+[1]*(len(times)-1),
                    'path'              : '',
                    'fnames'            : ['','']}

    if evolution=='T2':
        t1 = 15 

        nloadsteps = int((t1-t2)/tstep)
        load = np.zeros((nloadsteps+1, NLAT, NLON))
        times = np.arange(min(0, t2), t1+tstep, tstep)[::-1]

        for i, t in enumerate(np.arange(t2, t1+tstep, tstep)[::-1]):
            alpha = (t1-t)/(t1-t2)*10
            h = (t1-t)/(t1-t2)*h0
            load[i] = sphericalload(Lons, Lats, lonc, latc, h, alpha)
        
        metadata = {'Lon'               : Lons,
                    'Lat'               : Lats,
                    'nlat'              : NLAT,
                    'shape'             : Lons.shape,
                    '_alterationMask'   : np.zeros_like(Lats),
                    'areaProps'         : {},
                    'areaVerts'         : {},
                    'times'             : times,
                    'stageOrder'        : range(nloadsteps)+[nloadsteps-1]*(len(times)-nloadsteps),
                    'path'              : '',
                    'fnames'            : ['','']}
                    
    if evolution=='T3':
        t1 = 15

        nloadsteps = int((t1-t2)/tstep)
        load = np.zeros((nloadsteps+1, NLAT, NLON))
        times = np.arange(min(0, t2), t1+tstep, tstep)[::-1]

        for i, t in enumerate(np.arange(t2, t1+tstep, tstep)[::-1]):
            alpha = (t2-t)/(t2-t1)*10
            h = (t2-t)/(t2-t1)*h0
            load[i] = sphericalload(Lons, Lats, lonc, latc, h, alpha)
        
        metadata = {'Lon'               : Lons,
                    'Lat'               : Lats,
                    'nlat'              : NLAT,
                    'shape'             : Lons.shape,
                    '_alterationMask'   : np.zeros_like(Lats),
                    'areaProps'         : {},
                    'areaVerts'         : {},
                    'times'             : times,
                    'stageOrder'        : range(nloadsteps)+[nloadsteps-1]*(len(times)-nloadsteps),
                    'path'              : '',
                    'fnames'            : ['','']}

    tmpice = PersistentIceHistory(load, metadata)
    for t, stage in zip(tmpice.times, tmpice):
        np.savetxt(fname(t), stage)

    return tmpice

def gen_sstopo(nlon, nlat, spatial='B0', sigb=26):
    """
    Generate a circular exponential basin.

    Parameters
    ----------
    spatial - code for spatial distribution (see topo_spatial_cases)
        May be 'B0', 'B1', 'B2', 'B3'
    sigb    - the angular decay rate for exponential basin

    Returns
    -------
    sstopo - np.array(NLAT, NLON)
    """
    Lons, Lats = np.meshgrid(np.linspace(-np.pi, np.pi, nlon),
                             np.linspace(-np.pi/2, np.pi/2, nlat))
    thetab, phib, bmax, b0 = topo_spatial_cases[spatial]
    # convert colat to lat
    thetab = 90 - thetab
    # convert degrees to radians
    thetab, phib, sigb = np.radians([thetab, phib, sigb])

    # delta is angular distance from center of load
    delta = haversine(Lats, thetab, Lons, phib, r=1, radians=True)

    sstopo = bmax - b0*np.exp(-delta**2/2./sigb**2)
    return sstopo

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Benchmark the sea level equation') 
    parser.add_argument('--ntrunc', type=int, default=128, 
                            help=('Order number to truncate earth response,'
                                  'default 128, maximum 256.'))
    parser.add_argument('--dir', type=str, default='./',
                    help='directory into which to write results, default ./')
    parser.add_argument('--keep', default=False, action='store_true')

    args = parser.parse_args()

    try:
        benchlist = args.benchlist
    except:
        benchlist = []

    ntrunc = args.ntrunc
    drctry = args.dir

    earth = giapy.earth_tools.earthSphericalLap.SphericalEarth()
    earth.loadLoveNumbers('mod_M3-L70-V01', drctry=giapy.MODPATH+'/data/earth/')

    if True:

        onlineice = gen_offline_icehistory('L2', 'T2', tstep=0.5)
        offlineice = giapy.icehistory.OfflineIceHistory(np.arange(0,15.5,0.5)[::-1],
                        ['icefile_t{}.txt'.format(i) for i in np.arange(0,15.5,0.5)[::-1]],
                        (NLAT,NLON), 'icefiles/')

        print(offlineice.times) 
        print(offlineice.fnames)
        print('Offline ice load generated, {} files in icefile'.format(
                                                len(os.listdir('icefiles'))))
        offlinesim = giapy.giasim.GiaSimGlobal(earth, offlineice,
                                                topo=gen_sstopo(NLON, NLAT, 'B1'))

        onlinesim = giapy.giasim.GiaSimGlobal(earth, onlineice,
                                                topo=gen_sstopo(NLON, NLAT, 'B1'))

        onlinebenchmark = onlinesim.compute(out_times=onlineice.times,
                                eliter=5, ntrunc=ntrunc)

        offlinebenchmark = offlinesim.compute(out_times=offlineice.times,
                                eliter=5, ntrunc=ntrunc, verbose=True)

        print('Mean difference between final sstopo:{}'.format(
                np.mean(onlinebenchmark.sstopo[-1]-offlinebenchmark.sstopo[-1])))

        del onlinebenchmark

        def writeout(i, t, o):
            np.savetxt('output/topot{}.txt'.format(t), o.sstopo[i])

        offlinebenchmark = offlinesim.compute(out_times=offlineice.times,
                                eliter=5, ntrunc=ntrunc, intwriteout=writeout)

        print('Mean difference between final sstopo and written sstopo:{}'.format(
                            np.mean(np.loadtxt('output/topot0.0.txt') -
                            offlinebenchmark.sstopo[-1])))
 
        shutil.rmtree('icefiles')


