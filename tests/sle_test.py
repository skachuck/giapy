"""
sle_test.py
Author: Samuel B. Kachuck
Date: Sep 19, 2017

Benchmark the sea level equation.

Test cases from https://geofjv.troja.mff.cuni.cz/GIABenchmark.

"""

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

def gen_icehistory(spatial='L1', evolution='T1', tstep=0.02, **kwargs):
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

    return PersistentIceHistory(load, metadata)

def gen_sstopo(spatial='B0', sigb=26):
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
    Lons, Lats = np.meshgrid(np.linspace(-np.pi, np.pi, NLON),
                             np.linspace(-np.pi/2, np.pi/2, NLAT))
    thetab, phib, bmax, b0 = topo_spatial_cases[spatial]
    # convert colat to lat
    thetab = 90 - thetab
    # convert degrees to radians
    thetab, phib, sigb = np.radians([thetab, phib, sigb])

    # delta is angular distance from center of load
    delta = haversine(Lats, thetab, Lons, phib, r=1, radians=True)

    sstopo = bmax - b0*np.exp(-delta**2/2./sigb**2)
    return sstopo

def benchmark_interpers(sim, benchmark, ):
    """
    Generate interpolators from sim grid to benchmark test profiles.

    Parameters
    ----------
    sim - 
    benchmark - the result of sim.performConvolution
    """

    Lat = sim.ice.Lat*180/np.pi
    Lon = sim.ice.Lon*180/np.pi
    
    upl = RectBivariateSpline(Lon[0],Lat[:,0], 
                                      sim.harmTrans.spectogrd(benchmark.upl[-1].copy()).T)

    ut, vt = sim.harmTrans.getgrad(benchmark.hor[-1].copy())
    u =  RectBivariateSpline(Lon[0],Lat[:,0], ut.T)
    v =  RectBivariateSpline(Lon[0],Lat[:,0], vt.T)

    geo = RectBivariateSpline(Lon[0],Lat[:,0], 
                                      sim.harmTrans.spectogrd(benchmark.geo[-1].copy()).T)
    sstopo = RectBivariateSpline(Lon[0],Lat[:,0], 
                                      (benchmark.sstopo[0]-benchmark.sstopo[-1]).T)

    esl = benchmark.esl.array[-1]
    
    return upl, u, v, geo, sstopo, esl

def lonlatev(sim, lon=None, lat=None):
    """
    Helper function to parse profile coordinates.
    """
    Lat = sim.ice.Lat*180/np.pi
    Lon = sim.ice.Lon*180/np.pi
    
    if lon is not None:
        lonev, latev = lon*np.ones_like(Lat[:,0]), Lat[:,0]
        x = 90 - Lat[:,0]
        lonev, latev, x = lonev[::LATFAC], latev[::LATFAC], x[::LATFAC]
        
    elif lat is not None:
        lonev, latev = 180+Lon[0], (90-lat)*np.ones_like(Lon[0])
        lonev[lonev>=180] -= 360
        x = 180 + Lon[0]

    else:
        raise ValueError('must specify lon or lat of profile')
        
    return lonev, latev, x

def write_result(sim, benchmark, figname, lon=None, lat=None, drctry='./'):
    """
    Write out the result for a circle of longitude or latitude.
    """

    upl, u, v, geo, sstopo, esl = benchmark_interpers(sim, benchmark)
    
    lonev, latev, x = lonlatev(sim, lon, lat)
        
    header = '''  col1: longitude (figY_uvf.dat) and colatitude (figZ_uvf.dat) in degrees
  col2: vertical displacement (meter)
  col3: th-component of horizontal displ. (meter)
  col4: ph-component of horizontal displ. (meter)
  col5: gravitational potential increment (SI units)
  col6: Sea-surface variations wrt h_UF (metres)
  col7: Sea-level equation (metres)'''

    np.savetxt(drctry+'{}_SBK.dat'.format(figname),
        np.vstack([x,
                   upl.ev(lonev, latev), 
                   v.ev(lonev, latev), 
                   u.ev(lonev, latev),
                   -geo.ev(lonev, latev)*9.815,
                   geo.ev(lonev, latev)+esl,
                   sstopo.ev(lonev,latev)]).T,
           header=header)

def plot_profiles(sim, benchmark, lat=None, lon=None, data=None, diff=False):
    """
    Plot the profiles of a circle of longitude or latitude
    """
    
    upl, u, v, geo, sstopo, esl = benchmark_interpers(sim, benchmark)
    lonev, latev, x = lonlatev(sim, lon, lat)

    fig, axs = plt.subplots(2,3, figsize=(12,4))
    if not diff: 
        axs[0,0].plot(x, upl.ev(lonev,latev))
        axs[0,1].plot(x, v.ev(lonev,latev))
        axs[0,2].plot(x, u.ev(lonev,latev))
        axs[1,0].plot(x, -geo.ev(lonev,latev))
        axs[1,1].plot(x, sstopo.ev(lonev,latev))
        axs[1,2].plot(x, geo.ev(lonev,latev)+esl)
        
        if data is not None:
            data = np.loadtxt(data).T
            axs[0,0].plot(data[0], data[1], '--')
            axs[0,1].plot(data[0], data[2], '--')
            axs[0,2].plot(data[0], data[3], '--')
            axs[1,0].plot(data[0], data[4]/9.815, '--')
            axs[1,1].plot(data[0], data[6], '--')
            axs[1,2].plot(data[0], data[5], '--')

    else:
        data = np.loadtxt(data).T

        if lon is None: 
            lonev = data[0]
            latev = lat
        else:
            latev = data[0]
            lonev = lon

            axs[0,0].plot(data[0], upl.ev(lonev,latev)-data[1], '--')
            axs[0,1].plot(data[0], v.ev(lonev,latev)-data[2], '--')
            axs[0,2].plot(data[0], u.ev(lonev,latev)-data[3], '--')
            axs[1,0].plot(data[0], -geo.ev(lonev,latev)-data[4]/9.815, '--')
            axs[1,1].plot(data[0], sstopo.ev(lonev,latev)-data[6], '--')
            axs[1,2].plot(data[0], geo.ev(lonev,latev)-data[5], '--')

    return plt.gca()
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Benchmark the sea level equation')
    parser.add_argument('benchlist', nargs='+')
    parser.add_argument('--ntrunc', type=int, default=128, 
                            help=('Order number to truncate earth response,'
                                  'default 128, maximum 256.'))
    parser.add_argument('--dir', type=str, default='./',
                    help='directory into which to write results, default ./')

    args = parser.parse_args()

    try:
        benchlist = args.benchlist
    except:
        benchlist = []

    ntrunc = args.ntrunc
    drctry = args.dir

    earth = giapy.earth_tools.earthSphericalLap.SphericalEarth()
    earth.loadLoveNumbers('mod_M3-L70-V01', drctry=giapy.MODPATH+'/data/earth/')

    topoB1 = gen_sstopo('B1') 
    topoB2 = gen_sstopo('B2')
    topoB3 = gen_sstopo('B3')

    if 'all' in benchlist:
        benchlist = ['A', 'C2', 'D1', 'D2', 'E1', 'E2', 'D3', 'F1']

    if 'A' in benchlist:
        iceL1T1 = gen_icehistory('L1', 'T1', tstep=0.02)
        simA = giapy.giasim.GiaSimGlobal(earth, iceL1T1)
        benchmarkA = simA.performConvolution(out_times=np.linspace(11, 0, 12),
                                ntrunc=ntrunc)
        uplA = benchmarkA.upl[-1].copy()
        horA = benchmarkA.hor[-1].copy()
        geoA = benchmarkA.geo[-1].copy()

        # jmin = 2, so zero out n=0 and n=1
        ms, ns = giapy.giasim.spharm.getspecindx(NLAT-1)
        uplA[ns==0] = 0
        uplA[ns==1] = 0
        horA[ns==0] = 0
        horA[ns==1] = 0
        geoA[ns==0] = 0
        geoA[ns==1] = 0

        np.savetxt(drctry+'A_fig10_SBK.dat',
                np.vstack([90-simA.grid.Lat[:,0],
                           simA.harmTrans.spectogrd(uplA)[:,0], 
                           simA.harmTrans.getgrad(horA)[1][:,0], 
                           simA.harmTrans.spectogrd(geoA)[:,0]]).T,
                           header='Colat(deg)\tUplift(m)\tHorizontal(m)\tGeoid(m)')

        del iceL1T1, simA, benchmarkA

    if 'C2' in benchlist: 
        iceL2T1 = gen_icehistory('L2', 'T1', tstep=0.02)
        simC2 = giapy.giasim.GiaSimGlobal(earth, iceL2T1, topo=topoB1)
        benchmarkC2 = simC2.performConvolution(out_times=iceL2T1.times,
                        eliter=5, ntrunc=ntrunc, bathtub=True)

        fignums= ['10', '11', '12', '13']
        figprops= [{'lon':75}, {'lat':25}, {'lon':-40}, {'lat':100}]
        for prop, num in zip(figprops, fignums):
            write_result(simC2, benchmarkC2, 'C2_fig{}'.format(num), drctry=drctry, **prop)

        del iceL2T1, simC2, benchmarkC2

    if 'D1' in benchlist:
        iceL2T1 = gen_icehistory('L2', 'T1', tstep=0.02)
        simD1 = giapy.giasim.GiaSimGlobal(earth, iceL2T1, topo=topoB1)
        benchmarkD1 = simD1.performConvolution(out_times=iceL2T1.times,
                                eliter=5, ntrunc=ntrunc)

        fignums= ['10', '11', '12', '13']
        figprops= [{'lon':75}, {'lat':25}, {'lon':-40}, {'lat':100}]
        for prop, num in zip(figprops, fignums):
            write_result(simD1, benchmarkD1, 'D1_fig{}'.format(num), drctry=drctry, **prop)

        del iceL2T1, simD1, benchmarkD1

    if 'D2' in benchlist:
        iceL2T2 = gen_icehistory('L2', 'T2', tstep=0.02)
        simD2 = giapy.giasim.GiaSimGlobal(earth, iceL2T2, topo=topoB1)
        benchmarkD2 = simD2.performConvolution(out_times=iceL2T2.times,
                                eliter=5, ntrunc=ntrunc)

        fignums= ['10', '11', '12', '13']
        figprops= [{'lon':75}, {'lat':25}, {'lon':-40}, {'lat':100}]
        for prop, num in zip(figprops, fignums):
            write_result(simD2, benchmarkD2, 'D2_fig{}'.format(num), drctry=drctry, **prop)

        del iceL2T2, simD2, benchmarkD2

    if 'E1' in benchlist:
        iceL2T2 = gen_icehistory('L2', 'T2', tstep=0.02)
        simE1 = giapy.giasim.GiaSimGlobal(earth, iceL2T2, topo=topoB2)
        benchmarkE1 = simE1.performConvolution(out_times=iceL2T2.times,
                                eliter=20, ntrunc=ntrunc)

        fignums= ['10', '11', '12', '13']
        figprops= [{'lon':75}, {'lat':25}, {'lon':25}, {'lat':35}]
        for prop, num in zip(figprops, fignums):
            write_result(simE1, benchmarkE1, 'E1_fig{}'.format(num), drctry=drctry, **prop)

        del iceL2T2, simE1, benchmarkE1

    if 'E2' in benchlist:
        iceL3T2 = gen_icehistory('L3', 'T2', tstep=0.02)
        simE2 = giapy.giasim.GiaSimGlobal(earth, iceL3T2, topo=topoB3)
        benchmarkE2 = simE2.performConvolution(out_times=iceL3T2.times,
                                eliter=20, ntrunc=ntrunc)

        fignums= ['10', '11', '12', '13']
        figprops= [{'lon':75}, {'lat':25}, {'lon':25}, {'lat':35}]
        for prop, num in zip(figprops, fignums):
            write_result(simE2, benchmarkE2, 'E2_fig{}'.format(num), drctry=drctry, **prop)

        del iceL3T2, simE2, benchmarkE2

    if 'D3' in benchlist:
        iceL3T2 = gen_icehistory('L3', 'T2', tstep=0.02)
        simD3 = giapy.giasim.GiaSimGlobal(earth, iceL3T2, topo=topoB3)
        benchmarkD3 = simD3.performConvolution(out_times=iceL3T2.times,
                                eliter=20, ntrunc=ntrunc, bathtub=True)

        fignums= ['10', '11', '12', '13']
        figprops= [{'lon':75}, {'lat':25}, {'lon':25}, {'lat':35}]
        for prop, num in zip(figprops, fignums):
            write_result(simD3, benchmarkD3, 'D3_fig{}'.format(num), drctry=drctry, **prop)

        del iceL3T2, simD3, benchmarkD3

    if 'F1' in benchlist:
        Aearth = (4*np.pi*6371**2)
        iceL3T2 = gen_icehistory('L3', 'T2', tstep=0.02)
        topo = topoB3.copy()
        del topoB1, topoB2
        i = 0
        while True:
            simF1 = giapy.giasim.GiaSimGlobal(earth, iceL3T2, topo=topo)
            benchmarkF1 = simF1.performConvolution(out_times=iceL3T2.times,
                                eliter=20, ntrunc=ntrunc)

            dtopo = benchmarkF1.sstopo[-1] - topoB3
            err = simF1.grid.integrate(np.abs(dtopo))/Aearth
            print('Error at iter {}: {}'.format(i, err))
            if err < 1e-2 or i >= 10:
                break
            else:
                topo -= 0.5*dtopo
                del dtopo, benchmarkF1
            i += 1

        fignums= ['10', '11', '12', '13']
        figprops= [{'lon':75}, {'lat':25}, {'lon':25}, {'lat':35}]
        for prop, num in zip(figprops, fignums):
            write_result(simF1, benchmarkF1, 'F1_fig{}'.format(num), drctry=drctry, **prop)
