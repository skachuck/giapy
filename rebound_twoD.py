import numpy as np
import time
from scipy.interpolate import RectBivariateSpline

import giapy.ice_tools.icehistory as icehistory
import giapy.earth_tools.earth_two_d as earth_two_d
import giapy.data_tools.emergedata as emergedata
            
def perform_convolution(earth, ice, out_times=[0], t_rel=0, verbose=False): 
    #TODO Make this a class? Attach observers, like out_times? uplift?
    """Convolve an ice load and an earth response model in fft space.
    
    Parameters
    ----------
    earth - an object that has procedure earth.get_resp(t_dur)
    ice - an obect that has ice.fft(), ice.times
    out_times - an array of times at which to caluclate the convolution
    t_rel - the time relative to which uplift is considered (defaul present)
            (None for no relative)
    """
    time_start = time.clock()
    N = earth.N                             # use the resolution in earth
    Nrem = 1                                # number of intermediate steps

    # Make sure t_rel is in out_times
    if t_rel is not None and t_rel not in out_times:
        raise ValueError('t_rel must be in out_times')
    
    # Fourier transform the ice_hist
    ice_stages = ice.fft(N)
    
    # Initialize the uplift array
    uplift_f = np.zeros((len(out_times), N, N), dtype=complex)
    
    # Convolve each ice stage to the each output time
    for ice0, t0, ice1, t1 in \
                      zip(ice_stages, ice.times, ice_stages[1:], ice.times[1:]):
        delta_ice = (ice0 - ice1)/Nrem
        for inter_time in np.linspace(t0, t1, Nrem, endpoint=False):
            # Perform the time convolution for each output time
            for t_out in out_times[out_times <= inter_time]:
                t_dur = (inter_time-t_out)
                # 0.3 accounts for density difference between ice and rock
                uplift_f[t_out == out_times, :, :] += 0.3 *\
                                    delta_ice * earth.get_resp(t_dur)
    
    # Retransform the uplift
    # The normalization needs to be corrected for each dimension (ice.N/N)**2
    uplift = np.real(np.fft.ifft2(uplift_f, s=[N, N]))/((ice.N/N)**2)

    # Calculate uplift relative to t_rel (default, present)
    if t_rel is not None: 
        uplift = uplift[np.where(out_times==t_rel)] - uplift 

    if verbose: print 'Convolution completed, '+str(time.clock()-time_start)+'s'
    #TODO make resolution correction to grid.

    return uplift[:,:ice.shape[0], :ice.shape[1]]
    
def interp3d_emergence(uplift, data, out_times, verbose=False):
    """Interpolate uplift surfaces (xyz data at a specific t) to data locations 
    (non-grid) and data times (between times calculated). 
    
    Uses progressive linear interpolations: first the uplift at each outputted 
    time is interpolated to the data locations in data.locs, then they are
    interpolated to the data times in each location.
    
    Parameters
    ----------
    uplift (array-like) - size (times, lon, lat) array of uplift surfaces
    data - data whose data.locs are the locations to interpolate to.
    out_times - the times for the first index of the uplift array (should be
            of uplift eventually, yes?)
    """
    time_start = time.clock()
    ##########################################
    # STUFF TO FIX HEEEEEEERE!!!!!!!!!!!!!
    #N = np.shape(uplift)[-1]    
    N = uplift[0].shape
    # TODO These should be gotten from somewhere, right? uplift.grid??
    #X = np.arange(0, 4910000, 10000) ; Y = np.arange(0, 4710000, 10000)    
    X = np.linspace(0, 4900000, num=N[1], endpoint=True)
    Y = np.linspace(0, 4700000, num=N[0], endpoint=True)
    ##########################################
    
    # interp_data will be an array of size (N_output_times, N_locations)
    # for use in interpolating the calculated emergence to the locations and
    # times at which there are data in data
    interp_data = []
    # Interpolate the calculated uplift at each time on the Lat-Lon grid
    # to the data locations.
    for uplift_at_a_time in uplift:
        #interp_func = scipy.interpolate.interp2d(X, Y, uplift_at_a_time)
        #interp_emerge.append(np.array([interp_func(loc[0], loc[1]) for loc in emerge_data.locs]).flatten())
        interp_func = RectBivariateSpline(X, Y, uplift_at_a_time.T)
        # TODO data.locs currently as lon,lat list. Need Array. Need mapping
        interp_data.append(interp_func.ev(data.locs[:,0], data.locs[:,1]))
    interp_data = np.array(interp_data).T
    
    calc_vector = []
    # Interpolate the calculated uplifted at each time and data location
    # to the times of the data location.
    for interp, loc in zip(interp_data, data):
        calc_vector.append(np.interp(loc['data_dict']['times'],
        out_times[::-1], interp[::-1]))
    
    # flatten the array    
    calc_vector = np.array([item for l in calc_vector for item in l])
    
    if verbose: print 'Interpolation time: '+str(time.clock()-time_start)+'s'

    return calc_vector


def rebound_2d_taus_res(taus, earth, ice, emerge_data, verbose=False):
    time_start = time.clock()
    earth.set_taus(taus)
    out_times = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) 
    uplift = perform_convolution(earth, ice, out_times=out_times, verbose=verbose)
    calc_vector = interp3d_emergence(uplift, emerge_data, out_times, verbose=verbose)
    
    if verbose:
        print 'Cost function evaluatation time: '+str(time.clock()-time_start)+'s'
    
    return (calc_vector-emerge_data.long_data)/10
    
def rebound_2d_earth_res(params, earth, ice, emerge_data, verbose=False):
    time_start = time.clock()
    out_times = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    
    earth.set_mantle(params[0])
    earth.set_asth(params[0])
    earth.set_lith(params[1])
    earth.calc_taus_from_earth(earth.N)

    uplift = perform_convolution(earth, ice, out_times=out_times, verbose=verbose)
    calc_vector = interp3d_emergence(uplift, emerge_data, out_times, verbose=verbose)
    
    if verbose: 
        print 'Cost function evaluatation time: '+str(time.clock()-time_start)+'s'
    
    sig = 0.1*np.array(emerge_data.long_data)

    return (calc_vector-emerge_data.long_data)/10
   
def rebound_2d_earth_jac(xs, eps_f=10e-16):
    jac = []
    for i, x in enumerate(xs):
        xs = np.asarray(xs)
        # Determine the separation to use
        # Optimal one-pt separation is (eps_f*f/f'')^(1/2) ~ sqrt(eps_f)*x
        # Optimal two-pt separation is (eps_f*f/f''')^(1/3) ~ cbrt(eps_f)*x
        h = np.zeros(len(xs))
        h[i] = np.sqrt(eps_f*x)

        # Evaluate the function
        # One-pt
        #f1 = rebound_2d_earth_res(xs...)
        # Two-pt
        f1 = rebound_2d_earth_res(xs-h)
        f2 = rebound_2d_earth_res(xs+h)

        # Difference
        # One-pt
        #(f2-f1)/h
        # Two-pt
        jac.append((f2-f1)*0.5/h[i])

    # put them together
    jac = np.asarray(jac)
    return jac
    
def open_everything():
    ice = icehistory.load(u'./IceModels/eur_ice.p')
    earth = earth_two_d.load(u'./EarthModels/earth64.p')
    europe_data = emergedata.load(u'./Data/emergence_data/eur_emergence_data.p')
    
    return ice, earth, europe_data

def calc_tilts(uplift, Lon, Lat, r=6371):
    # central difference in lat and lon, throw out edges
    du_lat = uplift[2:, 1:-1]-uplift[:-2, 1:-1]
    du_lon = uplift[1:-1, 2:]-uplift[1:-1, :-2]
    
    dLat = (Lat[2:, 1:-1]-Lat[:-2, 1:-1])
    dLon = (Lon[1:-1, 2:]-Lon[1:-1, :-2])

    # dLon, dLat = np.meshgrid(dlon, dlat)
    # convert from degrees to kilometers 
    dX = dLon*r*np.pi/180*np.cos(Lat[1:-1, 1:-1]*np.pi/180)
    dY = dLat*r*np.pi/180

    # derivatives with respect to lon/lat
    du_lon_dx = du_lon/dX
    du_lat_dy = du_lat/dY
    # convert to derivaties with respect to km
    tilt = np.sqrt(du_lon_dx**2+du_lat_dy**2)

    return tilt

