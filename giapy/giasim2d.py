import numpy as np
#import pymc as pm
import time
import inspect

import spharm

from scipy.optimize import leastsq

try:
    from progressbar import ProgressBar, Percentage, Bar, ETA
except:
    pass

from giapy.map_tools import GridObject, redistributeOcean, sealevelChangeByMelt,\
                    volumeChangeLoad, sealevelChangeByUplift, oceanUpliftLoad

from giapy import GITVERSION, timestamp

class GiaSim(object):
    """
    
    Calculate and store Glacial Isostacy Simulation results, and compare
    with data. Must be called with an earth model (earth), an ice model (ice)
    and a grid (grid). To add a data source to be used in residual finding and
    interpolation, use GiaSim.attach_data.

    Parameters
    ----------
    earth    : giapy.earth_tools.EarthModel
        The earth model to be used in the simulation
    ice      : giapy.ice_tools.IceModel
        The ice model to be used in the simulation
    grid     : giapy.map_tools.GridObject
        The map associated with the earth and ice models
    datalist : list (optional)
        A list of giapy.data_tools objects to compare to results of simulation.
        Can be added later using GiaSim.attach_data

    Attributes
    ----------
    earth
    ice
    grid
    datalist
    esl
    priors
    old_params
    old_chi2

    Example
    -------
    Instantiate the GiaSim object
    >>> ice = giapy.ice_tools.icehistory.load(u'./IceModels/eur_ice.p')
    >>> earth = giapy.earth_tools.earth_two_d.EarthTwoLayer(1, 10)
    >>> earth.calc_taus(128)
    >>> grid = giapy.map_tools.GridObject(mapparm=giapy.plot_tools.eur_map_param)
    >>> data = giapy.data_tools.emerge_data.load(u'./Data/emergence_data/\
                                                    eur_emerge_data.p')

    >>> sim = giapy.GiaSim(earth, ice, grid, [data])
    >>> sim.attach_esl(giapy.data_tools.meltwater.gen_eustatic())
    >>> sim.set_out_times(np.arange(16, -1, -1))

    Use the GiaSim object to calculate uplift. Results are stored in
    sim.uplift with size (len(sim.out_times), sim.grid.shape)
    >>> sim.perform_convolution()
    """
    
    def __init__(self, earth, ice, grid):
        self.earth = earth
        self.ice = ice
        self.grid = grid

    def perform_convolution(self, out_times=None, emergeCorr=True, 
                            t_rel=0, verbose=False):  
        """Convolve an ice load and an earth response model in fft space.
        Calculate the uplift associated with stored earth and ice model.
        
        Parameters
        ----------
        out_times - an array of times at which to caluclate the convolution.
                    (default is to use previously stored values).
        t_rel - the time relative to which uplift is considered (defaul present)
                (None for no relative)
        emergeCorr : Bool
            Apply any attached corrections to uplift to get emergence
        """
        time_start = time.clock()

        earth = self.earth
        ice = self.ice

        N = earth.N                             # use the resolution in earth
        Nrem = 1                                # number of intermediate steps
    
        out_times = out_times or self.out_times
        self.out_times = out_times

        # Make sure t_rel is in out_times
        if t_rel is not None and t_rel not in out_times:
            raise ValueError('t_rel must be in out_times')
        
        # Fourier transform the ice_hist
        #TODO Use pyfftw instead.
        ice_stages = ice.fft(N, self.grid)
        
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
        
        # The resolution correction
        res = float(N)/ice.N
        shape = (np.ceil(res*ice.shape[0]), np.ceil(res*ice.shape[1]))
        # Retransform the uplift
        # The normalization needs to be corrected for each dimension (N/ice.N)**2
        uplift = np.real(np.fft.ifft2(uplift_f, s=[N, N]))*(res**2)
    
        # Calculate uplift relative to t_rel (default, present)
        if t_rel is not None: 
            uplift = uplift[np.where(out_times==t_rel)] - uplift 

        self.grid.update_shape(shape)
    
        # Correctly grid the uplift array by removing the fourier padding
        self.uplift = uplift[:, :shape[0], :shape[1]]

        if verbose: print 'Convolution time: {0}s'.format(time.clock()-\
                                                                time_start)

    def mw_corr(self, esl=None):
        """Apply the meltwater correction to transform uplift to emergence."""
        self.esl = esl or self.esl
            
        eslcorr = self.esl(self.out_times)
        self.uplift = self.uplift + eslcorr[:, np.newaxis, np.newaxis]

def performConvolution(self, earth, ice, grid, outTimes=None):  
    """Convolve an ice load and an earth response model in fft space.
    Calculate the uplift associated with stored earth and ice model.
    
    Parameters
    ----------
    out_times - an array of times at which to caluclate the convolution.
                (default is to use previously stored values).
    t_rel - the time relative to which uplift is considered (defaul present)
            (None for no relative)
    emergeCorr : Bool
        Apply any attached corrections to uplift to get emergence
    """
    time_start = time.clock()

    N = earth.N                             # use the resolution in earth
    DENICE      = 0.934          # g/cc
    DENWAT      = 0.999          # g/cc
    DENSEA      = 1.029          # g/cc
    GSURF       = 982.2          # cm/s^2
    DYNEperM    = DENSEA*GSURF*1e2
    NREM = 1                                # number of intermediate steps

    # Make sure t_rel is in out_times
    if t_rel is not None and t_rel not in out_times:
        raise ValueError('t_rel must be in out_times')
    
    # Fourier transform the ice_hist
    #TODO Use pyfftw instead.
    ice_stages = ice.fft(N, grid)
    
    # Initialize the uplift array
    uplift_f = np.zeros((len(outTimes), N, N), dtype=complex)
    
    # Convolve each ice stage to the each output time
    for ice0, t0, ice1, t1 in \
                      zip(ice_stages, ice.times, ice_stages[1:], ice.times[1:]):
        delta_ice = (ice0 - ice1)/Nrem
        for inter_time in np.linspace(t0, t1, NREM, endpoint=False):
            # Perform the time convolution for each output time
            for t_out in outTimes[outTimes <= inter_time]:
                t_dur = (inter_time-t_out)
                uplift_f[t_out == out_times, :, :] += DYNEperM *\
                                    delta_ice * earth.get_resp(t_dur)
    
    # The resolution correction
    res = float(N)/ice.N
    shape = (np.ceil(res*ice.shape[0]), np.ceil(res*ice.shape[1]))
    # Retransform the uplift
    # The normalization needs to be corrected for each dimension (N/ice.N)**2
    uplift = np.real(np.fft.ifft2(uplift_f, s=[N, N]))*(res**2)

    grid.update_shape(shape)

    # Correctly grid the uplift array by removing the fourier padding
    uplift = uplift[:, :shape[0], :shape[1]]

    return uplift
