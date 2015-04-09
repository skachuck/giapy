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

from giapy.map_tools import GridObject, rectifyMassBalance


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
    
    def __init__(self, earth, ice, grid, datalist=None):
        self.earth = earth
        self.ice = ice
        self.grid = grid

        if datalist is not None:
            for data in datalist:
                self.datalist = datalist
        else:
            self.datalist = []

        self.old_params = None
        self.old_chi2 = None
        self.priors = None

    def attach_data(self, data):
        self.datalist.append(data)

    def reset_data(self):
        self.datalist = []

    def remove_data(self, data):
        self.datalist.remove(data)

    def attach_esl(self, esl):
        self.esl = esl

    def set_out_times(self, out_times):
        self.out_times = out_times
        
    def leastsq(self, x0, func=None, args=None, priors=None, 
                    save_params=False, save_chi2=False, **kwargs):
        """Calculate the least squares minimum from starting point x0.

        Parameters
        ----------
        x0 - the initial guess vector
        arglist - 
        priors - list of parameter prior standard deviations 
        save_params - if True, save the param steps during optimization
        save_chi2 - if True, save the steps in chi2 during optimization
        **kwargs - see kwargs for scipy.optimize.leastsq
        """

        func = func or self.residualsEarth
        self.priors = priors
        self.old_params = [] if save_params else None
        self.old_chi2 = [] if save_chi2 else None

        m = leastsq(func, x0, args=(args,), Dfun=self.jacobian, 
                    col_deriv=1, **kwargs)
        
        if save_params: self.old_params = np.asarray(self.old_params)
        if save_chi2: self.old_chi2 = np.asarray(self.old_chi2)

        return m

    def residualsEarth(self, xs, arglist=None, verbose=False):
        """Calculate the residuals associated with stored data sources and
        earth parameters xs.
        """
        if not self.datalist:
            raise StandardError('self.datalist is empty. Use self.attach_data.')

        if arglist is None:
            self.earth.reset_params(*xs)
        else:
             self.earth.reset_params_list(xs, arglist)

        self.perform_convolution()
        
        res = []

        for data in self.datalist:
            res.append(data.residual(self, verbose=verbose))
        
        if self.priors is not None:
            res.append((xs-self.priors[:,0])/self.priors[:,1])

        res = np.concatenate(res)

        # If saving steps, save current step
        if inspect.stack()[1][3] != 'jacobian':
            if self.old_params is not None:
                self.old_params.append(self.earth.get_params())
            if self.old_chi2 is not None:
                self.old_chi2.append(res.dot(res))

        return res

    def residualsIce(self, xs, namelist=None, verbose=False):
        if not self.datalist:
            raise StandardError('self.datalist is empty. Use self.attach_data.')

        # Ascertain size of alteration
        
        self.ice.updateAreas(xs)
        self.perform_convolution()
        res = []
            
        for data in self.datalist:
            res.append(data.residual(self, verbose=verbose))
        
        if self.priors is not None:
            res.append((xs-self.priors[:,0])/self.priors[:,1])

        res = np.concatenate(res)

        # If saving steps, save current step
        if inspect.stack()[1][3] != 'jacobian':
            if self.old_params is not None:
                self.old_params.append(self.earth.get_params())
            if self.old_chi2 is not None:
                self.old_chi2.append(res.dot(res))

        return res

    def jacobian(self, xs, arglist=None, func=None, func_args=None,
                    eps_f=5.e-5):
        """Calculate the jacobian associated with stored data sources and
        parameters xs, with function evaluation error eps_f (default 5e-11).
        """
        func = func or self.residualsIce

        jac = []
        xs = np.asarray(xs)
        for i, x in enumerate(xs):
            # Determine the separation to use
            # Optimal one-pt separation is (eps_f*f/f'')^(1/2) ~ sqrt(eps_f)*x
            # Optimal two-pt separation is (eps_f*f/f''')^(1/3) ~ cbrt(eps_f)*x
            h = np.zeros(len(xs))
            h[i] = (eps_f**(1./3.))*max(x, 0.5)

            # Evaluate the function
            # One-pt
            #f1 = rebound_2d_earth_res(xs...)
            # Two-pt
            f1 = func(xs-h, arglist)
            f2 = func(xs+h, arglist)

            # Difference
            # One-pt
            #(f2-f1)/h
            # Two-pt
            jac.append((f2-f1)*0.5/h[i])

        # put them together
        jac = np.asarray(jac)
        # reset the function to initial value
        trash = func(xs, arglist)
        return jac

    def mcmc(self, N):
        # priors

        # observations
        @pm.deterministic
        def emerge(viscs):
            return self.data_vec(viscs)
        obs = pm.Normal('obs', emerge, observed=True)

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

        if emergeCorr:
            # Perform meltwater correction
            if hasattr(self, 'esl'): self.mw_corr()

        if verbose: print 'Convolution time: {0}s'.format(time.clock()-time_start)

    def mw_corr(self, esl=None):
        """Apply the meltwater correction to transform uplift to emergence."""
        self.esl = esl or self.esl
            
        eslcorr = self.esl(self.out_times)
        self.uplift = self.uplift + eslcorr[:, np.newaxis, np.newaxis]

class GiaSimGlobal(object):
    def __init__(self, earth, ice, grid=None):

        self.earth = earth
        self.ice = ice

        self.nlon, self.nlat = ice.shape
        
        # The grid used is a cylindrical projection (equispaced lat/lon grid
        # unless otherwise specified)
        if grid is not None:
            self.grid = grid
        else:
            self.grid = GridObject(mapparam={'projection': 'cyl'}, 
                                    shape=ice.shape)

        self.harmTrans = spharm.Spharmt(self.nlon, self.nlat, legfunc='stored')

    def performConvolution(self, out_times=None, ntrunc=None, topo=None,
                             paleotopo=False, verbose=False):  
        """Convolve an ice load and an earth response model in fft space.
        Calculate the uplift associated with stored earth and ice model.
        
        Parameters
        ----------
        out_times - an array of times at which to caluclate the convolution.
                    (default is to use previously stored values).
        ntrunc : int
           The truncation number of the spherical harmonic expansion. Default
           is from the earth model, must be <= ice model's and < grid.nlat
        t_rel - the time relative to which uplift is considered (default present)
                (None for no relative). Must be in out_times.
        emergeCorr : Bool
            Apply any attached corrections to uplift to get emergence
        """
 
        earth = self.earth
        ice = self.ice
 
        Nrem = 1                                # number of intermediate steps

        # Resolution
        ntrunc = ntrunc or ice.nlat-1
        if ntrunc >= self.nlat:
           raise ValueError('ntrunc must be < grid.nlat')
        ms, ns = spharm.getspecindx(ntrunc)     # the list of degrees, m, and
                                                # order numbers, n. Sizes of
                                                # (ntrunc+1)*(ntrunc+2)/2

        # Calculate earth response to correct order number
        #earth.calcResponse(ntrunc)
        
        # Store out_times
        out_times = out_times or self.out_times
        self.out_times = out_times
        if out_times is None:
           raise ValueError('out_times is not set')
 
        # Make sure t_rel is in out_times
        if t_rel is not None and t_rel not in out_times:
            raise ValueError('t_rel must be in out_times')
                
        # Initialize the uplift array
        uplift_f = np.zeros((len(out_times), (ntrunc+1)*(ntrunc+2)/2), dtype=complex)
        
        # Use progressbar to track calculation
        if verbose:
           try:
               widgets = ['Convolution: ', Bar(), ' ', ETA()]
               pbar = ProgressBar(widgets=widgets, maxval=len(ice.times)+1)
               pbar.start()
           except NameError:
               ImportError('progressbar not loaded')

        # Convolve each ice stage to the each output time.
        # Primary loop: over ice load changes.
        i=0
        for ice0, t0, ice1, t1 in ice.pairIter():
            # Find ice change between stages.
            dice = ice1 - ice0

            # Take/put water equiv ice change from/into ocean as water un/load.
            if topo is not None:
                # Construct paleo-ocean surface at t1 for mass redistribution.
                if paleotopo:
                    self.harmTrans.spectogrid(uplift_f[t1 == out_times, :])
                    # Add uplift and geoid to get ocean surface right.
                    paleo = topo + upl + geo
                else:
                    paleo = topo
            dice = rectifyIceMassBalance(dice*0.9, self.grid, paleo)

            # Transform load change into spherical harmonics.
            loadChangeSpec = self.harmTrans.gridtospec(dice)/Nrem
            
            # Check for mass conservation.
            if np.abs(dice[0]/dice.max()) >= 0.01:
                print("Ice change at time {0} doesn't conserve mass.".format(t0))
            # N.B. the n=0 load should be zero in cases of glacial isostasy, as 
            # mass is conserved during redistribution.

            # Secondary loop: over output times.
            for inter_time in np.linspace(t0, t1, Nrem, endpoint=False):
                # Perform the time convolution for each output time
                for t_out in out_times[out_times <= inter_time]:
                    t_dur = (inter_time-t_out)
                    respArray = earth.getResp(t_dur)
                    # Vertical Deformations
                    #respArray = respArray[:,0]+respArray[:,1]
                    # Horizontal Deformations
                    respArray = respArray[:,4]+respArray[:,5]
                    # 0.3 accounts for density difference between ice and rock
                    uplift_f[t_out == out_times, :] += 0.3 *\
                               loadChangeSpec * respArray[ns]
                               #np.tile(delta_ice, (respArray.shape[1],1))*\
                               #respArray[ns]
                               
            if verbose: pbar.update(i+1)
            i+=1
        if verbose: pbar.finish()
 
        return uplift_f 
