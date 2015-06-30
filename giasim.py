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

from .map_tools import GridObject, redistributeOcean, sealevelChangeByMelt,\
                    volumeChangeLoad, sealevelChangeByUplift, oceanUpliftLoad

from . import GITVERSION, timestamp

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
        raise NotImplemented()
        #TODO Get residual calculations out of data containers
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
        raise NotImplemented()
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

        if emergeCorr:
            # Perform meltwater correction
            if hasattr(self, 'esl'): self.mw_corr()

        if verbose: print 'Convolution time: {0}s'.format(time.clock()-\
                                                                time_start)

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
                            verbose=False, eliter=5):  
        """Convolve an ice load and an earth response model in fft space.
        Calculate the uplift associated with stored earth and ice model.
        
        Parameters
        ----------
        out_times : an array of times at which to caluclate the convolution.
                    (default is to use previously stored values).
        ntrunc : int
           The truncation number of the spherical harmonic expansion. Default
           is from the earth model, must be <= ice model's and < grid.nlat
        emergeCorr : Bool
            Apply any attached corrections to uplift to get emergence
        """
 
        DENICE      = 0.934          # g/cc
        DENWAT      = 0.999          # g/cc
        DENSEA      = 1.029          # g/cc
        GSURF       = 982.2          # cm/s^2
        DYNEperM    = DENSEA*GSURF*1e2
        NREM        = 1              # number of intermediate steps


        earth = self.earth
        ice = self.ice
        grid = self.grid

        # Resolution
        ntrunc = ntrunc or ice.nlat-1
        if ntrunc >= self.nlat:
           raise ValueError('ntrunc must be < grid.nlat')
        ms, ns = spharm.getspecindx(ntrunc)     # the list of degrees, m, and
                                                # order numbers, n. Sizes of
                                                # (ntrunc+1)*(ntrunc+2)/2
        
        # Store out_times
        out_times = out_times or self.out_times
        self.out_times = out_times
        if out_times is None:
           raise ValueError('out_times is not set')
        
        # Calculate times of intermediate removal stages.
        diffs = np.diff(ice.times)
        addRemovalTimes = []
        for i in range(1, NREM+1):
            addRemovalTimes.append(ice.times[:-1]+i*diffs/NREM)
        addRemovalTimes = np.array(addRemovalTimes).flatten()
        remTimes = np.union1d(ice.times, addRemovalTimes)[::-1]
        calcTimes = np.union1d(remTimes, out_times)[::-1]
                 
        # Initialize the return object
        uplObserver = TotalUpliftObserver(out_times, ntrunc, ns)
        horObserver = TotalHorizontalObserver(out_times, ntrunc, ns)
        loadObserver = LoadObserver(remTimes, ice.shape)
        eslUplObserver = TotalUpliftObserver(remTimes, ntrunc, ns)
        eslGeoObserver = GeoidObserver(remTimes, ntrunc, ns)
        topoObserver = TopoObserver(remTimes, ice.shape)
        eslObserver = EslObserver(remTimes)

        observerDict = GiaSimOutput(self)
        observerDict.addObserver('upl'   , uplObserver)
        observerDict.addObserver('hor'   , horObserver)
        observerDict.addObserver('load'  , loadObserver)
        observerDict.addObserver('eslUpl', eslUplObserver)
        observerDict.addObserver('eslGeo', eslGeoObserver)
        observerDict.addObserver('topo'  , topoObserver)
        observerDict.addObserver('esl'   , eslObserver)

        #out_times = np.union1d(remTimes, out_times)[::-1]

        # Use progressbar to track calculation
        if verbose:
           try:
               widgets = ['Convolution: ', Bar(), ' ', ETA()]
               pbar = ProgressBar(widgets=widgets, maxval=len(ice.times)+1)
               pbar.start()
           except NameError:
               raise ImportError('progressbar not loaded')

        for o in observerDict:
            o.loadStageUpdate(ice.times[0], topo=topo)

        esl = 0

        # Convolve each ice stage to the each output time.
        # Primary loop: over ice load changes.
        i=0     # i counts loop number for ProgressBar.
        for icea, ta, iceb, tb in ice.pairIter():
            # Find ice change between stages.

            # Take/put water equiv ice change from/into ocean as water un/load.
            if topo is not None:
                # Get index for starting time.
                nta = observerDict['eslUpl'].locateByTime(ta)
                # Collect the topography at the beginning of the step.
                Ta = observerDict['topo'].array[nta] + iceb - icea

                # Make the ocean volume change based on the ice change.
                dM = (iceb - icea) * DENICE/DENSEA
                dhwBarI = sealevelChangeByMelt(-grid.integrate(dM, km=False), 
                            Ta, grid)
                dhwI = volumeChangeLoad(dhwBarI, Ta)

                # First correction to topography is applied.
                Tb = Ta - dhwBarI
                esl += dhwBarI

                # First load is ice change and melt change.
                dLoad = dM + dhwI

                if eliter:
                    # Redistribute ocean by change in ocean floor.
                    upla = observerDict['eslUpl'].array[nta]
                    uplb = observerDict['eslUpl'].array[nta+1]
                    geoa = observerDict['eslGeo'].array[nta]
                    geob = observerDict['eslGeo'].array[nta+1]
                    dU = self.harmTrans.spectogrd(uplb-upla)
                    dG = self.harmTrans.spectogrd(geob-geoa)
                    dhwBarU = sealevelChangeByUplift(dU, Tb, grid)
                    dhwU = oceanUpliftLoad(dhwBarU, Tb, dU)

                    # Correct topography and load with uplift.
                    Tb = Tb + dU - dhwBarU
                    esl += dhwBarU
                    dLoad = dLoad + dhwU

                    # Get elastic response to the meltwater and uplift load.
                    elResp = earth.getResp(0.0)[ns,0]/100

                    # Find the elastic uplift in response to stage's load
                    # redistribution.
                    dUel = self.harmTrans.spectogrd(DYNEperM*elResp*\
                                self.harmTrans.grdtospec(dLoad))
                    dhwBarUel = sealevelChangeByUplift(dUel, Tb, grid)
                    dhwUel = oceanUpliftLoad(dhwBarUel, Tb, dUel)

                    Tb = Tb + dUel - dhwBarUel
                    esl += dhwBarUel
                    dLoad = dLoad + dhwUel

                # Iterate elastic responses until they are sufficiently small.
                for i in range(eliter):
                    # Need to save elastic uplift at each iteration to compare
                    # to previous steps for convergence.
                    dUelp = self.harmTrans.spectogrd(DYNEperM*elResp*\
                                self.harmTrans.grdtospec(dhwUel))
                    dhwBarUel = sealevelChangeByUplift(dUelp, Tb, grid)
                    dhwUel = oceanUpliftLoad(dhwBarUel, Tb, dUelp)

                    # Correct topography
                    Tb = Tb + dUelp - dhwBarUel
                    esl += dhwBarUel
                    dLoad = dLoad + dhwUel

                    if np.mean(np.abs(dUelp))/np.mean(np.abs(dUel)) <= 1e-2:
                        break
                    else:
                        dUel = dUel + dUelp
                        continue

            else:
                dLoad = (iceb-icea)*DENICE/DENSEA
                Tb = None

            for o in observerDict:
                # Topography and load for time tb are updated and saved.
                o.loadStageUpdate(tb, dLoad=dLoad, topo=Tb, esl=esl)

            # Transform load change into spherical harmonics.
            loadChangeSpec = self.harmTrans.grdtospec(dLoad)/NREM
            
            # Check for mass conservation.
            massConCheck = np.abs(loadChangeSpec[0]/loadChangeSpec.max())
            if  massConCheck>= 0.01:
                print("Load at {0} doesn't conserve mass: {1}.".format(ta,
                                                                massConCheck))
            # N.B. the n=0 load should be zero in cases of glacial isostasy, as 
            # mass is conserved during redistribution.

            # Secondary loop: over output times.
            for inter_time in np.linspace(tb, ta, NREM, endpoint=False)[::-1]:
                # Perform the time convolution for each output time
                for t_out in calcTimes[calcTimes <= inter_time]:
                    respArray = earth.getResp(inter_time-t_out)
                    for o in observerDict:
                        o.respStageUpdate(t_out, respArray, 
                                            DYNEperM*loadChangeSpec)
                               
            if verbose: pbar.update(i+1)
            i+=1
        if verbose: pbar.finish()

        # Don't keep the intermediate uplift stages for water redistribution
        observerDict.removeObserver('eslUpl')

        return observerDict


class GiaSimOutput(object):
    def __init__(self, inputs):
        self.GITVERSION = GITVERSION
        self.TIMESTAMP = timestamp()
        self.inputs = inputs
        self._observerDict = {}

    def __getitem__(self, key):
        return self._observerDict.__getitem__(key)

    def __iter__(self):
        return self._observerDict.itervalues() 

    def addObserver(self, name, observer):
        self._observerDict[name] = observer

    def removeObserver(self, name):
        del self._observerDict[name]


class AbstractGiaSimObserver(object):
    """The GiaSimObserver mediates between an earth model's respons function
    and the convolved result. It needs to store the harmonic response 
    """
    def __init__(self):
        pass

    def __getitem__(self, key):
        return self.array.__getitem__(key)
        
    def __iter__(self):
        return self.array.__iter__()
    
    @property
    def shape(self):
        return self.array.shape

    def initialize(self, outTimes, ntrunc):
        self.array = np.zeros((len(out_times), 
                                (ntrunc+1)*(ntrunc+2)/2), dtype=complex)

    def loadStageUpdate(self, *args, **kwargs):
        pass

    def respStageUpdate(self, *args, **kwargs):
        pass

    def update(self, tout, tdur, dLoad):
        pass

    def untransform(self):
        pass

    def locateByTime(self, time):
        if time not in self.outTimes:
            raise ValueError('time not in self.outTimes')
        return np.argwhere(self.outTimes == time)[0][0]

class AbstractEarthGiaSimObserver(AbstractGiaSimObserver):
    def __init__(self, outTimes, ntrunc, ns):
        self.initialize(outTimes, ntrunc, ns)
    def initialize(self, outTimes, ntrunc, ns):
        self.array = np.zeros((len(outTimes), 
                                (ntrunc+1)*(ntrunc+2)/2), dtype=complex)
        self.outTimes = outTimes
        self.ns = ns

    def respStageUpdate(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def update(self, tout, respArray, dLoad):
        if tout not in self.outTimes:
            return
        n = self.locateByTime(tout)
        resp = self.isolateRespArray(respArray)
        self.array[n] += resp * dLoad

    def isolateRespArray(self, respArray):
        raise NotImplemented()

class TotalUpliftObserver(AbstractEarthGiaSimObserver):
    def isolateRespArray(self, respArray):
        # 1/100 makes the response in m uplift / m ice
        return (respArray[self.ns,0] + respArray[self.ns,1])/100

class TotalHorizontalObserver(AbstractEarthGiaSimObserver):
    def isolateRespArray(self, respArray):
        # 1/100 makes the response in m displacement / m ice
        return (respArray[self.ns,2] + respArray[self.ns,3])/100

class GeoidObserver(AbstractEarthGiaSimObserver):
    def isolateRespArray(self, respArray):
        return respArray[self.ns,4]

class MOIObserver(AbstractEarthGiaSimObserver):
    pass

class AngularMomentumObserver(AbstractEarthGiaSimObserver):
    pass

class LoadObserver(AbstractGiaSimObserver):
    def __init__(self, outTimes, iceShape):
        self.initialize(outTimes, iceShape)

    def initialize(self, outTimes, iceShape):
        self.array = np.zeros((len(outTimes), 
                                iceShape[0], iceShape[1]))
        self.outTimes = outTimes

    def loadStageUpdate(self, tout, **kwargs):
        if 'dLoad' in kwargs.keys():
            self.update(tout, kwargs['dLoad'])

    def update(self, tout, load):
        if tout not in self.outTimes:
            return
        n = self.locateByTime(tout)
        self.array[n] = load

class TopoObserver(AbstractGiaSimObserver):
    def __init__(self, outTimes, shape):
        self.initialize(outTimes, shape)

    def initialize(self, outTimes, shape):
        self.array = np.zeros((len(outTimes), 
                                shape[0], shape[1]))
        self.outTimes = outTimes

    def loadStageUpdate(self, tout, **kwargs):
        if 'topo' in kwargs.keys():
            self.update(tout, kwargs['topo'])

    def update(self, tout, topo):
        if tout not in self.outTimes:
            return
        n = self.locateByTime(tout)
        self.array[n] = topo

class EslObserver(AbstractGiaSimObserver):
    def __init__(self, outTimes):
        self.array = np.zeros(len(outTimes))
        self.outTimes = outTimes

    def loadStageUpdate(self, tout, **kwargs):
        if 'esl' in kwargs.keys():
            self.update(tout, kwargs['esl'])

    def update(self, tout, esl):
        if tout not in self.outTimes:
            return
        n = self.locateByTime(tout)
        self.array[n] = esl

