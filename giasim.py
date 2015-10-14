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

from .map_tools import GridObject, sealevelChangeByMelt,\
                    volumeChangeLoad, sealevelChangeByUplift, oceanUpliftLoad,\
                    floatingIceRedistribute

from . import GITVERSION, timestamp

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
                            verbose=False, eliter=5, nrem=1):  
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
        NREM        = nrem           # number of intermediate steps


        earth = self.earth
        ice = self.ice
        grid = self.grid

        # Resolution
        ntrunc = ntrunc or ice.nlat-1
        if ntrunc >= self.nlat:
           raise ValueError('ntrunc must be < grid.nlat')
        ms, ns = spharm.getspecindx(ntrunc)     # the list of degrees, m, and
                                                # order numbers, n. Sizes of
                                                # (ntrunc+1)*(ntrunc+2)/2.
        
        # Store out_times
        if out_times is None:
            out_times = self.out_times
        else:
            out_times = out_times
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
        gravObserver = GravObserver(out_times, ntrunc, ns)
        loadObserver = HeightObserver(remTimes, ice.shape, 'dLoad')
        wloadObserver = HeightObserver(remTimes, ice.shape, 'dwLoad')
        geoObserver = GeoidObserver(out_times, ntrunc, ns)

        eslUplObserver = TotalUpliftObserver(remTimes, ntrunc, ns)
        eslGeoObserver = GeoidObserver(remTimes, ntrunc, ns)
        topoObserver = HeightObserver(remTimes, ice.shape, 'topo')
        eslObserver = EslObserver(remTimes)
        velObserver = VelObserver(out_times, ntrunc, ns)
        rslObserver = HeightObserver(remTimes, ice.shape, 'sstopo')

        observerDict = GiaSimOutput(self)
        observerDict.addObserver('upl'   , uplObserver)
        observerDict.addObserver('hor'   , horObserver)
        observerDict.addObserver('grav'  , gravObserver)
        observerDict.addObserver('load'  , loadObserver)
        observerDict.addObserver('wload' , wloadObserver)
        observerDict.addObserver('vel'   , velObserver)
        observerDict.addObserver('geo'   , geoObserver)

        observerDict.addObserver('eslUpl', eslUplObserver)
        observerDict.addObserver('eslGeo', eslGeoObserver)
        observerDict.addObserver('topo'  , topoObserver)
        observerDict.addObserver('esl'   , eslObserver)
        observerDict.addObserver('sstopo', rslObserver)


        # Use progressbar to track calculation if desired.
        if verbose:
           try:
               widgets = ['Convolution: ', Bar(), ' ', ETA()]
               pbar = ProgressBar(widgets=widgets, maxval=len(ice.times)+1)
               pbar.start()
           except NameError:
               raise ImportError('progressbar not loaded')

        for o in observerDict:
            o.loadStageUpdate(ice.times[0], sstopo=topo)

        esl = 0                 # Equivalent sea level assumed to start at 0.

        # Convolve each ice stage to the each output time.
        # Primary loop: over ice load changes.
        for icea, ta, iceb, tb in ice.pairIter():
            # Determine the water load redistribution for ice, uplift, and
            # geoid changes between ta and tb,
            if topo is not None:
                # Get index for starting time.
                nta = observerDict['eslUpl'].locateByTime(ta)
                # Collect the solid-surface topography at beginning of step.
                Ta = observerDict['sstopo'].array[nta]

                # Redistribute the ocean by change in ocean floor / surface.
                upla = observerDict['eslUpl'].array[nta]
                uplb = observerDict['eslUpl'].array[nta+1]
                geoa = observerDict['eslGeo'].array[nta]
                geob = observerDict['eslGeo'].array[nta+1]
                dU = self.harmTrans.spectogrd(uplb-upla)
                dG = self.harmTrans.spectogrd(geob-geoa)
                dhwBarU = sealevelChangeByUplift(dU-dG, Ta, grid)
                dhwU = oceanUpliftLoad(dhwBarU, Ta, dU-dG)

                # Update the solid-surface topography with uplift / geoid.
                Tb = Ta + dU - dG - dhwBarU
                esl += dhwBarU
                dLoad = dhwU
                dwLoad = dhwU                       # Save the water load

                # Redistribute ice, consistent with current floating ice. 
                dILoad, dhwBarI = floatingIceRedistribute(icea, iceb, Tb, grid,
                                                            DENICE/DENSEA)

                # Combine loads from ocean changes and ice volume changes.
                dLoad += dILoad
                esl += dhwBarI
                dwLoad += volumeChangeLoad(dhwBarI, Tb+DENICE/DENSEA*iceb)
                Tb -= dhwBarI
                

                # Calculate instantaneous (elastic and gravity) responses to
                # the load shift and redistribute ocean accordingly.
                # Note: WE DO NOT CURRENTLY RECHECK FOR FLOATING ICE LOADS.
                if eliter:
                    # Get elastic and geoid response to the water load.
                    elResp, geoResp = earth.getResp(0.0)[np.meshgrid(ns,[0,4])]
                    elResp *= 1e-2
                    #TODO make this a not hard-coded number (do in earth model?)
                    geoResp /= -982.22*100


                    # Find the elastic uplift in response to stage's load
                    # redistribution.
                    dUel = self.harmTrans.spectogrd(DYNEperM*elResp*\
                                self.harmTrans.grdtospec(dLoad))
                    dGel = self.harmTrans.spectogrd(DYNEperM*geoResp*\
                                self.harmTrans.grdtospec(dLoad))

                    dhwBarUel = sealevelChangeByUplift(dUel-dGel, Tb, grid)
                    dhwUel = oceanUpliftLoad(dhwBarUel, Tb, dUel-dGel)

                    Tb = Tb + dUel - dhwBarUel
                    esl += dhwBarUel
                    dLoad = dLoad + dhwUel
                    dwLoad += dhwUel

                    # Iterate elastic responses until they are sufficiently small.
                    for i in range(eliter):
                        # Need to save elasticuplift  and geoid  at each iteration to
                        # compare to previous steps for convergence.
                        dUelp = self.harmTrans.spectogrd(DYNEperM*elResp*\
                                    self.harmTrans.grdtospec(dhwUel))
                        dGelp = self.harmTrans.spectogrd(DYNEperM*geoResp*\
                                    self.harmTrans.grdtospec(dhwUel))

                        dhwBarUel = sealevelChangeByUplift(dUelp-dGelp, Tb, grid)
                        dhwUel = oceanUpliftLoad(dhwBarUel, Tb, dUelp-dGelp)

                        # Correct topography
                        Tb = Tb + dUelp - dGelp - dhwBarUel
                        esl += dhwBarUel
                        dLoad = dLoad + dhwUel
                        dwLoad += dhwUel

                        percChange = np.mean(np.abs(dUelp-dGelp))/\
                                     np.mean(np.abs(dUel-dGel))

                        if  percChange <= 1e-2:
                            break
                        else:
                            dUel = dUel + dUelp
                            dGel = dGel + dGelp
                            continue

                for o in observerDict:
                    # Topography and load for time tb are updated and saved.
                    o.loadStageUpdate(tb, dLoad=dLoad, 
                                      topo=Tb+iceb*(Tb + DENICE/DENSEA*iceb>=0), 
                                      esl=esl, dwLoad=dwLoad, sstopo=Tb)

            else:
                dLoad = (iceb-icea)*DENICE/DENSEA
                Tb = None

                for o in observerDict:
                    # Topography and load for time tb are updated and saved.
                    o.loadStageUpdate(tb, dLoad=dLoad)


            

            # Transform load change into spherical harmonics.
            loadChangeSpec = self.harmTrans.grdtospec(dLoad)/NREM
            
            # Check for mass conservation.
            massConCheck = np.abs(loadChangeSpec[0]/loadChangeSpec.max())
            if  verbose and massConCheck>= 0.01:
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
        # 1/100 makes the response in m uplift / dyne ice
        return (respArray[self.ns,0] + respArray[self.ns,1])/100

class TotalHorizontalObserver(AbstractEarthGiaSimObserver):
    def isolateRespArray(self, respArray):
        # 1/100 makes the response in m displacement / dyne ice
        return (respArray[self.ns,2] + respArray[self.ns,3])/100

class GeoidObserver(AbstractEarthGiaSimObserver):
    def isolateRespArray(self, respArray):
        # Divide the negative potential by PREM surface gravity,
        # 982.22 cm/s^2, to get the geoid shift. (negative because when the
        # potential at the surface decreases, the equipotential surface
        # representing the ocean must have risen.)
        #TODO make this a not hard-coded number (do in earth model?)
        # 1e-2 makes the response in m displacement / dyne ice
        return -respArray[self.ns,4]/982.22*1e-2

class GravObserver(AbstractEarthGiaSimObserver):
    def isolateRespArray(self, respArray):
        # 1e3 makes the response in miligals / dyne ice
        return respArray[self.ns,5]*1e3

class VelObserver(AbstractEarthGiaSimObserver):
    def isolateRespArray(self, respArray):
        # 3.1536e8 makes the response in mm/yr / dyne ice
        return respArray[self.ns,6]*3.1536e8

class MOIObserver(AbstractEarthGiaSimObserver):
    pass

class AngularMomentumObserver(AbstractEarthGiaSimObserver):
    pass

class HeightObserver(AbstractGiaSimObserver):
    def __init__(self, outTimes, iceShape, name):
        self.initialize(outTimes, iceShape)
        self.name = name

    def initialize(self, outTimes, iceShape):
        self.array = np.zeros((len(outTimes), 
                                iceShape[0], iceShape[1]))
        self.outTimes = outTimes

    def loadStageUpdate(self, tout, **kwargs):
        if self.name in kwargs.keys():
            self.update(tout, kwargs[self.name])

    def update(self, tout, load):
        if tout not in self.outTimes:
            return
        n = self.locateByTime(tout)
        self.array[n] = load

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

