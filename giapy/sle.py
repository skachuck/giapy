"""
sle.py
Author: Samuel B. Kachuck
Date  : 08 01 2015

    Main module for the giapy package. It provides the class GiaSimGlobal,
    which contains the method for computing the glacial isostatic adjustment to
    an earth with a given rheological decay spectrum to an ice history model.

Methods
-------
configure_giasim

Classes
-------
GlobalSLE
GiaSimGlobal
GiaSimOutput

"""
from __future__ import division

import numpy as np
from scipy.optimize import root
import spharm
import subprocess

from giapy.map_tools import GridObject

from giapy import GITVERSION, timestamp, MODPATH, call, os

class GlobalSLE(object):
    def __init__(self, earth, ice, grid=None, topo=None):
        """
        Compute glacial isostatic adjustment on a globe.

        Paramaters
        ----------
        earth : <giapy.code.earth_tools.earthSpherical.SphericalEarth>
        ice   : <giapy.code.icehistory.IceHistory / PersistentIceHistory>
        grid  : <giapy.code.map_tools.GridObject>
        topo  : numpy.ndarray

        Methods
        -------
        performConvolution
            

        """

        self.earth = earth

        self.ice = ice
        self.nlat, self.nlon = ice.shape
        
        # The grid used is a cylindrical projection (equispaced lat/lon grid
        # unless otherwise specified)
        if grid is not None:
            self.grid = grid
        else:
            self.grid = GridObject(mapparam={'projection': 'cyl'}, 
                                    shape=ice.shape)

        self.topo = topo
        
        # Precompute and store harmonic transform coefficients, for
        # computational efficiency, but at a memory cost.
        self.harmTrans = spharm.Spharmt(self.nlon, self.nlat, legfunc='stored')

    def performConvolution(self,*args, **kwargs):
        """Legacy function, see compute"""
        self.compute(*args, **kwargs)

    def compute(self, out_times=None, ntrunc=None, topo=None,
                            verbose=False, eliter=5, nrem=1, massconerr=1e-2,
                            bathtub=False, intwriteout=None):  
        """Convolve an ice load and an earth response model in fft space.
        Calculate the uplift associated with stored earth and ice model.
        
        Parameters
        ----------
        out_times : an array of times at which to caluclate the convolution.
                    (default is to use previously stored values).
        ntrunc : int
           The truncation number of the spherical harmonic expansion. Default
           is from the earth model, must be <= ice model's and < grid.nlat
        topo   : array
            Topography on which to compute. If None (default), assumes a flat
            topography. Must be the same shapt as ice.
        verbose : boolean
            Display progress on computation. Default is False.
        eliter  : int
            The maximum number of iterations allowed to compute initial elastic
            response to redistributed load at each stage. If 0, instantaneous
            elastic response is not computed. Default 5.
        nrem   : int
            Number of removal stages between the provided ice stages
            (intermediate steps are interpolated linearly). Default 1.
        massconerr : float
            acceptable conservation of mass error, estimated by the relative
            magnitude of the zeroth harmonic to the largest load harmonic.
            Default 1e-2.
        bathtub : Boolean. Keep coastlines fixed, if true (Default: False).
        intwriteout(i, t, observerDict) : callable
            A function that can write out the currently completed stage of the
            sea level equation. It must accept the current index, the time
            associated with that index, and the observerDict, which contains
            all the computed quantities, see GiaSimOutput.
       
        Results
        -------
        observerDict : GiaSimOutput
            A dictionary whose keys are fields of interest, such as uplift
            ('upl'), geoid ('geo'), and solid surface topography ('sstopo'),
            computed on the input grid at out_times.
        """
 
        #TODO make these optional keyword arguments
        DENICE      = 931.   #934.           # kg/m^3
        DENWAT      = 1000.  #999.           # kg/m^3
        DENSEA      = 1000.  #1029.          # kg/m^3
        GSURF       = 9.815          # m/s^2
        DENP        = DENICE/DENSEA 
        NREM        = nrem           # number of intermediate steps


        earth = self.earth
        ice = self.ice
        grid = self.grid
        if topo is None and self.topo is not None:
            topo = self.topo
        if topo is not None:
            assert topo.shape == ice.shape, 'Topo and Ice must have the same shape'

        # Resolution
        ntrunc = ntrunc or min(earth.nmax, ice.nlat-1)
        assert ntrunc <= ice.nlat-1, 'ntrunc > ice.nlat-1'
        ms, ns = spharm.getspecindx(ice.nlat-1)
        # npad is the indices in the larger (padded) array of spherical
        # harmonics that correspond to the smaller (response) array.
        npad = (ns <= ntrunc)
        
        # Store out_times
        if out_times is None:
            out_times = self.out_times
        else:
            out_times = np.asarray(out_times)
        self.out_times = out_times
        assert out_times is not None, 'out_times is not set'
        
        # Calculate times of intermediate removal stages.
        diffs = np.diff(ice.times)
        addRemovalTimes = []
        for i in range(1, NREM+1):
            addRemovalTimes.append(ice.times[:-1]+i*diffs/NREM)
        addRemovalTimes = np.array(addRemovalTimes).flatten()
        remTimes = np.union1d(ice.times, addRemovalTimes)[::-1]
        calcTimes = np.union1d(remTimes, out_times)[::-1]
        calcTimes = out_times.copy()

        # Initialize output observer         
        observerDict = initialize_output(self, out_times, calcTimes, ice.nlat-1, 
                                            ntrunc, ns, ice.shape) 

        for o in observerDict:
            o.loadStageUpdate(ice.times[0], sstopo=topo)

        esl = 0                 # Equivalent sea level assumed to start at 0.
        eslI = 0                 # Equivalent sea level assumed to start at 0.
        eslU = 0                 # Equivalent sea level assumed to start at 0.

        elRespArray = earth.getResp(0.)
        ssResp = np.zeros_like(ns) 
        ssResp[npad] = observerDict['SS'].isolateRespArray(elRespArray)

        if topo is not None: Ts = topo.copy()

        # Convolve each ice stage to the each output time.
        # Primary loop: over ice load changes.
        for icea, ta, iceb, tb in ice.pairIter():
            ################### LOAD STAGE CALCULATION ###################
            # Determine the water load redistribution for ice, uplift, and
            # geoid changes between ta and tb,
            if topo is not None and not bathtub:
                # Get index for starting time.
                nta = observerDict['SS'].locateByTime(ta)
                # Collect the solid-surface topography at beginning of step.
                Ta = observerDict['sstopo'].array[nta]

                # Increment the grounded ice load on the old topography
                dIwh = (np.maximum(0, Ta+DENP*iceb) -
                                    np.maximum(0, Ta+DENP*icea)) 
 
                # Volume change of grounded ice.
                dVi = -grid.integrate(dIwh, km=False)
                # Spread this volume change over the ocean
                dhwBarI = sealevelChangeByMelt(dVi, Ta+DENP*iceb, grid)
                dwLoadI = volumeChangeLoad(dhwBarI, Ta+DENP*iceb)
             

                # Update the solid surface with the sea surface increment dSS
                ssa, ssb = observerDict['SS'].array[[nta, nta+1]] 
                dSS = self.harmTrans.spectogrd(ssb-ssa)
                dhwBarU = sealevelChangeByUplift(dSS, Ta+DENP*iceb, 
                                                        grid)
                dwLoadU = oceanUpliftLoad(dhwBarU, Ta+DENP*iceb, dSS)

                # Collect the solid-surface at tb so far
                Tb = Ta + dSS - dhwBarU - dhwBarI

                # Transfer water load of marine ice to water load
                dwLoadM = (iceb>0)*dSS*(Tb<0)
             
                # Locate (newly) floating ice minus (newly) grounded ice
                dCalved = DENP*iceb*np.logical_xor(
                            ((Tb+DENP*iceb)<0)*((Ta+DENP*iceb)>0),
                            ((Tb+DENP*iceb)>0)*((Ta+DENP*iceb)<0)) 

                # Spread floating ice over the ocean
                dVc = grid.integrate(dCalved, km=False)
                dhwBarC = sealevelChangeByMelt(dVc, Tb+DENP*iceb, grid)
                dwLoadC = volumeChangeLoad(dhwBarC, Tb+DENP*iceb)

                Tb -= dhwBarC

                dwLoad = dwLoadI + dwLoadU + dwLoadC - dwLoadM

          
                dILoad = dIwh - dCalved + dwLoadM
                dLoad = dwLoad + dILoad + dCalved 
                
                eslU += dhwBarU
                eslI += dhwBarI + dhwBarC
                esl += dhwBarU + dhwBarI + dhwBarC

                SMALL = 1e-17
               
                dSSel = np.zeros(grid.shape)
                dSSelp = self.harmTrans.spectogrd((ssResp)*\
                            self.harmTrans.grdtospec(dLoad))

                # Calculate instantaneous (elastic and gravity) responses to
                # the load shift and redistribute ocean accordingly. 
                for i in range(eliter):
                    dhwBarUel = sealevelChangeByUplift(dSSelp, 
                                                        Tb+DENP*iceb, grid)
                    dhwUel = oceanUpliftLoad(dhwBarUel, 
                                                Tb+DENP*iceb, dSSel)

                    # Locate (newly) floating ice minus (newly) grounded ice
                    dCalvedp = DENP*iceb*np.logical_xor(
                                (Tb+dSSelp-dhwBarUel+DENP*iceb<0)*(Tb+DENP*iceb>0),
                                (Tb+dSSelp-dhwBarUel+DENP*iceb>0)*(Tb+DENP*iceb<0))

                    # Spread floating ice over the ocean
                    dVc = grid.integrate(dCalvedp, km=False)
                    dhwBarC = sealevelChangeByMelt(dVc, Tb+dSSelp-dhwBarUel+DENP*iceb, grid)
                    dwLoadC = volumeChangeLoad(dhwBarC, Tb+dSSelp-dhwBarUel+DENP*iceb)

                    dwLoad += dhwUel + dwLoadC
                    dILoad -= dCalvedp
                    dLoad += dhwUel + dwLoadC - dCalvedp
                    dCalved += dCalvedp

                    Tb += dSSelp - dhwBarUel - dhwBarC

                    eslU += dhwBarUel
                    eslI += dhwBarC
                    esl += dhwBarUel + dhwBarC

                    err = np.mean(np.abs(dSSelp))/SMALL
                    if err <= massconerr:
                        break
                    else:
                        dSSel += dSSelp
                        SMALL = np.mean(np.abs(dSSel))
                        # Get elastic and geoid response to the water load.
                        # Find the elastic uplift in response to stage's load
                        # redistribution.
                        dSSelp = self.harmTrans.spectogrd((ssResp)*\
                                self.harmTrans.grdtospec(dhwUel))
                        

                if eliter:
                    observerDict['SS'].array[nta+1] += self.harmTrans.grdtospec(dSSel) 

                for o in observerDict:
                    # Topography and load for time tb are updated and saved.
                    o.loadStageUpdate(tb, dLoad=dLoad, 
                                      topo=Tb+iceb*(Tb + DENP*iceb>=0), 
                                      eslU=eslU, eslI=eslI, esl=esl, dwLoad=dwLoad, sstopo=Tb,
                                      diLoad=dILoad, dcLoad=dCalved)

            elif topo is not None and bathtub:
              
                # Get index for starting time.
                nta = observerDict['SS'].locateByTime(ta)
                # Collect the solid-surface topography at beginning of step.
                Ta = observerDict['sstopo'].array[nta]

                # Redistribute the ocean by change in ocean floor / surface.
                ssa, ssb = observerDict['SS'].array[[nta, nta+1]] 
                dSS = self.harmTrans.spectogrd(ssb-ssa)
                dhwBarU = sealevelChangeByUplift(dSS, topo, 
                                                        grid, bathtub)
                dhwU = oceanUpliftLoad(dhwBarU, topo, dSS,
                                            bathtub)

                # Update the solid-surface topography with uplift / geoid.
                Tb = Ta + dSS - dhwBarU
                esl += dhwBarU
                eslU += dhwBarU
                dLoad = dhwU.copy()
                dwLoad = dhwU.copy()                # Save the water load

             
                dILoad = (iceb - icea) * (topo > 0) * DENP 
                dhwBarI = -grid.integrate(dILoad) / grid.integrate(topo<0)
                dIwLoad = volumeChangeLoad(dhwBarI, topo, bathtub)

                # Combine loads from ocean changes and ice volume changes.
                dLoad += dILoad + dIwLoad
                esl += dhwBarI
                eslI += dhwBarI
                dwLoad += volumeChangeLoad(dhwBarI, topo,
                                            bathtub)
                Tb -= dhwBarI
                

                # Calculate instantaneous (elastic and gravity) responses to
                # the load shift and redistribute ocean accordingly.
                # Note: WE DO NOT CURRENTLY RECHECK FOR FLOATING ICE LOADS.
                if eliter:
                    # Get elastic and geoid response to the water load.
                    # Find the elastic uplift in response to stage's load
                    # redistribution.
                    dSSel = self.harmTrans.spectogrd((ssResp)*\
                                self.harmTrans.grdtospec(dLoad)) 

                    dhwBarUel = sealevelChangeByUplift(dSSel, 
                                                        topo,
                                                        grid, bathtub)
                    dhwUel = oceanUpliftLoad(dhwBarUel, 
                                                topo, dSSel,
                                                bathtub)

                    Tb = Tb + dSSel - dhwBarUel
                    esl += dhwBarUel
                    dLoad = dLoad + dhwUel
                    dwLoad += dhwUel

                    # Iterate elastic responses until they are sufficiently small.
                    for i in range(eliter):
                        # Need to save elastic uplift and geoid at each iteration
                        # to compare to previous steps for convergence.
                        dSSelp = self.harmTrans.spectogrd((ssResp)*\
                                    self.harmTrans.grdtospec(dhwUel))
                      
                     

                        dhwBarUel = sealevelChangeByUplift(dSSelp, 
                                                            topo, grid, bathtub)
                        dhwUel = oceanUpliftLoad(dhwBarUel, 
                                                    topo, dSSelp, bathtub)

                        # Correct topography
                        Tb = Tb + dSSelp - dhwBarUel
                        esl += dhwBarUel
                        dLoad = dLoad + dhwUel
                        dwLoad += dhwUel

                        # Truncation error from further iteration
                        err = np.mean(np.abs(dSSelp))/np.mean(np.abs(dSSel))
                        if err <= massconerr:
                            break
                        else:
                            dSSel = dSSel + dSSelp
                       
                            continue
                if eliter:
                    observerDict['SS'].array[nta+1] += self.harmTrans.grdtospec(dSSel) 

                for o in observerDict:
                    # Topography and load for time tb are updated and saved.
                    o.loadStageUpdate(tb, dLoad=dLoad, 
                                      topo=Tb+iceb*(Tb + DENP*iceb>=0), 
                                      esl=esl, eslI=eslI, eslU=eslU, dwLoad=dwLoad, sstopo=Tb,
                                      diLoad=dILoad)



            else:
                dLoad = (iceb-icea)*DENP
                Tb = None

                for o in observerDict:
                    # Topography and load for time tb are updated and saved.
                    o.loadStageUpdate(tb, dLoad=dLoad)

            # Transform load change into spherical harmonics.
            loadChangeSpec = self.harmTrans.grdtospec(dLoad)/NREM
            
            # Check for mass conservation.
            massConCheck = np.abs(loadChangeSpec[0])/np.abs(loadChangeSpec.max())
            if  verbose and massConCheck >= massconerr:
                print("Load at {0} doesn't conserve mass: {1}.".format(ta,
                                                                massConCheck))
            # N.B. the n=0 load should be zero in cases of glacial isostasy, as 
            # mass is conserved during redistribution.

            ################# RESPONSE STAGE CALCULATION #################
            # Secondary loop: over output times.
            for inter_time in np.linspace(tb, ta, NREM, endpoint=False)[::-1]:
                # Propagate response to current load increment to future times.
                for t_out in calcTimes[calcTimes < inter_time]:
                    respArray = earth.getResp(inter_time-t_out)
                    for o in observerDict:
                        o.respStageUpdate(t_out, respArray, 
                                            DENSEA*loadChangeSpec) 

            if intwriteout is not None:
                intwriteout(nta+1, tb, observerDict)
            if verbose:
                print('Stage {} completed, time {}\r'.format(nta+1, tb))

        # Don't keep the intermediate uplift stages for water redistribution
        #observerDict.removeObserver('eslUpl', 'eslGeo') 

        return observerDict

# Legacy class definition.
GiaSimGlobal = GlobalSLE

def configure_giasim(configdict=None):
    """
    Convenience function for setting up a GiaSimGlobal object.

    The function uses inputs stored in giapy/data/inputs/, which can be
    downloaded from pages.physics.cornell.edu/~skachuck/giainputs.tar.gz or
    using the script giapy/dldata.

    Parameters
    ----------
    configdict : dict
        A dictionary with keys 'earth' and 'ice' and values of the
        names of these inputs. The key 'topo' is optional, and defaults to a
        flat initial topography.

    sim        : GiaSimGlobal object
    """

    DEFAULTCONFIG = {'earth': '75km0p04Asth_4e23Lith',
                     'ice'  : 'AA2_Tail_nochange5_hightres_Pers_288_square',
                     'topo' : 'sstopo288'}

    
    configdict = configdict or DEFAULTCONFIG
    
    assert configdict.has_key('earth'), 'GiaSimGlobal needs earth specified'
    assert configdict.has_key('ice'), 'GiaSimGlobal needs ice specified'

    #ppath = os.path.dirname(os.path.split(__file__)[0])
    dpath = MODPATH + '/data/inputs/' 
    ename = configdict['earth']+'.earth'
    iname = configdict['ice']+'.ice'

    filecheck = os.path.isfile(dpath+ename) and os.path.isfile(dpath+iname)
    
    topo = configdict.get('topo', None)
    if topo is not None:
        tname = topo+'.topo'
        filecheck = filecheck and os.path.isfile(dpath+tname)
    
    if not filecheck:
        with open(MODPATH+'/data/inputfilelist', 'r') as f:
            filelist = f.read().splitlines()
        if ename in filelist and iname in filelist and tname in filelist:
            print('Downloading inputs')
            p = subprocess.Popen(os.path.join(MODPATH+'/dldata'), shell=True,
                                cwd=ppath)
            p.wait()
        else:
            raise IOError('One of the inputs you specified does not exist.')

    earth = np.load(dpath+ename)
    ice = np.load(dpath+iname)
    if topo is not None:
        topo = np.load(dpath+tname)[2]
    
    sim = GlobalSLE(earth=earth, ice=ice, topo=topo)

    return sim

def volumeChangeLoad(h, topo, bathtub=False):
    """Compute ocean depth changes for a topographic shift h, consistent with
    sloping topographies.
    
    Parameters
    ----------
    h : float
        The topographic shift.
    topo : np.ndarray
        The topography to shift. (Altered topography is T - h)

    Returns
    -------
    hw : np.ndarray
        An array with shape topo.shape whose maximum magnitude is h, with
        decreasing magnitudes along slopes newly submerged or uncovered.
    """

    if bathtub:
        hw = h*(topo < 0)
    else:
        if h > 0:
            hw = (h - np.maximum(0, topo)) * (h > topo)
        elif h < 0:
            hw = (np.maximum(topo, h)) * (topo < 0)
        else:
            hw = 0*topo

    return hw

def sealevelChangeByMelt(V, topo, grid, bathtub=False):
    """Find the topographic lowering that alters the ocean's volume by V.

    Because of changing coastlines, a eustatic increase (decrease) of h will
    generally change the volume of the ocean by more (less) than with a
    'bathtub' ocean model. This function uses a Newton method with an initial
    guess based on the 'bathtub' model.

    Parameters
    ----------
    V : float
        The volume by which to alter the ocean.
    topo : np.ndarray
        The topography to alter.
    grid : <GridObject>
        The grid object assists with integration.
    
    Returns
    -------
    h : float
        The topographic shift consistent with changing / sloping coastlines.
        Note that the new topography after this shift is T - h.
    """
    if V == 0:
        return 0
    # Get first guess of eustatic h.
    h0 = V / grid.integrate(topo < 0, km=False)

    if bathtub:
        return h0
    
    # Use scipy.optimize.root to minimize volume difference.
    Vexcess = lambda h: V - grid.integrate(volumeChangeLoad(h, topo), km=False)
    h = root(Vexcess, h0)

    return h['x'][0]

def oceanUpliftLoad(h, Ta, upl, bathtub=False):
    """Compute ocean depth changes for a topographic shift h, consistent with
    sloping topographies.

    Note that the resultant topography is Tb = Ta + upl - h.

    
    Parameters
    ----------
    h : float
        The topographic shift.
    Ta : np.ndarray
        The topography to shift.     
    upl : np.ndarray
        The uplift additionally shifting the topography. Note that uplift and
        geoid affect sea level oppositely (opposite sign).

    Returns
    -------
    hw : np.ndarray
        An array with shape topo.shape whose maximum magnitude is h, with
        decreasing magnitudes along slopes newly submerged or uncovered.

    """
    if bathtub: 
        return (h - upl)*(Ta<0)

    # The new topography
    Tb = Ta + upl - h
    #               Newly submerged.            Newly emerged.
    hw = (h - upl - np.maximum(Ta, 0)*(Ta>0))*(Tb<0) + Ta*(Tb>0)*(Ta<0)
    return hw

def sealevelChangeByUplift(upl, topo, grid, bathtub=False):
    """Find the topographic lowering that alters the ocean's volume by V.

    Because of changing coastlines, a eustatic increase (decrease) of h will
    generally change the volume of the ocean by more (less) than with a
    'bathtub' ocean model. This function uses a Newton method with an initial
    guess based on the 'bathtub' model.

    Parameters
    ----------
    upl : np.ndarray
        The uplift additionally shifting the topography.Note that uplift and
        geoid affect sea level oppositely (opposite sign).
    topo : np.ndarray
        The topography to alter.
    grid : <GridObject>
        The grid object assists with integration.
    
    Returns
    -------
    h : float
        The topographic shift consistent with changing / sloping coastlines.
        Note that the new topography after this shift is T + upl - h.
        Techincally, upl = uplift - geoid.
    """
    if np.allclose(upl, 0):
        return 0

    # Average ocean floor uplift, for initial guess.
    h0 = grid.integrate(upl*(topo<0), km=False)/grid.integrate(topo<0, km=False)

    if bathtub:
        return h0

    # Use scipy.optimize.root to minimize volume difference..
    Vexcess = lambda h: grid.integrate(oceanUpliftLoad(h, topo, upl), km=False)
    h = root(Vexcess, h0)

    return h['x'][0]


def floatingIceRedistribute(I0, I1, S0, S1, grid, denp=0.9077):
    """Calculate load and topographic shift due to ice height changes.

    Calculate the water-equivalent load changes due to changing from ice
    heights I0 to I1 on a solid-surface topography (height of solid earth, NOT
    ice, relative to sea level at t0) S0. The load accounts for the fact that
    when ice is not grounded, it represents a neutral water load, and 
    appropriately updates the 'groundedness' where necessary. 

    The updated solid surface,            S1 = S0 - dhwBar.
    Floating ice can be identified where  (S1 + denp*I1) < 0.
    Topography (to top of ice) is         T1 = S1 + I1*(S1 + denp*I1 >= 0).

    Parameters
    ----------
    I0, I1 : np.ndarrays
        Ice heights (from solid surface to top of ice) at times t0 and t1
    S0 : np.ndarray
        Solid surface topography (height of solid earth relative to sea level)
        at t0.
    grid : <GridObject>
        The grid object assists with integration.
    denp : float
        The ratio of densities of ice and water (default = 0.9077). Used in
        transforming ice heights to equivalent water heights.

    Returns
    -------
    dLoad : np.ndarray
        The total water load induced by ice height chagnes from I0 to I1,
        taking into account floating ice and mass redistribution.
    dhwBar : float
        The topographic shift associated with the mass transfer.
    """
    
    # Find the water-equivalent load change relative to sea level at t0.
    #       
    dIwh = np.maximum(0, S1+denp*I1) - np.maximum(0, S0+denp*I0)
    # Water-equivalent change in grounded ice from t0 to t1
    dIwh = denp*(I1*(S1+denp*I1>=0) - I0*(S0+denp*I0>=0))
    #dIwh = np.maximum(0, S0+denp*I1)*(I1>0) - denp*I0
    #               Change in height of floating ice
    dCalved = denp*(I1*(S1+denp*I1<=0)*(I1>=0) - I0*(S0+denp*I0<=0)*(I0>=0))

   
                    
    # The change in water volume of the ocean is opposite the change in
    # grounded ice.
    dVo = -grid.integrate(dIwh, km=False)
                                            
    # The water load associated with the volume change.
    dhwBar = sealevelChangeByMelt(dVo, S1+denp*I1, grid)
    dwLoad =  volumeChangeLoad(dhwBar, S1+denp*I1) -\
                    (dIwh<0)*(S1+denp*I1<0)*(S1+denp*I1)

                                                                                  
    return dIwh, dwLoad, dCalved, dhwBar

def initialize_output(sim, out_times, calcTimes, nmax, ntrunc, ns, shape):
    earth = sim.earth
    # Initialize the return object to include...
    # ... values desired at output times
    #   [1] Uplift
    uplObserver = earth.TotalUpliftObserver(out_times, nmax, ntrunc, ns)
    #   [2] Horizontal deformation
    horObserver = earth.TotalHorizontalObserver(out_times, nmax, ntrunc, ns)
    #   [3] Uplift velocities
    velObserver = earth.VelObserver(out_times, nmax, ntrunc, ns)
    #   [4] Geoid perturbations
    geoObserver = earth.GeoidObserver(out_times, nmax, ntrunc, ns)
    #   [5] Gravitational acceleration perturbations
    gravObserver = earth.GravObserver(out_times, nmax, ntrunc, ns) 

    # ... and values needed to perform the convolution
    #   [1] Uplift for ocean redistribution
    SeaSurfaceObserver = earth.SeaSurfaceObserver(calcTimes, nmax, ntrunc, ns)
    #   [2] Geoid for ocean redistribution
    eslGeoObserver = earth.GeoidObserver(calcTimes, nmax, ntrunc, ns)
    #   [3] Topography (to top of ice) to find floating ice
    topoObserver = HeightObserver(calcTimes, shape, 'topo')
    #   [4] Load (total water + ice load in water equivalent)
    loadObserver = HeightObserver(calcTimes, shape, 'dLoad')
    #   [5] Water load
    wloadObserver = HeightObserver(calcTimes, shape, 'dwLoad') 
    iloadObserver = HeightObserver(calcTimes, shape, 'diLoad') 
    cloadObserver = HeightObserver(calcTimes, shape, 'dcLoad') 
    #   [6] Solid surface topography for ocean redistribution
    rslObserver = HeightObserver(calcTimes, shape, 'sstopo')
    #   [7] Eustatic sea level, with average uplift and geoid over oceans.
    eslObserver = EslObserver(calcTimes) 
    eslIObserver = EslObserver(calcTimes, 'eslI') 
    eslUObserver = EslObserver(calcTimes, 'eslU') 

    observerDict = GiaSimOutput(sim)
    observerDict.addObserver('upl'   , uplObserver)
    observerDict.addObserver('hor'   , horObserver)
    observerDict.addObserver('grav'  , gravObserver)
    observerDict.addObserver('load'  , loadObserver)
    observerDict.addObserver('wload' , wloadObserver)
    observerDict.addObserver('iload' , iloadObserver)
    observerDict.addObserver('cload' , cloadObserver)
    observerDict.addObserver('vel'   , velObserver)
    observerDict.addObserver('geo'   , geoObserver)

    observerDict.addObserver('SS', SeaSurfaceObserver) 
    observerDict.addObserver('topo'  , topoObserver)
    observerDict.addObserver('esl'   , eslObserver)
    observerDict.addObserver('eslI'   , eslIObserver)
    observerDict.addObserver('eslU'   , eslUObserver)
    observerDict.addObserver('sstopo', rslObserver)
    return observerDict

class GiaSimOutput(object):
    """A container object for computations of glacial isostatic adjustment.

    The results of the GIA computation can be accessed via object attributes or
    via a dictionary (e.g., if iteration is desired). The object's __repr__
    gives the computed attributes.

    Parameters
    ----------
    inputs : anything
        anything desired to be stored as inputs. Typically, this will be a
        GiaSimGlobal object that contains the ice load, earth model, and other
        necessary objects for reproducing the computation.

    Methods
    -------
    addObserver - add an observer to the watchlist
    removeObserver - remove an observer from the watchlist
    transformObservers - transform each observer in the watchlist, using each's
        own transform function (must be provided by observer).


    Data
    ----
    GITVERSION : the hash of the current giapy git HEAD
    TIMESTAMP : the datetime of creation (calculation)
    """
    def __init__(self, inputs):
        self.GITVERSION = GITVERSION
        self.TIMESTAMP = timestamp()
        self.inputs = inputs
        self._observerDict = {}

    def __getitem__(self, key):
        return self._observerDict.__getitem__(key)

    def __iter__(self):
        return self._observerDict.itervalues() 

    def __repr__(self):
        retstr = ''
        retstr = 'GIA computed on {}\n'.format(self.TIMESTAMP)
        retstr += 'Has observers: '+', '.join(self._observerDict)
        return retstr

    def addObserver(self, name, observer):
        self._observerDict[name] = observer
        setattr(self, name, observer)

    def removeObserver(self, *names):
        for name in names:
            del self._observerDict[name]
            delattr(self, name)

    def transformObservers(self, inverse=False):
        for obs in self:
            obs.transform(self.inputs.harmTrans, inverse=inverse)


class AbstractGiaSimObserver(object):
    """The GiaSimObserver mediates between an earth model's response function
    and the convolved result. It needs to store the harmonic response

    initialize and update must be overwritten for class to work, all others 
    simply pass.
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
        raise NotImplemented

    def loadStageUpdate(self, *args, **kwargs):
        pass

    def respStageUpdate(self, *args, **kwargs):
        pass

    def update(self, tout, tdur, dLoad):
        raise NotImplemented

    def transform(self, trans, inverse=True):
        pass

    def locateByTime(self, time):
        if time not in self.outTimes:
            raise ValueError('time not in self.outTimes')
        return np.argwhere(self.outTimes == time)[0][0]

    def nearest_to(self, time):
        """Return a field from outTimes nearest to time.
        """
        idx = (np.abs(self.outTimes-time)).argmin()
        return self[idx]


class AbstractEarthGiaSimObserver(AbstractGiaSimObserver):
    """Abstract class for results in spherical harmonic space, updated during
    the response stage.

    Must implement isolateRespArray to pull proper response curve from the
    computed earth model.
    """
    def __init__(self, outTimes, nmax, ntrunc, ns):
        self.initialize(outTimes, nmax, ns)
        self.npad = (ns <= ntrunc)
        self.ns = ns[self.npad]
        self.spectral = True

    def initialize(self, outTimes, ntrunc, ns):
        self.array = np.zeros((len(outTimes), 
                               int((ntrunc+1)*(ntrunc+2)/2)), dtype=complex)
        self.outTimes = outTimes
        self.ns = ns

    def respStageUpdate(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def update(self, tout, respArray, dLoad):
        if tout not in self.outTimes:
            return
        n = self.locateByTime(tout)
        resp = self.isolateRespArray(respArray)
        self.array[n][self.npad] += resp * dLoad[self.npad]

    def transform(self, trans, inverse=True):
        if not inverse and self.spectral:
            self.array = trans.spectogrd(self.array.T).T
            self.spectral = False
        elif inverse and not self.spectral:
            self.array = trans.grdtospec(self.array.T).T
            self.spectral = True

    def isolateRespArray(self, respArray):
        raise NotImplemented()


class HeightObserver(AbstractGiaSimObserver):
    """General observer for heights computed on the real-space grid and updated
    during the loadStage.
    """
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
    def __init__(self, outTimes, name='esl'):
        self.array = np.zeros(len(outTimes))
        self.outTimes = outTimes
        self.name=name

    def loadStageUpdate(self, tout, **kwargs):
        if self.name in kwargs.keys():
            self.update(tout, kwargs[self.name])

    def update(self, tout, esl):
        if tout not in self.outTimes:
            return
        n = self.locateByTime(tout)
        self.array[n] = esl

