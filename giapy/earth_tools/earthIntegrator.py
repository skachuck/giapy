"""
earthIntegrator.py

    Module for integrating the response of a spherical, non-rotating,
    self-gravitating, viscoelastic earth to a spherical harmonic load.
    The response involves integrating the displacement, stress, and
    gravitational fields seperately for the elastic and viscous equations of
    motion through the mantle (from the Core Mantle Boudnary to
    the Surface) and stepping the viscous displacements in time.

    The mantle integration is a two-point boundary value problem, with three
    conditions on the CMB and three on the surface. It can be performed
    numerically using a shooting method, implemented in SphericalEarthShooter,
    or a relaxation method, implemented in SphericalEarthRelaxer. Both are
    called with an instance of giapy.earthParams.EarthParams, an array of
    integration points in the mantle, and an order number. The relaxation
    method requires an initial guess of the solution, which can be calculated
    using get_t0_guess (uses the shooting method).

    The integration through time is an initial value problem (no discplament at
    time 0), and is implemented through a simple stepper,
    integrateRelaxationDirect, or through Scipy's ode integrator,
    integrateRelaxationScipy. Each of these requires a mantle integration
    object (described above) and an output object, SphericalEarthOutput, that
    organizes the recording of solution points.

    NOTE ON UNITS:
    This module uses cgs units.

    Module Contents
    ---------------
    SphericalEarthOutput
    SphericalEarthRelaxer
    SphericalElasSMat
    SphericalViscSMat
    SphericalEarthShooter

    integrateRelaxationScipy
    integrateRelaxationDirect
    get_t0_guess
    propMatElas
    propMatVisc
    rk4Earth

    Author: Samuel B. Kachuck
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import ode, odeint
from numba import jit, void, int64, float64

#from giapy.numTools.solvde import solvde
from giapy.numTools.solvdeJit import solvde

        

def integrateRelaxationScipy(f, out, atol=1e-6, rtol=1e-5):
    """Use Scipy ode for surface response to harmonic load.

    Parameters
    ----------
    f : a function for the viscous velocities
    out : an output object
        Must have methods out.out and out.converged and data out.times.
    """
    earth = f.earthparams
    paramSurf = earth.getParams(1.)
    vislim = 1./(paramSurf['den']*paramSurf['grav']*f.alpha)
    nz = len(f.zarray)

    # Get the t=0 response elastic, and save it
    f(0, np.zeros(2*nz))
    out.out(0, 0, 0, f)

    r = ode(f).set_integrator('vode', method='adams')
    #r = ode(f).set_integrator('dop853')
    r.set_initial_value(y=np.zeros(2*nz), t=0)
    timeswrite = out.times
    dts = timeswrite[1:]-timeswrite[:-1]


    for dt in dts:
        r.integrate(r.t+dt)
        out.out(r.t, r.y[nz-1], r.y[2*nz-1], f)
        
        if (vislim+r.y[nz-1]<(atol+rtol*abs(r.y[nz-1]))):
            out.converged(-vislim)
            break
            
        # Check for convergence, but continue until next write step.
        # N.B. need to check for convergence because of apparent bouncing in
        # the solution. The surface oscillates due to numerical error.
        # TODO LOOK INTO THIS!!!


def integrateRelaxationDirect(f, out, eps=1e-7):
    """Directly integrate ode for surface response to harmonic load.

    Parameters
    ----------
    f : function for the viscous velocities
    out : output object
        Must have methods out.out and out.converged and data out.times.
    """

    # initialize earth model
    earth = f.earthparams
    paramSurf = earth.getParams(1.)
    vislim = 1./(paramSurf['den']*paramSurf['grav']*f.alpha)

    # Initialize hard-coded integration parameters
    secs_per_year = 3.1536e+7  # seconds in a year

    dt = np.array([0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, .1, .1, .1,
        0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2, 0.2,  0.2,  0.2,
        0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2, 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
        0.5,  0.5,  0.5,  0.5, 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1., 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2., 5.,
        5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5., 5.,  5.,  5.,  5.,  5.,  5.,  
        5.])

    dt = (secs_per_year*1e+3*dt).tolist()
    tmax = out.times[-1]
    tstep = dt.pop(0)

    t = 0
    nz = len(f.zarray)
    disps = np.zeros(2*nz)
    converged = False

    # Get the t=0 response elastic, and save it
    f(0, np.zeros(2*nz))
    out.out(0, 0, 0, f)

    for i in range(len(dt)):
        vels = f(t, disps)

        # Step the total viscous deformation
        disps = disps + vels*tstep

        if t in out.times:
            out.out(t, disps[nz-1], disps[2*nz-1], f)
            

            if converged:
                out.converged()
                #for tfill in out.times[out.times>t]:
                    #returnArray.append([0, -vislim, 0, 0, 0, uhconv])
                break
            
        # Check for convergence, but continue until next write step.
        # N.B. need to check for convergence because of apparent bouncing in
        # the solution. The surface oscillates due to numerical error.
        # TODO LOOK INTO THIS!!!
        if (t>=tmax) or (eps is not None and vislim+disps[nz-1]<eps):
            converged = True

        t += tstep
        tstep = dt.pop(0)

class SphericalEarthOutput(object):
    """Class for organizing the recording of solution points while calculating
    viscoelastic relaxation in response to surface load.

    Attributes
    ----------
    times : array
        Array of times to write, in seconds.

    outArray : array (len(out.times), 6)
        Stores Ue, Uv, Ve, Vv, phi1, g1 at times.

    maxind : int
        The index of the most recent successful write step.

    Methods
    -------
    out (t, uv, f) : write surface element of uv and f.y at time t to outArray.
    converged () : fills outArray[maxind:] with outArray[maxind].
        Useful for terminating the ODE integration when a convergence criterion
        is met.
        
    """
    def __init__(self, times=None):
        secs_per_year = 3.1536e+7  # seconds in a year
        if times is not None:
            self.times = times[:]
        else:
            self.times = np.array([0., 0.2, 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 
                      10., 12., 13., 14., 15., 16., 18., 21., 25., 30., 
                      40., 50., 70., 90., 110., 130., 150.])            

        self.times = self.times*secs_per_year*1e+3
        #   Ue  Uv  Ve  Vv  phi1    g1  vels
        self.outArray = np.zeros((len(self.times), 7))

    def out(self, t, uv, vv, f):
        ind = np.argwhere(self.times == t)
        try:
            self.maxind = ind[0][0]
        except IndexError:
            raise IndexError("SphericalEarthOutput received a time t={0:.3f}".format(t)+
                            " that was not in its output times.")
        Ue, Ve, phi1, g1, vel = f.solout()
        self.outArray[ind, 0] = Ue
        self.outArray[ind, 1] = uv
        self.outArray[ind, 2] = Ve
        self.outArray[ind, 3] = vv
        self.outArray[ind, 4] = phi1
        self.outArray[ind, 5] = g1
        self.outArray[ind, 6] = vel

    def converged(self, vislim=None):
        n = len(self.times)-self.maxind-1
        vislim = vislim or self.outArray[self.maxind, 1]
        self.outArray[self.maxind+1:] = np.tile(
            [0, vislim, 0, 
                self.outArray[self.maxind, 3], 0, 0, 0], (n,1))


############################## RELAXATION METHOD ##############################

class SphericalEarthRelaxer(object):
    """Implements the relaxation method to integrate fields in the mantle.

    With initial guesses from Shooter method (e.g., from get_t0_guesses), takes 
    two iterations to settle on solution.

    Initilization Parameters
    ------------------------
    earthparams : EarthParams
    zarray : array
        Points for finite-element mesh. Should emphasize areas of rapidly
        changing or high viscosity.
    yEt0, yVt0 : arrays
        Initial guesses for elastic and viscous solutions, must be in
        simulation units.

    Call Parameters
    ---------------
    t : time (in seconds)
    uv : array of viscous radial displacements at all depth points.
    verbose : Bool
        Relaxation method will report errors while converging.
    stepmax : int
        The number of steps Solvde is allowed to take (default 10).

    Call Returns
    ------------
    vv : array of viscous radial displacements at all depth points.

    Attributes
    ----------
    yEt0, yVt0 : arrays
        Previous solutions at t~0. Used for subsequent integrations
        initialized at t=0 (e.g., next order number)
    yE, yV : arrays
        Previous solution at t>0. Used at subsequent time step calls.
    difeqElas, difeqVisc : SphericalElasSMat, SphericalViscSMat
        Classes that hold the finite element components of relaxation method.
    commArray : array
    commProfs : interp1d
        The array and interpolation object for communicating elastic
        (viscous) values to the differential equations for the next viscous
        (elastic) values.
        
    Methods
    -------
    changeOrder
    initCommProfs
    updateElProfs
    updateUvProfs
    solout
    """
    
    def __init__(self, earthparams, zarray, yEt0, yVt0, n):
        # t==0 Initial guesses.
        self.yEt0 = yEt0
        self.yVt0 = yVt0
        # t>=0 Initial guesses.
        self.yE = yEt0.copy()
        self.yV = yVt0.copy()
        self.earthparams = earthparams
        self.zarray = zarray
        self.nz = len(zarray)
        self.initCommProfs(zarray)

        self.difeqElas = SphericalElasSMat(n, zarray, earthparams,
                                self.commProfs)
        self.difeqVisc = SphericalViscSMat(n, zarray, earthparams,
                                self.commProfs)

        self.alpha = earthparams.getLithFilter(n=n)
        
    def __call__(self, t, disps, verbose=False, stepmax=10):
        self.updateUvProfs(disps[:self.nz], dim=False)
        slowc = 1
        #TODO make scales adaptive
        scalvElas = np.array([1, 0.1, 1, 0.2, 0.1, 0.2])
        scalvVisc = np.array([0.5, 0.2, 1., 0.2])
        #scalvElas = np.abs(self.yE).max(axis=1)
        #scalvVisc = np.abs(self.yV).max(axis=1)

        # Update propagators for elastic relaxation
        self.difeqElas.updateProps()
        # Solve for elastic variables using relaxation method.
        if self.difeqElas.n == 1: 
            indexv = np.array([0,4,3,1,5,2])
        else:
            indexv = np.array([3,4,0,1,5,2])
        #solvde = Solvde(stepmax, 1e-14, slowc, scalvElas, indexv, 3, 
        #                    self.yE, self.difeqElas, verbose)
        #self.yE = solvde.y              # Store results for next initial guess.
        self.yE = solvde(stepmax, 1e-14, slowc, scalvElas, indexv, 3,
                            self.yE, self.difeqElas, verbose)
        self.updateElProfs(self.yE)      # Update communication profiles.

        # Update propagators for viscous relaxation
        self.difeqVisc.updateProps()
        # Solve for viscous variables using relaxation method.
        if self.difeqVisc.n == 1:
            indexv = np.array([0,3,2,1])
        else:
            indexv = np.array([2,3,0,1])
        #solvde = Solvde(stepmax, 1e-14, slowc, scalvVisc, indexv, 2, 
        #                    self.yV, self.difeqVisc, verbose)
        #self.yV = solvde.y              # Store results for next initial guess.
        self.yV = solvde(stepmax, 1e-14, slowc, scalvVisc, indexv, 2,
                            self.yV, self.difeqVisc, verbose)

        rstar   = self.earthparams.norms['r']
        etastar = self.earthparams.norms['eta']
        velfac  = rstar/etastar         # Simulation units to [cm / s].

        # If this is the early time step, save for next n.
        if t <= 1e9:
            self.yEt0 = self.yE.copy()
            self.yVt0 = self.yV.copy()

        # All the rates are converted back to real units and accelerated by the
        # elastic (and gravitational) energy stored by the lithosphere.
        vels = velfac*self.yV[[0,1],:]*self.alpha
        return vels.flatten()

    def changeOrder(self, n):
        """Change order number to n.

        Update the finite-element classes to n, and reset the initial solution
        guesses to yEt0, yVt0.
        """
        self.difeqElas.updateProps(n=n)
        self.difeqVisc.updateProps(n=n)
        self.yE = self.yEt0.copy()
        self.yV = self.yVt0.copy()

    def initCommProfs(self, zarray):
        """Initialize the interpolation object for communicating profiles
        between the elastic and viscous solutions. The profiles are stored:

        self.commArray = [[ue, ve, phi1, g1, uv] at z0
                          ....
                          [ue, ve, phi1, g1, uv] at zm]
        """
        self.commArray = np.zeros((len(zarray), 5))

        self.commProfs = interp1d(self.zarray, self.commArray.T)

    def updateElProfs(self, y, dim=True):
        """Update the functions that communicate elastic values to the viscous
        equations. 
        
        If instructed to redimensionalize values (True by default), values are 
        dimensionalized. For internal use.
        """
        rstar  = self.earthparams.norms['r']
        mustar = self.earthparams.norms['mu']
        disfac = rstar/mustar if dim else 1.
        gfac   = 1./rstar if dim else 1.

        self.commArray[:,0] = disfac*y[0,:]
        self.commArray[:,1] = disfac*y[1,:]
        self.commArray[:,2] = y[4,:]
        self.commArray[:,3] = gfac*y[5,:]
        self.commProfs = interp1d(self.zarray, self.commArray.T)
        # Also update profiles in finite difference classes
        self.difeqElas.commProf = self.commProfs
        self.difeqVisc.commProf = self.commProfs

    def updateUvProfs(self, uv, dim=True):
        """Update functions that communicate viscous values to elastic
        equations.

        If instructed to redimensionalize values (True by default), values are 
        dimensionalized. For internal use.
        """
        rstar   = self.earthparams.norms['r']
        etastar = self.earthparams.norms['eta']
        velfac  = rstar/etastar if dim else 1.

        self.commArray[:,4] = velfac*uv
        self.commProfs = interp1d(self.zarray, self.commArray.T)

        # Also update profiles in finite difference classes
        self.difeqElas.commProf = self.commProfs
        self.difeqVisc.commProf = self.commProfs

    def solout(self):
        """Returns elastic surface values in physical units.
        """
        rstar  = self.earthparams.norms['r']
        mustar = self.earthparams.norms['mu']
        disfac = rstar/mustar
        gfac   = 1./rstar

        etastar = self.earthparams.norms['eta']
        velfac  = rstar/etastar         # Simulation units to [cm / s].

        vel = velfac*self.yV[0,-1]*self.alpha

        Ue, Ve = disfac*self.yE[[0,1], -1]
        phi1 = self.yE[4, -1]
        g1 = gfac*self.yE[5, -1]

        return Ue, Ve, phi1, g1, vel


def get_t0_guess(earthmodel, zarray, n=2):
    """Use shooter method for t=0 solutions.

    For input into the relaxation method.
    """
    earthSolver = SphericalEarthShooter(earthmodel, zarray, n)
    earthSolver.initProfs(zarray)
    yEt0 = earthSolver.calcElasProfile(n)
    yVt0 = earthSolver.calcViscProfile(n)

    # Return to simulation units for use in Solvde
    rstar   = earthmodel.norms['r']
    mustar  = earthmodel.norms['mu']
    etastar = earthmodel.norms['eta']

    yEt0[[0,1],:] *= mustar/rstar
    yEt0[5,:]     *= rstar
    yVt0[[0,1],:] *= etastar/rstar

    return yEt0, yVt0


class SphericalElasSMat(object):
    """Class that provides smatrix to Solvde for Elastic solutions."""
    def __init__(self, n, z, earthparams, commProf):
        self.n = n
        self.z = z
        self.earthparams = earthparams
        self.commProf = commProf

        self.mpt = len(self.z)
        self.updateProps()
        self.alpha_i = 1./self.earthparams.getLithFilter(n=n)

    def updateProps(self, n=None):
        if n is None: n = self.n
        self.n = n
        self.alpha_i = 1./self.earthparams.getLithFilter(n=n)

        zmids = 0.5*(self.z[1:]+self.z[:-1])
        self.A, self.b = propMatElas(zmids, self.n, 
                                        self.earthparams, self.commProf)

    def smatrix(self, k, k1, k2, jsf, is1, isf, indexv, s, y):
        if k == k1:      # Core-Mantle boundary conditions.
            rCore = self.earthparams.rCore
            paramsCore = self.earthparams(rCore)
            rhoCore = paramsCore['den']
            gCore = paramsCore['grav']
            if self.commProf is not None:
                ue, ve, phi1, g1, uvCore = self.commProf(rCore)
            else:
                uvCore = 0.
            
            denCore = self.earthparams.denCore
            difden = denCore-rhoCore
            G = self.earthparams.G

            rstar = self.earthparams.norms['r']
            mustar = self.earthparams.norms['mu']

            phi1_v = -G*rCore*rstar*\
                    (denCore-rhoCore)*\
                    uvCore/(2*self.n+1)

            if self.n == 1:
                # Radial displacement of CMB.
                s[3, 6+indexv[0]] = 1.
                s[3, 6+indexv[1]] = 0.
                s[3, 6+indexv[2]] = 0.
                s[3, 6+indexv[3]] = 0.
                s[3, 6+indexv[4]] = 0.
                s[3, 6+indexv[5]] = 0.
                s[3, jsf] = y[0,0]
            else:
                # Radial stress on the core.
                s[3, 6+indexv[0]] = -denCore*gCore*rstar/mustar
                s[3, 6+indexv[1]] = 0.
                s[3, 6+indexv[2]] = 1.
                s[3, 6+indexv[3]] = 0.
                s[3, 6+indexv[4]] = -denCore    
                s[3, 6+indexv[5]] = 0.          
                s[3, jsf] = (y[2,0] - denCore*gCore*rstar/mustar*y[0,0] - \
                                difden*gCore*uvCore - \
                                denCore*(y[4,0] + phi1_v))
                                            
            # Poloidal stress on the core.
            s[4, 6+indexv[0]] = 0.          
            s[4, 6+indexv[1]] = 0.          
            s[4, 6+indexv[2]] = 0.          
            s[4, 6+indexv[3]] = 1.          
            s[4, 6+indexv[4]] = 0.          
            s[4, 6+indexv[5]] = 0.          
            s[4, jsf] = y[3,0]

            # gravitational potential perturbation on core.
            s[5, 6+indexv[0]] = -G*difden*rstar*rstar/mustar
            s[5, 6+indexv[1]] = 0.
            s[5, 6+indexv[2]] = 0.
            s[5, 6+indexv[3]] = 0.
            s[5, 6+indexv[4]] = -self.n/rCore
            s[5, 6+indexv[5]] = 1.
            s[5, jsf] = (y[5,0] - G*difden*rstar*(uvCore+rstar/mustar*y[0,0])-\
                                self.n/rCore*(y[4,0] + phi1_v))

        elif k >= k2:     # Surface boundary conditions.
            paramsSurf = self.earthparams(1.)
            gSurf = paramsSurf['grav']
            rhoSurf = paramsSurf['den']
            G = self.earthparams.G
            
            if self.commProf is not None:
                ue, ve, phi1, g1, uvSurf = self.commProf(1.)
            else:
                uvSurf = 0.
            
            # The load is applied directly to the top of the viscoelastic
            # mantle (i.e., the bottom of the lithosphere), which means only
            # the load unsupported by the lithosphere (1/alpha) is applied.
            load = (self.alpha_i + rhoSurf*gSurf*uvSurf)

            rstar = self.earthparams.norms['r']
            mustar = self.earthparams.norms['mu']

            # Radial stress on surface.
            s[0, 6+indexv[0]] = 0.
            s[0, 6+indexv[1]] = 0.
            s[0, 6+indexv[2]] = 1.
            s[0, 6+indexv[3]] = 0.
            s[0, 6+indexv[4]] = 0.
            s[0, 6+indexv[5]] = 0.
            s[0, jsf] = (y[2, self.mpt-1] + load)

            # Poloidal stress on surface.
            s[1, 6+indexv[0]] = 0.
            s[1, 6+indexv[1]] = 0.
            s[1, 6+indexv[2]] = 0.    
            s[1, 6+indexv[3]] = 1.    
            s[1, 6+indexv[4]] = 0.    
            s[1, 6+indexv[5]] = 0.    
            s[1, jsf] = y[3, self.mpt-1]
                                      
            # gravitational potential perturbation on surface.
            s[2, 6+indexv[0]] = G*rhoSurf*rstar*rstar/mustar
            s[2, 6+indexv[1]] = 0.    
            s[2, 6+indexv[2]] = 0.    
            s[2, 6+indexv[3]] = 0.    
            s[2, 6+indexv[4]] = (self.n+1)
            s[2, 6+indexv[5]] = 1.    
            s[2, jsf] = (y[5, self.mpt-1] + (self.n+1)*y[4, self.mpt-1] + \
                            G*rhoSurf*rstar*rstar/mustar*y[0, self.mpt-1] + \
                            G*load*rstar/gSurf)

        else:           # Finite differences.
            zsep = (self.z[k] - self.z[k-1])
            A = 0.5*zsep*self.A[k-1]
            b = zsep*self.b[k-1]
            interior_smatrix_fast(6, k, jsf, A, b, y, indexv, s) 
            
            #s[:6, indexv] = -np.eye(6) - A
            #s[:6, 6+indexv] = np.eye(6) - A
            #s[:, jsf] =  (y[:, k] - y[:,k-1] - \
            #                A.dot(y[:, k]+y[:,k-1]) - b)            
        return s

class SphericalViscSMat(object):
    """Class that provides smatrix to Solvde for viscous solutions."""
    def __init__(self, n, z, earthparams, commProf):
        self.n = n
        self.z = z
        self.earthparams = earthparams
        self.commProf = commProf

        self.mpt = len(self.z)
        self.updateProps()
        self.alpha_i = 1./self.earthparams.getLithFilter(n=n)

    def updateProps(self, n=None):
        if n is None: n = self.n
        self.n = n
        self.alpha_i = 1./self.earthparams.getLithFilter(n=n)

        zmids = 0.5*(self.z[1:]+self.z[:-1])
        self.A, self.b = propMatVisc(zmids, self.n, 
                                        self.earthparams, self.commProf)

    def smatrix(self, k, k1, k2, jsf, is1, isf, indexv, s, y):
        if k == k1:      # Core-Mantle boundary conditions.
            rCore = self.earthparams.rCore
            paramsCore = self.earthparams(rCore)
            rhoCore = paramsCore['den']
            gCore = paramsCore['grav']

            ueCore, ve, phi1Core, g1, uvCore = self.commProf(rCore)
            
            denCore = self.earthparams.denCore
            difden = denCore-rhoCore
            G = self.earthparams.G

            if self.n == 1:
                # Radial displacement of CMB.
                s[2, 4+indexv[0]] = 1.
                s[2, 4+indexv[1]] = 0.
                s[2, 4+indexv[2]] = 0.
                s[2, 4+indexv[3]] = 0.
                s[2, jsf] = y[0,0]
            else:
                # Radial stress on the core.
                s[2, 4+indexv[0]] = 0.
                s[2, 4+indexv[1]] = 0.
                s[2, 4+indexv[2]] = 1.
                s[2, 4+indexv[3]] = 0.
                s[2, jsf] = (y[2,0] - denCore*gCore*ueCore - \
                                difden*gCore*uvCore - \
                                denCore*phi1Core)
            
            # Poloidal stress on the core.
            s[3, 4+indexv[0]] = 0.
            s[3, 4+indexv[1]] = 0.
            s[3, 4+indexv[2]] = 0.
            s[3, 4+indexv[3]] = 1.
            s[3, jsf] = y[3,0]

        elif k >= k2:     # Surface boundary conditions.
            paramsSurf = self.earthparams(1.)
            gSurf = paramsSurf['grav']
            rhoSurf = paramsSurf['den']
            G = self.earthparams.G
            
            ue, ve, phi1, g1, uvSurf = self.commProf(1.)
            
            # The load is applied directly to the top of the viscoelastic
            # mantle (i.e., the bottom of the lithosphere), which means only
            # the load unsupported by the lithosphere (1/alpha) is applied.
            load = (self.alpha_i + rhoSurf*gSurf*uvSurf)

            # Radial stress on surface.
            s[0, 4+indexv[0]] = 0.
            s[0, 4+indexv[1]] = 0.
            s[0, 4+indexv[2]] = 1.
            s[0, 4+indexv[3]] = 0.
            s[0, jsf] = (y[2, self.mpt-1] + load)

            # Poloidal stress on surface.
            s[1, 4+indexv[0]] = 0.
            s[1, 4+indexv[1]] = 0.
            s[1, 4+indexv[2]] = 0.
            s[1, 4+indexv[3]] = 1.
            s[1, jsf] = y[3, self.mpt-1]

        else:       # Finite differences.
            zsep = (self.z[k] - self.z[k-1])
            A = 0.5*zsep*self.A[k-1]
            b = zsep*self.b[k-1]
            interior_smatrix_fast(4, k, jsf, A, b, y, indexv, s)

            #s[:4, indexv] = -np.eye(4) - 0.5*zsep*A
            #s[:4, 4+indexv] = np.eye(4) - 0.5*zsep*A
            #s[:, jsf] =  (y[:, k] - y[:,k-1] - \
            #                zsep*(0.5*A.dot(y[:, k]+y[:,k-1]) + b))
            
        return s

@jit(void(int64, int64, int64, float64[:,:], float64[:], 
    float64[:,:], int64[:], float64[:,:]), nopython=True)
def interior_smatrix_fast(n, k, jsf, A, b, y, indexv, s):
    for i in range(n):
        rgt = 0.
        for j in range(n):
            if i==j:
                s[i, indexv[j]]   = -1. - A[i,j]
                s[i, n+indexv[j]] =  1. - A[i,j]
            else:
                s[i, indexv[j]]   = -A[i,j]
                s[i, n+indexv[j]] = -A[i,j]
            rgt += A[i,j] * (y[j, k] + y[j, k-1])
        s[i, jsf] = y[i, k] - y[i, k-1] - rgt - b[i]


##############################  SHOOTING METHOD  ##############################

class SphericalEarthShooter(object):
    """Implements the shooting method to integrate fields in the mantle.

    Uses scipy's odeint.

    Initilization Parameters
    ------------------------
    earthparams : EarthParams
    zarray : array
        Points for finite-element mesh. Should emphasize areas of rapidly
        changing or high viscosity.

    Call Parameters
    ---------------
    t : time (in seconds)
    uv : array of viscous radial displacements at all depth points.
    verbose : Bool
        Relaxation method will report errors while converging.

    Call Returns
    ------------
    vv : array of viscous radial displacements at all depth points.

    Attributes
    ----------
    yE, yV : arrays
        Previous solution at t>0. Used at subsequent time step calls.
    difeqElas, difeqVisc : SphericalElasSMat, SphericalViscSMat
        Classes that hold the finite element components of relaxation method.
    commArray : array
    commProfs : interp1d
        The array and interpolation object for communicating elastic
        (viscous) values to the differential equations for the next viscous
        (elastic) values.
        
    Methods
    -------
    derivativeElas
    calcElasProfile
    derivativeVisc
    calcViscProfile
    initProfs
    solout
    """

    def __init__(self, earthparams, zarray, n):
        self.earthparams = earthparams
        self.initProfs(zarray)
        self.nz = len(zarray)
        self.n = n
        self.alpha = earthparams.getLithFilter(n=n)

    def __call__(self, t, disps):
        self.commArray[:,4] = disps[:self.nz]
        self.profs = interp1d(self.zarray, self.commArray.T)
        # calculate the elastic part at the new time step
        # [Ue(z), Ve(z), Pe(z), Qe(z), phi1(z), g1(z)] @ time=t
        self.eProf = self.calcElasProfile(self.n)
        # feed the new elastic solution to recalculate velocities
        # [Uv(z), Vv(z), Pv(z), Qv(z)] @ time=t
        self.vProf = self.calcViscProfile(self.n)
        vels = self.vProf[[0,1],:]*self.alpha
        return vels.flatten()
    
    def derivativeElas(self, y, z, n):
        """
        Calculate the derivative of the stress and gravity perutrbation at a depth z.
        it starts by interpolating all the earth properties at the given depth, forms the 
        6x6 non-dimensional array for d/dr [mu*u/r, mu*v/r, p, q, r*g1, phi1].
        
        For use internally (solve_elastic_sys)
        """        
        # Interpolate the material parameters at z
        try:
            params = self.earthparams.getParams(z)
        except ValueError as e:
            if z > self.zarray.max():
                params = self.earthparams.getParams(self.zarray.max())
            elif z < self.zarray.min():
                params = self.earthparams.getParams(self.zarray.min())
        if z > self.zarray.max():
            z = self.zarray.max()

        a, b = propMatElas(z, n, self.earthparams, self.profs)
        
        return np.dot(a, y)+b

    def calcElasProfile(self, n):
        """Solves for Ue, Ve, Pe, Qe, phi1, and g1 profiles as functions of
        depth for spherical harmonic order number n using a Shooting Method.

        Requires material parameters of the Earth (stored in earth) and a
            previously calculated viscous profile (for initial conditions)

        Recall: Ue = radial elastic displacement
                Ve = divergent elastic displacement
                Pe = radial stress field
                Qe = divergent stress field
                phi1 = perturbation to gravitational potential
                g1 = derivative of phi1
        """
        # Extract important quantities from earth model
        denCore = self.earthparams.denCore
        G = self.earthparams.G
        r = self.earthparams.norms['r']
        rCore = self.earthparams.rCore
        muStar = self.earthparams.norms['mu']

        tmp, tmp, tmp, tmp, uv = self.profs(rCore)

        # Four initial starting vectors at Core Mantle Boundary
        # assuming Ue = Ve = g = 1 at CMB.
        params = self.earthparams.getParams(rCore)
        lam = params['bulk']
        mu = params['shear']
        rho = params['den']
        g = params['grav']
        grad_rho = params['dend']
        grad_rho_na = params['nonad']

        phi1_v = -G*rCore*r*\
                    (denCore-rho)*\
                    uv/(2*n+1)

        y0 = np.array([[muStar/r, 0, denCore*g, 0, 0, G*r*(denCore-rho)],
                    [0, muStar/r, 0, 0, 0, 0],
                    [0, 0, denCore, 0, 1, n/rCore],
                    [0, 0, (denCore-rho)*g*uv+denCore*phi1_v, 0, phi1_v, 
                            G*r*(denCore-rho)*uv+n*phi1_v/rCore]])
        if n==1:
            y0[0] = np.array([0, 0, 1., 0, 0, 0])
        
        # Propogate solution to Surface
        y = np.asarray([odeint(self.derivativeElas, y0i, self.zarray, 
                        args=(n, ), h0 = 0.001) for y0i in y0])
        
        # convert solution to real units
        y[:,:,0:2] = y[:,:,0:2]*r/muStar   # the displacements u and v
        y[:,:,5] = y[:,:,5]/r                     # gravity perturbation field
        
        # Apply Surface conditions.
        params = self.earthparams.getParams(1.)
        rho = params['den']
        g = params['grav']

        tmp, tmp, tmp, tmp, uv = self.profs(1.)
        # Update load = 1 dyne redistributed plus amount that has relaxed.
        # The load is applied directly to the top of the viscoelastic
        # mantle (i.e., the bottom of the lithosphere), which means only
        # the load unsupported by the lithosphere (1/alpha) is applied.

        load = 1./self.alpha+rho*uv*g
        
        # Initialize the boundary solver: a*x=b.
        a = y[0:3,-1,[2,3,5]]  # An array of the boundary elements from 
                               # each soln vector.
        a[:,2] += y[0:3, -1, 4]*(1+n)/r + G*rho*y[0:3, -1, 0]
            # taking the surface value of p (2), q (3), and g1 (5)
            # shape: [y0.p, y0.q, y0.g1+(1+n)/rstar*y0.phi1+4\pi G \rho*y0.u]
            #        [y1.p, y1.q, y1.g1+(1+n)/rstar*y1.phi1+4\pi G \rho*y1.u]]
            #        [y2.p, y2.q, y2.g1+(1+n)/rstar*y2.phi1+4\pi G \rho*y2.u]] 
        
        # b is a vector of the boundary values for p, q, and g1.
        b = np.array([-load, 
                      0, 
                      -G*load/g-y[3, -1, 4]*(1+n)/r-G*rho*y[3, -1, 0]])
        # Include the (known) core-mantle boundary starting vector
        b -= y[3,-1,[2,3,5]] 
        coeffs = np.r_[np.linalg.solve(a.T, b), 1]
        
        # sum for the final solution together
        #profile = y[0,:,:]*coeffs[0]+y[1,:,:]*coeffs[1]+y[2,:,:]*coeffs[2]+y[3,:,:]
        profile = y.T.dot(coeffs)
        
        # for communication to the viscous equation
        self.commArray[:,[0,1,2,3]] = profile[[0,1,4,5],:].T
        self.profs = interp1d(self.zarray, self.commArray.T)
            
        return profile

    def derivativeVisc(self, y, z, n):
        """
        Calculate the derivative of the stress and gravity perutrbation at a
        depth z. It starts by interpolating all the earth properties at the
        given depth, forms the 4x4 non-dimensional array for 
        d/dr [eta*u/r, eta*v/r, p, q].

        For use internally (solve_fluid_sys)
        """
        if z>self.zarray.max():
            z = self.zarray.max()

        a, b = propMatVisc(z, n, self.earthparams, self.profs)
        
        return np.dot(a, y)+b

    
    def calcViscProfile(self, n):
        """
        Solve for the depth profile, core to surface, of the 4-vector 
            y = [Uv,              Vv,                Pv,            Qv             ]
                [Radial velocity, Poloidal Velocity, Radial Stress, Poloidal Stress]
        """
        # Extract important quantities from earth model
        denCore = self.earthparams.denCore
        G = self.earthparams.G
        r = self.earthparams.norms['r']
        rCore = self.earthparams.rCore
        etaStar = self.earthparams.norms['eta']

        ue, ve, phi1, g1, uv = self.profs(rCore)

        # Four initial starting vectors at Core Mantle Boundary
        params = self.earthparams.getParams(rCore)
        eta = params['visc']
        rho = params['den']
        g = params['grav']

        # initial conditions
        y0 = np.array([[etaStar/r, 0, 0, 0], 
                    [0, etaStar/r, 0, 0],
                    [0, 0, ((denCore-rho)*g*uv+denCore*phi1+denCore*g*ue),0]])
        if n==1:                    # Different initial conditions for n=1 case
            y0[0] = np.array([0., 0., -1., 0.])
        
        # solve for profile
        y = np.asarray([odeint(self.derivativeVisc, y0i, self.zarray, args=(n, ), 
                            h0 = 0.001) for y0i in y0])

        # convert solution to real units
        y[:,:,0:2] = y[:,:,0:2]*r/etaStar   # the displacements u and v

        params = self.earthparams.getParams(1.)
        eta = params['visc']
        rho = params['den']
        g = params['grav']

        # update load
        tmp, tmp, tmp, tmp, uv = self.profs(1.)
        # The load is applied directly to the top of the viscoelastic
        # mantle (i.e., the bottom of the lithosphere), which means only
        # the load unsupported by the lithosphere (1/alpha) is applied.
        load = -1./self.alpha-rho*g*uv
        
        # initialize the boundary solver: a*x=b
        a = y[0:2,-1,[2,3]]  # an array of the boundary elements from each soln vector
                               # taking the surface value of p (2) and q (3)
                               # shape: [y0.p, y0.q]
                               #        [y1.p, y1.q]
            
        b = np.array([load, 0]) - y[2,-1,[2,3]] # a vector of the boundary values
                                             # for p and q
        
        try:
            coeffs = np.r_[np.linalg.solve(a.T, b), 1]
        except np.linalg.LinAlgError as e:
            print y
            print a
            print b
            raise e
        
        # put the final solution together
        profile = y.T.dot(coeffs)
        
        return profile

    def initProfs(self, zarray):
        # viscous deformation uv = $\int_0^t vis.u[\tau] d\tau$
        # for all depths in the depth range
        # Profiles for communication are:
        # [Ue    Ve    phi1  g1   Uv]
        self.zarray = zarray
        self.commArray = np.zeros((len(self.zarray), 5))
        self.profs = interp1d(self.zarray, self.commArray.T)

    def solout(self):
        rstar  = self.earthparams.norms['r']
        mustar = self.earthparams.norms['mu']
        disfac = rstar/mustar
        gfac   = 1./rstar

        Ue, Ve = disfac*self.eProf[[0,1], -1]
        phi1 = self.eProf[4, -1]
        g1 = gfac*self.eProf[5, -1]

        vel = self.vProf[0,-1]*self.alpha

        return Ue, Ve, phi1, g1, vel

##############################   Z-DERIVATIVES   ##############################

def propMatElas(zarray, n, earth, commProf=None):
    """Generate the propagator matrix at all points in zarray. Should have
    shape (len(zarray), 6, 6)
    
    """
    # Check for individual z call
    zarray = np.asarray(zarray)
    singz = False
    if zarray.shape == ():
        zarray = zarray[np.newaxis]
        singz = True
         
    params = earth.getParams(zarray)
    lam = params['bulk']
    mu = params['shear']
    rho = params['den']
    g = params['grav']
    grad_rho = params['dend']
    grad_rho_na = params['nonad']

    r = earth.norms['r']
    muStar = earth.norms['mu']
    G = earth.G

    # Dimensional values
    rstar_over_mustar = r/muStar
    cons2 = G*r*rstar_over_mustar

    beta_i = 1./(lam+2*mu)
    gamma = mu*(3*lam+2*mu)*beta_i

    z_i = 1./zarray
    
    delsq = -n*(n+1)

    a = np.zeros((len(zarray), 6, 6))
    ones = np.ones(len(zarray))

    for i in range(len(zarray)):
        a[i] = z_i[i]*np.array([[-2*lam[i]*beta_i[i], -lam[i]*delsq*beta_i[i],
        zarray[i]*beta_i[i], 0, 0, 0], 
                        [-1, 1, 0, zarray[i]/mu[i], 0, 0],
                        [4*(gamma[i]*z_i[i] - rho[i]*g[i]*rstar_over_mustar) +\
                            cons2*rho[i]*rho[i]*zarray[i], 
                                (2*gamma[i]*z_i[i]-rho[i]*g[i]*rstar_over_mustar)*delsq, 
                                -4*mu[i]*beta_i[i], -delsq, 0, zarray[i]*rho[i]],
                        [-2*gamma[i]*z_i[i]+rho[i]*g[i]*rstar_over_mustar, 
                                -((gamma[i]+mu[i])*delsq+2*mu[i])*z_i[i],
                                -lam[i]*beta_i[i], -3, 
                                rho[i], 0],
                        [0, 0, 0, 0, 0, zarray[i]],
                        [-cons2*(zarray[i]*grad_rho[i]+4*mu[i]*rho[i]*beta_i[i]), 
                                -2*cons2*rho[i]*mu[i]*delsq*beta_i[i],
                                -cons2*rho[i]*zarray[i]*beta_i[i], 
                                0, -delsq*z_i[i], -2]])
    #a[:,0,0] = -2*lam*beta_i
    #a[:,0,1] = -lam*delsq*beta_i
    #a[:,0,2] = zarray*beta_i

    #a[:,1,0] = -1*ones
    #a[:,1,1] = ones
    #a[:,1,3] = zarray/mu

    #a[:,2,0] = 4*(gamma*z_i - rho*g*rstar_over_mustar) +\
    #                cons2*rho*rho*zarray
    #a[:,2,1] = (2*gamma*z_i-rho*g*rstar_over_mustar)*delsq
    #a[:,2,2] = -4*mu*beta_i
    #a[:,2,3] = -delsq*ones
    #a[:,2,5] = zarray*rho

    #a[:,3,0] = -2*gamma*z_i+rho*g*rstar_over_mustar
    #a[:,3,1] = -((gamma+mu)*delsq+2*mu)*z_i
    #a[:,3,2] = -lam*beta_i
    #a[:,3,3] = -3*ones
    #a[:,3,4] = rho

    #a[:,4,5] = zarray
    #
    #a[:,5,0] = -cons2*(zarray*grad_rho+4*mu*rho*beta_i)
    #a[:,5,1] = -2*cons2*rho*mu*delsq*beta_i
    #a[:,5,2] = -cons2*rho*zarray*beta_i
    #a[:,5,4] = -delsq*z_i
    #a[:,5,5] = -2*ones

    #a *= z_i[:,np.newaxis,np.newaxis]

    # And the inhomogenous component of the propagator
    if commProf is None:
        uv = np.zeros(len(zarray))
    else:
        ue, ve, phi1, g1, uv = commProf(zarray)
    pot = G*r*grad_rho_na*uv/(2*n+1)
    b = np.zeros((len(zarray), 6))
    
    b[:,2] = -g*grad_rho_na*uv
    b[:,4] = pot
    b[:,5] = -(n+1)*pot
    
    if singz:
        return a[0], b[0] 
    else:
        return a, b

def propMatVisc(zarray, n, earth, commProf=None):
    """
    Calculate the derivative of the stress and gravity perutrbation at a
    depth z. It starts by interpolating all the earth properties at the
    given depth, forms the 4x4 non-dimensional array for 
    d/dr [eta*u/r, eta*v/r, p, q].

    For use internally (solve_fluid_sys)
    """
    # Check for individual z call
    zarray = np.asarray(zarray)
    singz = False
    if zarray.shape == ():
        zarray = zarray[np.newaxis]
        singz = True

    # Extract relevant parameters    
    r = earth.norms['r']             
    G = earth.G                      
                                     
    params = earth.getParams(zarray)
    eta = params['visc']             
    rho = params['den']              
    g = params['grav']               
    grad_rho_na = params['nonad']    
                                     
    ue, ve, phi1, g1, uv = commProf(zarray)
                                     
                                     
    delsq = -n*(n+1)

    z_i = 1./zarray
     
    a = np.zeros((len(zarray), 4, 4))  
    ones = np.ones(len(zarray))

    # Form propagation matrix
    for i in range(len(zarray)):
        a[i] = z_i[i]*np.array([[-2, -delsq, 0, 0], 
                        [-1, 1, 0, zarray[i]/eta[i]], 
                        [12*eta[i]*z_i[i], 6*delsq*eta[i]*z_i[i], 0, -delsq], 
                        [-6*eta[i]*z_i[i], -2*eta[i]*(2*delsq+1)*z_i[i], -1, -3]])

    #a[:,0,0] = -2*ones
    #a[:,0,1] = -delsq*ones

    #a[:,1,0] = -1*ones
    #a[:,1,1] = ones
    #a[:,1,3] = zarray/eta
    #
    #a[:,2,0] = 12*eta*z_i
    #a[:,2,1] = 6*delsq*eta*z_i
    #a[:,2,3] = -delsq*ones

    #a[:,3,0] = eta*z_i
    #a[:,3,1] = -2*eta*(2*delsq+1)*z_i
    #a[:,3,2] = -1*ones
    #a[:,3,3] = -3*ones

    #a *= z_i[:,np.newaxis,np.newaxis]
    
    # Form inhomogeneous vector
    b = np.zeros((len(zarray), 4))

    g_pert     = r*rho*g1
    nonad_pert = -g*uv*grad_rho_na
    el_pert    = ((-4*ue - delsq*ve)*rho*g*z_i +\
                            G*r*ue*(rho**2))

    for i in range(len(zarray)):
        b[i] = np.array([0, 0, g_pert[i]+nonad_pert[i]+el_pert[i], 
                            rho[i]*(phi1[i]+g[i]*ue[i])*z_i[i]      ])

    #b[:,2] = g_pert + nonad_pert + el_pert
    #b[:,3] = rho*(phi1+g+ue)*z_i

    if singz:
        return a[0], b[0]
    else:
        return a, b

##############################    OTHER STUFF    ##############################

def rk4Earth(npts, nskip):
    zarray = np.linspace(rCore, 1., npts)
    zsep = zarray[1:]-zarray[:-1]
    
    y = []
    
    for y0i in y0:
        yi = [y0i]
        ys = y0i
        for i, zs in enumerate(zip(zarray, zsep)):
            z, dz = zs
            dydx = earthSolver.derivativeElas(ys, rCore, n)
            ys = rk4(ys, dydx, z, dz, earthSolver.derivativeElas, (n,))
            if (i+1)%nskip == 0: 
                yi.append(ys)
        y.append(yi)
    
    y = np.asarray(y)

    return y


