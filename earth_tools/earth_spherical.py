"""
earth_spherical.py

    Module for spherical, non-rotating, self-gravitating, viscoelastic earth
    model.

    Author: Samuel B. Kachuck
"""

import numpy as np
from scipy.integrate import odeint, ode
from scipy.interpolate import interp1d
from progressbar import ProgressBar, Bar, Percentage

class EarthParams(object):
    def __init__(self):
        self.G = 4*np.pi*6.674e-8  # cm^3/g.s^2
        
        self.norms = {'r': 6.371e+8}    # cm
        self.rCore = 0.5495             # earth radii
        self.rStep = 0.001              # earth radii
        self.denCore = 9.927            # gm/cc

        self.paramNames = ['den', 'visc', 'shear', 'bulk', 'grav', 'dend',
                            'nonad']

    def getParams(self, z, depth=False, norm=True, unit='km'):
        """
        Return a dictionary of parameters interpolated to radius (depth) z.
        """
        if unit=='km':
            pass
        elif unit=='m':
            z*=1e3
        elif unit=='cm':
            z*=1e5
        else:
            raise ValueError('Unit not recognized. Use km, m, or cm')
            
        if depth:
            z = self.r - z

        vals = self._interpParams(z)
        

        return dict(zip(self.paramNames, vals)) 

    def readParams(self, fname):
        """
        Import parameters from a text file.
        """
        #TODO Write this.
        self.paramArray = np.r_[every, one, of, them].reshape((num, len(zz)))
        self._interpParams = interp1d(zz, self.paramArray)

    def hb_init(self, norm=1):
        """

        inputs:
           norm (boolean): if the profiles are to be normalized, this
                           normalizes depth by earth radii, shear and
                           bulk modulii by the shear modulus at the cmb
                           and the viscosity by the viscosity at the cmb.
        """
        self.rCore = 0.552 # in earth radii
        
        # these are the piecewise-linear profiles of the hadden bullen model
        zz = 1.e+5*np.array([2855., 2185., 985., 985., 635., 635., 335., 335.,
                                75., 75., 0.]) # cm
        den = np.array([5.527, 5.188, 4.539, 4.539, 4.200, 4.150, 3.700, 3.441, 3.313, 3.313, 3.313]) # gm/cc
        grav = np.array([1073.8,1010.2,995.8,995.8,998.7,998.7,995.5,995.5,983.2,983.2,983.2]) # cm/s^2
        shear = 1.e+12*np.array([2.954,2.555,1.837,1.837,1.462,1.396,.749,.697,.709,.709,.709]) # dyne/cm^2
        bulk = 1.e+12*np.array([6.349,5.407,3.490,3.490,2.665,2.698,1.836,1.706,1.017,1.017,1.017]) # dyne/cm^2
        visc = 1.e+22*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1., 1., 1]) # poise = g/cm.s
        
        zz = self.norms['r']-zz
        self.norms['mu'] = shear[0]
        self.norms['eta'] = visc[0]

        zzd = np.array([ 0.55187569,  0.65703971,  0.65703971,  0.84539319,  0.84539319,  0.90032962,
                      0.90032962,  0.94741799,  0.94741799,  0.98822791,  0.98822791,  1.        ])

        dend = np.array([-3.22353582, -3.33        , -3.44564917, -6.17076857, -6.17076857, 
                      -9.5565,     -9.5565,     -3.13649231, -3.13649231,  0.,          0.       ])

        nonad = self.norms['r']*np.array([0., 0., 0., 0., 0., 0.5/3.e+7, 0.5/3.e+7, 0., 0., 0., 0.])
        nonad = 0.*nonad
        
        
        # normalizing, if requested
        # we normalize only the deformation parameters and depths
        if norm == 1:
            zz = zz/self.norms['r']
            shear = shear/self.norms['mu']
            bulk = bulk/self.norms['mu']
            visc = visc/self.norms['eta']
        
        self.zz = zz
        self.paramArray = np.r_[den, visc, shear, 
                                bulk, grav, dend, nonad].reshape((7, len(zz)))
        self._interpParams = interp1d(zz, self.paramArray)



class SphericalEarthSolver(object):
    """Class of functions for calculating response of earth to unit spherical
    harmonic loads.
    """

    def __init__(self, earth):
        """
        params : 
            Parameters defining earth structure
        n : int
            Spherical harmonic order number
        """
        self.earth = earth
    
    def derivativeElas(self, y, z, n):
        """
        Calculate the derivative of the stress and gravity perutrbation at a depth z.
        it starts by interpolating all the earth properties at the given depth, forms the 
        6x6 non-dimensional array for d/dr [mu*u/r, mu*v/r, p, q, r*g1, phi1].
        
        For use internally (solve_elastic_sys)
        """        
        # Interpolate the material parameters at z
        try:
            params = self.earth.getParams(z)
        except ValueError as e:
            if z > self.zarray.max():
                params = self.earth.getParams(self.zarray.max())
            elif z < self.zarray.min():
                params = self.earth.getParams(self.zarray.min())
        if z > self.zarray.max():
            z = self.zarray.max()
        lam = params['bulk']
        mu = params['shear']
        rho = params['den']
        g = params['grav']
        grad_rho = params['dend']
        grad_rho_na = params['nonad']

        r = self.earth.norms['r']
        muStar = self.earth.norms['mu']
        G = self.earth.G

        # Dimensional values
        rstar_over_mustar = r/muStar
        cons2 = G*r*rstar_over_mustar

        beta_i = 1./(lam+2*mu)
        gamma = mu*(3*lam+2*mu)*beta_i

        z_i = 1./z
        
        delsq = -n*(n+1)
        
        a = z_i*np.array([[-2*lam*beta_i, -lam*delsq*beta_i, z*beta_i, 0, 0, 0], 
                        [-1, 1, 0, z/mu, 0, 0],
                        [4*(gamma*z_i - rho*g*rstar_over_mustar)+cons2*rho*rho*z, 
                                (2*gamma*z_i-rho*g*rstar_over_mustar)*delsq, 
                                -4*mu*beta_i, -delsq, 0, z*rho],
                        [-2*gamma*z_i+rho*g*rstar_over_mustar, 
                                -((gamma+mu)*delsq+2*mu)*z_i, -lam*beta_i, -3, 
                                rho, 0],
                        [0, 0, 0, 0, 0, z],
                        [-cons2*(z*grad_rho+4*mu*rho*beta_i), 
                                -2*cons2*rho*mu*delsq*beta_i,-cons2*rho*z*beta_i, 
                                0, -delsq*z_i, -2]])

            
        stress = -g*grad_rho_na*self.profs['uv'](z)
        pot = G*r*grad_rho_na*self.profs['uv'](z)/(2*n+1)
            
        b = np.array([0, 0, stress, 0, pot, -(n+1)*pot])
        
        return np.dot(a, y)+b

    def calcElasProfile(self, n):
        """Solves for Ue, Ve, Pe, Qe, phi1, and g1 profiles as functions of
        depth for spherical harmonic order number n.

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
        denCore = self.earth.denCore
        G = self.earth.G
        r = self.earth.norms['r']
        rCore = self.earth.rCore
        muStar = self.earth.norms['mu']

        uv = self.profs['uv'](self.earth.rCore)

        # Four initial starting vectors at Core Mantle Boundary
        params = self.earth.getParams(self.earth.rCore)
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
        
        # Propogate solution to Surface
        y = np.asarray([odeint(self.derivativeElas, y0i, self.zarray, 
                        args=(n, ), h0 = 0.001) for y0i in y0])
        
        # convert solution to real units
        y[:,:,0:2] = y[:,:,0:2]*r/muStar   # the displacements u and v
        y[:,:,5] = y[:,:,5]/r                     # gravity perturbation field
        
        # Apply Surface conditions
        params = self.earth.getParams(1.)
        rho = params['den']
        g = params['grav']

        # update load = 1 dyne redistributed plus amount that has relaxed
        load = 1+rho*self.profs['uv'](1.)*g
        
        # initialize the boundary solver: a*x=b
        a = y[0:3,-1,[2,3,5]]  # an array of the boundary elements from each soln vector
        a[:,2] += y[0:3, -1, 4]*(1+n)/r + G*rho*y[0:3, -1, 0]
                # taking the surface value of p (2), q (3), and g1 (5)
                # shape: [y0.p, y0.q, y0.g1+(1+n)/rstar*y0.phi1+4\pi g \rho*y0.u]
                #        [y1.p, y1.q, y1.g1+(1+n)/rstar*y1.phi1+4\pi g \rho*y1.u]]
                #        [y2.p, y2.q, y2.g1+(1+n)/rstar*y2.phi1+4\pi g \rho*y2.u]] 
        
        # b is a vector of the boundary values for p, q, and g1
        b = np.array([-load, 0, -G*load/g-y[3, -1, 4]*(1+n)/r-\
                            G*rho*y[3, -1, 0]])-y[3,-1,[2,3,5]] 
        coeffs = np.r_[np.linalg.solve(a.T, b), 1]
        
        # sum for the final solution together
        #profile = y[0,:,:]*coeffs[0]+y[1,:,:]*coeffs[1]+y[2,:,:]*coeffs[2]+y[3,:,:]
        profile = y.T.dot(coeffs)
        
        # for communication to the viscous equation
        self.profs['ue'] = interp1d(self.zarray, profile[0,:])
        self.profs['ve'] = interp1d(self.zarray, profile[1,:])
        self.profs['phi1'] = interp1d(self.zarray, profile[4,:])
        self.profs['g1'] = interp1d(self.zarray, profile[5,:])
            
        return profile

    def derivativeVisc(self, y, z, n):
        """
        Calculate the derivative of the stress and gravity perutrbation at a
        depth z. It starts by interpolating all the earth properties at the
        given depth, forms the 4x4 non-dimensional array for 
        d/dr [eta*u/r, eta*v/r, p, q].

        For use internally (solve_fluid_sys)
        """

        #eta = self.viscosity(z)
        #ue = selfi.profs['ue'](z)
        #rho = self.density(z)
        #g = self.grav(z)
        #grad_rho_na = self.nonadiab(z)

        if z>self.zarray.max():
            z = self.zarray.max()

        # Extract relevant parameters
        r = self.earth.norms['r']
        G = self.earth.G

        params = self.earth.getParams(z)
        eta = params['visc']
        rho = params['den']
        g = params['grav']
        grad_rho_na = params['nonad']

        ue = self.profs['ue'](z)
        ve = self.profs['ve'](z)
        uv = self.profs['uv'](z)
        g1 = self.profs['g1'](z)
        phi1 = self.profs['phi1'](z)
                
        delsq = -n*(n+1)

        z_i = 1./z
            
        a = z_i*np.array([[-2, -delsq, 0, 0], 
                        [-1, 1, 0, z/eta], 
                        [12*eta*z_i, 6*delsq*eta*z_i, 0, -delsq], 
                        [-6*eta*z_i, -2*eta*(2*delsq+1)*z_i, -1, -3]])
        
        gravity_perturbation = r*rho*g1
        nonadiab_perturbation = -g*uv*grad_rho_na
        elastic_perturbation = ((-4*ue - delsq*ve)*rho*g*z_i +\
                                    G*r*ue*(rho**2))

        b = np.array([0, 0, gravity_perturbation +\
                            nonadiab_perturbation +\
                            elastic_perturbation,
                            z_i*rho*phi1+g*ue])
        
        return np.dot(a, y)+b

    
    def calcViscProfile(self, n):
        """
        Solve for the depth profile, core to surface, of the 4-vector 
            y = [Uv, Vv, Pv, Qv].
        """
        # Extract important quantities from earth model
        denCore = self.earth.denCore
        G = self.earth.G
        r = self.earth.norms['r']
        rCore = self.earth.rCore
        etaStar = self.earth.norms['eta']

        uv = self.profs['uv'](rCore)
        ue = self.profs['ue'](rCore)
        phi1 = self.profs['phi1'](rCore)

        # Four initial starting vectors at Core Mantle Boundary
        params = self.earth.getParams(self.earth.rCore)
        eta = params['visc']
        rho = params['den']
        g = params['grav']
    
        # initial conditions
        y0 = np.array([[etaStar/r, 0, 0, 0], 
                    [0, etaStar/r, 0, 0],
                    [0, 0, ((denCore-rho)*g*uv+denCore*phi1+denCore*g*ue),0]])
        if n==1:                    # Different initial conditions for n=1 case
            y0[0] = array([0., 0., -1., 0.])
        
        # solve for profile
        y = np.asarray([odeint(self.derivativeVisc, y0i, self.zarray, args=(n, ), 
                            h0 = 0.001) for y0i in y0])
        
        # convert solution to real units
        y[:,:,0:2] = y[:,:,0:2]*r/etaStar   # the displacements u and v

        params = self.earth.getParams(1.)
        eta = params['visc']
        rho = params['den']
        g = params['grav']

        # update load
        load = -1-rho*g*self.profs['uv'](1.)
        
        # initialize the boundary solver: a*x=b
        a = y[0:2,-1,[2,3]]  # an array of the boundary elements from each soln vector
                               # taking the surface value of p (2) and q (3)
                               # shape: [y0.p, y0.q]
                               #        [y1.p, y1.q]
            
        b = np.array([load, 0]) - y[2,-1,[2,3]] # a vector of the boundary values
                                             # for p, q, and g1
        coeffs = np.r_[np.linalg.solve(a.T, b), 1]
        
        # put the final solution together
        #profile = y[0,:,:]*coeffs[0]+y[1,:,:]*coeffs[1]+y[2,:,:]
        profile = y.T.dot(coeffs)
        
        return profile
    
    def viscVelocity():
        # Solve elastic profile
        elasProfile = calcElasProfile()
        # Solve viscous profile
        viscProfile = calcViscProfile()
        return vs
    
    def timeEvolveBetter(self):
        # TODO Look up dynamic time-step optimization for saving profiles
        r = ode(f, jac)
        r.set_integrator()
        r.set_solout()
    
        while r.successful() and r.t<150:
            r.integrate()


    def timeEvolve(self, n=2):
        """
        Calculate and store the response to a unit load of order n.
        """
        # initialize earth model
        earth = self.earth
        paramSurf = self.earth.getParams(1.)
        vislim = paramSurf['den']*paramSurf['grav']

    
        # Initialize hard-coded integration parameters
        #TODO Generalize/Optimize all this.
        # depths for profile
        zarray = self.zarray = np.linspace(earth.rCore, 1., 50)
    
        # initialize time integration parameters (in s)
        t = 0
        secs_per_year = 3.1536e+7  # seconds in a year
    
        dt = np.array([0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, .1, .1, .1,
            0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2, 0.2,  0.2,  0.2,
            0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2, 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,
            0.5,  0.5,  0.5,  0.5, 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1., 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2., 5.,
            5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5., 5.,  5.,  5.,  5.,  5.,  5.,  5., 5., 5., 5., 5., 5.])
    
        dt = (secs_per_year*1e+3*dt).tolist()
        tmax = secs_per_year*1.5e+5
        times_write = secs_per_year*1e+3*np.array([0., 0.2, 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 
                                            10., 12., 13., 14., 15., 16., 18., 21., 25., 30., 
                                            40., 50., 70., 90., 110., 130., 150.])
        tstep = dt.pop(0)
        #tstep = secs_per_year*100
        #times_write = secs_per_year*1e+3*arange(0., 13.)
        #tmax = secs_per_year*12e+3.

        # initialize communication profiles.
        self.profs = {}
        # viscous deformation uv = $\int_0^t vis.u[\tau] d\tau$
        # for all depths in the depth range
        uv = np.zeros(len(zarray))
        self.profs['uv'] = interp1d(zarray, uv)
        self.profs['ue'] = None
        self.profs['ve'] = None
        self.profs['phi1'] = None
        self.profs['g1'] = None

        # initialize function outputs
        returnDict = {}
        returnDict['times'] = []
        returnDict['eUpl'] = []
        returnDict['vUpl'] = []
        returnDict['phi1'] = []
        returnDict['g1'] = []
    
        pbar = ProgressBar(widgets=['n = {0}:'.format(n), Bar(), Percentage()])
        #while (t <= tmax): #and abs(1./3313.+uv[-1])>=1e-5):
        for i in pbar(range(len(dt))):
            # calculate the elastic part at the new time step
            # [Ue(z), Ve(z), Pe(z), Qe(z), phi1(z), g1(z)] @ time=t
            eProf = self.calcElasProfile(n)
            # feed the new elastic solution to recalculate velocities
            # [Uv(z), Vv(z), Pv(z), Qv(z)] @ time=t
            vProf = self.calcViscProfile(n)
        
            # update the total viscous deformation
            # not done in cycle because it is time-step dependent
            uv = uv + vProf[0,:]*tstep
            self.profs['uv'] = interp1d(zarray, uv)
                    
            # save at an adequate number of time steps
            #TODO Save at more dynamic times
            #TODO Save things better - observers?
            if t in times_write:
                returnDict['vUpl'].append(uv[-1])
                returnDict['eUpl'].append(eProf[0,-1])
                returnDict['times'].append(t/secs_per_year)
                returnDict['phi1'].append(eProf[4,-1])
                returnDict['g1'].append(eProf[5,-1])
                
            if (t>=tmax) or (1+uv[-1]*vislim<1e-5):
                break
            t+=tstep
            tstep=dt.pop(0)

        for key, val in returnDict.iteritems():
            returnDict[key] = np.array(val)
        
        return returnDict       
    

class SphericalEarth(object):
    """A class for calculating, storing, and recalling 

    Stores decay profiles for each spherical order number up to N for
        1) Surface Uplift
        2) Elastic Uplift
        3) Viscous Uplift Rate
        4) Gravitational potential
        5) Geoid
    responses to a unit load.

    ntrunc : int
        maximum spherical order number
    """

    def __init__(self):
        pass

    def __repr__(self):
        return self._desc
    
    def get_resp(self, t_dur):
        """Return an NDarray ((ntrunc+1)*(ntrunc+2)/2, 5) of the responses to a unit load applied for
        time t_dur.
        """
        pass
    
    def save(self):
        pass

    def setDesc(self, string):
        pass
    
    def reset_params_list(self, params, arglist, visclog=False):
        pass

    def reset_params(self):
        pass
        
    def calcResponse(self):
        pass



