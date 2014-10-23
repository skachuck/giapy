"""
earth_spherical.py

    Module for fully spherical earth model.

    Author: Samuel B. Kachuck
"""

import numpy as np

class EarthParams(object):
    

class SphericalEarthSolver(object):

    def __init__(self, params, n):
        """
        params : 
            Parameters defining earth structure
        n : int
            Spherical harmonic order number
        """
        self.params = params
        self.n = n
    
    def derivativeElas():
        """
        Calculate the derivative of the stress and gravity perutrbation at a depth z.
        it starts by interpolating all the earth properties at the given depth, forms the 
        6x6 non-dimensional array for d/dr [mu*u/r, mu*v/r, p, q, r*g1, phi1].
        
        For use internally (solve_elastic_sys)
        """        
        lam = self.bulkmod(z)
        mu = self.shearmod(z)
        rho = self.density(z)
        g = self.grav(z)
        grad_rho = self.dengrad(z)
        grad_rho_na = self.nonadiab(z)

        rstar_over_mustar = self.r_star/self.mu_star
        cons2 = self.g_cons*self.r_star*rstar_over_mustar

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

            
        stress = -g*grad_rho_na*self.uv(z)
        pot = self.g_cons*self.r_star*grad_rho_na*self.uv(z)/(2*n+1)
            
        b = np.array([0, 0, stress, 0, pot, -(n+1)*pot])
        
        return np.dot(a, y)+b

    def calcElasProfile():

    def derivativeVisc():
        """
        Calculate the derivative of the stress and gravity perutrbation at a
        depth z. It starts by interpolating all the earth properties at the
        given depth, forms the 4x4 non-dimensional array for 
        d/dr [eta*u/r, eta*v/r, p, q].

        For use internally (solve_fluid_sys)
        """

        eta = self.viscosity(z)
        ue = self.ue(z)
        rho = self.density(z)
        g = self.grav(z)
        grad_rho_na = self.nonadiab(z)
                
        delsq = -n*(n+1)

        z_i = 1./z
            
        a = z_i*np.array([[-2, -delsq, 0, 0], 
                        [-1, 1, 0, z/eta], 
                        [12*eta*z_i, 6*delsq*eta*z_i, 0, -delsq], 
                        [-6*eta*z_i, -2*eta*(2*delsq+1)*z_i, -1, -3]])
        
        gravity_perturbation = self.r_star*rho*self.g1(z)
        nonadiab_perturbation = -g*self.uv(z)*grad_rho_na
        elastic_perturbation = ((-4*ue - delsq*self.ve(z))*rho*g*z_i +\
                                    self.g_cons*self.r_star*ue*(rho**2))

        b = np.array([0, 0, gravity_perturbation +\
                            nonadiab_perturbation +\
                            elastic_perturbation,
                            z_i*rho*(self.phi1(z)+g*ue)])
        
        return np.dot(a, y)+b

    
    def calcViscProfile():
    
    def viscVelocity():
        # Solve elastic profile
        elasProfile = calcElasProfile()
        # Solve viscous profile
        viscProfile = calcViscProfile()
    
    
    return vs
    
    def timeEvolve():
        # TODO Look up dynamic time-step optimization for saving profiles
        r = ode()
        r.set_integrator()
        r.set_solout()
    
        while r.successful() and r.t<150:
            r.integrate()
        
    

class SphericalEarth(object):

    def __init__(self):
    
    def get_resp(self, t_dur):
    
    def reset_params_list(self, params, arglist, visclog=False):

    def reset_params(self):
        

