"""
    NOTE ON UNITS:
    This module uses cgs units.
"""
import numpy as np
from scipy.interpolate import interp1d

from giapy import MODPATH

############# UNITS ############
# 1 dyne = 1 g cm / s^2 = 1e-5 N = 1e-5 kg m / s^2
# 1 GPa  = 1e9 Pa = 1e9 N/m^2
# 1 GPa = 1e10 dyne / cm^2

class EarthParams(object):
    """Store and interpolate Earth's material parameters.

    Uses PREM (Dziewonski & Anderson 1981) for density and elastic parameters.

    Parameters
    ----------
    model : str
        The elastic model to use. Needs to be .txt in MODPATH/data/earth/ with
        columns:      radius   density   shear   bulk (or first lame)   gravity 
        (default PREM).
    visArray : np.ndarray
        The array of depths and viscosities (in poise, 1e-1 Pa s). If visArray
        is None, assumes a uniform 1e21 Pa s mantle. visArray should be a 2xN 
        array of depths and viscosities. Can be changed later using 
        EarthParams.addViscosity.
    D : float
        The flexural rigidity of the lithosphere (in N). Can be changed later
        using EarthParams.addLithosphere (with either D, the flexural rigidity,
        or H, the elastic thickness of the lithospehre).
    bulk : boolean
        Indicates whether the model uses bulk of lame parameters so that it can
        be converted to the first lame parameter if necessary (default True).
    normmode: 'dim', 'larry', or 'love'
        Indicates which normalization to use for the parameters. This can be
        either 'dim' for dimensional parameters (in cgs, for now), 'larry'
        which nondimensionalizes elastic parameters, radii, and viscosities, or
        'love' which nondimensionalizes everything for use with direct Love
        number computation.
    """
    def __init__(self, model='prem', visArray=None, D=0, bulk=True,
                    normmode='larry', G=6.674e-11):        
        self.G = 4*np.pi*G                      # m^3/kg.s^2
        
        self.normmode = 'larry'
        self.norms = {'r'  :     6.371e+8 ,     # cm
                      'eta':     1e+22    ,     # poise = g/cm.s    
                      'mu' :     293.8e+10,     # dyne/cm^2
                      'g'  :     981.56   }     # cm/s^2

        try:
            locprem = np.loadtxt(MODPATH+'/data/earth/'+model+'.txt')
        except:
            raise

        self.norms = {'r'  : locprem[-1,0]*1e3, # m
                      'eta': 1e21             , # Pa s
                      'mu' : locprem[1,3]*1e9 , # N/m^2
                     'g'  :locprem[-1,4]   } #   m/s^2

        self.rCore = locprem[0,0]/locprem[-1,0]       # earth radii
        self.denCore = locprem[0,1]                # g/cc

        self.z = locprem[1:,0]/locprem[-1, 0]              # Normalized depths in mantle.
        z = self.z
        locprem[1:, [2,3]] /= locprem[1, 3]           # Normalized elastic props by
                                                # shear modulus.
        # Convert the bulk modulus to the first lame parameter.
        if bulk:
            locprem[1:, 2] = locprem[1:, 2] - (2./3.*locprem[1:, 3])

        # Create density gradient, g/cc.earthRadii
        dend = np.gradient(locprem[1:,1])/np.gradient(z)

        # Fillers for nonadiabatic density gradients and visocisity.
        filler = np.zeros((len(z), 1))

        # Make interpolation array
        # 0     Density
        # 1     BulkMod
        # 2     ShearMod
        # 3     Gravity
        # 4     Density Gradient
        # 5     Non-adiabatic density Gradient
        # 6     Viscosity
        self._paramNames = ['den', 'bulk', 'shear', 'grav', 'dend',
                            'nonad', 'visc']
        self._paramArray = np.concatenate((locprem[1:,1:5], dend[:,np.newaxis], 
                                            filler, filler), axis=1).T

        self._interpParams = interp1d(z, self._paramArray)
 

        # Set up viscosity profile with uniform viscosity
        if visArray is None:
            try:
                visArray = locprem[1:,5]
                if visArray[-1] == visArray[-2] >= 1e11:
                    visArray[-2:] = visArray[-3]
                    H = (z[-1]-z[-2])*6371

                self._paramArray[6] = visArray
                self._interpParams = interp1d(z, self._paramArray)

            except:
                self._paramArray[6] = np.ones_like(z)
                self._interpParams = interp1d(z, self._paramArray)
        else:
            self.addViscosity(visArray)
        

        try:
            self.addLithosphere(H=H)       
        except:
            self.addLithosphere(D=D)

        # Flexural rigidity is assumed 0 (no lithosphere)
        #self.D = D

        self.normalize(normmode)

    def __call__(self, z, depth=False):
        return self.getParams(z, depth)

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['_interpParams']
        return odict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._interpParams = interp1d(self.z, self._paramArray)

    def normalize(self, normmode='love'):
        """Normalize or dimensionalize the parameters.
        
        normmode: 'dim', 'larry', or 'love'
            Indicates which normalization to use for the parameters. This can be
            either 'dim' for dimensional parameters (in cgs, for now), 'larry'
            which nondimensionalizes elastic parameters, radii, and viscosities, or
            'love' which nondimensionalizes everything for use with direct Love
            number computation.
            """

        assert normmode in ['love', 'larry', 'dim'], \
            'normmode {} not supported. See docstring.'.format(normmode)

        # If the object is already in this mode, do nothing.
        if normmode == self.normmode: return

        # Putting units back in.
        if normmode == 'dim':
            if self.normmode == 'love': 
                re = self.norms['r']
                g0 = self.norms['g']
                rhobar = g0/self.G/re

                self._paramArray[[0,4,5]] *= rhobar
                self._paramArray[[1,2]] *= rhobar*re*g0   
                self._paramArray[3] *= g0
                self._paramArray[6] *= self.norms['eta']

                self.rCore *= re
                self.denCore *= rhobar

            elif self.normmode == 'larry':
                self._paramArray[[0,4,5]] *= 1.
                self._paramArray[[1,2]] *= self.norms['mu']
                self._paramArray[3] *= 1.
                self._paramArray[6] *= self.norms['eta']

                self.rCore *= self.norms['r']
                self.denCore *= 1.
                
        # Removing (some) units.
        else:
            # First, redimensionalize,
            self.normalize('dim')
            # then nondimensionalize.
            if normmode == 'love':
                re = self.norms['r']
                g0 = self.norms['g']
                rhobar = g0/self.G/re

                self._paramArray[[0,4,5]] /= rhobar
                self._paramArray[[1,2]] /= rhobar*re*g0   
                self._paramArray[3] /= g0
                self._paramArray[6] /= self.norms['eta']

                self.rCore /= re
                self.denCore /= rhobar
            elif normmode == 'larry':
                self._paramArray[[0,4,5]] /= 1.
                self._paramArray[[1,2]] /= self.norms['mu']
                self._paramArray[3] /= 1.
                self._paramArray[6] /= self.norms['eta']

                self.rCore /= self.norms['r']
                self.denCore /= 1.

        # Set the normmode for reference later and recreate interpolation
        # object.
        self.normmode = normmode
        self._interpParams = interp1d(self.z, self._paramArray)

    def getParams(self, z, depth=False):
        """
        Return a dictionary of parameters interpolated to radius (depth) z.
        """            
        if depth:
            z = self.r - z

        vals = self._interpParams(z)

        return dict(zip(self._paramNames, vals))

    def addViscosity(self, visArray, etaStar=None):
        """visArray is an 2xN array of depths zi and viscosities at those
           depths, in poise."""
        if etaStar is not None:
            self.norms['eta'] = etaStar
        visArray = np.asarray(visArray)
        visArray[1] /= self.norms['eta']        # Normalize viscosities
        self._alterColumn(6, visArray)


    def addNonadiabatic(self, nonad, normed=True):
        """Introduce a nonadiabatic density gradient in the mantle.

        Parameters
        ----------
        nonad : np.ndarray
            A 2xN array of depths and nonadiabatic density gradients.
            NOTE: must be in g/cc / cm (if normed=False) or
                             g/cc / earth radii (if normed=True).
        normed : boolean
            Are the gradients already normalized to earth radii?
        """
        if not normed:
            nonad[0] = nonad[0]*self.norms['r']
        self._alterColumn(5, nonad)

    def addLithosphere(self, D=None, H=None):
        """Append a lithosphere with flexural rigidity D (N m) or of thickness
        H (km)
        """
        if D is not None:
            self.D = D
        elif H is not None:

            re = self.norms['r']
            g0 = self.norms['g']
            rhobar = g0/self.G/re

            paramSurf = self(1.)
            lam = paramSurf['bulk']#34.3 * 1e+10 # dyne / cm^2
            mu =  paramSurf['shear']#26.6 * 1e+10 # dyne / cm^2
            pois = lam/(2*(lam+mu))
            young = mu*(3*lam + 2*mu)/(mu + lam)*rhobar*g0*re #Pa
            # 1e8 converts km^3 dyne / cm^2 to N m
            # 1e9 converts km^3 to m^3 for D to have units N m
            self.D = young * H**3 / (12*(1-pois**2))*1e9
        else:
            raise ValueError('Muse specify either D (in N m) or H (in km)')

    def getLithFilter(self, k=None, n=None):
        """Return the Lithospheric filter value for loads and rates.
        """
        if k is not None:
            pass
        elif n is not None:
            k =  (n + 0.5)/self.norms['r']  # m
        else:
            raise ValueError('Must specify k (m^-1)  or n.')

        re = self.norms['r']
        g0 = self.norms['g']
        rhobar = g0/self.G/re

        paramSurf = self(1.)
        rho = paramSurf['den']*rhobar   # kg / m^3
        g = paramSurf['grav']*g0        # m/s^2
        # 1e1 converts rho*g in dyne/cm^3 to N/m^3
        return 1 + k**4 * self.D / (rho * g)

    def effectiveElasticThickness(self):


        re = self.norms['r']
        g0 = self.norms['g']
        rhobar = g0/self.G/re

        paramSurf = self(1.)
        lam = paramSurf['bulk']#34.3 * 1e+10 # dyne / cm^2
        mu =  paramSurf['shear']#26.6 * 1e+10 # dyne / cm^2
        pois = lam/(2*(lam+mu))
        young = mu*(3*lam + 2*mu)/(mu + lam)*rhobar*g0*re #Pa
         
        # 1e-8 converts (N m) / (dyne / cm^2) to km^3
        # 1e9 converts km^3 to m^3 for D to have units N m
        return (12 * (1-pois**2) * self.D / young *1e-9)**(0.333)

    def _alterColumnPresDisc(self, col, zy):
        """
        Alter a parameter column, while preserving all discontinuities.

        Results in new depth array, with all depth discontinuities from both
        the original parameter array and new column.
        """
        z = zy[0]
        y = zy[1]

        # Create interpolator to join new array into old
        interpY = interp1d(z, y) 
        
        # Find discontinuities in new column
        idalt = locateDiscontinuities(z)
        zdalt = z[idalt]
        # Find discontinuities in old array
        idold = locateDiscontinuities(self.z)
        zdold = self.z[idold]
        # Only need to keep disconstinuities once, in case of overlap
        zd = np.union1d(zdalt, zdold)
        # The new z array is created by unioning and then adding
        # discontinuities back in (once, by previous line)
        zu = np.union1d(z, self.z)
        znew = np.sort(np.r_[zu, zd])

        # The discontinuities in the new array.
        idnew = locateDiscontinuities(znew)
        zdnew = znew[idnew]

        # We interpolate the new column and old array to new z array
        newcolumn = interpY(znew)
        newcolumn = np.interp(znew, z, y)

        newparamArray = self._interpParams(znew) 

        # ans replace discontinuities one at a time
        for i, zi in zip(idnew, zdnew):
            if zi in zdalt:
                itmp, = np.where(zi == z)
                newcolumn[i] = y[itmp[0]]
            if zi in zdold:
                itmp, = np.where(zi == self.z)
                newparamArray[:,i-1] = self._paramArray[:,itmp[0]]

        # Put the new column into the array
        newparamArray[col] = newcolumn

        # and reset all the class data.
        self._paramArray = newparamArray 
        self.z = znew
        self._interpParams = interp1d(self.z, self._paramArray)

    def _alterColumn(self, col, zy):
        self._alterColumnPresDisc(col, zy)
        #z = zy[0]
        #y = zy[1]
        #interpY = interp1d(z, y) 
        #self.z = np.union1d(z, self.z)
        #self._paramArray = self._interpParams(self.z)
        #self._paramArray[col] = interpY(self.z)
        #self._interpParams = interp1d(self.z, self._paramArray)

def locateDiscontinuities(z):
    """Locate where in an array a value is repeated.
    
    Note: it returns the index of the first of each pair of repeated values.
    """

    trash, uniqueInds = np.unique(z, return_inverse=True)
    i, = np.where((uniqueInds[1:]-uniqueInds[:-1]) == 0) 
    return i

def layered_gravity(rs, ds, G=6.674e-11):
    gs = np.zeros_like(rs)
    G43p = 4*np.pi*G/3.
    gs[0] = G43p*rs[0]*ds[0]
    for i, rd in enumerate(zip(rs[1:], ds[1:]), start=1):
        r, d = rd
        gs[i] = (gs[i-1]*rs[i-1]**2 + G43p*d*(r**3-rs[i-1]**3))/r**2

    return gs
