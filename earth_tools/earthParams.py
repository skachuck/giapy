"""

"""
import numpy as np
from scipy.interpolate import interp1d

############ PRELIMINARY EARTH REFERENCE MODEL FOR MANTLE #############
#          (first entry is the core)
#          Radius     Density     Bulk mod    Shear mod   Gravity
#          (km)       (g/cc)      (GPa)       (GPa)       (cm/s^2)
prem = \
np.array([[ 3480.    ,  9.90349  ,   644.1   ,     0.    ,    1068.23],
          [ 3480.    ,  5.56645  ,   655.6   ,   293.8   ,    1068.23],
          [ 3600.    ,  5.50642  ,   644.    ,   290.7   ,    1052.04],
          [ 3630.    ,  5.49145  ,   641.2   ,   289.9   ,    1048.44],
          [ 3630.    ,  5.49145  ,   641.2   ,   289.9   ,    1048.44],
          [ 3800.    ,  5.40681  ,   609.5   ,   279.4   ,    1030.95],
          [ 4000.    ,  5.30724  ,   574.4   ,   267.5   ,    1015.8 ],
          [ 4200.    ,  5.20713  ,   540.9   ,   255.9   ,    1005.35],
          [ 4400.    ,  5.1059   ,   508.5   ,   244.5   ,     998.59],
          [ 4600.    ,  5.00299  ,   476.6   ,   233.1   ,     994.74],
          [ 4800.    ,  4.89783  ,   444.8   ,   221.5   ,     993.14],
          [ 5000.    ,  4.78983  ,   412.8   ,   209.8   ,     993.26],
          [ 5200.    ,  4.67844  ,   380.3   ,   197.9   ,     994.67],
          [ 5400.    ,  4.56307  ,   347.1   ,   185.6   ,     996.98],
          [ 5600.    ,  4.44317  ,   313.3   ,   173.    ,     999.85],
          [ 5600.    ,  4.44317  ,   313.3   ,   173.    ,     999.85],
          [ 5701.    ,  4.38071  ,   299.9   ,   154.8   ,    1001.43],
          [ 5701.    ,  3.99214  ,   255.6   ,   123.9   ,    1001.43],
          [ 5771.    ,  3.97584  ,   248.9   ,   121.    ,    1000.38],
          [ 5771.    ,  3.97584  ,   248.9   ,   121.    ,    1000.38],
          [ 5871.    ,  3.8498   ,   218.1   ,   105.1   ,     998.83],
          [ 5971.    ,  3.72378  ,   189.9   ,    90.6   ,     996.86],
          [ 5971.    ,  3.54325  ,   173.5   ,    80.6   ,     996.86],
          [ 6061.    ,  3.48951  ,   163.    ,    77.3   ,     993.61],
          [ 6151.    ,  3.43578  ,   152.9   ,    74.1   ,     990.48],
          [ 6151.    ,  3.3595   ,   127.    ,    65.6   ,     990.48],
          [ 6221.    ,  3.3671   ,   128.7   ,    66.5   ,     987.83],
          [ 6291.    ,  3.37471  ,   130.3   ,    67.4   ,     985.53],
          [ 6291.    ,  3.37471  ,   130.3   ,    67.4   ,     985.53],
          [ 6346.6   ,  3.38076  ,   131.5   ,    68.2   ,     983.94],
          [ 6346.6   ,  2.900    ,    75.3   ,    44.1   ,     983.94],
          [ 6356.    ,  2.900    ,    75.3   ,    44.1   ,     983.32],
          [ 6356.    ,  2.600    ,    52.    ,    26.6   ,     983.32],
          [ 6371.    ,  2.600    ,    52.    ,    26.6   ,     982.22]])

class EarthParams(object):
    """Store and interpolate Earth's material parameters.

    Uses PREM (Dziewonski & Anderson 1981) for density and elastic parameters.
    if visArray is None, assumes a uniform 1e21 Pa s mantle.

    visArray should be a 2xN array or depths and viscosities.
    """
    def __init__(self, visArray=None):        
        self.G = 4*np.pi*6.674e-8               # cm^3/g.s^2
        
        self.norms = {'r'  :     6.371e+8,      # cm
                      'eta':     1e+22   ,      # poise = g/cm.s    
                      'mu' :     2.938e12}      # dyne/cm^2

        self.rCore = prem[0,0]/prem[-1,0]       # earth radii
        self.denCore = prem[0,1]                # g/cc

        z = prem[1:,0]/prem[-1, 0]              # Normalized depths in mantle.
        prem[1:, [2,3]] /= prem[1, 3]           # Normalized elastic props by
                                                # shear modulus.

        # Create density gradient, g/cc.earthRadii
        dend = np.gradient(prem[1:,1])/np.gradient(z)

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
        self._paramArray = np.concatenate((prem[1:,1:], dend[:,np.newaxis], 
                                            filler, filler), axis=1).T

        self._interpParams = interp1d(z, self._paramArray)

        zDisc = locateDiscontinuities(z)        # Save discontinuities.
        self.z = np.union1d(z, zDisc)

        self._paramArray = self._interpParams(self.z)
        self._interpParams = interp1d(self.z, self._paramArray)

        # Set up viscosity profile with uniform viscosity
        if visArray is None:
            visArray = np.array([[z[0]      , z[-1]      ],
                                 [1.0       , 1.0        ]])
        self.addViscosity(visArray)

    def __call__(self, z, depth=False):
        return self.getParams(z, depth)

    def getParams(self, z, depth=False):
        """
        Return a dictionary of parameters interpolated to radius (depth) z.
        """            
        if depth:
            z = self.r - z

        vals = self._interpParams(z)

        return dict(zip(self._paramNames, vals))

    def addViscosity(self, visArray, muStar=1e22):
        """visArray is an 2xN array of depths zi and viscosities at those
           depths."""
        visArray[1] /= visArray[1,0]        # Normalize viscosities
        self.alterColumn(6, visArray)


    def addNonadiabatic(self, z, nonad):
        self.alterColumn(5, nonad)

    def alterColumn(self, col, zy):
        z = zy[0]
        y = zy[1]

        interpY = interp1d(z, y)
        zDisc = locateDiscontinuities(z)
        z = np.union1d(z, zDisc)

        # Generate new interpolation array
        self.z = np.union1d(self.z, z)

        self._paramArray = self._interpParams(self.z)

        self._paramArray[col] = interpY(self.z)
        self._interpParams = interp1d(self.z, self._paramArray)


def locateDiscontinuities(z, eps=None):
    eps = eps or z.max()*1e-5
    trash, uniqueInds = np.unique(z, return_inverse=True)
    i, = np.where((uniqueInds[1:]-uniqueInds[:-1]) == 0)
    zDisc = z[i]
    zDisc = np.r_[z[i]-eps, z[i]+eps]
    return zDisc

