"""
icehistory.py

    Objects and methods for reading, manipulating, and using maps of ice
    heights over time.

    Author: Samuel B. Kachuck
"""

import numpy as np
import cPickle as pickle

def load(filename):
    return pickle.load(open(filename, 'rb'))

class Ice2d(object):
    """Store and use a small ice model."""
    def __init__(self):
        self.times = np.array([20, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8])
        self._alterDict = {}
        self._desc = ''
        #icefile = '/Users/skachuck/Documents/Work Documents/GIA_Modeling/2d_model/Fullice_Aleksey.dat'
        #Lat, Lon, ice_stages = icehistory.readice(icefile, 471, 491)

    def __str__(self):
        return self._desc+self.printAreas()
        
    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))
    
    def load_data(self, filename):
        self.heights = pickle.load( open( filename, 'rb' ) )
        self.shape = np.shape(self.heights[0,:,:])
        self.N = 512
        
    def readice(self, filename, Nx, Ny):
        """Read in a full comma-delimitted ice file in x-y-z format.
    
        Parameters
        ----------
            filename : the file to be read
            Nx, Ny   : number of lon, lat sites

        Example
        -------
        >>> ice = Ice2d()
        >>> ice.readice(u'./path/to/Fullice_Aleksey.dat', 471, 491)
        """
        rawdata = np.loadtxt(filename, delimiter=',')
        self.lon = (np.reshape(rawdata[:,0], (-1, Nx, Ny))[0, :, :])[0,:]
        self.lat = (np.reshape(rawdata[:,1], (-1, Nx, Ny))[0, :, :])[:,0]
        self.heights = np.reshape(rawdata[:,2], (-1, Nx, Ny))
        self.N = 512
        self.shape = np.shape(self.heights[0,:,:])
        
    def fft(self, N):
        """Return the truncated FFT of ice heights to order N."""
        fullfft = np.fft.fft2(self.heights, s=[self.N,self.N])
        truncfft = np.zeros((11, N, N), dtype=complex)
        truncfft[:,:N/2,:N/2]=fullfft[:,          :N/2,           :N/2]
        truncfft[:,:N/2,N/2:]=fullfft[:,          :N/2, self.N-N/2:   ]
        truncfft[:,N/2:,:N/2]=fullfft[:,self.N-N/2:   ,           :N/2]
        truncfft[:,N/2:,N/2:]=fullfft[:,self.N-N/2:   , self.N-N/2:   ]
        return truncfft

    def addArea(self, name, verts, prop, latlon=True):
        """Add an area to the model to be altered proportionally by amout prop.

        Parameters
        ----------
        name : string
            The name of the area (user convenience)
        verts : list of tuples
            The lon/lat (or map coord pairs) defining the area edges
        prop : float
            The proportion by which to alter the area
        latlon : bool (default True)
            Indicates whether vertices are in lon/lat or map coords

        Result
        ------
        New entry in self.alterDict under key 'name' with values 'verts' 
        and 'prop'.
        """
        self._alterDict[name] = {'verts':verts, 'prop':prop, 'latlon':latlon}

    def editArea(self, name, verts=None, prop=None):
        """Edit an area previously defined and named by addArea."""
        if verts is not None: self._alterDict[name]['verts'] = verts
        if prop is not None: self._alterDict[name]['prop'] = prop

    def alterAreas(self, grid):
        """Multiply the areas in self.heights by their props.

        Parameters
        ----------
        grid : GridObject
            the map grid associated with the ice model            
        """
        grid.update_shape(self.shape)
        for areaDict in self._alterDict.values():
            areaind = grid.selectArea(areaDict['verts'], areaDict['latlon'])
            self.heights[:, areaind] *= areaDict['prop'] 

    def printAreas(self):
        """Print the list of alterations made to the ice model"""
        arealist = ''
        for name, areaDict in self._alterDict.iteritems():
            arealist += '\t {0}: {1}\n'.format(name, areaDict[prop])
        return arealist

class IceHistory(object):
    """An object for loading and using large ice models."""
    def __init__(self):
        pass
    
    def singiter(self):
        def singGenerator(self):
            for f in self.filenames:
                self.load_data(f)
                yield self.heights
        return singGenerator
