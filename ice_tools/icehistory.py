import numpy as np
import cPickle as pickle

def load(filename):
    return pickle.load(open(filename, 'rb'))

class Ice2d(object):
    def __init__(self):
        self.times = np.array([20, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8])
        #icefile = '/Users/skachuck/Documents/Work Documents/GIA_Modeling/2d_model/Fullice_Aleksey.dat'
        #Lat, Lon, ice_stages = icehistory.readice(icefile, 471, 491)
        
    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))
    
    def load_data(self, filename):
        self.heights = pickle.load( open( filename, 'rb' ) )
        self.shape = np.shape(self.heights[0,:,:])
        self.N = 512
        
    def readice(self, filename, Nx, Ny):
        """Read in a full comma-delimitted ice file in x-y-z format.
    
        Parameters:
            filename - the file to be read
            Nx - number of latitude sites
            Ny - number of longitude sites
    
        Returns:
            lat, lon - size (Nx, Ny) arrays of latitude and longitude measures
            height - size (num of stages, Nx, Ny) array of ice heights
        """
        rawdata = np.loadtxt(filename, delimiter=',')
        self.lon = (np.reshape(rawdata[:,0], (-1, Nx, Ny))[0, :, :])[0,:]
        self.lat = (np.reshape(rawdata[:,1], (-1, Nx, Ny))[0, :, :])[:,0]
        self.heights = np.reshape(rawdata[:,2], (-1, Nx, Ny))
        self.N = 512
        self.shape = np.shape(self.heights[0,:,:])
        
    def fft(self, N):
        fullfft = np.fft.fft2(self.heights, s=[self.N,self.N])
        truncfft = np.zeros((11, N, N), dtype=complex)
        truncfft[:,:N/2,:N/2]=fullfft[:,          :N/2,           :N/2]
        truncfft[:,:N/2,N/2:]=fullfft[:,          :N/2, self.N-N/2:   ]
        truncfft[:,N/2:,:N/2]=fullfft[:,self.N-N/2:   ,           :N/2]
        truncfft[:,N/2:,N/2:]=fullfft[:,self.N-N/2:   , self.N-N/2:   ]
        return truncfft
        
class IceHistory(object):
    def __init__(self):
        pass
    
    def singiter(self):
        def singGenerator(self):
            for f in self.filenames:
                self.load_data(f)
                yield self.heights
        return singGenerator
