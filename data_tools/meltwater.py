import cPickle as pickle
import numpy as np
from scipy.interpolate import interp1d

class 

def generate_meltwater_interpolator(filename):
    times = np.array([ 0,  6,  7,  8,  
                       9, 10, 11, 12,
                      13, 14, 15, 16, 
                      17, 18, 19, 20])
    
    meters  = np.array([   0,    0,     -4, -14.2,
                         -26,  -44,    -61,   -63,
                         -78,  -85,   -110,  -113,
                        -117, -120, -122.5,  -125])
    
    sealevel_curve = interp1d(times, meters)

    pickle.dump(sealevel_curve, open(filename, 'w'))
