from giapy import pickle
import numpy as np
from scipy.interpolate import interp1d

def gen_eustatic():
    times = np.array([ 0,  6,  7,  8,  
                       9, 10, 11, 12,
                      13, 14, 15, 16, 
                      17, 18, 19, 20])
    
    meters  = np.array([   0,    0,     -4, -14.2,
                         -26,  -44,    -61,   -63,
                         -78,  -85,   -110,  -113,
                        -117, -120, -122.5,  -125])
    
    sealevel_curve = interp1d(times, meters, bounds_error=False, fill_value=0)

    return sealevel_curve

def readEustatic(fname):
    esl = np.loadtxt(fname)
    return interp1d(esl[:,0]/1000., esl[:,1], bounds_error=False, fill_value=0)
