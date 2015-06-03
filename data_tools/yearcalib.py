import numpy as np
from scipy.interpolate import interp1d
from . import C14TABLE

def uncalib_bloom(yrs):
    """Uncalibrate a set of dates calibrated by Bloom.
    
    Bloom used an older c14 calibration curve, so we need to undo his
    calibration prior to applying the newer one.
    """

    ind = np.where(np.logical_and(yrs>=700, yrs<2100))
    yrs[ind] = yrs[ind]+100

    ind = np.where(np.logical_and(yrs>=2100, yrs<6000))
    yrs[ind] = (yrs[ind]+1100)/1.5

    return yrs

class C14corr(object):
    """Calibrate c14 dates to calibrated years before present.
    
    Parameters
    ----------
    t : array
        The times to calibrate.

    Returns
    -------
    calib_t : array
        Calibrated times.
    """

    # Generate the c14 corrector
    c14array = np.loadtxt(C14TABLE, delimiter=',')
    ind = np.argsort(c14array[:,1])
    
    def __init__(self):
        self.c14corr = interp1d(self.c14array[self.ind,1], 
                                self.c14array[self.ind,0])
    def __call__(self, t):
        return self.c14corr(t)

c14corr = C14corr()
