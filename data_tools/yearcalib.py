import numpy as np

def uncalib_bloom(yrs):

    ind = np.where(np.logical_and(yrs>=700, yrs<2100))
    yrs[ind] = yrs[ind]+100

    ind = np.where(np.logical_and(yrs>=2100, yrs<6000))
    yrs[ind] = (yrs[ind]+1100)/1.5

    return yrs
