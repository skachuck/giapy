"""
giaflat.py

Author: Samuel B. Kachuck
Date: September 9, 2018

    Compute GIA on a flat earth.
"""

import numpy as np

def compute_2d_uplift_stage(t, ice, dx, dy, rate=False, **ekwargs):

    padfac = 1
    ny, nx = ice.shape
    upl = np.zeros((ny*padfac, nx*padfac))
    dload = np.zeros((ny*padfac, nx*padfac))
    padsl = np.s_[0:ny, 0:nx]

    taus, elup, alpha = calc_earth(nx, ny, dx, dy, **ekwargs)

    # t = 0 is defined with no uplift.
    if t <= 0: return upl

    for ice0, bas0, t0, ice1, bas1, t1 in ice.pairIter():
        # Only propagate response from earlier times.
        if t0 >= t:
            break
        dIload = (thickness_above_floating(ice1,bas1)
                            -thickness_above_floating(ice0,bas0)) 
        dWload = 0#ocean_uplift_load(bas0, bas1)
        dload[padsl] = dIload# + dWload

        dload_f = np.fft.fft2(dload)
        dur = t - t0

        unit_resp = -(1 - np.exp(dur/taus*alpha/1e3))
        if rate:
            upl += np.real(0.3*np.fft.ifft2(unit_resp/taus/1e3*dload_f))
        else:    
            # propagate response to current stage
            upl += np.real(0.3*np.fft.ifft2(unit_resp/alpha*dload_f))
            # Add elastic uplift off lithosphere
            upl += np.real(0.3*np.fft.ifft2((1-1./alpha)*elup*dload_f)) 

    return upl

def propagate_2d_adjustment(t, upl, **ekwargs):
    """
    Propagate the effect of load change from t0 to t1 to future times
    """

    padfac = 1
    nx, ny = 128, 192
    upl = np.zeros((len(ice.times)-1, ny*padfac, nx*padfac))
    dload = np.zeros((1, ny*padfac, nx*padfac))
    padsl = np.s_[..., 0:ny, 0:nx]

    i = 0
    for ice0, t0, ice1, t1 in ice.pairIter():
        dload[padsl] = ice1-ice0
        dload_f = np.fft.fft2(dload)
        durs = ice.times[1:,None,None] if i == 0 else ice.times[1:-i,None,None]
        unit_resp = -(1 - np.exp(durs/taus*alpha/1000))
        # propagate response to current and all future stages
        upl[i:] += np.real(0.3*np.fft.ifft2(unit_resp/alpha*dload_f))
        # Add elastic uplift off lithosphere
        upl[i:] += np.real(0.3*np.fft.ifft2((1-1./alpha)*elup*dload_f[0])) 
        i += 1

    return upl

def thickness_above_floating(thk, bas, beta=0.9):
    """Compute the (water equivalent) thickness above floating.

    thk - ice thickness
    bas - topography
    beta - ratio of ice density to water density (defauly=0.9)
    """
    #   Segment over ocean, checks for flotation    Over land
    taf = (beta*thk+bas)*(beta*thk>-bas)*(bas<0) + beta*thk*(bas>0)
    return taf

def ocean_uplift_load(bas0, bas1):
    """Compute the (water equivalent) ocean load from uplift/subsidence.

    bas0, bas1 - the base at beginning and end of time step. 
    """
    ocup = -(bas1-bas0)*(bas1<0)
    newsub = - np.maximum(bas0, 0)*(bas0>0)*(bas1<0)
    neweme = bas0*(bas1>0)*(bas0<0)

    return ocup + newsub + neweme


def calc_earth(nx,ny,dx,dy,return_freq=False,**kwargs):
    """Compute the decay constants, elastic uplift, and lithosphere filter.

    Parameters
    ----------
    nx, ny : int
        Shape of flat earth grid.
    dx, dy : float
        Grid spacing.
    
    Returns
    -------

    """ 
    freqx = np.fft.fftfreq(nx, dx)
    freqy = np.fft.fftfreq(ny, dx)
    freq = np.sqrt(freqx[None,:]**2 + freqy[:,None]**2)

    u = kwargs.get('u', 1e0)
    u1 = kwargs.get('u1', None)
    u2 = kwargs.get('u2', None)
    h = kwargs.get('h', None)
    g = kwargs.get('g', 9.8)
    rho = kwargs.get('rho', 3313)
    mu = kwargs.get('mu', 26.6)
    fr23 = kwargs.get('fr23', 1.)

    # Error catching. For two viscous layers, u1, u2, and u3 must be set.
    assert (u1 is not None) == (u2 is not None) == (h is not None), 'Two layer model must have u1 and u2 set.'

    if u1 is not None:
        # Cathles (1975) III-21
        c = np.cosh(freq*h)
        s = np.sinh(freq*h)
        ur = u2/u1
        ui = 1./ur
        r = 2*c*s*ur + (1-ur**2)*(freq*h)**2 + ((ur*s)**2+c**2)
        r = r/((ur+ui)*s*c + freq*h*(ur-ui) + (s**2+c**2))

        u = u1

    else:
        r = 1


    # taus is in kyr
    taus = -2*u*np.abs(freq/g/rho * 1e8/np.pi)*r

    # elup is in m
    elup = -rho*g/2/mu/freq*1e-6
    elup[0,0] = 0

    # alpha is dimensionless
    alpha = 1 + freq**4*fr23/g/rho*1e11

    if return_freq:
        return freq, taus, elup, alpha
    else:
        return taus, elup, alpha
