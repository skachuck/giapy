import numpy as np
import spharm

def get_gravrate(result, t0=-0.1, t1=0.1, specout=False):
    n0 = result['grav'].locateByTime(t0)
    n1 = result['grav'].locateByTime(t1)

    # x5 changes units to microGal / yr and the whole thing is negative because
    # the gravity anomaly is measured oppositely to the gravity perturbation.
    gravrate = -(result['grav'][n0] - result['grav'][n1])*5
    # Transform back if desired.
    if not specout:
        gravrate = result.inputs.harmTrans.spectogrd(gravrate)

    return gravrate

def grav_response_less_hydroisostacy(result, t0, trem=1):
    """Get grav response at t0 from ice mass changes ONLY from trem on.
    """

    ms, ns = spharm.getspecindx(result.inputs.nlat-1)

    # Identify the load stage associated with first ice mass change desired.
    remind = np.argmin(np.abs(result['load'].outTimes - trem))

    # Identify the maximum load change stage (can only consider ice mass
    # changes before t0.
    maxind = np.argwhere(result['load'].outTimes - t0 < 0)
    if maxind.shape[0]:
        maxind = min(maxind)
    else:
        maxind = None

    # Index slice for load.
    rslice = slice(remind, maxind)
    # Iterator for times, total load changes, and water load changes.
    loadzip = zip(result['load'].outTimes[rslice], 
                  result['load'][rslice], 
                  result['wload'][rslice])

    dg = np.zeros(result.inputs.grid.shape)
    for t, dL, dW in loadzip:
        # Subtract water load change from total chagne to isolate ice changes.
        dLspec = result.inputs.harmTrans.grdtospec(dL - dW)
        respArray = result.inputs.earth.getResp(t - t0)
        # Multiply the resp array and dLspec with things to get right units.
        tmpresp = respArray[ns, 5]*1e3 * dLspec*101068.38
        dg += result.inputs.harmTrans.spectogrd(tmpresp)

    return dg

def grav_response_from_hydroisostacy(result, t0, trem=1):
    """Get grav response at t0 from ice mass changes ONLY from trem on.
    """

    ms, ns = spharm.getspecindx(result.inputs.nlat-1)

    # Identify the load stage associated with first ice mass change desired.
    remind = np.argmin(np.abs(result['wload'].outTimes - trem))

    # Identify the maximum load change stage (can only consider ice mass
    # changes before t0.
    maxind = np.argwhere(result['wload'].outTimes - t0 < 0)
    if maxind.shape[0]:
        maxind = min(maxind)
    else:
        maxind = None

    # Index slice for load.
    rslice = slice(remind, maxind)
    # Iterator for times, total load changes, and water load changes.
    loadzip = zip(result['wload'].outTimes[rslice],
                  result['wload'][rslice])

    dg = np.zeros(result.inputs.grid.shape)
    for t, dW in loadzip:
        # Subtract water load change from total chagne to isolate ice changes.
        dLspec = result.inputs.harmTrans.grdtospec(dW)
        respArray = result.inputs.earth.getResp(t - t0)
        # Multiply the resp array and dLspec with things to get right units.
        tmpresp = respArray[ns, 5]*1e3 * dLspec*101068.38
        dg += result.inputs.harmTrans.spectogrd(tmpresp)

    return dg

def get_gravrate_less_modern_iceloss(result, trem=1):
    """Return present day gravrate from GIA result less the gravity signal from
    more contemporary iceloss (from the trem thousand years). We explicitly
    include the gravity signal from the hydroisostacy caused (primarily by the
    local deglaciation?).
    """


    gravrate = get_gravrate(result)

    dgn1 = grav_response_less_hydroisostacy(result, -0.1, trem)
    dgp1 = grav_response_less_hydroisostacy(result, 0.1, trem)

    gravrate += (dgn1 - dgp1)*5

    return gravrate


def sphgauss_lp(lmax, lpsig):
    """Compute a Gaussian low-pass filter of halfwidth lpsig for spherical order
    numbers up to (and including) lmax.
    """
    ws = np.array([w for w in _yield_sphgauss_lp(lmax, lpsig)])
    return ws 

def sphgauss_hp(lmax, hpsig):
    """Compute a Gaussian high-pass filter of halfwidth hpsig for spherical order
    numbers up to (and including) lmax. NOTE: W_hp = 1. - W_lp.
    """
    ws = 1. - sphgauss_lp(lmax, hpsig)
    return ws

def sphgauss_bp(lmax, lpsig, hpsig):
    ws = sphgauss_lp(lmax, lpsig) * sphgauss_hp(lmax, hpsig)
    return ws



def _yield_sphgauss_lp(lmax, r, a=6371.):
    """Generate a Gaussian low-pass filter in spherical harmonics.

    Implements the reccurrence relation from 

    Parameters
    ----------
    lmax : The largest order number to take the filter to.
    r    : The halfwidth of the gaussian filer (in km).
    a    : The radius of the Earth (in km).

    Yields
    ------
    Value of the filter for successive order numbers.
    """
    # The minimum value of the filter, for stability (to ensure non-negative
    # filter values only)
    EPS = 1e-12

    b = np.log(2)/(1 - np.cos(r/a))
    bi = 1./b

    # W0, the initial $W_{l-2}$, to be updated with each step.
    # Also, the normalization.
    Wlm2 = 1.
    # W1, the initial $W_{l-1}$, to be updated with eac step.
    Wlm1 = (((1 + np.exp(-2*b)) / (1 - np.exp(-2*b))) - bi)

    l = 1
    while l <= lmax:
        if l==0: yield Wlm2
        elif l==1: yield Wlm1 * Wlm2
        else:
            if Wlm1 * Wlm2 > EPS:
                # The recurrence relationship.
                Wl = -(2*l + 1.)*bi*Wlm1 + 1.
                # Save values for later, keeping the normalization going.
                Wlm2, Wlm1 = Wlm1*Wlm2, Wl
                yield Wl * Wlm2
            else:
                yield EPS
        l += 1
