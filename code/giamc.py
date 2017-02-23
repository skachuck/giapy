"""
giamc.py

    Methods and classes to support MCMC calculations using the emcee module.

    Author: Samuel B. Kachuck
"""

import numpy as np
import time
import sys
import os

import emcee

from .. import GITVERSION, timestamp, pickle

def gen_metadatastr(metadatadict):
    """Generate a string of line-separated metadata from a dictionary.

    This string can be written onto the beginning of a results file to carry
    around important information for the run (as defined by fields in
    metadatadict). The first line of the string contains the number of lines of
    the metadata (to ease automated reading in later). The function adds the
    git hash of the current version of the code (GITVERSION) and the time of
    metadata creation (timestamp).

    Parameters
    ----------
    metadatadict : dict
        Dictionary with metadata categories (keys) and values (values).

    Results
    -------
    metadatastr : str
        A string representation of the metadata dictionary, with keys and
        values tab separated and successive entries line separated. The first
        line contrains the number of lines following. 

    Notes
    -----
    Some things of interest might be:
        - ndim, nwalkers, nblobs (these are necessary for get_metadatadict to work)
        - arguments (i.e., data, earth model, ice model areas)
    """

    metadatadict['GITVERSION'] = GITVERSION
    metadatadict['timestamp'] = timestamp()

    metadatastr = ''
    linecount = 1

    for key, value in metadatadict.iteritems():
        metadatastr += '{0}\t{1}\n'.format(key, str(value).replace('\n', ''))
        linecount += 1

    # Append the linecount to the beginning.
    metadatastr = '{0}\n'.format(linecount) + metadatastr
    
    return metadatastr

def get_metadatadict(fname):
    """Retrieve the metadata from an MCMC result as a dictionary.

    Parameters
    ----------
    fname : str
        The file path of the MCMC result. The first line of the file must be
        the number of lines of metadata.

    Results
    -------
    metadatadict : dict
        A dictionary of the metadata.

    """
    with open(fname, 'r') as f:
        metadatadict = {}
        linecount = int(f.readline())
        for i in range(linecount-1):
            line = f.readline().replace('\n','').split('\t')
            metadatadict[line[0]] = line[1]
        metadatadict['linecount'] = int(linecount)

    try:
        metadatadict['ndim'] = int(metadatadict['ndim'])
        metadatadict['nwalkers'] = int(metadatadict['nwalkers'])
        metadatadict['nblobs'] = int(metadatadict['nblobs'])
    except KeyError:
        pass

    return metadatadict


def readMCMCresult(fname, metadata=False):
    """Read in a saved MCMC result.

    Returns an array of samples, blobs, and (if metadata is True) the metadata.
    """
    metadatadict = get_metadatadict(fname)
    linecount = metadatadict['linecount']

    nwalkers = metadatadict['nwalkers'] 
    ndim     = metadatadict['ndim']
    nblobs   = metadatadict['nblobs']

    fulloutput = np.loadtxt(fname, skiprows=linecount)
    fulloutput = fulloutput.reshape((-1, nwalkers, ndim+2+nblobs))
    fulloutput = fulloutput.transpose([1,0,2])

    probs = fulloutput[:,:,1]
    samples = fulloutput[:,:,2:2+ndim]
    blobs = fulloutput[:,:,2+ndim:2+ndim+nblobs]

    if metadata:
        return probs, samples, blobs, metadatadict
    else:
        return probs, samples, blobs
    
def make_uniform_lnprior(lower=None, upper=None):
    """Generate a boxcar (uniform) prior between lower and upper bounds.
    The function returns a normalized distribution if both bounds are finite,
    an unnormalized distribution if either bound is missing.

    Parameters
    ----------
    lower  : float or np.ndarray
    upper  : float or np.ndarray
        The lower and upper bounds of the parameter(s). Must have same shape as
        params.
    """
    assert lower is not None or upper is not None, \
    'Must specifiy either lower or upper bound of uniform prior'

    # Interpret input: making Nones into -np.inf or np.inf where necessary.
    try:
        if lower is None:
            lower = np.repeat(-np.inf, len(upper))
        if upper is None:
            upper = np.repeat(np.inf, len(lower))
    except:
        lower = lower or -np.inf
        upper = upper or np.inf

    # If both lower and upper are specified (and therefore finite), we can
    # create a normalized uniform distribution.
    if np.all(np.isfinite(np.r_[lower, upper])):
        lnvol = np.sum(np.log(1./(upper - lower)))
    # Otherwise, assume the uniform dist is of order unity.
    else:
        lnvol = 0.0

    def uniform_lnprior(params, *args, **kwargs):
        assert np.asarray(params).shape == np.asarray(upper).shape ==\
            np.asarray(lower).shape, 'Input shapes incompatible.' 

        if np.all(np.logical_and( lower <= params, params <= upper)): 
            return lnvol
        else:
            return -np.inf

    return uniform_lnprior

def sampleOut(sampler, pos, lnprob0, blobs0, fname, nsteps, 
                verbose=False, resCov=None, burnin=0, resCovDump=0,
                corrSkip=1):
    """Iteratively sample and store from an emcee Sampler starting at pos.

    If the output file exists, the results are appended. If not, it is created.

    Parameters
    ----------
    sampler : <emcee.EnsembleSampler>
        An instance of the emcee EnsemleSampler class.
    pos : np.ndarray
        The starting position og the sampler. Must have a shape 
        (nwalkers, ndim).
    fname : str
        The name of the output file to write to.
    nsteps : int
        The number of steps to take in this sampling.
    blobs: bool
        If the sampler also returns predictive blobs to store (default False).
    verbose : bool
        If terminal feedback on calculation is desired (default False).
    resCov : <giapy.numTools.stats.OnlineSamplingCovariance>
        An updatable covariance object. Default None. If not None, it is
        assumed the post-probability function of sampler outputs all residual
        values as blobs after the actual blobs. The residuals are removed from
        blobs before the sampled blobs are stored.
    burnin : int
        Number of steps of nsteps to take as burn-in (not stored). Note that
        the number of written-out steps will be nsteps - burnin.
    resCovDump : int
        Number of steps at which to write save the covariance matrix out to a
        file str(hash(filename))+'.p'. Default 0 means never to dump (can dump
        after sampling from calling function).
    corrSkip : int
        Number of samples to skip before writing out (to avoid correlated
        samples during MCMC stepping).
    """
    # Check if the file exists and, if not,
    try:
        f = open(fname, 'r')
        f.close()
    # create it.
    except IOError:
        f = open(fname, 'w')
        f.close()

    fnoext = os.path.splitext(fname)[0]

    if verbose: 
        tstart = time.time()
        outmsg = 'Taking step {0:d}/{1:d}\033[K\r'
        donemessage = ''

    for i, step in enumerate(sampler.sample(pos, lnprob0=lnprob0, blobs0=blobs0, 
                                iterations=nsteps, storechain=False), start=2):
        if verbose:
            sys.stdout.write(outmsg.format(i, nsteps))
            sys.stdout.flush()

        # If burning in or skipping correlation, skip write-out.
        if i < burnin or i%corrSkip != 0:
            continue

        # For each step we create an output dump.
        output = ''
        # Iterate over the walkers.
        for k in range(step[0].shape[0]):

            # Write the logProbability.
            output += '{0:d}\t{1:e}\t'.format(k, step[1][k])
            # Write out all parameters.
            for param in step[0][k]:
                output += '{0:f}\t'.format(param)
            # Write out the blobs, if asked to.
            if resCov is not None:
                res = step[3][k][-resCov.m:]
                blobs = step[3][k][:-resCov.m]
                if np.isfinite(step[1][k]):
                    resCov.update(res)
            else:
                blobs = step[3][k]

            for blob in blobs:
                output += '{0}\t'.format(blob)
            # End the line.
            output += '\n'
        # Dump the output.
        with open(fname, 'a') as f:
            f.write(output)

        if resCov is not None and resCovDump and i%resCovDump == 0:
            pickle.dump(resCov, open(fnoext+'_resCov.p', 'w'), -1)

    # Post sampling message.
    if verbose:
        ttotal = time.time() - tstart
        if ttotal > 3600:
            ttotal = ttotal / 3600
            unit = 'hrs'
        elif ttotal > 60:
            ttotal = ttotal / 60
            unit = 'min'
        else:
            unit = 's'
        avgaccept = np.mean(sampler.naccepted)/nsteps
        donemessage += '{0:3d} steps in {1:.3f} {2}, with {3:.3f} accepted.\033[K\n'
        sys.stdout.write(donemessage.format(nsteps, ttotal, unit, avgaccept))

def autocorr(samples, nburn=0):
    """Compute the average autocorrelation time (in steps) between walkers for
    each variable in an MCMC sampling.

    Parameters
    ----------
    samples : np.ndarray (nwalkers, nsamples, ndim)
    nburn   : int, initial number of samples to skip.
    
    Returns
    -------
    avgauto : np,ndarray (ndim)
        The average autocorrelation over walkers for each variable.
    """

    avgauto = np.array([np.mean([emcee.autocorr.integrated_time(walker) 
                                        for walker in var.T]) 
                                        for var in samples[:,nburn:,:].T])

    return avgauto

def flatten_walkers(samplelist, nburnlist=None, ncorrlist=None):
    """Combine samples from independent samplers (i.e., computers).

    (Can be used on blobs as well.)

    Parameters
    ----------
    samplelist : list of np.ndarray (nwalkers, nsamples, ndim)
    nburnlist  : list of ints corresponding to burn in correspinding sample.
    ncorrlist  : list of ints corresponding to the number to skip in
        corresponding sample

    Returns
    -------
    long_samples : np.ndarray (nsamples, ndim)
        nsamples = Sum [(nsamples_i - nburn_i) / ncorr_i]
    """
    
    ndims = np.array([samples.shape[2] for samples in samplelist])
    assert np.all(ndims == ndims[0]), 'All samples must have same ndim.'
    ndim = ndims[0]

    if nburnlist is not None:
        assert len(nburnlist) == len(samplelist), \
            'Must specify nburn for each samples.'
    if ncorrlist is not None:
        assert len(ncorrlist) == len(samplelist), \
            'Must specify ncorr for each samples.'

    nburnlist = nburnlist or np.zeros(len(samplelist))
    ncorrlist = ncorrlist or np.ones(len(samplelist))

        
    long_samples = np.vstack([s[:,nburn::ncorr,:].reshape((-1, ndim)) 
                                for s, nburn, ncorr 
                                in zip(samplelist, nburnlist, ncorrlist)])



    return long_samples

    
        

