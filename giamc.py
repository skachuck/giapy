"""
giamc.py

Author: Samuel B. Kachuck

This module uses the emcee module to perform Markov Chain Monte Carlo sampling
to invert data for earth parameters.
"""
import numpy as np
import emcee

def lnlike(theta, other_params):
    """

    For boxcar (uniform) prior in range, return 0 in range, -np.inf outside.
    """
    return log_likelihood_of_data_given_theta

def lnprior(theta):
    return prior_of_theta

def lnprob(theta, other_params):
    if not np.isfinite(lnprior):
        return -np.inf
    return lnprior + lnlike

def giamc(blah blah blah):
    ndim = ndim
    nwalkkers = nwalkers
    starting_position = ...
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                        args=(other_params), threads=threads)
    pos, prob, state = sampler.run_mcmc(pos, nsteps)

def forward_model(theta):
    earth.change_stuff(theta)
    sim = giasim.GiaSimGlobal(earth, ice, grid)
    result = sim.performConvolution(topo=topo, eliter=5)

    emergecalc = giadata.calcEmergence(result, emergedata)

"""
For writing out during steps. sampler.sample is a generator.
f = open("chain.dat", "w")
f.close()

for result in sampler.sample(pos, nsteps, storechain=False):
    position = result[0]
    f = open("chain.dat", "a")
    for k in range(position.shape[0]):
        f.write("{0:4d} {1:s}\n".formate(k, " ".join(position[k])))
    f.close()
"""


