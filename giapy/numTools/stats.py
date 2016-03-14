"""
stats.py
Author: Samuel B. Kachuck

A module intended to hold numerical tools for statistical analysis.
"""

import numpy as np

class OnlineSamplingCovariance(object):
    """Generate means and covariant matrices online without having to store
    individual samples.

    The algorithm is a robust form of Welford's method. It is not as robust as
    possible (see two pass algorithms).

    Parameters
    ----------
    n : int
        The number of samples included so far.
    mean : float or np.ndarray
        The current vector (or singlet) of means.
    m2n : float or np.ndarray
        The current product of residuals.
    univariate : bool
        Is the distribution univariate? Default False. Note that it is not
        necessary to change this for a univariate distribution, only a
        convenience in calling it up later.
    """
    def __init__(self, m, n=0, mean=None, m2n=None):
        self.n = n
        self.mean = mean
        self.m2n = m2n

        if m > 1:
            self.univariate = False
        elif m == 1:
            self.univariate = True
        else:
            raise ValueError('')

        self.m = m

        self.n = n
        if mean is None and m2n is None:
            if self.univariate:
                self.mean = 0.
                self.m2n = 0.
            else:
                self.mean = np.zeros(m)
                self.m2n = np.zeros((m, m))
        elif mean is not None and m2n is not None:
            self.mean = mean
            self.m2n = m2n
        else:
            raise ValueError('Specify mean and m2n or neither.')

        
    def update(self, x):
        """Update the covariance matrix with a newly sampled pt/vector.

        Parameters
        ----------
        x : float or np.ndarray
            Newly sampled point or vector to include.
        """
        delta = x - self.mean
        if self.univariate:
            self.m2n = self.m2n + self.n/(self.n + 1.) * delta**2
        else:
            self.m2n = self.m2n + self.n/(self.n + 1.) * np.outer(delta, delta)
        self.mean = self.mean + delta/(self.n + 1.)
        self.n = self.n + 1
        
    def combine(self, other):
        """Combine two covariance matrices together into one.
        
        Parameters
        ----------
        self, other : <OnlineSamplingCovariance>
            The two online covariances to combine.

        Results
        -------
        combination : <OnlineSamplingCovariance>
            The combination of the covariances. The returned object has updated
            means and sample counts (combination.n = self.n + other.n).
        """
        
        assert self.univariate==other.univariate
        if not self.univariate:
            assert self.m2n.shape == other.m2n.shape
            
        delta = other.mean - self.mean
        n = self.n + other.n
        mean = (self.n*self.mean + other.n*other.mean)
        
        if self.univariate:
            m2n = self.m2n + other.m2n + (delta**2)*self.n*other.n
        else:
            m2n = self.m2n + other.m2n + np.outer(delta, delta)*self.n*other.n
            
        return OnlineSamplingCovariance(self.m, n=n, mean=mean, m2n=m2n)
    
    @property
    def cov(self):
        """The population covariance matrix."""
        if self.n < 2:
            return np.nan
        else:
            return self.m2n/(self.n - 1)
