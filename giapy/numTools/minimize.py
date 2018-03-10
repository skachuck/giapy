"""
minimze.py
author: Samuel B. Kachuck
date: January 4, 2018

Minimization routines. Implemented are a Levenberg-Marquardt as in [1] and the
geodesic Levenberg-Marquardt from [2].

References:
[1] Press, Flannery, Teukolsky, and Vetterling. Numerical Recipes.
    Cambridge University Press, Cambridge UK.

[2] Transtrum and Sethna (2012). Improvements to the Levenberg-Marquardt
    algorithm for nonlinear least-squares minimization. Downloaded from
    http://arxiv.org/abs/1201.5885.
"""

import numpy as np

def lm_minimize(f, x0, jac=None, lup=5, ldo=10, fargs=(), fkwargs={}, jargs=(),
                jkwargs={}, keep_steps=False, j=None, r=None):
    SAFE = 0.5

    x = np.atleast_1d(x0)
    if keep_steps:
        xs = [x]
    if r is None:
        r = f(x, *fargs, **fkwargs)
    l = 100
    I = np.eye(len(x))
    if jac is None:
        jac = lambda xp: jacfridr(f, xp, np.ones_like(x), ndim=len(r),
                                    fargs=fargs, fkwargs=fkwargs)
    if j is None:
        j = jac(x, *jargs, **jkwargs)

    C = 0.5*r.dot(r)

    MAXSTEP = 10
    i = 0

    while i<=MAXSTEP: 
        i += 1

        g = j.T.dot(j) + l*I
        gradC = j.T.dot(r)

        xnew = x - SAFE*np.linalg.inv(g).dot(gradC)
        rnew = f(xnew, *fargs, **fkwargs)
        Cnew = 0.5*rnew.dot(rnew)

        if Cnew < C:
            x = xnew
            r = rnew
            Cnew = C
            l = l/ldo

            if keep_steps:
                xs.append(x)
            
            if np.mean(r.dot(r)) < 1e-5:
                if keep_steps:
                    return x, xs, j, r
                else:
                    return x
            else: 
                j = jac(x, *jargs, **jkwargs)

        else:
            l = l*lup
    if keep_steps:
        return x, xs, j, r
    else:
        return x

def geolm_minimize(f, x0, jac=None, lup=5., ldo=10., fargs=(), fkwargs={}, jargs=(),
                jkwargs={}, keep_steps=False, j0=None, r0=None, geo=False, l0=100,
                maxstep=100, maxfeval=200, maxjeval=50, verbose=True):
    """
    Geodesic-accelerated Levenberg-Marquardt for nonlinear least-squares.

    Parameters
    ----------
    f - the function whose squared sum is to be minimized (reqidual function)
    x0 - the starting point for minimizaiont
    jac - jacobian function of the residuals (default is numerical dfridr)
    lup, ldo - Levenberg-Marquardt parameter stepping factor for up (fail) and
        down (success). (Default are 5 and 10, respectively.)
    fargs, fkwargs - arguments for the residual function (default None)
    jargs, jkwargs - arguments for the jacobian function (default None)
    keep_steps - Boolean for keeping intermediary steps (defautl False)
    j0, r0 - initial residual and jacobian values (default is to recompute).
    geo - Boolean for whether to use geodesic acceleration.

    Returns
    -------
    If keep_steps is False, the location of the minimum. Otherwise, (x, xs, j,r)
        a tuple of the location of the minimum (x), the steps (xs), the final
        jacobian (j) and the final residuals (r).
    """
    SAFE = 0.75
    ALPHA = 2.
    h=0.1

    x = np.atleast_1d(x0)
    r = r0 or f(x, *fargs, **fkwargs)
    n = len(r)
    converged = LMConverged(n)

    verbose_str = '''
Iter {} nfevals {} njevals {} accept {}
---------------------------------------------
    C    = {}
    l    = {}
    x    = {}
    dx   = {}
    |dx| = {}'''

    if geo: 
        verbose_str+='''
    av   = {}'''
    
    jevals = 1
    fevals = 1
    nbad = 0

    if keep_steps:
        xs = [x]
        rs = [r]
        fs = [1]
        js = [1]
        nbads = [0]

    l = l0
    I = np.eye(len(x))
    if jac is None:
        jac = lambda xp: jacfridr(f, xp, np.ones_like(x), ndim=n,
                                    fargs=fargs, fkwargs=fkwargs)
    j = j0 or jac(x, *jargs, **jkwargs)

    C = 0.5*r.dot(r)/n


    i = 0

    while i<=maxstep and jevals <= maxjeval and fevals <= maxfeval: 
        i += 1

        g = j.T.dot(j) + l*I
        gradC = j.T.dot(r)

        try:
            gi = np.linalg.inv(g)
        except:
            print('PROBLEM IN INV, l={}'.format(l))
            break

        dx1 = - gi.dot(gradC)

        if not geo:
            dx2 = 0
        else:
            k = 2/h*((f(x + h*dx1, *fargs, **fkwargs) - r)/h - j.dot(dx1))
            fevals += 1
            dx2 = - 0.5*gi.dot(j.T.dot(k))

            truncerr = 2*np.sqrt(dx2.dot(dx2))/np.sqrt(dx1.dot(dx1)) 


        if geo and truncerr > ALPHA:
            accept = False
        else:
            dx = SAFE*(dx1 + 0.5*dx2)
            xnew = x + dx 
            rnew = f(xnew, *fargs, **fkwargs)
            fevals += 1
            Cnew = 0.5*rnew.dot(rnew)/n
            accept = Cnew < C

            if verbose:
                if not geo:
                    strarg = (i, fevals, jevals, accept, Cnew, l, xnew, dx,
                                np.linalg.norm(dx))
                else:
                    strarg = (i, fevals, jevals, accept, Cnew, l, xnew, dx, 
                                np.linalg.norm(dx), truncerr)

                print(verbose_str.format(*strarg))


        if accept:
            x = xnew
            r = rnew
            C = Cnew
            l = l/ldo

            if keep_steps:
                xs.append(x)
                rs.append(r)
                fs.append(fevals)
                js.append(jevals)
                nbads.append(nbad)
            
            if converged(r, dx):
                if keep_steps:
                    return x, xs, rs, r, j, i, fevals, jevals, fs, js, nbads
                else:
                    return x
            else: 
                j = jac(x, *jargs, **jkwargs)
                jevals += 1

        else:
            l = l*lup
            nbad += 1
    if keep_steps: 
        return x, xs, rs, r, j, i, fevals, jevals, fs, js, nbads
    else:
        return x

class LMConverged(object):
    def __init__(self, n, atol=1e-2, nsteps=5, dxrtol=1e-4, dxatol=1e-6):
        self.n = n
        self.atol=atol
        self.nsteps=nsteps
        self.dxatol = dxatol
        self.dxrtol = dxrtol
        self.Cs = []
        self.dxs = []
    def __call__(self, r, dx):
        C = 0.5*r.dot(r)/self.n
        tests = [C<self.atol]
        if len(self.Cs)>self.nsteps:
            self.Cs.pop(0)
            self.Cs.append(C)
            tests.append(np.all(np.abs(np.diff(self.Cs))<self.dxrtol))
            self.dxs.pop(0)
            self.dxs.append(dx)
            tests.append(np.all(self.dxs<self.dxrtol))
        else:
            self.Cs.append(C)
            self.dxs.append(np.linalg.norm(dx))
        tests.append(np.all(np.abs(dx)<self.dxatol))
        return np.any(tests)
