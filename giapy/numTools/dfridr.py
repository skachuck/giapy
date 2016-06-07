"""
dfridr.py

    Numerical differentiation of functions. The methods are a translation of
    dfridr in [1], chapter , and an adaptation for functions of multiple
    variables (gradfridr) and multivalue functions (multivariate_dfridr and
    jacfridr).

    Author: Samuel B. Kachuck

    References:
        [1] Press, Flannery, Teukolsky, and Vetterling. Numerical Recipes.
        Cambridge University Press, Cambridge UK.
"""

import numpy as np

def dfridr(f, x, h, fargs=(), fkwargs={}, full_output=False):
    """Returns the derivative of a function f at a point x by Ridders' method
    of polynomial extrapolation. The value h is input as an estimated intial
    stepsize; it need not be small, but rather should be an increment in x over
    which f changes substantially. An estimate of the error in the derivative
    is returned as err.
    """

    NTAB=10                     # Sets the maximum size of tableau
    CON = 1.4; CON2 = CON*CON   # Stepsize decreased by CON at each iteration.
    BIG = np.finfo(np.float64).max
    SAFE = 2.0                  # Return when error is SAFE worse than the best
                                # so far.

    a = np.zeros((NTAB, NTAB))

    hh = h
    a[0,0] = (f(x+hh, *fargs, **fkwargs) - f(x-hh, *fargs, **fkwargs))/(2.0*hh)
    # Initialize zeroth order return.
    ans = a[0,0]
    err = BIG

    for i in range(1, NTAB):
        # Successive columns in the Neville tableau will go to smaller stpsizes
        # and higher orders of extrapolation.
        hh /= CON
        a[0,i] = (f(x+hh, *fargs, **fkwargs) - f(x-hh, *fargs, **fkwargs))/(2.0*hh)   # Try new, smaller stepsize.
        fac = CON2
        for j in range(1, i+1):
            # Compute extrapolations of various orders, requiring no new
            # function evaluations.
            a[j, i] = (a[j-1, i]*fac - a[j-1,i-1])/(fac-1.0)
            fac = CON2*fac
            errt = max(abs(a[j,i] - a[j-1,i]), abs(a[j,i] - a[j-1,i-1]))
            # The error strategy is to compare each new extrapolation to one
            # order lower, both at the present stepsize and the previous one.

            if errt <= err:
                # If error is decreased, save the improved answer.
                err=errt
                ans = a[j,i]

        if abs(a[i,i] - a[i-1,i-1]) >= SAFE*err:
            # If higher order is worse by a significant SAFE factor, then quit
            # early.
            break
    if full_output:
        return ans, err, hh, a
    else:
        return ans

def gradfridr(f, x, h, fargs=(), fkwargs={}):
    """Apply dfridr to each dimension of x independently.
    """
    x = np.atleast_1d(x)
    h = np.atleast_1d(h)
            
    grad = np.zeros_like(x)
                    
    for i, hh in enumerate(h):
        f1arg = make_f_1arg(f, x, i)
        grad[i] = dfridr(f1arg, x[i], hh, fargs, fkwargs)

    return grad


def multivalued_dfridr(f, x, h, fargs=(), fkwargs={}, full_output=False):
    """Returns the derivative of a function f at a point x by Ridders' method
    of polynomial extrapolation. The value h is input as an estimated intial
    stepsize; it need not be small, but rather should be an increment in x over
    which f changes substantially. An estimate of the error in the derivative
    is returned as err.
    """

    NTAB=10                     # Sets the maximum size of tableau
    CON = 1.4; CON2 = CON*CON   # Stepsize decreased by CON at each iteration.
    BIG = np.finfo(np.float64).max
    SAFE = 2.0                  # Return when error is SAFE worse than the best
                                # so far.

    
    hh = h
    tmp = (f(x+hh, *fargs, **fkwargs) - f(x-hh, *fargs, **fkwargs))/(2.0*hh)
    try:
        ndim = len(tmp)
    except:
        raise TypeError("f isn't multivalued, use dfridr.")

    a = np.zeros((NTAB, NTAB, ndim))
    a[0,0] = tmp
    err = BIG

    for i in range(1, NTAB):
        # Successive columns in the Neville tableau will go to smaller stpsizes
        # and higher orders of extrapolation.
        hh /= CON
        a[0,i] = (f(x+hh, *fargs, **fkwargs) - f(x-hh, *fargs, **fkwargs))/(2.0*hh)   # Try new, smaller stepsize.
        fac = CON2
        for j in range(1, i+1):
            # Compute extrapolations of various orders, requiring no new
            # function evaluations.
            a[j, i] = (a[j-1, i]*fac - a[j-1,i-1])/(fac-1.0)
            fac = CON2*fac
            errt = max(np.abs(a[j,i] - a[j-1,i]).max(), 
                        np.abs(a[j,i] - a[j-1,i-1]).max())
            # The error strategy is to compare each new extrapolation to one
            # order lower, both at the present stepsize and the previous one.

            if errt <= err:
                # If error is decreased, save the improved answer.
                err=errt
                ans = a[j,i]

        if np.all(abs(a[i,i] - a[i-1,i-1]) >= SAFE*err):
            # If higher order is worse by a significant SAFE factor, then quit
            # early.
            break
    if full_output:
        return ans, err, hh, a
    else:
        return ans

def jacfridr(f, x, h, ndim, fargs=(), fkwargs={}):
    x = np.atleast_1d(x)
    h = np.atleast_1d(h)
            
    jac = np.zeros((len(x), ndim))
                    
    for i, hh in enumerate(h):
            f1arg = make_f_1arg(f, x, i)
            jac[i] = multivalued_dfridr(f1arg, x[i], hh, fargs, fkwargs)
                                                
    return jac

def make_f_1arg(f, x, axis=0):
    """Make a function of multiple variables into a function of one variable
    x[axis] with the other x[i != axis] values fixed.
    """
    xold = x.copy()
    def f_1arg(x1, *args, **kwargs):
        xnew = xold.copy()
        xnew[axis] = x1
        return f(xnew, *args, **kwargs)
    return f_1arg
