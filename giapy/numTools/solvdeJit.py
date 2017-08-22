"""
solvde.py

    Two-point boundary condition ODE solution using a relaxation method. The
    method is a translation from the C code in [1], chapter 12.2.

    Author: Samuel B. Kachuck

    References:
        [1] Press, Flannery, Teukolsky, and Vetterling. Numerical Recipes.
        Cambridge University Press, Cambridge UK.
"""
import numpy as np
from numba import jit, void, int64, float64

def solvde(itmax, conv, slowc, scalv, indexv, nb, y, difeq, verbose=False,
            it_count=False):
    """Driver routine for solution of two-point boundary value problems by
    relaxation accelerated using numba.
    
    Parameters
    ----------
    itmax : int, the maximum number of iterations allowed before failure.
    conv : float, the average error tolerance.
    slowc : float, the error, above which, to slow convergence (by slowc/err).
    scalv : array
        An array the same length as the solution vector, giving a typical scale
        of the solution variables, for error calculation.
    indexv : array
        An array giving the indices of the reordered solution variables so that
        the variables constrained by boundary conditions are first.
    nb : int, the number of bottom boundary conditions.
    y : array
        The initial estimate of the solution. The solution is changed in place,
        so input a copy if you wish to save the original guess.
    difeq : class
        A class that provides the smatrix routine for filling the s matrix with
        boundary conditions and internal couplings. It's call signature must be
        difeq.smatrix(self, k, k1, k2, jsf, is1, isf, indexv, s, y)
    verbose : Boolean, print step number and error info (default: False)
    it_count : return iteration number in addition to solution (default: False)
    """

    ne, m = y.shape
    nvars = ne*m
    k1=0; k2=m
    #indexv = np.asarray(indexv)

    c = np.zeros((ne, ne-nb+1, m+1))
    s = np.zeros((ne, 2*ne+1))

    # Set up row and column markers.
    j1=0
    j2=nb
    j3=nb
    j4=ne
    j5=j4+j1
    j6=j4+j2
    j7=j4+j3
    j8=j4+j4
    j9=j8+j1

    ic1=0
    ic2=ne-nb
    ic3=ic2
    ic4=ne
    jc1=0
    jcf=ic3

    for it in xrange(itmax):        # Primary iteration loop.
        k = k1                 # Boundary conditions at first point.
        s = difeq.smatrix(k, k1, k2, 2*ne, ne-nb, 
                                    ne, indexv, s, y)
        pinvs(ne-nb, ne, ne, 2*ne, 0, k1, s, c, np.zeros(nb, dtype=int), np.zeros(nb))

        for k in xrange(k1+1, k2):    # Finite difference equations at
            kp=k                        # all point pairs.
            s = difeq.smatrix(k, k1, k2, 2*ne, 0, 
                                        ne, indexv, s, y)
            red(0, ne, 0, nb, nb, ne, 2*ne, ne-nb, 0, ne-nb, kp, s, c)
            pinvs(0, ne, nb, 2*ne, 0, k, s, c, np.zeros(ne, dtype=int), np.zeros(ne))

        k = k2                     # Final boundary conditions.
        s = difeq.smatrix(k, k1, k2, 2*ne, 0, 
                                    ne-nb, indexv, s, y)
        red(0, ne-nb, ne, ne+nb, ne+nb, 2*ne, 2*ne, 
                    ne-nb, 0, ne-nb, k2, s, c)
        pinvs(0, ne-nb, ne+nb, 2*ne, ne-nb, k2, s, c, np.zeros(ne-nb, dtype=int),
                    np.zeros(ne-nb))
        bksub(ne, nb, ne-nb, k1, k2, c)      # Backsubstitution.

        # Convergence check, accumulate average error.
        err = errest(ne, k1, k2, indexv, scalv, c)
        err = err/nvars

        # Reduce correction when error is large.
        fac = slowc/err if err > slowc else 1.
        
        # Apply corrections.
        for j in range(ne):
            jv = indexv[j]
            y[j, k1:k2] -= fac*c[jv, 0, k1:k2]
        
        if verbose:
            print("Iter.")
            print("{:<11}".format("Error")+"{:<11}".format("FAC"))
            print("{:<8}".format(it))
            print("{0:5f}{1:<3}".format(err, ' ')+"{0:5f}{1:<3}".format(fac, ' '))

        if err < conv:
            # jit doesn't like return in for loop. Consider break.
            break
    if it_count:
        return y, it+1
    else:
        return y,
    
    # jit won't raise errors, consider flag.
    #raise ValueError('Too many iterations in solvde')
            
@jit(void(int64, int64, int64, int64, int64, int64, 
            float64[:,:], float64[:,:,:], int64[:], float64[:]), nopython=True)
def pinvs(ie1, ie2, je1, jsf, jc1, k, s, c, indxr, pscl):
    """Diagonalize the square subsection of the s matrix, and store the
    recursion coefficients in c; used internally by Solvde."""
    
    iesize = ie2-ie1
    #indxr = indxr*0
    je2 = je1 + iesize

    # Implicit pivoting, as in NR 2.1.
    #pscl = np.max(np.abs(s[ie1:ie2, je1:je2]), axis=1)
    for i in range(ie1, ie2):
        big = 0.
        for j in range(je1, je2):
            if abs(s[i,j]) > big:
                big = abs(s[i,j])
        pscl[i-ie1] = 1./big

    # jit doesn't raise errors.
    #if np.any(big == 0):
    #    raise ValueError('Singular matrix - row all 0, in pinvs')
    #pscl = 1./pscl
    #
    for im in range(0, iesize):
        piv = 0.
        for i in range(ie1, ie2):       # Find pivot element.
            if indxr[i-ie1] == 0:
                big = 0.0
                for j in range(je1, je2):
                    if abs(s[i,j]) > big:
                        jp = j
                        big = abs(s[i,j])
    
                if big*pscl[i-ie1] > piv:
                    ipiv = i
                    jpiv = jp
                    piv = big*pscl[i-ie1]

    #    # jit doesn't raise errors.
    #    #if s[ipiv, jpiv] == 0:
    #    #    raise ValueError('Singular matrix in routine pinvs')
    #
        indxr[ipiv-ie1] = jpiv+1
        pivinv = 1./s[ipiv, jpiv]
        for j in range(je1, jsf+1):
            s[ipiv, j] *= pivinv
        #s[ipiv, je1:jsf+1] *= pivinv
        s[ipiv, jpiv] = 1.
    
        for i in range(ie1, ie2):
            if indxr[i-ie1] != jpiv+1:
                if s[i, jpiv] != 0.:
                    dum = s[i, jpiv]
                    #s[i, je1:jsf+1] -= dum*s[ipiv, je1:jsf+1]
                    for j in range(je1, jsf+1):
                        s[i, j] -= dum*s[ipiv, j]
                    s[i, jpiv] = 0.
    
    jcoff = jc1-je2
    icoff = ie1-je1
    #irows = indxr+icoff-1

    for i in range(ie1, ie2):
        irow = indxr[i-ie1]+icoff
        for j in range(je2, jsf+1):
            c[irow-1, j+jcoff, k] = s[i, j]

@jit(void(int64, int64, int64, int64, int64, float64[:,:,:]), nopython=True)
def bksub(ne, nb, jf, k1, k2, c):
    nbf=ne-nb
    im = 1
    for k in range(k2-1,-1,-1):
        if k == k1: im=nbf+1
        kp = k+1
        for j in range(nbf):
            xx=c[j, jf, kp]
            # Pythonic: c[im-1:ne, jf, k] -= c[im-1:ne, j,k]*c[j, jf, kp]
            for i in range(im-1, ne):
                c[i, jf, k] -= c[i,j,k]*xx
    # Pythonic: c[:nb,0,k1:k2] = c[nbf:nb+nbf,jf,k1:k2]
    for i in range(nb):
        for k in range(k1,k2):
            c[i,0,k] = c[nbf+i,jf,k]
    # Pythonic: c[nb:nb+nbf,0,k1:k2] = c[:nbf,jf,k1+1:k2+1]
    for i in range(nbf):
        for k in range(k1,k2):
            c[nb+i,0,k] = c[i,jf,k+1] 

@jit(void(int64, int64, int64, int64, int64, int64, int64, int64, 
            int64, int64, int64, float64[:,:], float64[:,:,:]), nopython=True)
def red(iz1, iz2, jz1, jz2, jm1, jm2, jmf, ic1, jc1, jcf, kc, s, c):
    """Reduce columns jz1..jz21 of the s matrix, using previous results
    stored in the c matrix. Only columns jm1..jm2-1 and jmf are affected by
    the prior results. Used internally by Solvde."""
    loff = jc1-jm1
    for ic, j in zip(range(ic1, ic1+jz2-jz1), range(jz1, jz2)):
        for l in range(jm1, jm2):
            vx = c[ic, l+loff, kc-1]
            #Pythonic: s[iz1:iz2, l] -= s[iz1:iz2, j]*vx
            for i in range(iz1, iz2):
                s[i, l] -= s[i, j]*vx
        vx=c[ic,jcf,kc-1]
        #Pythonic: s[iz1:iz2, jmf] -= s[iz1:iz2, j]*vx
        for i in range(iz1, iz2):
            s[i, jmf] -= s[i, j]*vx

@jit(float64(int64, int64, int64, int64[:], 
        float64[:], float64[:,:,:]), nopython=True)
def errest(ne, k1, k2, indexv, scalv, c):
    err = 0.
    for j in range(ne):
        jv = indexv[j]
        errj = 0.0; vmax = 0.0;
        km = 0
        for k in range(k1, k2):
            vz = np.abs(c[jv, 0, k])
            if vz > vmax:
                vmax = vz
                km = k+1
            errj += vz
        err += errj/scalv[j]
    return err

@jit(void(int64, int64, int64, float64[:,:], float64[:], 
    float64[:,:], int64[:], float64[:,:]), nopython=True)
def interior_smatrix_fast(n, k, jsf, A, b, y, indexv, s):
    """Generates the s matrix used by solvde for interior points for a linear
    system characterized by linear differential operator A and inhomogeneity b.
    i.e., dy/dx = A(x).y + b(x).
    """
    for i in range(n):
        rgt = 0.
        for j in range(n):
            if i==j:
                s[i, indexv[j]]   = -1. - A[i,j]
                s[i, n+indexv[j]] =  1. - A[i,j]
            else:
                s[i, indexv[j]]   = -A[i,j]
                s[i, n+indexv[j]] = -A[i,j]
            rgt += A[i,j] * (y[j, k] + y[j, k-1])
        s[i, jsf] = y[i, k] - y[i, k-1] - rgt - b[i]
