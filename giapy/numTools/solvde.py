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

from scipy.special import lpmv

class solvde(object):
    """Driver routine for solution of two-point boundary value problems by
    relaxation."""

    def __init__(self, itmax, conv, slowc, scalv, indexv, nb, y, difeq,
                    verbose=False, keep_steps=False, slowc_corr=None,
                    it_count=False):
        self.y = y
        ne, m = y.shape
        nvars = ne*m
        k1=0; k2=m
        indexv = np.asarray(indexv)

        self.c = np.zeros((ne, ne-nb+1, m+1))
        self.s = np.zeros((ne, 2*ne+1))

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

        # Keep steps keeps all objective values
        if keep_steps:
            self.steps = []
            self.errs = []
            

        for it in xrange(itmax):        # Primary iteration loop.
            k = k1                 # Boundary conditions at first point.
            self.s = difeq.smatrix(k, k1, k2, 2*ne, ne-nb, 
                                        ne, indexv, self.s, y)
            self.pinvs(ne-nb, ne, ne, 2*ne, 0, k1)

            for k in xrange(k1+1, k2):    # Finite difference equations at
                kp=k                        # all point pairs.
                self.s = difeq.smatrix(k, k1, k2, 2*ne, 0, 
                                            ne, indexv, self.s, y)
                self.red(0, ne, 0, nb, nb, ne, 2*ne, ne-nb, 0, ne-nb, kp)
                self.pinvs(0, ne, nb, 2*ne, 0, k)

            k = k2                     # Final boundary conditions.
            self.s = difeq.smatrix(k, k1, k2, 2*ne, 0, 
                                        ne-nb, indexv, self.s, y)
            self.red(0, ne-nb, ne, ne+nb, ne+nb, 2*ne, 2*ne, 
                        ne-nb, 0, ne-nb, k2)
            self.pinvs(0, ne-nb, ne+nb, 2*ne, ne-nb, k2)
            self.bksub(ne, nb, ne-nb, k1, k2)      # Backsubstitution.

            # Convergence check, accumulate average error.
            err = 0
            
            for j in range(ne):
                jv = indexv[j]
                errj = 0.0; vmax = 0.0
                km = 0
                for k in range(k1, k2):
                    vz = np.abs(self.c[jv, 0, k])
                    if vz > vmax:
                        vmax = vz
                        km = k+1
                    errj += vz
                err += errj/scalv[j]
            err = err/nvars
            
            if it == 0 and slowc_corr is not None and err < slowc:
                slowc = min(slowc, slowc_corr*err)

            if keep_steps:
                self.errs.append(err)
                self.steps.append(y.copy())

            # Reduce correction when error is large.
            fac = slowc/err if err > slowc else 1.
            
            # Apply corrections.
            for j in range(ne):
                jv = indexv[j]
                y[j, k1:k2] -= fac*self.c[jv, 0, k1:k2]
            
            if verbose:
                print "Iter."
                print "{:<11}".format("Error")+"{:<11}".format("FAC")
                print "{:<8}".format(it)
                print "{0:5f}{1:<3}".format(err, ' ')+"{0:5f}{1:<3}".format(fac, ' ')

            if err < conv: 
                self.y = y
                # Steps are zero-indexed.
                self.it = it + 1 
                return

        #raise ValueError('Too many iterations in solvde')
    def __getitem__(self, key):
        return self.y.__getitem__(key)
    def __iter__(self):
        return self.y.__iter__()
    def copy(self):
        return self.y.copy()

    def pinvs(self, ie1, ie2, je1, jsf, jc1, k):
        """Diagonalize the square subsection of the s matrix, and store the
        recursion coefficients in c; used internally by Solvde."""

        s = self.s
        
        iesize = ie2-ie1
        indxr = np.zeros(iesize, dtype=int)
        pscl = np.zeros(iesize)
        je2 = je1 + iesize
   
        indxr = np.zeros(iesize)
        # Implicit pivoting, as in NR 2.1.
        big = np.abs(s[ie1:ie2, je1:je2]).max(axis=1)
        if np.any(big == 0):
            raise ValueError('Singular matrix - row all 0, in pinvs')
        pscl = 1./big
        
        for im in range(0, iesize):
            piv = 0.
            for i in range(ie1, ie2):       # Find pivot element.
                if indxr[i-ie1] == 0:
                    big = 0.0
                    for j in range(je1, je2):
                        if abs(s[i,j]) > big:
                            jp = j
                            big = abs(s[i,j])
                    #jp = np.argmax(np.abs(s[i,je1:je2]))+je1
                    #big = np.abs(s[i, jp])
        
                    if big*pscl[i-ie1] > piv:
                        ipiv = i
                        jpiv = jp
                        piv = big*pscl[i-ie1]
            if s[ipiv, jpiv] == 0:
                raise ValueError('Singular matrix in routine pinvs')
        
            indxr[ipiv-ie1] = jpiv+1
            pivinv = 1./s[ipiv, jpiv]
            s[ipiv, je1:jsf+1] *= pivinv
            s[ipiv, jpiv] = 1.
        
            for i in range(ie1, ie2):
                if indxr[i-ie1] != jpiv+1:
                    if s[i, jpiv] != 0.:
                        dum = s[i, jpiv]
                        s[i, je1:jsf+1] -= dum*s[ipiv, je1:jsf+1]
                        s[i, jpiv] = 0.
        
        jcoff = jc1-je2
        icoff = ie1-je1
        irows = indxr.astype(int)+icoff-1

        for i in range(ie1, ie2):
            irow = int(indxr[i-ie1]+icoff)
            for j in range(je2, jsf+1):
                self.c[irow-1, j+jcoff, k] = s[i, j]
        self.s = s

    def bksub(self, ne, nb, jf, k1, k2):
        c = self.c

        nbf=ne-nb
        im = 1
        for k in range(k2)[::-1]:
            if k == k1: im=nbf+1
            kp = k+1
            for j in range(nbf):
            #    xx=c[j, jf, kp]
            #    for i in range(im-1, ne):
            #        c[i, jf, k] -= c[i,j,k]*xx
                c[im-1:ne, jf, k] -= c[im-1:ne, j,k]*c[j, jf, kp]

        c[:nb,0,k1:k2] = c[nbf:nb+nbf,jf,k1:k2]
        c[nb:nb+nbf,0,k1:k2] = c[:nbf,jf,k1+1:k2+1]

        self.c = c

    def red(self, iz1, iz2, jz1, jz2, jm1, jm2, jmf, ic1, jc1, jcf, kc):
        """Reduce columns jz1..jz21 of the s matrix, using previous results
        stored in the c matrix. Only columns jm1..jm2-1 and jmf are affected by
        the prior results. Used internally by Solvde."""
        s = self.s
        loff = jc1-jm1
        for ic, j in zip(range(ic1, ic1+jz2-jz1), range(jz1, jz2)):
            for l in range(jm1, jm2):
                vx = self.c[ic, l+loff, kc-1]
                #if kc==5:
                #    print self.c[ic, l+loff, kc-1]
                s[iz1:iz2, l] -= s[iz1:iz2, j]*vx
            vx=self.c[ic,jcf,kc-1]
            s[iz1:iz2, jmf] -= s[iz1:iz2, j]*vx
        self.s = s

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

################### EXAMPLE OF USE ######################
def main_sfroid(n, mm):
    M = 40; MM = 4
    NE = 3; NB = 1; NYJ=NE; NYK=M+1

    mpt = M+1
    indexv = np.zeros(NE)
    x = np.zeros(M+1)
    scalv = np.zeros(NE)
    y = np.zeros((NYJ, NYK))
    itmax = 100

    c2 = np.array([1., 16.])
    conv = 1e-14
    slowc = 1.
    h = 1./M

    if not n+mm % 2 :
        indexv = [0, 1, 2]
    else:
        indexv = [1, 0, 2]

    anorm = 1.
    if mm != 0:
        q1 = n
        for i in range(1, mm+1):
            anorm = -0.5*anorm*(n+1)*float(q1/i)
            q1 -= 1
    for k in range(M):                  # Initial guess.
        x[k] = k*h
        fac1 = 1.-x[k]*x[k]
        fac2 = np.exp((-mm/2.)*np.log(fac1))
        y[0, k] = lpmv(mm, n, x[k])*fac2
        deriv = -((n-mm+1)*lpmv(mm, n+1, x[k]) - \
                    (n+1)*x[k]*lpmv(mm, n, x[k]))/fac1
        y[1, k] = mm*x[k]*y[0, k]/fac1 + deriv*fac2
        y[2, k] = n*(n+1)-mm*(mm+1)

    x[M] = 1.
    y[0, M] = anorm
    y[2, M] = n*(n+1)-mm*(mm+1)
    y[1, M] = y[2, M]*y[0, M]/(2.*(mm+1.))

    scalv[0] = abs(anorm)
    scalv[1] = y[1, M] if y[1, M] > scalv[0] else scalv[0]
    scalv[2] = y[2, M] if y[2, M] > 1. else 1.

    for c2i in c2:
        difeq = SfroidDifeq(mm, n, mpt, h, c2i, anorm, x)
        solvde = Solvde(itmax, conv, slowc, scalv, indexv, NB, y, difeq)

        print 'lamda = '+str(solvde.y[2, 0] + mm*(mm+1))+'\n'

    return solvde

def gen_sfroid(n, mm, c2):
    M = 40; MM = 4
    NE = 3; NB = 1; NYJ=NE; NYK=M+1

    mpt = M+1
    indexv = np.zeros(NE)
    x = np.zeros(M+1)
    scalv = np.zeros(NE)
    y = np.zeros((NYJ, NYK))
    itmax = 100

    #c2 = np.array([1., 16.])
    conv = 1e-14
    slowc = 1.
    h = 1./M

    if not n+mm % 2 :
        indexv = [0, 1, 2]
    else:
        indexv = [1, 0, 2]

    anorm = 1.
    if mm != 0:
        q1 = n
        for i in range(1, mm+1):
            anorm = -0.5*anorm*(n+1)*float(q1/i)
            q1 -= 1
    for k in range(M):                  # Initial guess.
        x[k] = k*h
        fac1 = 1.-x[k]*x[k]
        fac2 = np.exp((-mm/2.)*np.log(fac1))
        fac2 = fac1**(-mm/2.)
        y[0, k] = lpmv(mm, n, x[k])*fac2
        deriv = -((n-mm+1)*lpmv(mm, n+1, x[k]) - \
                    (n+1)*x[k]*lpmv(mm, n, x[k]))/fac1
        y[1, k] = mm*x[k]*y[0, k]/fac1 + deriv*fac2
        y[2, k] = n*(n+1)-mm*(mm+1)

    x[M] = 1.
    y[0, M] = anorm
    y[2, M] = n*(n+1)-mm*(mm+1)
    y[1, M] = y[2, M]*y[0, M]/(2.*(mm+1.))

    scalv[0] = abs(anorm)
    scalv[1] = max(y[1,M], scalv[0])
    scalv[2] = max(y[2, M], 1.)

    difeq = SfroidDifeq(mm, n, mpt, h, c2, anorm, x)

    return difeq, y

class SfroidDifeq(object):
    def __init__(self, mm, n, mpt, h, c2, anorm, x):
        self.mm = mm
        self.n = n
        self.mpt = mpt
        self.h = h
        self.c2 = c2
        self.anorm = anorm
        self.x = x

    def smatrix(self, k, k1, k2, jsf, is1, isf, indexv, s, y):

        n = self.n
        mm = self.mm
        mpt = self.mpt
        h = self.h
        c2 = self.c2
        anorm = self.anorm
        
        if k == k1:
            if not n+mm % 2:
                s[2, 3+indexv[0]]=1.
                s[2, 3+indexv[1]]=0.                
                s[2, 3+indexv[2]]=0.
                s[2, jsf] = y[0, 0]
            else:
                s[2, 3+indexv[0]]=0.   
                s[2, 3+indexv[1]]=1.
                s[2, 3+indexv[2]]=0.
                s[2, jsf]=y[1, 0]
        elif k > k2-1:
            s[0, 3+indexv[0]] = -(y[2, mpt-1]-c2)/(2*(mm+1.))
            s[0, 3+indexv[1]] = 1.
            s[0, 3+indexv[2]] = -y[0, mpt-1]/(2*(mm+1.))
            s[0, jsf] = y[1, mpt-1] - (y[2, mpt-1]-c2)*y[0, mpt-1]/(2*(mm+1.))
            s[1, 3+indexv[0]] = 1.
            s[1, 3+indexv[1]] = 0.
            s[1, 3+indexv[2]] = 0.
            s[1, jsf] = y[0, mpt-1]-self.anorm
        else:
            s[0, indexv[0]] = -1.
            s[0, indexv[1]] = -0.5*self.h
            s[0, indexv[2]] = 0.
            s[0, 3+indexv[0]] = 1.
            s[0, 3+indexv[1]] = -0.5*h
            s[0, 3+indexv[2]] = 0.
            temp1 = self.x[k]+self.x[k-1]
            temp = h/(1.-temp1*temp1*0.25)
            temp2 = 0.5*(y[2, k]+y[2, k-1])-c2*0.25*temp1*temp1
            s[1, indexv[0]] = temp*temp2*0.5
            s[1, indexv[1]] = -1. - 0.5*temp*(mm+1.)*temp1
            s[1, indexv[2]] = 0.25*temp*(y[0, k]+y[0, k-1])
            s[1, 3+indexv[0]] = s[1, indexv[0]]
            s[1, 3+indexv[1]] = 2.+s[1, indexv[1]]
            s[1, 3+indexv[2]] = s[1, indexv[2]]
            s[2, indexv[0]] = 0.
            s[2, indexv[1]] = 0.
            s[2, indexv[2]] = -1.
            s[2, 3+indexv[0]] = 0.
            s[2, 3+indexv[1]] = 0.
            s[2, 3+indexv[2]] = 1.
            s[0, jsf] = y[0, k]-y[0, k-1]-0.5*h*(y[1, k]+y[1, k-1])
            s[1, jsf] = y[1, k]-y[1, k-1]-temp*((self.x[k]+self.x[k-1])*\
                        0.5*(mm+1.)*(y[1, k]+y[1, k-1]) - temp2*\
                        0.5*(y[0, k]+y[0, k-1]))
            s[2, jsf] = y[2, k] - y[2, k-1]

        return s
