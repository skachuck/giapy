import numpy as np
import cPickle as pickle

def _list2UT(l, N):
    """Return a list of the rows of an NxN upper triangular"""
    mat = []
    start = 0
    for i in range(N):
        mat.append(l[start:start+N-i])
        start += N-i
    return mat

def _UT2list(m):
    """Return the elements of an NxN upper triangular in row major order"""
    return np.array([item for thing in m for item in thing])
    
def _unsort(l, ind):
    """Scramble and return a list l by indexes ind"""
    l0 = np.zeros(len(l))
    for i, x in zip(ind, l):
        l0[i]=x
    return l0

def _symmetrize(A):
    """Form a symmetric matrix out of a list of upper triangle elements
    
    Example:
        >>> A = [[0, 1, 2], [3, 4], [5]]
        >>> symmetrize(A)
        array([[ 0.,  1.,  2.],
               [ 1.,  3.,  4.],
               [ 2.,  4.,  5.]])
    """
    N = len(A)
    As = np.zeros((N, N))
    for i, t in enumerate(A):
        As[i, i:]=t
    return As + As.T - np.diag(As.diagonal())
    
def _tile_arrays(A):
    """Take a symmetric array, A, size N x N, where N is even and return
    an array [ N x N ;  N - 1  ]
             [ N - 1 ; A[1, 1] ]
    such that the lower to exploit symmetries.
    
    Example (N=2): 
        >>> A =  array([[0, 1, 2],
        >>> ...        [1, 3, 4],
        >>> ...        [2, 4, 6]])
        >>> tile_arrays(tmp)
        array([[ 0.,  1.,  2.,  1.],
               [ 1.,  3.,  4.,  3.],
               [ 2.,  4.,  6.,  4.],
               [ 1.,  3.,  4.,  3.]])
    """    
        
    N = 2*(np.shape(A)[0]-1)                                        
    A_ref = np.zeros((N,N))
    A_ref[:N/2+1,:N/2+1]=A
    A_ref[N/2:,:N/2+1] = A_ref[N/2:0:-1,:N/2+1]
    A_ref[:,N/2:] = A_ref[:,N/2:0:-1]
    return A_ref
    
def _list_to_fft_mat(lis, index, N):
    lis = _unsort(lis, index)
    mat = _list2UT(lis, N/2+1)
    mat = _symmetrize(mat)                 # Make use of symmetries to
    mat = _tile_arrays(mat)                # to construct the whole array.
    return mat
            
def form_wl(N):
    #fwl = 10*N
    fwl = 6000.
    wl = np.array([[fwl/np.sqrt(i**2+j**2) if (i!=0 or j!=0) else 16000 
                                           for i in range(j, N)] 
                                           for j in range(N)])
    return _symmetrize(wl)
    
def yield_wl(i, j, fwl=6000):
    if i == j == 0:
        return 16000
    else:
        return fwl/np.sqrt(i**2+j**2)
        
def prop(u, d, k):
    """Return the propagator for a layer of depth d and viscosity u with a 
    harmonic load of order k
    
    As defined in Cathles 1975, p41, III-12.
    """
    y = k*d
    s = np.sinh(y)
    c = np.cosh(y)
    cp = c + y*s
    cm = c - y*s
    sp = s + y*c
    sm = s - y*c
    s = y*s
    c = y*c
    
    return np.array([[cp, c, sp/u, s/u], 
                    [-c, cm, -s/u, sm/u], 
                    [sp*u, s*u, cp, c], 
                    [-s*u, sm*u, -c, cm]])
                    
def exp_decay_const(earth, i, j):
    """Return the 2D exponential decay constant for order harmonic unit loads. 
    
    Parameters
    ----------
    earth - the earth object containing the parameters...
    i, j (int) - the x, y order numbers of the harmonic load
    
    Returns
    -------
    tau - the exponential decay constant for harmonic load i,j
    
    The resulting decay, dec(i, j) = 1-np.exp(elapsed_time * 1000./tauc(i, j)),
    along with lithospheric filter can be returned using earth.get_resp.
    """
    wl = yield_wl(i, j, fwl=600*10)         # wavelength
    ak = 2*np.pi/wl                         # wavenumber
    
    # ----- Interior to Surface integration ----- #
    # uses propagator method from Cathles 1975
    
    # initialize two interior boundary vectors
    # assuming solution stays finite in the substratum (Cathles 1975, p41)
    cc = np.array([[1., 0., 1., 0.], 
                [0., 1., 0., 1.]])
    
    # Determine the necessary start depth (start lower for longer wavelengths)
    # to solve a roundoff problem with the matrix method.
    #TODO look into stability problem here
    if wl < 40:
        lstart = 9
    elif wl < 200:
        lstart = 6
    elif wl < 500:
        lstart = 4
    else:
        lstart = 0
    
    # integrate from starting depth to surface, layer by layer
    for dd, uu, in zip(earth.d[lstart:], earth.u[lstart:]):
        p = prop(uu, dd, ak)
        for k, c in enumerate(cc):
            cc[k,:] = p.dot(c)
    
    # initialize the inegration constants
    x = np.zeros(2)
    # solve for them, assuming 1 dyne normal load at surface
    x[0] =  cc[1,2]/(cc[0,2]*cc[1,3]-cc[0,3]*cc[1,2])
    x[1] = -cc[0,2]/(cc[0,2]*cc[1,3]-cc[0,3]*cc[1,2])
    
    # multiply into the solution
    for k in range(2):
        cc[k,:]=x[k]*cc[k,:]
    # form the final solution
    cc = np.sum(cc, axis=0)
    
    # As cc[1] gives the surface velocity, the exponential time constant is its
    # reciprocal (see Cathles 1975, p43)
    #TODO reevaluate this expression
    tau = 19556.*ak*100./cc[1]
    
    return tau

class EarthTwoDBase(object):
    """A Base class for 2D, flat earth models. Provides methods for saving,
    adding descriptions, and returning response.

    User must define a method for generating taus, 
    """
    def __init__(self):
        self.taus = None
        self.ak = None
        self.alpha = None
        self.index = None
        self.N = None

    def __call__(self, t_dur):
        return self.get_resp(t_dur)
        
    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    def get_resp(self, t_dur):
        """Calculate and return earth response to a unit load in an fft_mat.
        
        Parameters
        ----------
        t_dur (float) - the duration, in cal ka BP, of applied load
        """            
        # Convert tau list to fft matrix
        taus = _list_to_fft_mat(self.taus, self.index, self.N)
        
        resp = (1-np.exp(t_dur*1000/taus))/self.alpha
        return resp

    def set_N(self, N):
        self.N = N

class EarthNLayer(EarthTwoDBase):
    """Return the isostatic response of a 2D earth with n viscosity layers.
    
    The response is calculated from a viscous profile overlain by an elastic
    lithosphere in the fourier domain.
    """
    
    def __init__(self, u=None, d=None, fr23=10.):
        if u is None:
            self.u = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.018])
        else:
            self.u = u
        if d is None:
            self.d = np.array([400.,300.,300.,300.,300.,300.,
                               300.,200.,215.,175.,75.       ])
        else:
            self.d = d
            
        self.fr23=fr23
                
    def reset_params(self, mant_vis=None, asth_vis=None, asth_thi=None,
                    fr23=None, N=None):
        """Set the full mantle rheology and calculate the decay constants.

        self.N must have been set already.
        """
        #TODO defaults should be the values already in earth
        if mant_vis is not None:
            # Assumes last element is asthenosphere viscosity
            self.u[:-1] = np.repeat(mant_vis, len(self.u[:-1]))
        if asth_vis is not None:
            self.u[-1] = asth_vis
        if asth_thi is not None:
            self.d[-1] = asth_thi
        self.fr23 = fr23 or self.fr23
        N = N or self.N
        self.calc_taus(self.N)

    def set_taus(self, taus):
        self.taus=taus
        
    def calc_taus(self, N=None):
        """Generate and store a list of exponential decay constants.
        
        The procedure sets class data: 
            N (the maximum order number calculated)
            taus (decay constants by increasing wavenumber)
            ak (wavenumbers in increasing order)
            index (the sorting key to reorder taus to increasing wavenumber)
            alpha (the lithospheric filter values in FFTmat format)

        Parameters
        ----------
        N (int) - the maximum order number. Resulting earth parameters ak,
        taus, and alpha will be size NxN when in FFTmat format.

        For description of formats, see help(_list_to_fft_mat)
        """
        self.N = N or self.N
        taus = [[exp_decay_const(self, i, j) for i in xrange(j, N/2+1)] 
                                                for j in xrange(N/2+1)]
       #TODO Generalize to arbitrary wavelengths 
        wl = np.array([[6000/np.sqrt(i**2+j**2) if (i!=0 or j!=0) else 16000 
                                    for i in range(j, N/2+1)] 
                                    for j in range(N/2+1)])
        wl = _UT2list(wl)
        self.ak = 2*np.pi/np.array(wl)
        
        # Sort by increasing order number
        self.index = range(len(self.ak))
        self.index.sort(key=self.ak.__getitem__)
        self.ak = self.ak[self.index]
        self.taus = _UT2list(taus)[self.index]
        
        # the Lithosphere filter, sorted by wave number
        #TODO reevaluate this term (48?)
        self.alpha = (1.+((self.ak/0.062832)**4)*self.fr23*48)
        
        # Augment the decay times by the Lithosphere filter
        self.taus = self.taus/self.alpha                                       
                                                
        # Turn the Lithosphere filter and taus into a matrix that matches the 
        # frequencey matrix from an NxN fft.
        self.alpha = _list_to_fft_mat(self.alpha, self.index, self.N)
        
        
class EarthTwoLayer(EarthTwoDBase):
    """Return the isostatic response of a flat earth with two layers.
    
    The response is calculated analytically in the fourier domain from a
    uniform mantle of viscosity u overlain by an elastic lithosphere with
    flexural rigidty fr23.
    """
    def __init__(self, u, fr23, g=10, rho=3.313):
        self.u = u
        self.fr23 = fr23
        self.g = g
        self.rho = rho

    def reset_params(self, u=None, fr23=None, N=None):
        self.u = u or self.u
        self.fr23 = fr23 or self.fr23
        N = N or self.N
        self.calc_taus(N)

    def calc_taus(self, N=None):
        """Generate and store a list of exponential decay constants.
        
        The procedure sets class data: 
            N (the maximum order number calculated)
            taus (decay constants in flattend upper diagonal list)
            ak (wavenumbers in increasing order)
            index (the sorting key to reorder taus to increasing wavenumber)
            alpha (the lithospheric filter values in FFTmat format)

        Parameters
        ----------
        N (int) - the maximum order number. Resulting earth parameters ak,
        taus, and alpha will be size NxN when in FFTmat format.

        For description of formats, see help(_list_to_fft_mat)
        """
        self.N = N or self.N
        
       #TODO Generalize to arbitrary wavelengths 
        wl = np.array([[6000/np.sqrt(i**2+j**2) if (i!=0 or j!=0) else 16000 
                                    for i in range(j, N/2+1)] 
                                    for j in range(N/2+1)])
        wl = _UT2list(wl)
        self.ak = 2*np.pi/np.array(wl)

        # Sort by increasing order number
        self.index = range(len(self.ak))
        self.index.sort(key=self.ak.__getitem__)
        self.ak = self.ak[self.index]

        #TODO scaling matches solutions if mult by 32394513.999996319?
        self.taus = -2*self.u*self.ak/self.g/self.rho*32394513.999996319
        
        # the Lithosphere filter, sorted by wave number
        #TODO reevaluate this term (48?)
        self.alpha = (1.+((self.ak/0.062832)**4)*self.fr23*48)
        
        # Augment the decay times by the Lithosphere filter
        self.taus = self.taus/self.alpha                                       
                                                
        # Turn the Lithosphere filter and taus into a matrix that matches the 
        # frequencey matrix from an NxN fft.
        self.alpha = _list_to_fft_mat(self.alpha, self.index, self.N)

class EarthThreeLayer(EarthTwoDBase):
    """Return the isostatic response of a flat earth with three layers.
    
    The response is calculated analytically in the fourier domain from a
    two layer mantle whose lower layer, of viscosity u1, is overlain layer
    of viscosity u2 and width h, which in turn is overlain by an elastic
    lithosphere with flexural rigidty fr23.
    """
    def __init__(self, u1, u2, fr23, h, g=10, rho=3.313):
        self.g = g
        self.rho = rho
        self.u1 = u1
        self.u2 = u2
        self.fr23 = fr23
        self.h = h

    def reset_params(self, u1=None, u2=None, fr23=None, h=None, N=None):
        self.u1 = u1 or self.u1
        self.u2 = u2 or self.u2
        self.fr23 = fr23 or self.fr23
        self.h = h or self.h
        N = N or self.N
        self.calc_taus(N)    

    def calc_taus(self, N):
        """Generate and store a list of exponential decay constants.
        
        The procedure sets class data: 
            N (the maximum order number calculated)
            taus (decay constants in flattend upper diagonal list)
            ak (wavenumbers in increasing order)
            index (the sorting key to reorder taus to increasing wavenumber)
            alpha (the lithospheric filter values in FFTmat format)

        Parameters
        ----------
        N (int) - the maximum order number. Resulting earth parameters ak,
        taus, and alpha will be size NxN when in FFTmat format.

        For description of formats, see help(_list_to_fft_mat)
        """
        self.N = N
        
       #TODO Generalize to arbitrary wavelengths 
        wl = np.array([[6000/np.sqrt(i**2+j**2) if (i!=0 or j!=0) else 16000 
                                    for i in range(j, N/2+1)] 
                                    for j in range(N/2+1)])
        wl = _UT2list(wl)
        self.ak = 2*np.pi/np.array(wl)

        # Sort by increasing order number
        self.index = range(len(self.ak))
        self.index.sort(key=self.ak.__getitem__)
        self.ak = self.ak[self.index]
    
        c = np.cosh(ak*self.h)
        s = np.sinh(ak*self.h)
        u = self.u2/self.u1
        ui = 1./u
        r = 2*c*s*u + (1-u)*(ak*self.h)**2 + (u*s**2+c**2)
        r = r/((u+ui)*s*c + ak*self.h(u-ui) + (s**2+c**s))

        self.taus = -2*self.u*self.ak/self.g/self.rho*r
        
        # the Lithosphere filter, sorted by wave number
        #TODO reevaluate this term (48?)
        self.alpha = (1.+((self.ak/0.062832)**4)*self.fr23*48)
        
        # Augment the decay times by the Lithosphere filter
        self.taus = self.taus/self.alpha                                       
                                                
        # Turn the Lithosphere filter and taus into a matrix that matches the 
        # frequencey matrix from an NxN fft.
        self.alpha = _list_to_fft_mat(self.alpha, self.index, self.N)
        

def check_k(earth):
    """Check whether different 2D k matrices are identical, useful for
    debugging.
    """
    N = earth.N

    wl = form_wl(N/2+1)
    wl = _tile_arrays(wl)
    ak_wl = 2*np.pi/wl

    ki = np.reshape(np.tile(
        np.fft.fftfreq(N, 6000/(2*np.pi)/N), N), (N, N))
    ak_fft = np.sqrt(ki**2 + ki.T**2)       
    ak_fft[0,0] = 2*np.pi/16000

    ak_ea = _list_to_fft_mat(earth.ak, earth.index, N)

    print "ak_wl and ak_fft are close: "+str(np.allclose(ak_wl, ak_fft))
    print "ak_wl and ak_ea are close:  "+str(np.allclose(ak_wl, ak_ea))
    print "ak_fft and ak_ea are close: "+str(np.allclose(ak_fft, ak_ea))

        
def load(filename):
    return pickle.load(open(filename, 'r'))
