"""
odeint.py

    Methods for integration of Ordinary Differential Equations (ODEs). The code
    is a translation from the C code in [1], chapter 17.

    Author: Samuel B. Kachuck

    References:
        [1] Press, Flannery, Teukolsky, and Vetterling. Numerical Recipes.
        Cambridge University Press, Cambridge UK.
"""

import numpy as np
import numexpr as ne
from numba import jit, void, int64, float64

class Odeint(object):
    """Stratagy pattern for ODE solving."""

    MAXSTP = 50000              # Take at most MAXSTP steps.
    EPS = np.finfo(float).eps

    def __init__(self, derivs, ystart, x1, x2, stepper, atol, rtol, h, hmin,
	nsave=None, xsave=None, **kwargs):
        """Constructor sets everything up. The routine integrates starting
	values ystart[0..nvar-1] from x1 to x2 with absolute tolerance atol and
	relative tolerance rtol. The quantity h1 should be set as a guessed
	first stepsize (can be zero). An Ouput object should be input to
	control the saving of intermediate values. On ouput, nok and nbad are
	the number of good and bad (but retired and dixed) steps taken, and
	ystart is replaced by values at the end of the integration interval.
	derivs is the user-supplied routine for calculating the right-hand side
	derivative."""

        nvar = len(ystart)
        self.y = np.zeros(nvar)
        self.dydx = np.zeros(nvar)
        self.ystart = ystart
        for i, yi in enumerate(self.ystart):
            self.y[i] = yi

        self.nok=0
        self.nbad=0

        self.x1 = x1
        self.x2 = x2
        self.x = x1

        self.hmin = hmin
        if xsave is None:
            self.out = Output(x1, x2, nsave)
        else:
            self.out = ArbitraryArrayOutput(x1, x2, xsave)

        self.dense = self.out.dense
        self.derivs = derivs
	self.h = np.sign(x2-x1)*h

        self.stepper = stepper(self.y, self.dydx, self.x, atol, rtol, self.dense, **kwargs)


    def integrate(self):
	"""Do the actual integration"""
        out = self.out
        s = self.stepper

        self.derivs(self.x, self.y, self.dydx)
        if self.dense:                          # Store initial values
            out.out(-1, self.x, self.y, s, self.h)
        else:
            out.save(self.x,self.y)

        for nstp in range(self.MAXSTP):
            if (self.x+self.h*1.0001-self.x2)*(self.x2-self.x1) > 0.0:
                self.h = self.x2-self.x         # If stepsize can overshoot, decrease.
            self.x, self.y = s.step(self.h, self.derivs)         # Take a step.
            
            if (s.hdid == self.h):
                self.nok += 1
            else:
                self.nbad += 1

            if self.dense:
                out.out(nstp, self.x, self.y, s, s.hdid)
            else:
                out.save(self.x,self.y)

            if (self.x-self.x2)*(self.x2-self.x1) >= 0.0:            # Are we done?
                for i,yi in enumerate(self.y):     # Update ystart.
                    self.ystart[i]=yi
                if abs(out.xsave[-1]-self.x2)>100.0*abs(self.x2)*self.EPS:
                    out.save(self.x, self.y)       # Make sure last step gets saved
                return out                 # Normal exit

            if abs(s.hnext) <= self.hmin:
                raise ValueError("Step size too small in Odeint")
            self.h = s.hnext

        raise ValueError("Too many steps in routine Odeint")

class StepperBase(object):
    def __init__(self, y, dydx, x, atol, rtol, dense):
	self.x = x
	self.y = y
	self.dydx = dydx
        self.dydxnew = dydx.copy()
	self.atol = atol
	self.rtol = rtol
	self.dense = dense
	self.n = len(y)
	self.neqn = self.n	        # neqn = n except for StepperStoerm.
	self.yout = np.zeros(self.n)	# New value of y 
	self.yerr = np.zeros(self.n)	# and error estimate

class Output(object):
    def __init__(self, x1, x2, nsave):
	self.nsave = nsave
        #TODO Allocate arrays first.
        self.xsave = []
        self.ysave = []
	if self.nsave > 0:
	    self.dense = True
	elif self.nsave < 0:
	    self.dense = False
	    return
	else:
	    self.dense = False

	if self.dense:
	    self.x1 = x1
	    self.x2 = x2
	    self.xout = x1
	    self.dxout = float(x2-x1)/nsave

    def save_dense(self, stepper, xout, h):
	self.ysave.append(stepper.denseOut(xout, h))
	self.xsave.append(xout)

    def save(self, x, y):
	self.ysave.append(y)
	self.xsave.append(x)

    def out(self, nstp, x, y, stepper, h):
        """Typically called by Odeint to produce dense output. Input variables
        are nstp, the current step number, the current values of x and y, the
        stepper s, and the stepsize h. A call with nstp=-1 saves the initial
        values. THe routine checks whether x is greater than the desired output
        point xout. If so, it calls save_dense.
        """
	if not self.dense:
	    raise ValueError("dense output not set in Output!")
	if nstp == -1:
	    self.save(x, y)
	    self.xout += self.dxout
	else:
	    while (x-self.xout)*(self.x2-self.x1) > 0.0:
		self.save_dense(stepper, self.xout, h)
		self.xout += self.dxout

class ArbitraryOutput(Output):
    def __init__(self, x1, x2, xsave=None):
        self.xsave = []
        self.ysave = []

        if xsave is None:
            self.dense = False
            return
        else:
            self.xsaveiter = iter(xsave)
            self.dense = True

        if self.dense:
            self.x1 = x1
            self.x2 = x2
            self.xout = x1

    def out(self, nstp, x, y, stepper, h):
        if not self.dense:
            raise ValueError("dense output not set in Output!")
        if nstp == -1:
            self.save(x, y)
            self.xout = self.xsaveiter.next()
        else:
            while (x-self.xout)*(self.x2-self.x1) > 0.0:
		self.save_dense(stepper, self.xout, h)
		self.xout = self.xsaveiter.next()

class ArbitraryArrayOutput(Output):
    def __init__(self, x1, x2, xsave=None):
        self.xsavelist = []
        self.ysave = []

        if xsave is None:
            self.dense = False
            return
        else:
            self.xsave = np.asarray(xsave)
            self.dense = True

        if self.dense:
            self.x1 = x1
            self.x2 = x2

    def save_dense(self, stepper, xout, h):
	self.ysave.append(stepper.denseOut(xout, h))
        self.xsavelist.append(xout)

    def save(self, x, y):
	self.ysave.append(y)
        self.xsavelist.append(x)

    def out(self, nstp, x, y, stepper, h):
        if not self.dense:
            raise ValueError("dense output not set in Output!")
        if nstp == -1:
            self.save(x, y)
        else:
            # Locate x range between x and x+h
            xout = self.xsave[np.logical_and(self.xsave<=x,
                            self.xsave>x-h)]
	    self.save_dense(stepper, xout, h)


class StepperDopr5(StepperBase):
    """Dormand-Prince fifth-order Runge-Kutta step with monitoring of local
    truncation error to ensure accuracy and adjust stepsize.
    
    Attributes
    ----------


    Methods
    -------
    step (htry, derivs)
    dy (h, derivs)
    prepare_dense(h, derivs)
    denseOut(x, h)
    error ()
    
    """
    EPS = np.finfo(float).eps

    # Data
    c2=0.2; c3=0.3; c4=0.8; c5=8./9.; a21=0.2; a31=3./40.; a32=9./40.
    a41=44./45.; a42=-56./15.; a43=32./9.; a51=19372./6561.;
    a52=-25360./2187.; a53=64448./6561.; a54=-212./729.; a61=9017./3168.; 
    a62=-355./33.; a63=46732./5247.; a64=49./176.; a65=-5103./18656.;
    a71=35./384.; a73=500./1113.; a74=125./192.; a75=-2187./6784.;
    a76=11./84.; e1=71./57600.; e3=-71./16695.; e4=71./1920.;
    e5=-17253./339200.; e6=22./525.; e7=-1./40.

    def __init__(self, *args, **kwargs):
        super(StepperDopr5, self).__init__(*args)
        #TODO Save as single array?
        self.k1 = np.zeros(self.n)
        self.k2 = np.zeros(self.n)
        self.k3 = np.zeros(self.n)
        self.k4 = np.zeros(self.n)
        self.k5 = np.zeros(self.n)
        self.k6 = np.zeros(self.n)

        # Dopr5 Stepsize Controller
	self.beta = kwargs.get('beta', 0.0) 
	self.alpha = 0.2 - self.beta*0.75
	self.safe = kwargs.get('safe', 0.9)
	self.minscale = kwargs.get('minscale', 0.2)
	self.maxscale = kwargs.get('maxscale', 10.)
        self.errold = 1e-4
        self.reject = False

    def step(self, htry, derivs):
	"""Attempts a step with stepsize htry. On output, y and x are replaced
	by their new values, hdid is the stepsize that was actually
	accomplished, and hnext is the estimated next stepsize."""

	self.h = htry

	self.dy(self.h, derivs)
	while True:
	    self.dy(self.h, derivs)
	    err = self.error()
	    if self.success(err):
		break
	    if abs(self.h) <= abs(self.x)*self.EPS:
		raise ValueError("stepsize underflow in StepperDopr5")

	if self.dense:
	    self.prepareDense(self.h, derivs)

	self.dydx = self.dydxnew.copy()
	self.y = self.yout
	self.xold = self.x
	self.hdid = self.h
	self.x += self.hdid
        return self.x, self.y

    def dy(self, h, derivs):
	"""Given values for n variables y[0..n-1] and their derivatives
	dydx[0..n-1] known at x, use the fifth-order Dormand-Prince Runge-Kutta
	method to advance the solution over an interval h and store the
	incremented variables in yout[0..n-1]. Also store an estimate of the
	local truncation error in yerr using the embedded fourth-order method.
	"""
        y = self.y
        x = self.x
        dydx = self.dydx

        ytemp = np.zeros_like(y)

	#ytemp = y + h*self.a21*dydx                        # First step.
	#derivs(x+self.c2*h, ytemp, self.k2)	            # Second step.
        #k2 = self.k2
	#ytemp = y + h*(self.a31*dydx + self.a32*k2)
	#derivs(x+self.c3*h, ytemp, self.k3)	            # Third step.
        #k3 = self.k3
	#ytemp = y + h*(self.a41*dydx + self.a42*k2 + \
        #                self.a43*k3)
	#derivs(x+self.c4*h, ytemp, self.k4)	            # Fourth step.
        #k4 = self.k4
	#ytemp = y + h*(self.a51*dydx + self.a52*k2 + \
        #                self.a53*k3 + self.a54*k4)
	#derivs(x+self.c5*h, ytemp, self.k5)                 # Fifth step.
        #k5 = self.k5
	#ytemp = y + h*(self.a61*dydx + self.a62*k2 + \
        #                self.a63*k3 + self.a64*k4 +\
        #                self.a65*k5)
	#xph = x + h
	#derivs(xph, ytemp, self.k6)	                    # Sixth step.
	## Accumulate increments with proper weights
	#self.yout = y + h*(self.a71*dydx  + self.a73*self.k3 + \
        #                    self.a74*self.k4 + self.a75*self.k5 + \
        #                    self.a76*self.k6)

        firstStep(ytemp, y, h, dydx, self.n)

        derivs(x+self.c2*h, ytemp, self.k2)
        secondStep(ytemp, y, h, dydx, self.k2, self.n)

        derivs(x+self.c3*h, ytemp, self.k3)
        thirdStep(ytemp, y, h, dydx, self.k2, self.k3, self.n)

        derivs(x+self.c4*h, ytemp, self.k4)
        fourthStep(ytemp, y, h, dydx, self.k2, self.k3, self.k4, self.n)

        derivs(x+self.c5*h, ytemp, self.k5)
        fifthStep(ytemp, y, h, dydx, self.k2, self.k3, self.k4, self.k5,
                    self.n)

	xph = x + h
	derivs(xph, ytemp, self.k6)
        #sixthStep(self.yout, y, h, dydx, self.k3, self.k4, self.k5,
        #            self.k6, self.n)
        self.yout = y + h*(self.a71*dydx  + self.a73*self.k3 + \
                            self.a74*self.k4 + self.a75*self.k5 + \
                            self.a76*self.k6)

	derivs(xph, self.yout, self.dydxnew)

	self.yerr = h*(self.e1*dydx + self.e3*self.k3 + self.e4*self.k4 + \
                        self.e5*self.k5 + self.e6*self.k6 + \
                        self.e7*self.dydxnew)

    def prepareDense(self, h, derivs):
	"""Store coefficients of interpolating polynomial for dense output in
	rcont1...rcont5"""
	# Data
        d1=-12715105075./11282082432.; d3=87487479700./32700410799.;
        d4=-10690763975./1880347072.; d5=701980252875./199316789632.;
        d6=-1453857185./8226518144.;  d7=69997945./29380423.;

	self.rcont1 = self.y
	ydiff = self.yout - self.y
	self.rcont2 = ydiff
	bspl = h*self.dydx - ydiff
	self.rcont3 = bspl
	self.rcont4 = ydiff - h*self.dydxnew - bspl
	self.rcont5 = h*(d1*self.dydx + d3*self.k3 + d4*self.k4 + d5*self.k5 + \
		    d6*self.k6 + d7*self.dydxnew)

    def denseOut(self, x, h):
	"""Evaluate interpolating polynomial for y at location x, where
	xold <= x <= xold+h."""
	#s = (x-self.xold)/h
        #s1 = 1.-s

	#ynew = self.rcont1 + s*(self.rcont2 + s1*(self.rcont3 + s*(self.rcont4 + \
	#			s1*self.rcont5)))
    
        ynew = np.zeros((len(x), self.n))

        denseOutJit(ynew, x, self.xold, h, self.rcont1, self.rcont2,
                    self.rcont3, self.rcont4, self.rcont5, len(x),
                    self.n)

	return ynew

    def error(self):
	"""Use yerr to compute norself.m of scaled error estimate. A value less than
	one means the step was successful."""
	err = 0
	for yi, youti, yerri in zip(self.y, self.yout, self.yerr):
	    sk = self.atol + self.rtol*max(abs(yi), abs(youti))
	    err += (yerri/sk)**2
	return np.sqrt(err/len(self.y))

    def success(self, err):
	"""Returns True if err<=1, False otherwise. If step was successful,
	sets hnext to the estimated optimal stepsize for the next step. If the
	step failed, reduces h appropriately for another try."""

	if err <= 1.:		# Step succeeded. Compute hnext.
	    if err == 0:
		scale = self.maxscale
	    else:
		scale = self.safe*err**(-self.alpha)*self.errold**self.beta
		if scale<self.minscale: scale=self.minscale
		if scale>self.maxscale: scale=self.maxscale

	    if self.reject:			 # Don' let step increase if last
		self.hnext = self.h*min(scale, 1.)    # one was just rejected.
	    else:
		self.hnext = self.h*scale

	    self.errold = max(err, 1e-4)        # Bookkeeping for next call.
	    self.reject = False
            return True

	else:                   # Truncation error too large, reduce stepsize.
	    scale = max(self.safe*err**(-self.alpha), self.minscale)
	    self.h *= scale
	    self.reject = True
	    return False


@jit(void(float64[:], float64[:], float64, float64[:], int64), nopython=True)
def firstStep(ytemp, y, h, dydx, ny):
    for i in range(ny):
        ytemp[i] = y[i] + h*0.2*dydx[i]

@jit(void(float64[:], float64[:], float64, float64[:], float64[:], int64), nopython=True)
def secondStep(ytemp, y, h, dydx, k2, ny):
    for i in range(ny):
        ytemp[i] = y[i] + h*(3./40.*dydx[i] + 9./40.*k2[i])

@jit(void(float64[:], float64[:], float64, float64[:], float64[:], float64[:], int64), nopython=True)
def thirdStep(ytemp, y, h, dydx, k2, k3, ny):
    for i in range(ny):
        ytemp[i] = y[i] + h*(44./45.*dydx[i] - 56./15.*k2[i] + \
                        32./9.*k3[i])

@jit(void(float64[:], float64[:], float64, float64[:], float64[:], float64[:], float64[:], int64), nopython=True)
def fourthStep(ytemp, y, h, dydx, k2, k3, k4, ny):
    for i in range(ny):
        ytemp[i] = y[i] + h*(19372./6561.*dydx[i] - 25360./2187*k2[i] + \
                        64448./6561.*k3[i] - 212./729.*k4[i])

@jit(void(float64[:], float64[:], float64, float64[:], float64[:], float64[:], float64[:], float64[:], int64), nopython=True)
def fifthStep(ytemp, y, h, dydx, k2, k3, k4, k5, ny):
    for i in range(ny):
        ytemp[i] = y[i] + h*(9017./3168.*dydx[i] - 355./33.*k2[i] + \
                        46732./5247.*k3[i] + 49./176.*k4[i] -\
                        5103./18656.*k5[i])

@jit(void(float64[:], float64[:], float64, float64[:], float64[:], float64[:], float64[:], float64[:], int64), nopython=True)
def sixthStep(yout, y, h, dydx, k3, k4, k5, k6, ny):
    for i in range(ny):
        yout[i] = y[i] + h*(35./384.*dydx[i]  + 500./1113.*k3[i] + \
                        125./192.*k4[i] - 2187./6784.*k5[i] + \
                        11./84.*k6[i])

@jit(void(float64[:,:], float64[:], float64, float64, float64[:], float64[:],
        float64[:], float64[:], float64[:], int64, int64), nopython=True)
def denseOutJit(ynew, x, xold, h, rcont1, rcont2, rcont3, rcont4, rcont5, nx, ny):
    for i in range(nx):
        s = (x[i]-xold)/h
        s1 = 1.-s
        for j in range(ny):
            ynew[i,j] =  rcont1[j] + s*(rcont2[j] + s1*(rcont3[j] +
                            s*(rcont4[j] + s1*rcont5[j])))

            

def rk4(y, dydx, x, h, derivs, args=()):
    """Given values for the variables y[0..n-1] and their derivatives
    dydx[0..n-1] known at x, use the fourth-order Runge-Kutta method to advance
    the solution over an interval h and return the incremented variables as
    yout[0..n-1]. The user supplies the routine derivs(x,y), which returns
    derivatives dydx at x.
    """
    n = len(y)
    hh = h*0.5
    h6 = h/6.
    xh = x+hh

    yt = y + hh*dydx		    # First step.
    dyt = np.zeros(n)
    derivs(xh, yt, dyt, *args)	    # Second step.
    yt = y + hh*dyt
    dym = np.zeros(n)
    derivs(xh, yt, dym, *args)	    # Third step.
    yt = y + h*dym
    dym += dyt
    derivs(x+h, yt, dyt, *args)    # Fourth step.

    # Accumulate increments with proper weights
    yout = y + h6 * (dydx + dyt + 2*dym)

    return yout

class VanDerPol(object):
    def __init__(self, eps):
        self.eps = eps
    def __call__(self, x, y, dydx):
        dydx[0] = y[1]
        dydx[1] = ((1.0-y[0]*y[0])*y[1]) - y[0]/self.eps

###### TESTING ######
if __name__ == '__main__':
    pass
