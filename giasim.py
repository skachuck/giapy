import numpy as np
import time

from scipy.optimize import leastsq

class GiaSim(object):
    """Calculate and store Glacial Isostacy Simulation results, and compare
    with data. Must be called with an earth model (earth), an ice model (ice)
    and a grid (grid). To add a data source to be used in residual finding and
    interpolation, use GiaSim.attach_data.

    Parameters
    ----------
    earth - the earth model to be used in the simulation
    ice - the ice model to be used in the simulation
    grid - the map associated with the earth and ice models
    """
    
    def __init__(self, earth, ice, grid, datalist=None):
       self.earth = earth
       self.ice = ice
       self.grid = grid

       if datalist is not None:
           for data in datalist:
               self.datalist = datalist
       else:
           self.datalist = []

    def attach_data(self, data):
        self.datalist.append(data)

    def reset_data(self):
        self.datalist = []

    def remove_data(self, data):
        self.datalist.remove(data)

    def attach_esl(self, esl):
        self.esl = esl

    def set_out_times(self, out_times):
        self.out_times = out_times
        
    def leastsq(self, x0, argdict=None, priors=None, full_output=0, 
                    save_params=False, save_chi2=False):
        """Calculate the least squares minimum from starting point x0.

        Optional Arguments
        ------------------
        argdict - 
        priors - list of parameter prior standard deviations
        full_output - 
        save_params - if True, save the param steps during optimization
        save_chi2 - if True, save the steps in chi2 during optimization
        """

        self.priors = priors

        if save_params:
            self.old_params = []
        else:
            self.old_params = None

        if save_chi2:
            self.old_chi2 = []
        else:
            self.old_chi2 = None

        m = leastsq(self.residuals, x0, args=(argdict,), Dfun=self.jacobian, 
                    col_deriv=1, full_output=full_output)
        return m

    def residuals(self, xs, argdict=None, verbose=False):
        """Calculate the residuals associated with stored data sources and
        earth parameters xs.
        """
        if not self.datalist:
            raise StandardError('self.datalist is empty. Use self.attach_data.')

        if self.old_params is not None:
            self.old_params.append(self.earth.get_params())

        if argdict is None:
            self.earth.reset_params(*xs)
        else:
             self.earth.reset_params_list(xs, argdict)

        self.perform_convolution()
        if hasattr(self, 'esl'): self.mw_corr()
        res = []

        for data in self.datalist:
            res.append(data.residual(self, verbose=verbose))
        
        if self.priors:
            res.append((xs-self.priors[:,0])/self.priors[:,1])

        res = np.concatenate(res)
        if self.old_chi2 is not None:
            self.old_chi2.append(res.dot(res))

        return res

    def jacobian(self, xs, argdict=None, eps_f=5e-11):
        """Calculate the jacobian associated with stored data sources and
        parameters xs, with function evaluation error eps_f (default 5e-11).
        """
        jac = []
        xs = np.asarray(xs)
        for i, x in enumerate(xs):
            # Determine the separation to use
            # Optimal one-pt separation is (eps_f*f/f'')^(1/2) ~ sqrt(eps_f)*x
            # Optimal two-pt separation is (eps_f*f/f''')^(1/3) ~ cbrt(eps_f)*x
            h = np.zeros(len(xs))
            h[i] = (eps_f**(1./3.))*x

            # Evaluate the function
            # One-pt
            #f1 = rebound_2d_earth_res(xs...)
            # Two-pt
            f1 = self.residuals(xs-h, argdict)
            f2 = self.residuals(xs+h, argdict)

            # Difference
            # One-pt
            #(f2-f1)/h
            # Two-pt
            jac.append((f2-f1)*0.5/h[i])

        # put them together
        jac = np.asarray(jac)
        return jac

    def perform_convolution(self, out_times=None, t_rel=0, verbose=False):  
        """Convolve an ice load and an earth response model in fft space.
        Calculate the uplift associated with stored earth and ice model.
        
        Parameters
        ----------
        out_times - an array of times at which to caluclate the convolution.
                    (default is to use previously stored values).
        t_rel - the time relative to which uplift is considered (defaul present)
                (None for no relative)
        """
        time_start = time.clock()

        earth = self.earth
        ice = self.ice

        N = earth.N                             # use the resolution in earth
        Nrem = 1                                # number of intermediate steps
    
        out_times = out_times or self.out_times
        self.out_times = out_times

        # Make sure t_rel is in out_times
        if t_rel is not None and t_rel not in out_times:
            raise ValueError('t_rel must be in out_times')
        
        # Fourier transform the ice_hist
        ice_stages = ice.fft(N)
        
        # Initialize the uplift array
        uplift_f = np.zeros((len(out_times), N, N), dtype=complex)
        
        # Convolve each ice stage to the each output time
        for ice0, t0, ice1, t1 in \
                          zip(ice_stages, ice.times, ice_stages[1:], ice.times[1:]):
            delta_ice = (ice0 - ice1)/Nrem
            for inter_time in np.linspace(t0, t1, Nrem, endpoint=False):
                # Perform the time convolution for each output time
                for t_out in out_times[out_times <= inter_time]:
                    t_dur = (inter_time-t_out)
                    # 0.3 accounts for density difference between ice and rock
                    uplift_f[t_out == out_times, :, :] += 0.3 *\
                                        delta_ice * earth.get_resp(t_dur)
        
        # The resolution correction
        res = float(N)/ice.N
        shape = (np.ceil(res*ice.shape[0]), np.ceil(res*ice.shape[1]))
        # Retransform the uplift
        # The normalization needs to be corrected for each dimension (N/ice.N)**2
        uplift = np.real(np.fft.ifft2(uplift_f, s=[N, N]))*(res**2)
    
        # Calculate uplift relative to t_rel (default, present)
        if t_rel is not None: 
            uplift = uplift[np.where(out_times==t_rel)] - uplift 

        self.grid.update_shape(shape)
    
        # Correctly grid the uplift array by removing the fourier padding
        self.uplift = uplift[:, :shape[0], :shape[1]]

        if verbose: print 'Convolution time: {0}s'.format(time.clock()-time_start)

    def mw_corr(self, esl=None):
        """Apply the meltwater correction to transform uplift to emergence."""
        self.esl = esl or self.esl
            
        eslcorr = self.esl(self.out_times)
        self.uplift = self.uplift + eslcorr[:, np.newaxis, np.newaxis]   
