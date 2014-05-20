import numpy as np
import time

class GiaSim(object):
    """
    # Set up the simulation with earth and ice model (and grid?)
    simulation = GiaSim(earth, ice)
    
    simulation.add_data(emerge_data)
    simulation.add_data(tilt_data)
    
    simulation.calc_residual(params)
        perform_convolution -> store uplift
        send uplift, out_times, grid to data objects
        for data_source in data objects:
            res = data objects.interpolate(simulation.uplift, etc.
        put residuals together
        return residuals
    """
    
    def __init__(self, earth, ice, grid):
       self.earth = earth
       self.ice = ice
       self.grid = grid
       self.datalist = []

    def attach_data(self, data):
        self.datalist.append(data)

    def residuals(self, xs, verbose=False):
        if not self.datalist:
            raise StandardError('self.datalist is empty. Use self.attach_data.')
        
        self.earth.reset_params(*xs)
        self.perform_convolution()
        res = []

        for data in self.datalist:
            res.append(data.residual(self, verbose=verbose))

        return np.concatenate(res)

    def jacobian(self, xs, eps_f=5e-11):
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
            f1 = self.residuals(xs-h)
            f2 = self.residuals(xs+h)

            # Difference
            # One-pt
            #(f2-f1)/h
            # Two-pt
            jac.append((f2-f1)*0.5/h[i])

        # put them together
        jac = np.asarray(jac)
        return jac.T

    def perform_convolution(self, out_times=None, t_rel=0, verbose=False):  
        """Convolve an ice load and an earth response model in fft space.
        
        Parameters
        ----------
        earth - an object that has procedure earth.get_resp(t_dur)
        ice - an obect that has ice.fft(), ice.times
        out_times - an array of times at which to caluclate the convolution
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
