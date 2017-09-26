import numpy as np


from giapy.map_tools import haversine

class TiltData(object):
    def __init__(self):
       self.locs = np.array([[5, 60.3], [6, 62], [10, 63.75],
                             [16, 68.5], [27.25, 71]])
       self.long_data = np.array([1.3, 1.3, 1.6, 1.1, 0.6])
       self.long_time_i = np.array([12, 12, 12, 12, 12])
       self.long_time_f = np.array([0,0,0,0,0])
       self.long_recnbrs = np.array([1, 2, 3, 4, 5])

    def transform_locs(self, basemap, inverse=False):
        xs, ys = basemap(self.locs[:,0], self.locs[:,1], inverse=inverse)
        self.locs[:,0], self.locs[:,1] = xs, ys

    def interp(self, simobject, verbose=False):


        grid = simobject.grid
        uplift = simobject.uplift[np.where(simobject.out_times==13)[0][0]]

        Lon, Lat = grid.basemap(*np.meshgrid(grid.x, grid.y), inverse=True)
    
        tilt = calc_tilts(uplift, Lon, Lat)
    
        interp_func = grid.create_interper(tilt.T)
        calc_vector = interp_func.ev(self.locs[:,0], self.locs[:,1])
    
        return calc_vector

    def residual(self, simobject, verbose=False):
        calc_vector = self.interp(simobject, verbose)

        return (calc_vector-self.long_data)/0.1

def calcTilts(uplift, Lon, Lat, r=6371):
    """Calculate the gradient magnitude of an uplift plane.
    
    Uses two-point central difference in the body and one-point difference
    along the edges.

    Parameters
    ----------
    uplift - the uplift at a single time.
    Lon, Lat - the Lon/Lat meshgrid associated with the uplift array.
    """

    if uplift.shape != Lon.shape != Lat.shape:
        raise ValueError('uplift {0}, Lon {1}, and Lat {2} must all have same\
                            shape'.format(uplift.shape, Lon.shape, Lat.shape))

    # intialize arrays to shape of one uplift plane
    ushape = uplift.shape[-2:]
    du_lat = np.zeros(ushape)
    du_lon = np.zeros(ushape)
    dX = np.zeros(ushape)
    dY = np.zeros(ushape)

    # differences in horizontal (longitudinal) direction
    du_lon[:, 1:-1] = uplift[:, 2:]-uplift[:, :-2]  # body
    du_lon[:, 0] = uplift[:, 1]-uplift[:, 0]        # left
    du_lon[:, -1] = uplift[:, -1]-uplift[:, -2]     # right

    dX[:, 1:-1] = haversine(Lat[:, :-2], Lat[:, 2:], Lon[:, :-2], Lon[:, 2:])
    dX[:, 0] = haversine(Lat[:, 0], Lat[:, 1], Lon[:, 0], Lon[:, 1])
    dX[:, -1] = haversine(Lat[:, -2], Lat[:, -1], Lon[:, -2], Lon[:, -1])

    # differences in vertical (latitudinal) direction
    du_lat[1:-1, :] = uplift[2:, :]-uplift[:-2, :]  # body
    du_lat[0, :] = uplift[1, :]-uplift[0, :]        # bottom
    du_lat[-1, :] = uplift[-1, :]-uplift[-2, :]     # top

    dY[1:-1, :] = haversine(Lat[:-2, :], Lat[2:, :], Lon[:-2, :], Lon[2:, :])
    dY[0, :] = haversine(Lat[0, :], Lat[1, :], Lon[0, :], Lon[1, :])
    dY[-1, :] = haversine(Lat[-2, :], Lat[-1, :], Lon[-2, :], Lon[-1, :])

    # derivatives with respect to lon/lat
    du_lon_dx = du_lon/dX
    du_lat_dy = du_lat/dY
    # calculate magnitude of gradient vectors
    tilt = np.sqrt(du_lon_dx**2+du_lat_dy**2)

    return tilt
