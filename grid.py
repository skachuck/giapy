import numpy as np

class EarthGrid(object):
    def __init__(self, res):
        if np.size(res)==1:
            self.res = (res, res)
        self.lat_list = np.arange(-90, 90, 180./self.res[0])
        self.lon_list = np.arange(-180, 180, 360./self.res[1])
        self.lat_mesh, self.lon_mesh = np.meshgrid(self.lat_list, self.lon_list)
        
    def select_area(self, array, latmin, latmax, lonmin, lonmax):
        lat_ind = np.logical_and(self.lat_mesh>latmin, self.lat_mesh<latmax)
        lon_ind = np.logical_and(self.lon_mesh>lonmin, self.lon_mesh<lonmax)
        
        nlat = len(self.lat_list[np.logical_and(self.lat_list>latmin, 
                                                self.lat_list<latmax)])
        nlon = len(self.lon_list[np.logical_and(self.lon_list>lonmin, 
                                                self.lon_list<lonmax)])
        
        return np.reshape(array[np.logical_and(lat_ind, lon_ind)], (nlon, nlat))
    
europe = {'latmax': 82, 'latmin': 49, 'lonmax': 75, 'lonmin': -15}
    
europe_plot_box = {'lon_0': 35, 'lat_0': 60.15, 'llcrnrlon':-5, 'urcrnrlon':99, 'llcrnrlat':35, 'urcrnrlat':80}

    m = basemap.Basemap(width=4900000,height=4700000,
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',area_thresh=1000.,projection='lcc',\
            lat_1=50.,lat_2=89.9,lat_0=72,lon_0=20.)