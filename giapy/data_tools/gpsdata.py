import numpy as np
from giapy.data_tools.abstractDataClasses import AbsGeoDatum, \
                                                AbsGeoDataContainer

class GPSDatum(AbsGeoDatum):
    pass

class GPSData(AbsGeoDataContainer):
    def __init__(self, data=[]):
        self.data = data
        self.form_long_vectors()

    def __iter__(self):
        return self.data.__iter__()

    def form_long_vectors(self):
        """Update the long lists: long_data and locs with currently
        encapsulated data.
        
        Certain numbers are stored twice, for convenience in calculating the 
        residuals, in the form of long lists. These are every emergence height, 
        every emergence time, and every lon,lat pair. The one-time storage 
        overhead is worth the time saved from not recreating these lists on
        every iteration of an inversion.
        """
        
        self.long_vu = []
        self.locs = []
        for loc in self:
            self.locs.append(loc.loc)
            self.long_vu.append(loc.vu)
        self.long_vu = np.array(self.long_vu)
        self.locs = np.array(self.locs)
