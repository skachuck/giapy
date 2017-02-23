from ... import pickle

class AbsGeoTimeSeries(object):
    """Wrapper for time-series data which carries the time-series' meta-data.
    """
    def __init__(self, timeseries, **metadata):
        self.timeseries = timeseries

        if 'loc' in metadata.keys():
            metadata['lon'] = metadata['loc'][0]
            metadata['lat'] = metadata['loc'][1]
            del metadata['loc']
        elif 'lon' not in metadata.keys() or 'lat' not in metadata.keys():
            raise ValueError('lon and lat or loc=(lon,lat) must be specified.')

        for name, value in metadata.items():
            setattr(self, name, value)

    def __getitem__(self, key):
        return self.timeseries.__getitem__(key)

    def __iter__(self):
        return self.timeseries.__iter__()

    def __len__(self):
        return self.timeseries.shape[0]

    @property
    def loc(self):
        return self.lon, self.lat

    @property
    def ts(self):
        return self.timeseries[:,0]

    @property
    def ys(self):
        return self.timeseries[:,1]

class AbsGeoTimeSeriesContainer(object):
    """Abstract wrapper for geographic data (data defined at many locations
    around the globe). It provides a basic functions and ability to filter the
    data by time, rectangular lat/lon box, or by recnbr.

    Notes
    -----
    Concrete classes must store data in iterable self.data.
    For filters, individual entry datum must have datum.lon and datum.lat, data
    must be of shapt (num, 2), where first colum is time, and must have
    datum.recnbr.
    """
    def __init__(self):
        raise NotImplemented()

    def __getitem__(self, key):
        return self.data.__getitem__(key)
        
    def __iter__(self):
        return self.data.itervalues()

    def __len__(self):
        return len(self.data)

    @property
    def lons(self):
        return self.locs[:,0]

    @property
    def lats(self):
        return self.locs[:,1]


    def save(self, filename):
        """Save the EmergeData interface object, with empty data list
        
        Parameters
        ----------
        filename (str) - path to the file to save
        """
        pickle.dump(self, open(filename, "wb"))

    def by_time(self, time, scale=1):
        """Yield locations whose data bounds a certain time, for interpolation 
        of emergence.
        
        Parameters
        ----------
        time - the time to locate
        scale - if the data and time are not in the same units (cal years / ka)   
        """
        for loc in self.data:
            if ( max(loc.ts) > time/scale
                and min(loc.ts) < time/scale):
                yield loc
                
    def by_loc(self, latmin=-90, latmax=90, lonmin=-180, lonmax=180):
        """Yield locations whose locations are within a lat/lon box
        
        Parameters
        ----------
        latmin / latmax / lonmin / lonmax - the lat/lon bounding box
        """
        for loc in self.data:
            if ((latmin <= loc.lat <= latmax) 
                and (lonmin <= loc.lon <= lonmax)):
                yield loc
        
    def by_recnbr(self, nbrs):
        """Yield locations identified by provided recordnumbers.
        
        Parameters
        ----------
        nbrs - a list of record numbers to select (ADD ABILITY TO SELECT ONLY
                ONE NUMBER).
        """
        for loc in self:
            if loc.recnbr in nbrs:
                yield loc

    @classmethod
    def filter(cls, func, args, filename=None):
        """Filter data and return a new EmergeData object with the new data.
        
        Parameters
        ----------
        func (str) - the name of the filtering method. Options include: 
                    by_time, by_loc, by_recnbr. See documentation for those 
                    methods for details.
        args - arguments to provide to the procedure (see documentation)
        filename - to save new data immediately, provide a filename. Filtered
                    data can be saved later by name.save(filename)
        """
        filteredData = {}

        for loc in func(**args):
            filteredData[loc.recnbr] = loc

        filtered = cls(filteredData)
                
        if filename is not None:
            filtered.save(filename)
            
        return filtered

    def plot_loc(self, map, **kwarg):
        """Plot the locations of data on a map object"""
        x, y = map(*zip(*self.locs))
        map.plot(x, y, ls='None', **kwarg)

