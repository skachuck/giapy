"""
This module includes procedures for importing and manipulating emergence data.
"""

import numpy as np
import cPickle as pickle
import time

class EmergeData(object):
    """
    """
    def __init__(self, data=[]):

        self.data = data     
        self.form_long_vectors()
        
    def __getitem__(self, key):
        return self.data.__getitem__(key)
        
    def __iter__(self):
        return self.data.__iter__()

    def interp(self, simobject, verbose=False):
        """Interpolate uplift surfaces (xyz data at a specific t) to data
        locations (non-grid) and data times (between times calculated). 
        
        Uses progressive linear interpolations: first the uplift at each 
        outputted time is interpolated to the data locations in data.locs, 
        then they are interpolated to the data times in each location.
        
        Parameters
        ----------
        simobject 
        """
        time_start = time.clock()
        
        # Extract attributes from simobject
        uplift = simobject.uplift
        out_times = simobject.out_times
        grid = simobject.grid

        # interp_data will be an array of size (N_output_times, N_locations)
        # for use in interpolating the calculated emergence to the locations
        # and times at which there are data in data
        interp_data = []
        # Interpolate the calculated uplift at each time on the Lat-Lon grid
        # to the data locations.
        for uplift_at_a_time in uplift:
            interp_func = grid.create_interper(uplift_at_a_time.T)
            interp_data.append(interp_func.ev(self.locs[:,0], self.locs[:,1]))
        interp_data = np.array(interp_data).T
        
        calc_vector = []
        # Interpolate the calculated uplifted at each time and data location
        # to the times of the data location.
        for interp, loc in zip(interp_data, self.data):
            calc_vector.append(np.interp(loc['data_dict']['times'],
            out_times[::-1], interp[::-1]))
        
        # flatten the array    
        calc_vector = np.array([item for l in calc_vector for item in l])
        
        if verbose: 
            print 'Interpolation time: {0}s'.format(time.clock()-time_start)
    
        return calc_vector

    def residual(self, simobject, verbose=False):
        calc_vector = self.interp(simobject, verbose)

        return (calc_vector-self.long_data)/1.0

    def import_from_file(self, filename):
        """Reads emergence data from a file where it is stored in the form...
        
        Stores the emergence data in a list self.data of dictionaries with keys:
                lat
                lon
                year      (of publication)
                desc      (of dataset)
                comm      (ent)
                auth      (ors of publication)
                tect      (key for tectonic activity, 1 or 0)
                tectup    (the magnitude of tectonic uplift)
                age       (some sort of age)
                recnbr    (the unique reference number)
                data_dict (a dictionary with keys 'times' and 'emerg' for ordered
                            lists of the data)
        
        Stores the locations in a list [[lon, lat]]
        Stores all the data in the form 
                            
        Parameters
        ----------
        filename - the file containing the data to be imported
        """
        
        self.data = []                         # Initialize the array,
        self.long_data = []
        self.long_time = []
        self.locs = []
        
        f = open(filename, 'r')                 # open the file,
        line = f.readline()                     # read the first line to initiate
        while line:                             # the while loop.
            # read off the metadata for the location
            lat, lon, year, num = np.array(line.split('\t')[:4]).astype(np.float)
            unit, sign, typ, age, tect, tectup, recnbr = np.array(
                                    f.readline().split('\t')[:7]).astype(np.float)
            auth = f.readline().split('\t')[0]
            desc = f.readline().split('\t')[0]
            comm = f.readline().split('\t')[0]
            
            if 100 <= recnbr < 400:         # apply a correction for misentered
                comm, desc = desc, comm     # data.
            
            times = []                      # initiate the data_dict
            emerg = []
            data = {}
            if recnbr < 400 or recnbr >= 500:
                for i in range(int(num)):
                    line = np.array(f.readline().split('\t')[:2]).astype(np.float)
                    if line[0] in times:
                        pass
                    else:
                        times.append(line[0])
                        self.long_time.append(line[0])
                        emerg.append(line[1])
                        self.long_data.append(line[1])
            else:
                # Newer data were given time bounds to read in as well (we're 
                # skipping them for now and reading in the most-likely time).
                for i in range(int(num)):
                    line = np.array(f.readline().split('\t')[:5]).astype(np.float)
                    if line[0] in times:
                        pass
                    else:
                        times.append(line[0])
                        self.long_time.append(line[0])
                        emerg.append(line[3])
                        self.long_data.append(line[3])
  
            # Post processing of data based on metadata
            if unit == 2: emerg = np.array(emerg)*0.3048          # ft -> m
            if sign >=10: 
                times = np.array(times)/1000.
                sign = sign/10
            if sign == 1: times = -1*np.array(times)
            
            data['times']=times
            data['emerg']=emerg
            data['error']=[]  
            
            badpts = [137, 41, 232, 234, 230, 231, 228, 229, 235, 236, 310,
                        200, 183, 318, 319, 320, 295, 296]

            # ignore bad pts
            if int(recnbr) in badpts: 
                pass
            # and pts whose locations are already in
            elif [lon, lat] in self.locs:
                print ('Recnbr {0} at [{1}, {2}] is a duplicate loc'\
                        .format(recnbr, lon, lat))
            else:
                # Form the dictionary entry for this location
                self.data.append(dict(zip(['lat', 'lon', 'year', 'desc', 'comm', 
                                            'auth', 'tect', 'tectup', 'age', 
                                            'recnbr', 'data_dict'], 
                                            [lat, lon, int(year), desc, comm, 
                                            auth, int(tect), tectup, int(age), 
                                            int(recnbr), data])))
                self.locs.append([lon, lat])
            line = f.readline()                 # step on.
            
        self.locs = np.array(self.locs)
        f.close()
        
    def form_long_vectors(self):
        """Update the long lists: long_data, long_time, and locs with currently
        encapsulated data.
        
        Certain numbers are stored twice, for convenience in calculating the 
        residuals, in the form of long lists. These are every emergence height, 
        every emergence time, and every lon,lat pair. The one-time storage 
        overhead is worth the time saved from not recreating these lists on
        every iteration of an inversion.
        """
        
        self.long_data = []
        self.long_time = []
        self.locs = []
        for loc in self.data:
            self.locs.append([loc['lon'], loc['lat']])
            for point in loc['data_dict']['emerg']:
                self.long_data.append(point)
            for point in loc['data_dict']['times']:
                self.long_time.append(point)
        self.locs = np.array(self.locs)
        
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
            if ( max(loc['data_dict']['times']) > time/scale
                and min(loc['data_dict']['times']) < time/scale):
                yield loc
                
    def by_loc(self, latmin=-90, latmax=90, lonmin=-180, lonmax=180):
        """Yield locations whose locations are within a lat/lon box
        
        Parameters
        ----------
        latmin / latmax / lonmin / lonmax - the lat/lon bounding box
        """
        for loc in self.data:
            if ((latmin <= loc['lat'] <= latmax) 
                and (lonmin <= loc['lon'] <= lonmax)):
                yield loc
        
    def by_recnbr(self, nbrs):
        """Yield locations identified by provided recordnumbers.
        
        Parameters
        ----------
        nbrs - a list of record numbers to select (ADD ABILITY TO SELECT ONLY
                ONE NUMBER).
        """
        for loc in self.data:
            if loc['recnbr'] in nbrs:
                yield loc

    def filter(self, func, args, filename=None):
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
        filtered = EmergeData([loc for loc in 
                                        self.__getattribute__(func)(**args)])
        filtered.form_long_vectors()
                
        if filename is not None:
            filtered.save(filename)
            
        return filtered

    def plot_loc(self, map, kwarg={}):
        """Plot the locations of data on a map object"""
        x, y = map(*zip(*self.locs))
        map.plot(x, y, ls='None', **kwarg)

def load(filename):
    return pickle.load(open(filename, 'r'))
