"""
This module includes procedures for importing and manipulating emergence data.
"""

import numpy as np
import time

from .yearcalib import uncalib_bloom, c14corr
from .abstractDataClasses import AbsGeoTimeSeries, AbsGeoTimeSeriesContainer

from .. import timestamp


def calcEmergence(sim, emergedata, smooth=True):
    #TODO This function shouldn't have to know what's inside any object, let
    # alone a complicated one, like sim. Consider calculating rsl first and
    # passing it and calculated times in (ts, rsl, emergedata).
    # To reference to present day
    #u0 = sim.inputs.harmTrans.spectogrd(sim['topo'][-1])
    u0 = sim['rsl'][-1]

    uAtLocs = []
    for ut in sim['rsl']:
        ut = u0 - ut
        interpfunc = sim.inputs.grid.create_interper(ut.T)
        uAtLocs.append(interpfunc.ev(emergedata.lons, emergedata.lats))

    uAtLocs = np.array(uAtLocs).T

    data = {}
    for uAtLoc, loc in zip(uAtLocs, emergedata):
        if smooth:
            ts = np.union1d(np.sort(loc.ts), np.linspace(0, loc.ts.max()))
        else:
            ts = np.sort(loc.ts)
        timeseries = np.array([ts, 
                    np.interp(ts, 
                                sim.inputs.out_times[::-1], uAtLoc[::-1])]).T
        data[loc.recnbr] = EmergeDatum(timeseries, 
                                        lat=loc.lat, 
                                        lon=loc.lon,
                                        desc=loc.desc,
                                        recnbr=loc.recnbr)

    return EmergeData(data)

class EmergeDatum(AbsGeoTimeSeries):
    def __repr__(self):
        return self.desc

class EmergeData(AbsGeoTimeSeriesContainer):
    """
    """
    def __init__(self, data={}):
        self.data = data
        self.form_long_vectors()
        self.W = None
        self.TIMESTAMP = timestamp()
        
    def transform_locs(self, basemap, inverse=False):
        xs, ys = basemap(self.locs[:,0], self.locs[:,1], inverse=inverse)
        self.locs[:,0], self.locs[:,1] = xs, ys

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
        
        # Use map coordinates from sim.grid.basemap for interpolation
        x, y = grid.basemap(self.locs[:,0], self.locs[:,1])

        # interp_data will be an array of size (N_output_times, N_locations)
        # for use in interpolating the calculated emergence to the locations
        # and times at which there are data in data
        interp_data = []
        # Interpolate the calculated uplift at each time on the Lat-Lon grid
        # to the data locations.
        for uplift_at_a_time in uplift:
            interp_func = grid.create_interper(uplift_at_a_time.T)
            interp_data.append(interp_func.ev(x, y))
        interp_data = np.array(interp_data).T
        
        calc_vector = []
        # Interpolate the calculated uplifted at each time and data location
        # to the times of the data location.
        for interp, loc in zip(interp_data, self.data):
            calc_vector.append(np.interp(loc[:,0],
            out_times[::-1], interp[::-1]))
        
        # flatten the array    
        calc_vector = np.array([item for l in calc_vector for item in l])
        
        if verbose: 
            print 'Interpolation time: {0}s'.format(time.clock()-time_start)
    
        return calc_vector

    def residual(self, simobject, verbose=False):
        calc_vector = self.interp(simobject, verbose)
        res = calc_vector - self.long_data
        if self.W is None:
            return res/1.0
        else:
            return self.W.dot(res)

    def findLocFromLongdata(self, num):
        """Return location index and point index for index from self.long_data.
        """
        nums = np.array([len(loc) for loc in self])
        cumnums = nums.cumsum()
        index = (np.arange(len(self))[cumnums < num]).max()
        return index+1, num-cumnums[index]-1 
        
    def form_long_vectors(self):
        """Update the long lists: long_data, long_time, and locs with currently
        encapsulated data.
        
        Certain numbers are stored twice, for convenience in calculating the 
        residuals, in the form of long lists. These are every emergence height, 
        every emergence time, and every lon,lat pair. The one-time storage 
        overhead is worth the time saved from not recreating these lists on
        every iteration of an inversion.
        """
        
        self.long_data = np.array([None])
        self.long_time = np.array([None])
        self.locs = np.array([None, None])
        for loc in self:
            self.locs = np.vstack((self.locs, loc.loc))
            self.long_time = np.concatenate((self.long_time, loc.ts))
            self.long_data = np.concatenate((self.long_data, loc.ys))
        self.long_data = self.long_data[1:]
        self.long_time = self.long_time[1:]
        self.locs = self.locs[1:]

def importEmergeDataFromFile(filename):
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
    
    data = {}                         # Initialize the array,
    
    # A list of bad points
    badpts = [137, 41, 203, 232, 234, 230, 231, 228, 229, 235, 236, 310,
                    200, 183, 318, 319, 320, 295, 296,
                    203, # Sabine high islands - uncertain age, high scatter
                    300 # Malaysia - uncertain calibration
                    ]



    locs = []
    
    f = open(filename, 'r')                 # open the file,
    line = f.readline()                     # read the first line to initiate
    while line:                             # the while loop.
        # read off the metadata for the location
        lat, lon, year, num = np.array(line.split('\t')[:4]).astype(np.float)
        unit, sign, typ, ageky, tect, tectup, recnbr = np.array(
                                f.readline().split('\t')[:7]).astype(np.float)
        auth = f.readline().split('\t')[0].strip('\' \" ')
        desc = f.readline().split('\t')[0].strip('\' \" ')
        comm = f.readline().split('\t')[0].strip('\' \" ')
        
        if 100 <= recnbr < 400:         # apply a correction for misentered
            comm, desc = desc, comm     # data.
        
        locs.append([lon, lat])
        timeseries = np.zeros((num, 2))     # initialize the time series array.
        if recnbr < 400 or recnbr >= 500:
            ncol = 2
            ecol = 1
        else:
            # Newer data were given time bounds to read in as well (we're 
            # skipping them for now and reading in the most-likely time).
            ncol = 5
            ecol = 3

        for i in range(int(num)):
            line = np.array(f.readline().split('\t')[:ncol]).astype(np.float)
            if line[0] in timeseries[:,0]:
                pass
            else:
                timeseries[i] = [line[0], line[ecol]]
        

        # Post processing of data based on metadata
        if unit == 2: timeseries[:,1] = timeseries[:,1]*0.3048      # ft -> m
        if sign >=10: 
            timeseries[:,0] = timeseries[:,0]/1000.
            sign = sign/10
        if sign == 1: timeseries[:,0] = -1*timeseries[:,0]
       
            # Correction for time calibration
        #if ageky == 1:
        #    pass
        #elif ageky == 2:
        #    try:
        #        times = c14corr(times*1000.)/1000.
        #    except:
        #        print desc, recnbr, times
        #        raise
        #elif ageky == 5:
        #    times = uncalib_bloom(times*1000.)/1000.
        #    times = c14corr(times*1000.)/1000.        

        # Ignore bad pts
        if int(recnbr) in badpts: 
            pass
        # and pts whose locations are already in
        #elif [lon, lat] in locs:
        #    dupnbr = [loc.recnbr for loc in data 
        #                if loc.lon==lon and loc.lat==lat]
        #    dupyr = [loc.year for loc in data 
        #                if loc.lon==lon and loc.lat==lat]
        #    print ('Recnbr {0} at [{1}, {2}] is a duplicate loc with {3}'\
        #            .format(recnbr, lon, lat, dupnbr))
        #    print ('    pubdates are {0} and {1}').format(year, dupyr)
        else:
            # Form the dictionary entry for this location
            data[int(recnbr)] = EmergeDatum(timeseries, lat=lat, lon=lon,
                                    year=int(year), desc=unicode(desc,
                                    errors='ignore'), comm=comm,
                                    auth=auth, tect=int(tect), tectup=tectup,
                                    ageky=int(ageky), recnbr=recnbr)

        line = f.readline()                 # Step on.
    f.close()                               # Close the file.

    emergeData = EmergeData(data)
        
    emergeData.form_long_vectors()
    return emergeData
