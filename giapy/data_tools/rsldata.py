import numpy as np
import numpy.fft as fft 
import re

from urllib2 import urlopen, HTTPError
from scipy.signal import get_window
from scipy.stats import pearsonr
from giapy.data_tools.abstractDataClasses import AbsGeoTimeSeries, \
                                                AbsGeoTimeSeriesContainer

def calcRSL(sim, rsldata, smooth=True, noise=0):
    #TODO This function shouldn't have to know what's inside any object, let
    # alone a complicated one, like sim. Consider calculating rsl first and
    # passing it and calculated times in (ts, rsl, emergedata).
    # To reference to present day
    #u0 = sim.inputs.harmTrans.spectogrd(sim['topo'][-1])
    u0 = sim['sstopo'][-1]

    uAtLocs = []
    for ut in sim['sstopo']:
        ut = u0 - ut
        interpfunc = sim.inputs.grid.create_interper(ut.T*1000)
        uAtLocs.append(interpfunc.ev(rsldata.lons, rsldata.lats))

    uAtLocs = np.array(uAtLocs).T

    calcTimes = sim['sstopo'].outTimes
    if np.all(np.diff(calcTimes)<0):
        reverse = True
    else:
        reverse=False
        

    data = {}
    for uAtLoc, loc in zip(uAtLocs, rsldata):
        if smooth:
            ts = np.union1d(np.sort(loc.ts), np.linspace(0, loc.ts.max()))
        else:
            ts = np.sort(loc.ts)

        # Sim objects output times in thousands of years relative to present
        tsint = (2015 - ts)/1000.

        if reverse:
            timeseries = np.array([ts, 
                    np.interp(tsint, 
                                sim['sstopo'].outTimes[::-1], uAtLoc[::-1])]).T
        else:
            timeseries = np.array([ts, 
                    np.interp(tsint, 
                                sim['sstopo'].outTimes, uAtLoc)]).T

        #Apply the noise (default 0)
        scatter = noise*np.random.randn(len(timeseries[:,1]))
        timeseries[:,1] = timeseries[:,1] + scatter
        data[loc.stid] = RSLDatum(timeseries, 
                                        lat=loc.lat, 
                                        lon=loc.lon,
                                        sitename=loc.sitename,
                                        stid=loc.stid)

    return RSLData(data=data)

class RLR(np.ndarray):
    """Download a Revised Local Reference for sea level from the PSMSL.
    
    Data is stored as an ndarray, with the first column ([:,0]) storing the
    dates of the records (in yearly decimal format, e.g., 1947.23), the second
    ([:,1]) the sea level records, and the third and fourth error information.

    Missing data points are stored as -99999, and a list of the indices of
    non-absent data are stored in RLR.ind.

    Parameters
    ----------
    num : int
        The unique locator code for the RLR data to download
    typ : 'monthly' or 'annual'
        Download either monthly or annual data

    Reference
    ---------
    Simon J. Holgate, Andrew Matthews, Philip L. Woodworth, Lesley J. Rickards,
    Mark E. Tamisiea, Elizabeth Bradshaw, Peter R. Foden, Kathleen M. Gordon,
    Svetlana Jevrejeva, and Jeff Pugh (2013) New Data Systems and Products at
    the Permanent Service for Mean Sea Level. Journal of Coastal Research:
    Volume 29, Issue 3: pp. 493 - 504. doi:10.2112/JCOASTRES-D-12-00175.1.
    """
    def __new__(cls, num, typ='monthly', *args):
        # option parsing
        if typ not in ['monthly', 'annual']:
            raise ValueError("typ must be 'monthly' or 'annual'")
        
        # the data array
        _url = 'http://www.psmsl.org/data/obtaining/rlr.'+typ+'.data/'
        try:
            _response = urlopen(_url+str(num)+'.rlrdata')
        except HTTPError:
            raise ValueError("{0} is an invalid station id".format(num))
        
        # Format of data is: date | mean sea level | missing record | flag
        if typ=='annual':
            # In annual data, third column indicates whether the record was
            # missing 30 days ('Y') or not ('N')
            converters = {2: lambda s: (0 if s=='N' else 1)}
        else:
            converters = None
        _data = np.loadtxt(_response, delimiter=';', converters=converters)
        # Protect against sites with only one data point
        _data = np.atleast_2d(_data)

        # the metadata
        _murl = 'http://www.psmsl.org/data/obtaining/stations/'
        _meta = urlopen(_murl+str(num)+'.php')
        _txt = _meta.read()
        _sitename = re.findall('<h1>(.*)</h1>', _txt, flags=re.DOTALL)[0]
        _tmarker = '<!-- Beginning of data table -->(.*)<!-- End of data table -->'
        _m = re.findall(_tmarker, _txt, flags=re.DOTALL)
        _rows = re.findall('<tr>.*?</tr>', _m[0], flags=re.DOTALL)
        _table = [[cell[4:-5] for cell in 
                    re.findall('<td>.*?</td>', row, flags=re.DOTALL)]
                    for row in _rows]

        if _table[3][0] in ['GLOSS ID', 'GLOSS ID:']:       # some locs have a
            _metadata = {'sitename' : _sitename,            # gloss id
                         'stid' : int(_table[0][1]),
                         'lat'  : float(_table[1][1]),
                         'lon'  : float(_table[2][1]),
                         #'gloss': float(_table[3][1][-7:-4]),
                         'coastcode' : int(_table[4][1]),
                         'stcode' : int(_table[5][1]),
                         'country' : _table[6][1],
                         'yrmin' : int(_table[7][1][:4]),
                         'yrmax' : int(_table[7][1][-4:]),
                         'comp' : float(_table[8][1]),
                         'frcode' : _table[9][1],
                         'update' : _table[10][1],
                         'type' : typ
                         }
        else:                                               # some don't
            _metadata = {'sitename' : _sitename,
                         'stid' : int(_table[0][1]),
                         'lat'  : float(_table[1][1]),
                         'lon'  : float(_table[2][1]),
                         'coastcode' : int(_table[3][1]),
                         'stcode' : int(_table[4][1]),
                         'country' : _table[5][1],
                         'yrmin' : int(_table[6][1][:4]),
                         'yrmax' : int(_table[6][1][-4:]),
                         'comp' : float(_table[7][1]),
                         'frcode' : _table[8][1],
                         'update' : _table[9][1],
                         'type' : typ
                         }
        
        # store indices of time where data are absent
        _inds = np.where(_data[:,1]!=-99999)[0]

        obj = np.asarray(_data).view(cls)
        obj.metadata = _metadata
        obj.inds = _inds

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', None)
        self.inds = getattr(obj, 'inds', None)
    
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(RLR, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.inds,self.metadata)
        # Return a tuple that replaces the
        # parent's __setstate__ tuple with our
        # own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.inds, self.metadata = state[-2], state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(RLR, self).__setstate__(state[0:-2])
        
    def __str__(self):
        return 'RLR( {0}, {1} )'.format(self.metadata['sitename'],
                        self.metadata['type'])

    def trend(self, coeffs=False):
        """Calculate the best linear fit for data, returns coeffs or line.
        """
        inds = self.inds
        A = np.array([self[inds,0], np.ones(len(inds))])
        w = np.linalg.lstsq(A.T, self[inds,1])[0]
        if coeffs:
            return w
        else:
            return w[0]*self[:,0]+w[1]

    def detrended(self):
        """Remove the linear trend from the RLR data.
        """
        inds = self.inds
        result = self.copy()#[self.inds,:]
        result[inds,1] = result[inds,1]-self.trend()[inds]#[self.inds, :]
        return result

    def window_filter(self, window):
        """Filters the data using a window in Fourier space.

        See docstring for scipy.signal.window
        """
        if self.metadata['type'] == 'monthly':
            d = 0.08333
        else:
            d = 1
        
        inds = self.inds
        result = self.copy()#[self.inds,:]

        fs = fft.fftshift(fft.fftfreq(len(self.inds), d=d))
        self_fft = fft.fftshift(fft.fft(self.detrended()[inds, 1]))
        win = get_window(window, len(inds))
        result[inds,1] = fft.ifft(fft.ifftshift(win*self_fft))
        result[inds,1] = result[inds,1] + self.trend()[inds]#[self.inds]

        return result

    def runavg_filter(self, w, nmiss=0, trimmed=True, align='left'):
        """Computes the running average of w samples.

        When a gap occurs in the data, it is given zero weight in the filter,
        and the weighting is renormalized. Thus, some gaps can be filled. If
        the gap cannot be filled, it is flagged as missing.

        Mitchum 1987 (cited by Clarke 1992)
        """
        result = self.copy()

        if nmiss<0:
            raise ValueError('nmiss must be non-negative')

        if align=='left':
            start = w
            finish = self.shape[0]
        elif align=='cent':
            start = w/2
            finish = self.shape[0]-w/2
        elif align=='right':
            start = 0
            finish = self.shape[0]-w

        for i in range(start, finish):
            if align=='left':
                block = self[i-w:i, 1]
            elif align=='cent':
                block = self[i-w/2:i+w/2, 1]
            elif align=='right':
                block = self[i:i+w, 1]
            block = block[block!=-99999]
            if len(block)<=nmiss: 
                result[i, 1] = -99999
            else:
                result[i, 1] = block.mean()
                    
        # reset the data not absent indices
        if trimmed: 
            if align=='left':
                result = result[w:,:]
            elif align=='cent':
                result = result[w/2:-w/2,:]
            elif align=='right':
                result = result[:-w,:]

        result.inds = np.where(result[:,1]!=-99999)[0]
        return result

    def time_filter(self, tmin, tmax):
        """Filter the data to a specific time window [tmin, tmax]"""
        filtinds = np.where(np.logical_and(self[:,0]>tmin, self[:,0]<=tmax))
        result = self[filtinds]
        # correct data not absent indices
        result.inds = np.where(result[:,1]!=-99999)[0]

        return result

    def plot(self, ax, *args, **kwargs):
        """Plots the data on ax, skipping missing points"""
        ax.plot(self[self.inds, 0], self[self.inds, 1], *args, **kwargs)
        return ax

class RSLDatum(AbsGeoTimeSeries):
    def __repr__(self):
        return 'RLR( {0}, {1} )'.format(self.sitename, self.type)

class RSLData(AbsGeoTimeSeriesContainer):
    def __init__(self, nbrs=None, data=None, typ='monthly'):
        if nbrs is not None:
            self.data = {}
            self.download(nbrs, typ)
        elif data is not None:
            self.data = data
        self.form_long_vectors()

    def __getitem__(self, key):
        return self.data.__getitem__(key)
        
    def __iter__(self):
        return self.data.itervalues()

    def __len__(self):
        return len(self.data)

    def download(self, nbrs, typ):
        for n in nbrs:
            try:
                loc = RLR(n, typ=typ)
                self.data[n] = RSLDatum(loc, **loc.metadata)
            except ValueError as e:
                print e.message
                continue
    
    def filter_by_time(self, tmin, tmax, replace=False):
        result = {}
        for n, loc in self.data.iteritems():
            if loc.metadata['yrmin']<=tmin and loc.metadata['yrmax']>=tmax:
                result[n] = loc
        if replace:
            self.data = result
        else:
            return RSLData(nbrs=None, data=result)

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

def download_all_rlr(ns=range(1,2400), typ='monthly'):
    return RSLData(ns, typ=typ)
        

def argmaxcontsect(inds):
    """Given indices, returns start index and end index for largest continuous
    section
    """
    diff = inds[1:]-inds[:-1]
    cumsum = cumsummax = icurr = imax = 0

    for i, x in enumerate(diff):
        if x == 1:
            cumsum += 1
            if cumsum > cumsummax:
                cumsummax = cumsum
                imax = icurr
        else:
            cumsum = 0
            icurr = i+1

    return inds[imax], inds[imax+cumsummax]

def corrcoef(loc1, loc2, shift=0):
    """Calculate the correlation between two RLR locations. Can shift one
    location in time by shift keyword.
    """

    filt1 = loc1.copy()
    filt2 = loc2.copy()

    filt1[:,0] += shift 
    
    maxyr = min(filt1[:,0].max(), filt2[:,0].max())
    minyr = max(filt1[:,0].min(), filt2[:,0].min())
    filt1 = filt1.time_filter(minyr, maxyr)
    filt2 = filt2.time_filter(minyr, maxyr)

    inds = np.intersect1d(filt1.inds, filt2.inds)
    return pearsonr(filt1[inds,1], filt2[inds,1])[0]
