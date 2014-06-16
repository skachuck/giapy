import numpy as np
import numpy.fft as fft 
import re
from urllib2 import urlopen, HTTPError
from scipy.signal import get_window

class RLR(np.ndarray):
    """Download a Revised Local Reference for sea level from the PSMSL.

    
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
        
        _url = 'http://www.psmsl.org/data/obtaining/rlr.'+typ+'.data/'
        try:
            _response = urlopen(_url+str(num)+'.rlrdata')
        except HTTPError:
            raise ValueError("{0} is an invalid station id".format(num))
        
        if typ=='annual':
            converters = {2: lambda s: (0 if s=='N' else 1)}
        else:
            converters = None
        _data = np.loadtxt(_response, delimiter=';', converters=converters)

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

        if _table[3][0] in ['GLOSS ID', 'GLOSS ID:']:
            _metadata = {'sitename' : _sitename,
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
        else:
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
        
        _inds = np.where(_data[:,1]!=-99999)[0]

        obj = np.asarray(_data).view(cls)
        obj.metadata = _metadata
        obj.inds = _inds

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', None)

    def __str__(self):
        return 'RLR( {0}, {1} )'.format(self.metadata['sitename'],
                        self.metadata['type'])

    def trend(self):
        inds = self.inds
        A = np.array([self[inds,0], np.ones(len(inds))])
        w = np.linalg.lstsq(A.T, self[inds,1])[0]
        return w[0]*self[:,0]+w[1]

    def detrended(self):
        result = self[self.inds,:]
        result[:,1] = result[:,1]-self.trend()[self.inds, :]
        return result

    def window_filter(self, window):
        """Filters the data using a window in Fourier space.

        See docstring for scipy.signal.window
        """
        if self.metadata['type'] == 'monthly':
            d = 0.08333
        else:
            d = 1
        
        result = self[self.inds,:]

        fs = fft.fftshift(fft.fftfreq(len(self.inds), d=d))
        self_fft = fft.fftshift(fft.fft(self.detrended()[:, 1]))
        win = get_window(window, len(self.inds))
        result[:,1] = fft.ifft(fft.ifftshift(win*self_fft))
        result[:,1] = result[:,1] + self.trend()[self.inds]

        return result

    def plot(self, ax, *args, **kwargs):
        ax.plot(self[self.inds, 0], self[self.inds, 1], *args, **kwargs)
        ax.set_ylabel('mm')
        ax.set_xlabel('year')
        return ax

class RSLData(object):
    def __init__(self, nbrs=None, data=None):
        if nbrs is not None:
            self.data = {}
            self.download(nbrs)
        elif data is not None:
            self.data = data

    def download(nbrs):
        for n in nbrs:
            try:
                loc = RLR(n)
                self.data[n] = loc
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
        

def argmaxcontsect(inds):
    """Given indices, returns start index and endindex for largest continuous
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


class CosSinEstimator(object):
    def __init__(self, ts, ys, fs):
        self.fs = fs
        
        Ts, Fs = np.meshgrid(ts, fs)

        c0 = np.ones(len(ts))[:, np.newaxis].T
        coss = np.cos(2*np.pi*Ts*Fs)
        sins = np.sin(2*np.pi*Ts*Fs)

        G = np.concatenate([c0, ts[:,np.newaxis].T, coss, sins]).T

        self.ms = linalg.inv(G.T.dot(G)).dot(G.T).dot(ys)

    def __call__(self, ts):
        
        Ts, Fs = np.meshgrid(ts, self.fs)

        c0 = np.ones(len(ts))[:, np.newaxis].T
        coss = np.cos(2*np.pi*Ts*Fs)
        sins = np.sin(2*np.pi*Ts*Fs)

        G = np.concatenate([c0, ts[:,np.newaxis].T, coss, sins]).T

        return G.dot(self.ms)
        
