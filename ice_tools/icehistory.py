"""
icehistory.py

    Objects and methods for reading, manipulating, and using maps of ice
    heights over time.

    Author: Samuel B. Kachuck
"""

import numpy as np
import cPickle as pickle
import os

from giapy.map_tools import loadXYZGridData

from progressbar import ProgressBar, ETA, Percentage, Bar

def load(filename):
    return pickle.load(open(filename, 'rb'))

class Ice2d(object):
    """Store, use, and manipulate an ice model small enough to be stored in a
    single array. (In practice, this means less than ...)
    
    Attributes
    ----------
    heights : array
        3D array of ice heights with dimensions [stages, xdim, ydim]
    shape : tuple
        (xdim, ydim)
    times : array
        Times corresponding to the defined ice stages (in order)
    N : int
        The optimal order number for Fourier decomposition of the ice model.

    >>> ice = Ice2D()
    >>> ice.readice('./IceModels/Fullice_Aleksey.dat', 471, 491)
    """
    def __init__(self):
        self.times = np.array([20, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8])
        self._alterDict = {}
        self._desc = ''
        self.stageOrder = None                       # Used for loading cycle

    def __str__(self):
        return self._desc+'\n'+self.printAreas()
        
    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))
    
    def load_data(self, filename):
        self.heights = pickle.load( open( filename, 'rb' ) )
        self.shape = np.shape(self.heights[0,:,:])
        self.N = 512
        
    def readice(self, filename, Nx, Ny):
        """Read in a full comma-delimitted ice file in x-y-z format.
    
        Parameters
        ----------
            filename : the file to be read
            Nx, Ny   : number of lon, lat sites

        Example
        -------
        >>> ice = Ice2d()
        >>> ice.readice(u'./path/to/Fullice_Aleksey.dat', 471, 491)
        """
        rawdata = np.loadtxt(filename, delimiter=',')
        self.lon = (np.reshape(rawdata[:,0], (-1, Nx, Ny))[0, :, :])[0,:]
        self.lat = (np.reshape(rawdata[:,1], (-1, Nx, Ny))[0, :, :])[:,0]
        self.heights = np.reshape(rawdata[:,2], (-1, Nx, Ny))
        self.N = 512
        self.shape = np.shape(self.heights[0,:,:])
        
    def fft(self, N, grid):
        """Return the truncated FFT of ice heights to order N."""

        heights = self.alterAreas(grid)
        fullfft = np.fft.fft2(heights, s=[self.N,self.N])
        truncfft = np.zeros((11, N, N), dtype=complex)
        truncfft[:,:N/2,:N/2]=fullfft[:,          :N/2,           :N/2]
        truncfft[:,:N/2,N/2:]=fullfft[:,          :N/2, self.N-N/2:   ]
        truncfft[:,N/2:,:N/2]=fullfft[:,self.N-N/2:   ,           :N/2]
        truncfft[:,N/2:,N/2:]=fullfft[:,self.N-N/2:   , self.N-N/2:   ]

        if self.stageOrder is None:
            return truncfft
        else:
            return truncfft[self.stageOrder]

    def addArea(self, name, verts, prop, latlon=True):
        """Add an area to the model to be altered proportionally by amount prop.

        Parameters
        ----------
        name : string
            The name of the area (user convenience)
        verts : list of tuples
            The lon/lat (or map coord pairs) defining the area edges
        prop : float
            The proportion by which to alter the area
        latlon : bool (default True)
            Indicates whether vertices are in lon/lat or map coords

        Result
        ------
        New entry in self.alterDict under key 'name' with values 'verts' 
        and 'prop'.
        """
        prop = np.asarray(prop)
        if prop.shape != self.times.shape and prop.shape != ():
            raise ValueError('prop must either be one number, or one for eac\
                time in self.times')
        self._alterDict[name] = {'verts':verts, 'prop':prop, 'latlon':latlon}

    def editArea(self, name, prop=None, verts=None):
        """Edit an area previously defined and named by addArea."""
        if verts is not None: self._alterDict[name]['verts'] = verts
        if prop is not None:
            prop = np.asarray(prop)
            if prop.shape != self.times.shape and prop.shape != ():
                raise ValueError('prop must either be one number, or one for\
                    each time in self.times')
            self._alterDict[name]['prop'] = prop

    def updateAreas(self, updatelist, namelist=None):
        if namelist is None: namelist = self._alterDict.keys()
        # Method uses list operations to parse updatelist into areas
        updatelist = list(updatelist)
        for name in namelist:
            # need length of prop, but need to catch length 1
            try: n = len(self._alterDict[name]['prop'])
            except TypeError: n = 1
            try:
                # Take the appropriate numer of alterations out ...
                props = updatelist[0] if n==1 else updatelist[:n]
                # ... delete them from the list ...
                del updatelist[:n]
                # ... and use them to update that areas props.
                self.editArea(name, prop=props)
            except IndexError:
                raise IndexError('updatelist had insufficient length for areas\
                in namelist')

    def alterAreas(self, grid):
        """Multiply the areas in self.heights by their props.

        Parameters
        ----------
        grid : GridObject
            the map grid associated with the ice model            
        """
        grid.update_shape(self.shape)
        alteredIce = self.heights.copy()
        for areaDict in self._alterDict.values():
            areaind = grid.selectArea(areaDict['verts'], areaDict['latlon'])
            if areaDict['prop'].shape == ():
                alteredIce[:, areaind] *= areaDict['prop']
            else:
                for icestage, prop in zip(alteredIce, areaDict['prop']):
                    icestage[areaind] *= prop

        return alteredIce

    def printAreas(self):
        """Print the list of alterations made to the ice model"""
        arealist = ''
        for name, areaDict in self._alterDict.iteritems():
            arealist += '\t {0}: {1}\n'.format(name, areaDict['prop'])
        return arealist

    def isGrounded(self, time, topo, interp=False):
        """Generate a Boolean array indicating the location of grounded ice.

        Grounded ice is ice which 
        
        Parameters
        ----------
        time : float
            The stage at which to check. Must be in self.times if interp is
            False.
        topo : ndarray
            The paleotopography at the stage of interest, must be shape
            ice.shape.
        interp : Boolean
            Indicates whether interpolation between explicitly defined stages
            is allowed.
        """ 
        if topo.shape != self.shape:
            raise ValueError('Shapes are incompatible')

        # Pull the ice at the desired time
        if time in self.times:
            icetime = self.heights[time==self.times]
        elif interp:
            #TODO implement this
            raise NotImplementedError()
        else:
            raise ValueError('If interp==False, time must be in self.times')

        return (icetime>topo*0.9)*np.sign(icetime)

    def groundedReplace(self, time, topo, areaind):
        """Replace the ice in a certain area with just-grounded ice.

        Parameters
        ----------
        time : float
            A time in self.times
        topo : ndarray
            The paleotopography at time
        areaind : list
            The indices defining the area. See GridObject.selectArea.
        """
        # Pull the ice at the desired time
        if time in self.times:
            icetime = self.heights[time==self.times]
        else:
            raise ValueError('time must be in self.times')
        
        # 1.666 = rho_asth * rho_w / rho_ice (rho_asth - rho_w)
        #       ~ 3 * 1 / 0.9 (3-1) 
        # which accounts for 10% above water level, and a zero-order isostatic
        # correction (assumes equilibrium).
        return 1.666*topo[areaind]*(icetime[areaind]!=0)

    def printMW(self, grid, oceanarea=3.61e8):
        """Print equivalent meters meltwater for the glaciers in _alterDict.

        Parameters
        ----------
        grid : GridObject
        oceanarea : float
            the area of the ocean to convert volumes to heights, 
            default = 3.14e8 km^2, current area.
        """
        for time, icestage in zip(self.times, self.heights):
            print '\n{0} ka BP'.format(time)
            print '------------------------------'
            # Get the glacier volumes by integrating on the grid.
            vols = grid.integrateAreas(icestage, self._alterDict)
            for name, vol in vols.iteritems():
                print '\t{0} : {1} mMW'.format(name, vol/oceanarea)

    def calcMW(self, grid, arealist=None, oceanarea=3.61e8):
        """
        """
        # Set up the dictionary for return - keys are area names, values are
        # lists of MW in time.
        if arealist is None:
            returndict = {}
            for name in self._alterDict.iterkeys():
                returndict[name] = np.zeros(len(self.times))
            returndict['whole'] = np.zeros(len(self.times))
        pbar = ProgressBar(widgets=['Integrating: ',' ', Bar(), ' ', ETA()])
        pbar.start()
        for i, icestage in enumerate(self.heights):
            vols = grid.integrateAreas(icestage, self._alterDict)
            for name, mwlist in returndict.iteritems():
                mwlist[i] = vols[name]/oceanarea
            pbar.update(i + 1)
        pbar.finish()

        return returndict
    
    def appendLoadCycle(self, esl, tLGM=None):
        """Identify glaciating stages with deglaciating stage of same ESL.

        Parameters
        ----------
        esl : scipy.interp1d
            Eustatic Sea Level interpolating object
        tLGM : float
            The time of Last Glacial Maximum in ka.

        Results
        -------
        Appends the glaciation times to self.times and the matching stage
        files to self.icefiles.
        """
        tLGM = tLGM or self.times[np.argwhere(esl.y == esl.y.min())]
        # To split the esl curve into glaciation/deglaciation sections
        iLGM = np.argmin(np.abs(esl.x - tLGM))

        stageOrder = range(len(self.times))

        tReturn = []
        nReturn = []

        for nStage, tStage in enumerate(self.times):
            eslStage = esl(tStage)
            # collect all indices where ESL passes through stage's ESL
            indices = np.argwhere(esl.y[iLGM:]-eslStage >= 0)
            iUps = indices[1:][indices[1:]-indices[:-1]>1]
            for i in np.r_[[indices.min()], iUps]:
                # Find the time and ESL just before match to ice stage
                tDown = esl.x[iLGM:][i-1]
                yDown = esl.y[iLGM:][i-1]
                dt = esl.x[iLGM:][i] - tDown
                dy = esl.y[iLGM:][i] - yDown
                tReturn.append(tDown + dt/dy*(eslStage-yDown))
                nReturn.append(nStage)

        # Remove multiple stage additions?

        # Append the load cycle
        self.times = np.r_[tReturn, self.times]
        self.stageOrder = np.r_[nReturn, stageOrder]

        # Sort it all into decreasing order
        sortInd = np.argsort(self.times)[::-1]
        self.times = self.times[sortInd]
        self.stageOrder = self.stageOrder[sortInd]
        print('{0} stages added for the load cycle.'.format(len(nReturn)))


class IceHistory(object):
    """An object for loading and using large ice models.

    Give the instance a path to the folder containing the individual stages to
    interactively select the ice stage files and input their corresponding ages
    in thousand years before present.
    
    Parameters
    ----------
    path : str
        The path to the ice filenames
    dataFormat : dict
        A dictionary of options for np.loadtxt, used when loading a stage.
    shape : tuple
        The shape of each stage array. (Default assumes square array)

    Attributes
    ----------
    times : array
        Array of times at which ice heights are defined
    fnames : list
        list of filenames
    path : str
        The path to the ice filenames
    dataFormat : dict
        A dictionary of options for np.loadtxt, used when loading a stage.
    shape : tuple
        The shape of each stage array
    Lon, Lat : NDarray
        meshgrid representations of the Lon/Lat grids for the ice model.
    """
    def __init__(self, path=None, dataFormat={}, shape=None):
        self.path = os.path.abspath(path)
        self.dataFormat = dataFormat

        self.fnames = []
        self.times = []

        for fname in os.listdir(self.path):
            resp = raw_input('Include '+fname+'? [y/n/end] ')
            if resp == 'y':
                year = raw_input('\tTime of file (in ka bp): ')
                self.fnames.append(fname)
                self.times.append(float(year))
            elif resp == 'end':
                break
            elif resp == 'n':
                continue

        # Extract shape, Lon, and Lat info from first file.
        try:
            trial = self.load(self.fnames[0], shape=shape, lonlat=True)
            self.Lon = trial[0]
            self.Lat = trial[1]
            self.shape = self.Lon.shape
            if shape is None: print('Shape assumed {0}'.format(self.shape))
        except ValueError as e:
            raise e

    def load(self, fname, lonlat=False, dataFormat=None):
        dataFormat = dataFormat or self.dataFormat
        data =  loadXYZGridData(self.path+fname, shape=self.shape,\
                                lonlat=lonlat, **dataFormat)
        return data

    def pairIter(self, transform=None):
        """Iterate over consecutive pairs of ice stages, loading only one at
        each iteration

        Parameters
        ----------
        transform : transformation function
            If the data are to be transformed
        """
        ice1, t1 = self.load(self.fnames[0]), self.times[0]
        if transform is not None:
            ice1 = transform(ice1)
        
        for time, fname in zip(self.times[1:], self.fnames[1:]):
            ice0, t0 = ice1, t1
            ice1, t1 = self.load(fname), time
            yield ice0, t0, ice1, t1

    def appendLoadCycle(self, esl, tLGM=None):
        """Identify glaciating stages with deglaciating stage of same ESL.

        Parameters
        ----------
        esl : scipy.interp1d
            Eustatic Sea Level interpolating object
        tLGM : float
            The time of Last Glacial Maximum in ka.

        Results
        -------
        Appends the glaciation times to self.times and the matching stage
        files to self.icefiles.
        """
        tLGM = tLGM or self.times[np.argwhere(esl.y == esl.y.min())]
        # To split the esl curve into glaciation/deglaciation sections
        iLGM = np.argmin(np.abs(esl.x - tLGM))

        tReturn = []
        nReturn = []

        for nStage, tStage in enumerate(self.times):
            eslStage = esl(tStage)
            # collect all indices where ESL passes through stage's ESL
            indices = np.argwhere(esl.y[iLGM:]-eslStage >= 0)
            iUps = indices[1:][indices[1:]-indices[:-1]>1]
            for i in np.r_[[indices.min()], iUps]:
                # Find the time and ESL just before match to ice stage
                tDown = esl.x[iLGM:][i-1]
                yDown = esl.y[iLGM:][i-1]
                dt = esl.x[iLGM:][i] - tDown
                dy = esl.y[iLGM:][i] - yDown
                tReturn.append(tDown + dt/dy*(eslStage-yDown))
                nReturn.append(nStage)

        #TODO Remove repeated stage additions?

        # Append the load cycle
        self.times = np.r_[tReturn, self.times]
        self.fnames = np.r_[self.fnames[nReturn], self.fnames]

        # Sort it all into decreasing order
        sortInd = np.argsort(self.times)[::-1]
        self.times = self.times[sortInd]
        self.fnames = self.fnames[sortInd]
        print('{0} stages added for the load cycle.'.format(len(sortInd)))

    def decimate(self, n, suf=''):
        """Reduce an ice load by a power n of 2. Files are resaved with suffix
            suf.
        """
        newfnames = []
        for fname in self.filenames:
            ice = self.load(fname)
            ice = ice[::2**n,::2**n]
            newfname = fname+suf
            newfnames.append(newfname)
            np.savetxt(newfname.format(2**n), ice)
        self.fnames = newfnames
        self.Lons, self.Lats = lons[::2**n,::2**n], lats[::2**n,::2**n]
