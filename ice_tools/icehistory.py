"""
icehistory.py

    Objects and methods for reading, manipulating, and using maps of ice
    heights over time.

    Author: Samuel B. Kachuck
"""

import numpy as np
import cPickle as pickle
import os

from . import GlacierBounds
from giapy.map_tools import loadXYZGridData

from progressbar import ProgressBar, ETA, Percentage, Bar


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
        path = path or os.path.curdir
        self.path = os.path.abspath(path)+'/'
        self.dataFormat = dataFormat

        self.fnames = []
        self.times = []

        for fname in os.listdir(self.path):
            resp = raw_input('Include '+fname+'? [Time (in ka bp)/n/end] ')
            try:
                year = float(resp)
                self.fnames.append(fname)
                self.times.append(year)
            except:
                if resp == 'end':
                    break
                elif resp == 'n':
                    continue
                else:
                    'Did not understand. Add file using self.addFile'
        # Sort files by decreasing time
        self.sortByTime()

        # Extract shape, Lon, and Lat info from first file.
        try:
            trial = loadXYZGridData(path+self.fnames[0], shape=shape,
                lonlat=True, **dataFormat)
            self.Lon = trial[0]
            self.Lat = trial[1]
            self.shape = self.Lon.shape
            self.nlat = len(np.union1d(ice.Lat.flatten(), ice.Lat.flatten()))
            if shape is None: print('Shape assumed {0}'.format(self.shape))
        except ValueError as e:
            raise e

        self.stageOrder = range(len(self.times))
        # Used for alterations.
        self.areaProps = None
        self.areaNames = None

    def __getitem__(self, key):
        return self.load(self.fnames[self.stageOrder[key]])

    def __iter__(self, alter=True):
        for stageNum in self.stageOrder:
            stage = self.load(self.fnames[stageNum])
            if self.areaProps is not None:
                self.alterStage(stage, stageNum)
            yield stage

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
            If the data are to be transformed before yielding.
        """
        stage0 = self.stageOrder[0]
        ice1, t1 = self.load(self.fnames[stage0]), self.times[0]
        if self.areaProps is not None:
            self.alterStage(ice1, stage0)
        if transform is not None:
            ice1 = transform(ice1)
        
        #for time, fname in zip(self.times[1:], self.fnames[1:]):
        for i, stage in enumerate(self.stageOrder[1:], start=1):
            time = self.times[i]
            fname = self.fnames[stage]
            ice0, t0, ice1, t1 = ice1, t1, self.load(fname), time
            if self.areaProps is not None:
                self.alterStage(ice1, stage)
            if transform is not None:
                ice1 = transform(ice1)
            yield ice0, t0, ice1, t1

    def appendLoadCycle(self, esl, verbose=False):
        """Identify glaciating stages with deglaciating stage of same ESL.

        Parameters
        ----------
        esl : scipy.interp1d
            Eustatic Sea Level interpolating object.

        Results
        -------
        Appends the glaciation times to self.times and the matching stage
        files to self.icefiles.
        """
        # To split the esl curve into glaciation/deglaciation sections
        iLGM = np.argmin(esl.y)
        glaTimes = esl.x[iLGM:]
        glaESLs = esl.y[iLGM:]

        tReturn = []
        nReturn = []

        for nStage, tStage in enumerate(self.times):
            # Collect all indices before ESL passes through esl(tStage)
            switches = np.sign(glaESLs - esl(tStage))
            switchInd = np.argwhere((switches[1:]-switches[:-1]) != 0)
            # Two sign changes occur is ESL at deglaciation stage is same as ESL
            # during load cycle to append. Find any actual zeros and take only
            # one index.
            zeroInd = np.argwhere(switches == 0) - 1
            for k in zeroInd:
                keep = np.logical_and(switchInd!=k, switchInd!=k+1)
                switchInd = np.r_[switchInd[keep].flatten(), k]

            # Interpolate to times at which ESL = esl(tStage)
            tUp = np.array([glaTimes[i+1] for i in switchInd])
            yUp = np.array([glaESLs[i+1] for i in switchInd])
            dt = tUp - glaTimes[switchInd]
            dy = yUp - glaESLs[switchInd]
            glaStageTimes = glaTimes[switchInd] + \
                                dt/dy*(esl(tStage) - glaESLs[switchInd])

            # Check for ESL load redundencies - keep only first
            if np.any([i in tReturn for i in glaStageTimes]):
                continue

            # Fill in lists.
            tReturn.extend(glaStageTimes.flatten())
            nReturn.extend(np.repeat(nStage, len(glaStageTimes)))

        # Append the load cycle to unloading times
        self.times = np.r_[tReturn, self.times]
        self.stageOrder = np.r_[nReturn, range(len(self.times))]

        # Sort it all into decreasing order
        sortInd = np.argsort(self.times)[::-1]
        self.times = self.times[sortInd]
        self.stageOrder = self.stageOrder[sortInd]
        if verbose:
            print('{0} stages added for the load cycle.'.format(len(nReturn)))

    def sortByTime(self, dec=True):
        """Sort the fnames by times, decreasing by default."""
        sortInd = np.argsort(self.times)[::-1]
        self.times = np.asarray(self.times)[sortInd]
        self.fnames = np.asarray(self.fnames)[sortInd]

    def decimate(self, n, suf=''):
        """Reduce an ice load by a power n of 2. Files are resaved with suffix
            suf.
        """
        #TODO add a way of keeping load cycle fnames separate.
        newfnames = []
        for fname in self.fnames:
            ice = self.load(fname)
            ice = ice[::2**n,::2**n]
            # Append the decimated filename, putting the suffix behind the
            # extension.
            f, ext = os.path.splitext(fname)
            newfname = f+suf+ext
            newfnames.append(newfname)
            # Save the deicmated file
            np.savetxt(newfname, ice.flatten())
        self.fnames = newfnames
        self.Lon = self.Lon[::2**n,::2**n]
        self.Lat = self.Lat[::2**n,::2**n]
        self.shape = self.Lon.shape

    def createAlterationAreas(self, grid, props, areaNames=None):
        """Create alteration areas for proportional ice height changes.

        The area definitions and proportions are stored in two dictionaries, 
        whose keys are the names of the areas. The reason for saving both is
        that the area definitions occassionally are changed, and for
        consistency and reproducibility it will be useful to have an altered
        ice model carry with it all the information needed for the alteration.
        (The only thing not stored by the ice model internally are the raw ice
        model files.)

        Parameters
        ----------
        grid : giapy.map_tools.GridObject
            Required for now to locate the lonlat indices of the areas.
        props : list
            A list of numbers by which to multiply ice heights in the area. It
            must have as many elements as there are areas, and the elements
            must either be singlets or else as long as there are stages
            (assumes the order follows self.stageOrder)
        areaNames : list
            A list of the area names to include. (Default None). If None,
            assumes all the areas in GlacierBounds.areaNames. See
            help(GlacierBounds) for more information about area names.
        """
        #TODO DO IT WITHOUT THE GRID OBJECT!
        areaNames = areaNames or GlacierBounds.areaNames
        
        assert len(props) == len(areaNames)

        # GlacierBounds.outputAsDict outputs  a dictionary of names, one for
        # each area in areaNames, with the values the vertices of the area.
        self.areaVerts = GlacierBounds.outputAsDict(areaNames)
        # The alteration mask is an lat/lon array mapping membership to a
        # glacier area to an integer, for fast area locating later on
        # (grid.selectArea is quite slow). The integer of each glacier is
        # stored in the area 
        self._alterationMask = np.zeros(self.shape, dtype=int)
        self.areaProps = {}
        for area, prop in zip(areaNames, props):
            self.areaProps[area] = prop
            inds = grid.selectArea(self.areaVerts[area])
            self._alterationMask += hash(area)*inds

    def updateAlterationAreas(self, updateDict):
        """Change the multiplicative factor for each area set by
        self.createAlterationAreas.

        Parameters
        ----------
        props : list
            A list of numbers by which to multiply ice heights in the area. It
            must have as many elements as there are areas, and the elements
            must either be singlets or else as long as there are stages
            (assumes the order follows self.stageOrder)

        """
        for area, prop in updateDict.iteritems():
            self.areaProps[area] = prop 

    def alterStage(self, stage, stageNum):
        """Multiplies each area in stage by the appropriate factor.

        Parameters
        ----------
        stage : 2D array
            Array of ice heights, as would be returned by self.load.
        stageNum : int
            The associated stage number, e.g., from self.stageOrder, so that if
            an area is being changed at each stage, the correct number can be
            retrieved.
        """
        #TODO Fix this type checking
        for name, prop in self.areaProps.iteritems():
            if (isinstance(prop, list) or \
                    isinstance(prop, np.ndarray)):
                stage[self._alterationMask==hash(name)] *= prop[stageNum]
            else:
                stage[self._alterationMask==hash(name)] *= prop

class PersistentIceHistory(IceHistory):
    def __init__(self, icehistory):
        self.stageArray = np.array([icehistory.load(stage) 
                            for stage in icehistory.fnames])

        # Copy important info from icehistory
        self.Lon = icehistory.Lon
        self.Lat = icehistory.Lat
        self.nlat              = icehistory.nlat 
        self.shape             = icehistory.shape
        self._alterationMask   = icehistory._alterationMask.copy()
        self.areaProps         = icehistory.areaProps.copy()
        self.areaVerts         = icehistory.areaVerts.copy()
        self.times             = icehistory.times.copy()
        self.stageOrder        = icehistory.stageOrder[:]
        self.fnames            = icehistory.fnames[:]
        self.fnameDict         = dict(zip(icehistory.fnames,
                                    range(len(icehistory.fnames))))
    def load(self, fname, **kwargs):
        return self.stageArray[self.fnameDict[fname]] 

    def applyAlteration(self, name):
        pass



def printMW(ice, grid, areaNames=None, oceanarea=3.61e8):
    """Print equivalent meters meltwater for the glaciers.

    Parameters
    ----------
    grid : GridObject
    oceanarea : float
        the area of the ocean to convert volumes to heights, 
        default = 3.14e8 km^2, current area.
    """
    
    areaNames = areaNames or GlacierBounds.areaNames
    areas = GlacierBounds.outputAsList(areaNames)
        
    s = ''
    for column in ['ka BP']+areaNames+[' Total']:
        s += '{column:{align}{width}} '.format(column=column, align='^',
                                                width=7)
    print(s)

    for i, stage in enumerate(ice):
        s = '{num:{align}{width}{base}}  '.format(num=ice.times[i], align='<',
                                                    width=7, base='.2f')
        # Get the glacier volumes by integrating on the grid.
        vols = grid.integrateAreas(stage, areas)
        for area in vols:
            s += '{num:{align}{width}{base}} '.format(num=area['vol']/oceanarea, 
                                                        align='>', width=7, base='.3f')
        print(s)
