"""
icehistory.py

    Objects and methods for reading, manipulating, and using maps of ice
    heights over time.

    Author: Samuel B. Kachuck
"""

import numpy as np

import os, time

from giapy import pickle
from giapy.map_tools import loadXYZGridData


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
            try:
                year = float(os.path.splitext(fname)[0])/1000.
            except:
                year = ''
            resp = raw_input('Include '+fname+'? [y/Time (in ka bp)/n/end] '+str(year))
            if resp == 'y':
                pass
            elif resp == 'end':
                break
            elif resp == 'n':
                continue
            else:
                try:
                    year = float(resp)
                except:
                    'Did not understand. Add file using self.addFile' 

            self.fnames.append(fname)
            self.times.append(year)
        # Sort files by decreasing time
        self.sortByTime()

        # Extract shape, Lon, and Lat info from first file.
        try:
            trial = loadXYZGridData(path+self.fnames[0], shape=shape,
                lonlat=True, **dataFormat)
            self.Lon = trial[0]
            self.Lat = trial[1]
            self.shape = self.Lon.shape
            self.nlat = len(np.union1d(self.Lat.ravel(), self.Lat.ravel()))
            if shape is None: print('Shape assumed {0}'.format(self.shape))
        except ValueError as e:
            raise e

        self.stageOrder = range(len(self.times))
        # Used for alterations.
        self.areaProps = None
        self.areaNames = None
        self._alterationMask = np.zeros(self.shape)

    def __getitem__(self, key):
        return self.load(self.fnames[self.stageOrder[key]])

    def __iter__(self, alter=True):
        for stageNum in self.stageOrder:
            stage = self.load(self.fnames[stageNum])
            if self.areaProps is not None:
                self.alterStage(stage, stageNum)
            yield stage

    def _getMetaData(self):
        metadata = {'Lon'               : self.Lon,
                    'Lat'               : self.Lat,
                    'nlat'              : self.nlat,
                    'shape'             : self.shape,
                    '_alterationMask'   : self._alterationMask.copy(),
                    'areaProps'         : self.areaProps.copy(),
                    'areaVerts'         : self.areaVerts.copy(),
                    'times'             : self.times[:],
                    'stageOrder'        : self.stageOrder[:],
                    'path'              : self.path,
                    'fnames'            : self.fnames[:]}
        return metadata

    def copy():
        """Create a copy of the ice model, including alteration dicts."""
        pass

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

    def createAlterationAreas(self, grid, props, areaNames=None, areaVerts=None):
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
        areaVerts : dict
            A dictionary of area names (as keys) and lists of lon/lat vertices 
            (as values). If defined, it is preferentially used over areaNames.
        """
        #TODO DO IT WITHOUT THE GRID OBJECT!
        if areaVerts is None:
            areaNames = areaNames or areaNamesGlacierBounds.areaNames
            # GlacierBounds.outputAsDict outputs  a dictionary of names, one for
            # each area in areaNames, with the values the vertices of the area.
            self.areaVerts = GlacierBounds.outputAsDict(areaNames)
        else:
            self.areaVerts = areaVerts
            areaNames = areaVerts.keys()
        
        assert len(props) == len(areaNames)

        # The alteration mask is an lat/lon array mapping membership to a
        # glacier area to an integer, for fast area locating later on
        # (grid.selectArea is quite slow). The integer of each glacier is
        # stored in the area 
        self._alterationMask = np.zeros(self.shape, dtype=int)
        self.areaProps = {}
        for area, prop in zip(areaNames, props):
            self.areaProps[area] = prop
            inds = grid.selectArea(self.areaVerts[area])
            self._alterationMask[inds] = hash(area)

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


def loadIceStages(icehistory):
    """Create a PersistentIceHistory from an IceHistory.

    Whereas IceHistory objects store links to the files containing the ice
    heights at each stage, a PersistentIceHistory stores the entire ice model
    in memory. Each stage is loaded and stacked into an array. All original
    meta-data is passed along.

    Parameters
    ----------
    icehistory : giapy.icehistory.IceHistory
        The ice history whose stages are to be loaded.
    """
    # Load the ice heights from each stage
    stageArray = np.array([icehistory.load(stage) 
                            for stage in icehistory.fnames])
    metadata = icehistory._getMetaData()

    return PersistentIceHistory(stageArray, metadata)

class PersistentIceHistory(IceHistory):
    """Store an ice model in memory (np.ndarray) for faster access.

    Whereas IceHistory objects store links to the files containing the ice
    heights at each stage, a PersistentIceHistory stores the entire ice model
    in memory. Each stage is loaded and stacked into an array. All original
    meta-data is passed along.

    Parameters
    ----------
    iceArray : numpy.ndarray
        The array of ice heights at each stage. Must have shape 
        (nstages, nlon, nlat).
    metadata : dict
        The metadata, e.g., as collected from _getMetaData().
    """
    def __init__(self, iceArray, metadata):

        self.stageArray = iceArray
        # Copy important info from icehistory
        for attr, value in metadata.iteritems():
            setattr(self, attr, value)

        self.fnameDict = dict(zip(self.fnames,
                                    range(len(self.fnames))))

    def copy(self):
        metadata = self._getMetaData()
        return PersistentIceHistory(self.stageArray.copy(), metadata)

    def __getitem__(self, key):
        return self.stageArray[self.stageOrder[key]]

    def __iter__(self, alter=True):
        for stageNum in self.stageOrder:
            stage = self.stageArray[stageNum]
            if self.areaProps is not None:
                self.alterStage(stage, stageNum)
            yield stage

    def pairIter(self, transform=None):
        """Iterate over consecutive pairs of ice stages, loading only one at
        each iteration

        Parameters
        ----------
        transform : transformation function
            If the data are to be transformed before yielding.
        """
        stage0 = self.stageOrder[0]
        ice1, t1 = self[0], self.times[0]
        if self.areaProps is not None:
            self.alterStage(ice1, stage0)
        if transform is not None:
            ice1 = transform(ice1)
        
        #for time, fname in zip(self.times[1:], self.fnames[1:]):
        for i, stage in enumerate(self.stageOrder[1:], start=1):
            time = self.times[i]
            fname = self[stage]
            ice0, t0, ice1, t1 = ice1, t1, self.stageArray[stage], time
            if self.areaProps is not None:
                self.alterStage(ice1, stage)
            if transform is not None:
                ice1 = transform(ice1)
            yield ice0, t0, ice1, t1


    def applyAlteration(self, names=None):
        """Applies all the proprtional alterations in self.areaProps and
        returns a new object. The original object is unchanged.

        Parameters
        ----------
        names : list or str
            The name(s) of the areas to alter. Must be a name in 
            self.areaProp. Default is to do them all.
        """
        # Generate the copy.
        altIce = self.copy()
        # Interpret the input.
        if names is not None:
            if isinstance(names, str):
                names = [names]
        else:
            names = altIce.areaProps.keys()

        # Apply the alterations location by location.
        for name in names:
            prop = altIce.areaProps[name]
            if (isinstance(prop, list) or \
                    isinstance(prop, np.ndarray)):
                # If the area has a different prop for each stage, it is
                # applied here by creating a stacked mask.
                mask = np.outer(prop, 
                                (altIce._alterationMask ==
                                hash(name))).reshape(altIce.stageArray.shape)
            else:
                # Otherwise, there's only one mask for the whole stack
                # of stages.
                mask = prop*(altIce._alterationMask == hash(name))

            # Everything outside the area stay the same.
            mask[mask == 0] = 1.
            # The whole thing is multiplied
            altIce.stageArray *= mask
            # and the area is removed from the alteration list.
            del altIce.areaProps[name]
            del altIce.areaVerts[name]
        # The alteration mask is zeroed.
        altIce._alterationMask *= 0
        return altIce

    def interp_to_t(self, t):
        """Interpolate the ice history to an interior time t (not checked).

        Parameters
        ----------
        t : the time to interpolate the ice history to.
        """

        assert ice.times.min()<=t<=ice.times.max(), 't must be interior.'

        itup = np.argwhere(self.times > t)[-1][0]
        itdo = itup + 1
        tup = self.times[itup]
        
        dicedt = (self[itdo] - self[itup])/(self.times[itdo] - tup)

        return self[itup] + (dicedt * (t - tup))

    def insert_interp_stage(self, t):
        """Insert an ice stage at t DURING DEGLACTIATION by interpolating the
        ice stage and putting it into the appropriate place in the stageArray. 
        The stageOrder list and time list are both corrected.

        NOTE: It only works for deglaciation stages, and the procedure DOES NOT
        CHECK.

        Parameters
        ----------
        t : the time at which to interpolate and insert.
        """

        assert t not in ice.times, 't must not be in ice.times.'

        newice = self.interp_to_t(t)
        itup = np.argwhere(ice.times > t)[-1][0]
        insertLoc = self.stageOrder[itup]

        # Get indices for stage Orders that need to be incremented to make room
        # for new stage.
        stageOrderFixes = self.stageOrder > insertLoc

        self.stageArray = np.vstack([self.stageArray[:insertLoc+1],
                                     newice[None, :, :],
                                     self.stageArray[insertLoc+1:]])

        self.stageOrder[stageOrderFixes] = self.stageOrder[stageOrderFixes] + 1
        self.stageOrder = np.r_[self.stageOrder[:itup+1], 
                                insertLoc,
                                self.stageOrder[itup+1:]]
        self.times = np.r_[self.times[:itup+1], t, self.times[itup+1]]

class OfflineIceHistory(object):
    """
    An ice history that is read in stage-by-stage, as they are made available.
    """
    def __init__(self, times, fnames, shape, drctry='./', readargs=[],
                    readkwargs={}, *args, **kwargs):
        self.times = times
        self.fnames = fnames
        self.shape = shape
        self.drctry = drctry
        self.readargs = readargs
        self.readkwargs = readkwargs
        # Need nlat for sle.
        self.nlat = self.shape[0]

    def __getitem__(self, key):
        stage = self.read_icestage(self.drctry+self.fnames[key],
                                *self.readargs, **self.readkwargs)
        return stage

    def pairIter(self):
        t0, icefname0 = self.times[0], self.fnames[0]
        ice0 = self.read_icestage(self.drctry+icefname0, *self.readargs,
                                    **self.readkwargs)
        for t1, next_ice_load_path in zip(self.times[1:], self.fnames[1:]):
            while not os.path.exists(self.drctry+next_ice_load_path):
                time.sleep(1)
            if os.path.isfile(self.drctry+next_ice_load_path):
                ice1 = self.read_icestage(self.drctry+next_ice_load_path, *self.readargs,
                                        **self.readkwargs)
                yield ice0, t0, ice1, t1
            else:
                raise ValueError("%s isn't a file!" % next_ice_load_path)

            # Save current step for next step
            ice0, t0 = ice1, t1

    def read_icestage(self, fname):
        """Overwritable method for reading in the ice stage."""
        return np.loadtxt(fname).reshape(self.shape)

def printMW(ice, grid, areaVerts=None, areaNames=None, oceanarea=3.61e8):
    """Print equivalent meters meltwater for the glaciers.

    Parameters
    ----------
    grid : GridObject
    oceanarea : float
        the area of the ocean to convert volumes to heights, 
        default = 3.14e8 km^2, current area.
    """
    
    if areaVerts is None:
        assert areaNames, 'need to specify areaVerts or areaNames'
        areaNames = areaNames or GlacierBounds.areaNames
        areaVerts = GlacierBounds.outputAsDict(areaNames)
        
    s = ''
    for column in ['ka BP']+areaVerts.keys()+[' Total']:
        s += '{column:{align}{width}} '.format(column=column, align='^',
                                                width=7)
    print(s)

    #TODO allow limiting by time.

    for i, stage in enumerate(ice):
        s = '{num:{align}{width}{base}}  '.format(num=ice.times[i], align='<',
                                                    width=7, base='.2f')
        # Get the glacier volumes by integrating on the grid.
        vols = grid.integrateAreas(stage, areaVerts)
        for area in areaVerts.keys()+['whole']:
            s += '{num:{align}{width}{base}} '.format(num=vols[area]/oceanarea, 
                                                        align='>', width=7, base='.3f')
        print(s)

class GlacierBounds(object):
    """An object containing lon/lat pairs of vertices for polygons bounding
    separate glacier systems. The following areas are defined:

    EUROPE
    fen     : Mainlaind Fennoscandia
    eng     : England
    fullbar : Barents Sea including islands
    sval    : Svalbard (and adjacent Barents Sea)
    fjl     : Franz Josefland
    nz      : Novaya Zemlya
    bar     : Barents Sea excluding islands (marine areas)

    NORTH AMERICA
    laut    : Laurentian
    cor     : Cordilleran
    naf     : Southern fringe of Laurentian
    inu     : Inutian
    grn     : Greenland

    OTHER
    ice     : Iceland
    want    : West Antarctica
    eant    : East Antarctica
    """

    areaNames = ['eng', 'fen', 'fullbar', 'laur', 'cor', 'naf', 'inu', 'grn',
                    'ice', 'want', 'eant']

    # England
    eng = [(-5, 68.57), (-3.7, 68.57), (5.67, 49.82), (5.67, 49.82),
           (-10.23, 49.82), (-10.23, 60), (-5, 60)]

    # Mainland Fennoscandia
    #fen = [(-20, 50), (-15, 60), (-5, 70), (-5, 85), (180, 85), (180, 40),
    #       (-20, 40)]
    
    fen = [(5.67, 49.82), (-3.7, 68.57), (-0.78, 68.5), (10.47, 68.57), 
           (17.63, 70.29), (23.78, 71.18), (29.52, 70.9), (39.6, 68.11),
           (47.7, 63.87), (47.7, 40), (5.67, 40)]

    # Barents Sea including islands
    fullbar = [( -1.0, 81.33), (36, 85.07),   (80, 82.4),    (75, 78.),  
               (75, 76.01), (59.06, 72.11),(57.5, 69.06), 
               
               (47.7, 63.87), 
               (39.6, 68.11), (29.52, 70.9), (23.78, 71.18), (17.63, 70.29), 
               (10.47, 68.57), (-0.78, 68.57)]

    # Svalbard
    #sval = [(7.18, 81.33), (36, 81.33), (36, 76.64), (24, 75), (12.5, 75.7)]
    sval = [(-1.0, 81.33), (36, 85.07), (36, 76.64), (24, 75), (12.5, 75.7)]

    # Franz Josefland
    #fjl = [(43, 79.9), (43, 82.4), (80, 82.4), (75, 78.0), (50, 78)]
    fjl = [(36, 85.07),  (80, 82.4), (75, 78.0), (50, 78)]

    # Novaya Zemlya
    #nz = [(50.93, 71.01), (50.93, 72.42), (58.12, 76.64), (75, 78),
    #      (69.38, 76.01), (59.06, 72.11), (57.5, 69.06)]
    nz = [(50.93, 71.01), (50.93, 72.42), (50, 78), (75, 78),
          (75, 76.01), (59.06, 72.11), (57.5, 69.06)]


    # Barents Sea marine areas
    #bar = [(-1.0, 81.33), (12.5, 75.7), (24, 75), (36, 76.64), (36, 85.07), 
    #       (43, 82.4),    (43, 79.9),   (50, 78), (75, 78.),   (58.12, 76.64), 
    #       (50.93, 72.42), (50.93, 71.01), (57.5, 69.06), (59.06, 72.11), 
    #       (69.38, 76.01), (75, 78), (80.7, 74.7), (75.8, 74.7), 
    #       (47.7, 63.87), (39.6, 68.11), (29.52, 70.9), (23.78, 71.18), 
    #       (17.63, 70.29), (10.47, 68.57), (-0.78, 68.57)]
    bar = [(-1.0, 81.33), (12.5, 75.7), (24, 75), (36, 76.64), (36, 85.07), 
           (50, 78),

           (50.93, 72.42), (50.93, 71.01), (57.5, 69.06),

           (47.7, 63.87), (39.6, 68.11), (29.52, 70.9), (23.78, 71.18), 
           (17.63, 70.29), (10.47, 68.57), (-0.78, 68.57)]

    ###########  NORTH AMERICA  #########
    # Laurentian
    laur = [(-140, 74.5), (-70, 74.5), (-65, 70), (-57.5, 65), (-55, 60),
            (-50, 40), (-55, 60), (-73, 45), (-82.5, 45), (-90, 47.5),
            (-95, 48.5), (-105, 49.5), (-112.5, 52), (-125, 60), (-130, 65),
            (-140, 65)]

    # Cordilleran
    cor = [(-165, 65), (-130, 65), (-125, 60), (-120, 55), (-112.5, 47), 
           (-125, 45), (-165, 45)]

    # NAFringe
    naf = [(-112.5, 52), (-105, 49.5), (-95, 48.5), (-90, 47.5), (-82.5, 45),
           (-73, 45), (-55, 60), (-50, 54), (-50, 30), (-102.5, 30),
           (-102.5, 47), (-112.5, 47)]

    # Inutian
    inu = [(-140, 85), (-45, 85), (-55, 83), (-60, 82), (-65, 81), (-70, 79.5),
            (-75, 77), (-70, 74.5), (-140, 74.5)]

    # Greenland
    grn = [(-75, 77), (-70, 79.5), (-65, 81), (-60, 82), (-55, 83), (-45, 85),
           (-5, 85), (-5, 75), (-15, 70), (-30, 65), (-35, 60), (-55, 60), 
           (-57.5, 65), (-60, 70), (-70, 74.5), (-75, 77)]

    # Iceland
    ice = [(-35, 60), (-30, 65), (-15, 70), (-5, 75), (-5, 60), (-35, 60)]

    # West Antarctica
    want = [(-180, -55), (0, -55), (0, -90), (-180, -90)]

    # East Antarctica
    eant = [(-180, -84.5), (-150, -86), (-90, -87.5), (-7.5, -87.5), 
            (-7.5, -55), (172, -55), (172, -82.5), (180, -84.5), (180, -90),
            (-180, -90)]

    @classmethod
    def outputAsList(cls, names=None):
        names = names or cls.areaNames
        outputList = []
        for name in names:
            outputList.append({ 'name':name,
                                'vert':getattr(cls, name)})
        return outputList

    @classmethod
    def outputAsDict(cls, names=None):
        names = names or cls.areaNames
        outputDict = {}
        for name in names:
            outputDict[name] = getattr(cls, name)
        return outputDict

glacierbounds = GlacierBounds()
