import numpy as np
import os

from giapy.data_tools.tiltdata import calcTilts

def read_t_files(directory, filenames, data_col=2):
    """Read in a full comma-delimitted ice file in x-y-z format.

    Parameters:
        filename - the file to be read
        Nx - number of latitude sites
        Ny - number of longitude sitesb

    Returns:
        lat, lon - size (Nx, Ny) arrays of latitude and longitude measures
        height - size (num of stages, Nx, Ny) array of ice heights
    """
    
    # initiliaze arrays by reading in first file
    rawdata = np.loadtxt(directory+filenames[0], delimiter=',', comments='#')
    # find the number of  latitude points
    Nlat = len(set(rawdata[:,1]))
    # find the number of longitude points 
    Nlon = len(set(rawdata[:,0]))

    Lon = np.reshape(rawdata[:,0], (Nlat, Nlon))*np.pi/180.
    Lat = np.reshape(rawdata[:,1], (Nlat, Nlon))*np.pi/180.
    
    height = rawdata[:,data_col]
    
    for filename in filenames[1:]:
        rawdata = np.loadtxt(directory+filename, delimiter=',', comments='#')
        height = np.append(height, rawdata[:,data_col])
    
    height = np.reshape(height, (-1, Nlat, Nlon))
        
    return Lat, Lon, height

def write_case_files(casename, result): 
    try: 
        os.mkdir(casename)
    except:
        pass 

    coltit = 'longitude\tlatitude\tTotUpl\tTotUpl\tRateUpl\tGeoid\t'
    coltit += 'emergence\twload\tload\tload\ticeload\tocean\ttopomap0'

    result.upl.transform(result.inputs.harmTrans, inverse=False)
    result.vel.transform(result.inputs.harmTrans, inverse=False)
    result.geo.transform(result.inputs.harmTrans, inverse=False)

    outTimes = result.upl.outTimes 
    fnames = []

    u0 = result.sstopo.nearest_to(0)
    for i, t in enumerate(outTimes):
        ai = np.vstack([result.inputs.grid.Lon.flatten(), 
                   result.inputs.grid.Lat.flatten(), 
                   result.upl[i].flatten(),
                   result.upl[i].flatten(),
                   result.vel[i].flatten(),
                   result.geo[i].flatten(),
                   (result.sstopo[i] - u0).flatten(),
                   result.wload[i].flatten(),
                   result.load[i].flatten(),
                   result.load[i].flatten(),
                   (result.load[i]- result.wload[i]).flatten(),
                   (result.sstopo[i]<0).flatten(),
                   result.inputs.topo.flatten()]).T

        fname = '{}/py_file_{}.txt'.format(casename, i+1)
        header = 'case: {} at {}\n'.format(casename, t) + coltit
        np.savetxt(fname, ai, header=header)

        fnames.append(fname)

    with open('{}/py_file_inf.txt'.format(casename), 'w') as f:
        f.write('case: {}\n'.format(casename))
        f.write('date: {}\n'.format(result.TIMESTAMP))
        f.write('vers: {}\n'.format(result.GITVERSION))
        f.write('files: {}\n'.format('\t'.join(fnames)))
        f.write('mMW: {}\n'.format('\t'.join([str(t) for t in result.esl.array])))
        f.write('mMW: {}\n'.format('\t'.join([str(result.inputs.grid.integrate(icet)/3.61e8) 
                                                for icet in results.inputs.ice])))
        f.write('ages: {}\n'.format('\t'.join([str(t) for t in outTimes])))
        

def write_data_files(casename, result, emergedata=None, rsldata=None,
                        gpsdata=None, tiltdata=None):
    try: 
        os.mkdir(casename)
    except:
        pass 
    
    result.upl.transform(result.inputs.harmTrans, inverse=False)

    if emergedata is not None:
        u0 = result['sstopo'].nearest_to(0)
        coltit = 'recnbr\tlongitude\tlatitude\temerge_i'
        outTimes = result.sstopo.outTimes 

        uAtLocs = []
        for ut in result['sstopo']:
            ut = u0 - ut
            interpfunc = result.inputs.grid.create_interper(ut.T)
            uAtLocs.append(interpfunc.ev(emergedata.lons, emergedata.lats))

        output = np.zeros((len(emergedata.lons), len(outTimes)+3))

        output[:, 3:] = np.asarray(uAtLocs).T
        output[:, 0] = [loc.recnbr for loc in emergedata]
        output[:, 1] = emergedata.lons
        output[:, 2] = emergedata.lats

        fname = '{}/py_file_emerge.txt'.format(casename)
        header = 'case: {} emergence interpolation\n'.format(casename) + coltit
        np.savetxt(fname, output, header=header)

    if rsldata is not None:
        coltit = 'recnbr\tlongitude\tlatitude\trsl_i'
        outTimes = result.sstopo.outTimes
        u0 = result['sstopo'].nearest_to(0)

        uAtLocs = []
        for ut in result['sstopo']:
            ut = u0 - ut
            interpfunc = result.inputs.grid.create_interper(ut.T)
            uAtLocs.append(interpfunc.ev(rsldata.lons, rsldata.lats))

        output = np.zeros((len(rsldata.lons), len(outTimes)+3))

        output[:, 3:] = np.asarray(uAtLocs).T
        output[:, 0] = [loc.stid for loc in rsldata]
        output[:, 1] = rsldata.lons
        output[:, 2] = rsldata.lats

        fname = '{}/py_file_rsl.txt'.format(casename)
        header = 'case: {} rsl interpolation\n'.format(casename) + coltit
        np.savetxt(fname, output, header=header)


    if tiltdata is not None:
        coltit = 'recnbr\tlongitude\tlatitude\tstart\tend\tcalc\tobs'
        writeout = 'case: {} tilt interpolation\n'+coltit+'\n'
        for ti, tf, loc, obs, recnbr in zip(tiltdata.long_times_i,
                                    tiltdata.long_times_f, tiltdata.locs,
                                    tiltdata.long_data, tiltdata.long_recnbrs):
            diffup = (result.uplift.nearest_to(tf) -
                            result.uplift.nearest_to(ti))
            tilt = calcTilts(diffup, result.inputs.grid.Lon,
                                    result.inputs.grid.Lat)
            calc = result.inputs.grid.interp(tilt, loc[0], loc[1])
            writeout += '{}\n'.format('\t'.join([recnbr, loc[0], loc[1], ti,
                                        tf, calc, obs]))
        with open('{}/py_file_tilt.txt'.format(casename)) as f:
            f.write(writeout)
    #with open('{}/py_file_inf.txt'.format(casename), 'w') as f:
    #    f.write('case: {}\n'.format(casename))
    #    f.write('date: {}\n'.format(result.TIMESTAMP))
    #    f.write('vers: {}\n'.format(result.GITVERSION))
    #    f.write('files: {}\n'.format('\t'.join(fnames)))
    #    f.write('mMW: {}\n'.format('\t'.join([str(t) for t in result.esl.array])))
    #    f.write('ages: {}\n'.format('\t'.join([str(t) for t in outTimes])))
