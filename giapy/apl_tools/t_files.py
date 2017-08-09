import numpy as np
import os

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

    os.mkdir(casename)

    coltit = 'longitude\tlatitude\tTotUpl\tTotUpl\tRateUpl\tGeoid\t'
    coltit += 'emergence\twload\tload\tload\ticeload\tocean\ttopomap0'

    result.upl.transform(result.inputs.harmTrans, inverse=False)
    result.vel.transform(result.inputs.harmTrans, inverse=False)
    result.geo.transform(result.inputs.harmTrans, inverse=False)

    outTimes = result.upl.outTimes 
    fnames = []

    for i, t in enumerate(outTimes):
        ai = np.vstack([result.inputs.grid.Lon.flatten(), 
                   result.inputs.grid.Lat.flatten(), 
                   result.upl[i].flatten(),
                   result.upl[i].flatten(),
                   result.vel[i].flatten(),
                   result.geo[i].flatten(),
                   (result.sstopo[i] - result.sstopo[-1]).flatten(),
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
        f.write('ages: {}\n'.format('\t'.join([str(t) for t in outTimes])))
        

