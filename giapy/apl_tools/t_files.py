import numpy as np

def read_t_files(directory, filenames, data_col=2):
    """Read in a full comma-delimitted ice file in x-y-z format.

    Parameters:
        filename - the file to be read
        Nx - number of latitude sites
        Ny - number of longitude sites

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