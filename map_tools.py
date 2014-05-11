import numpy as np
from scipy.interpolate import RectBivariateSpline

# Grid Interpolation

# Needs a basemap.Basemap object, m

Lon, Lat = m(X, Y, inverse=True)

X, Y = m(Lon, Lat)

x = np.linspace(0, 4900000, num=491, endpoint=True)
y = np.linspace(0, 4700000, num=471, endpoint=True)

X, Y = np.meshgrid(x, y)

europe = {'latmax': 82, 'latmin': 49, 'lonmax': 75, 'lonmin': -15}

lon = np.linspace(europe['lonmin'], europe['lonmax'], nlon)
lat = np.linspace(europe['latmin'], europe['latmax'], nlat)

Lon_interp, Lat_interp = np.meshgrid(lon, lat)
X_interp, Y_interp = m(Lon_interp, Lat_interp) 

interpolator = RectBivariateSpline(x, y, uplift)

uplift_interp = interpolator(X_interp, Y_interp)

def haversine(lat1, lat2, lon1, lon2, r=6371, radians=False):
    """Calculate the distance bewteen two sets of lat/lon pairs.
    """
    if not radians:
        lat1, lat2, lon1, lon2 = np.radians([lat1, lat2, lon1, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    ds = r*2*np.arcsin(np.sqrt(np.sin(dlat/2)**2 + 
                        np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2))

    return ds
