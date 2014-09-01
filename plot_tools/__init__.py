"""Module data define regular areas for plots"""

from mpl_toolkits.basemap import Basemap

# SQUARE AREAS
class SquareAreas(object):
    """An object with square areas defined"""
    sval = {'latmin':76, 'latmax':81, 'lonmin':10, 'lonmax':30}
    nam = {'latmin':-90, 'latmax':90, 'lonmin':-180, 'lonmax':180}
    namc = {'latmin':30, 'latmax':50, 'lonmin':-80, 'lonmax':-50}
    eur = {'latmax': 82, 'latmin': 49, 'lonmax': 75, 'lonmin': -15}

# MAP PARAMETERS
# m = basemap.Basemap(**param_dict)
class MapParams(object):
    """An object with basemap parameters defined"""
    eur = {'width':4900000,'height':4700000,
                'rsphere':(6378137.00,6356752.3142),\
                'resolution':'l','area_thresh':1000.,'projection':'lcc',\
                'lat_1':50.,'lat_2':89.9,'lat_0':72,'lon_0':20.}
    scan = {'llcrnrlon':-10,'llcrnrlat':45,'urcrnrlon':60,'urcrnrlat':69,
                'rsphere':(6378137.00,6356752.3142),
                'resolution':'l','area_thresh':1000.,'projection':'lcc',
                'lat_1':45.,'lat_2':69,'lat_0':57,'lon_0':25.}
    sval = {'llcrnrlon':10,'llcrnrlat':76,'urcrnrlon':30,'urcrnrlat':81,\
                'rsphere':(6378137.00,6356752.3142),\
                'resolution':'i','area_thresh':1000.,'projection':'lcc',\
                'lat_1':76.,'lat_2':81,'lat_0':78.5,'lon_0':20.}
    fjl = {'llcrnrlon':46,'llcrnrlat':79,'urcrnrlon':68,'urcrnrlat':82,\
                'rsphere':(6378137.00,6356752.3142),\
                'resolution':'i','area_thresh':250.,'projection':'lcc',\
                'lat_0':80.5,'lon_0':57.}
    namc = {'llcrnrlon':-80, 'llcrnrlat':30, 'urcrnrlon':-50, 'urcrnrlat':50,
                'rsphere':(6378137.00,6356752.3142),
                'resolution':'l', 'area_thresh':1000., 'projection':'merc'}
    nam = {'llcrnrlon':-110, 'llcrnrlat':20, 'urcrnrlon':-50, 'urcrnrlat':60,
                'rsphere':(6378137.00,6356752.3142),
                'resolution':'l', 'area_thresh':1000., 'projection':'merc'}
    glob = {'llcrnrlon':-180, 'llcrnrlat':-70, 
                'urcrnrlon':180, 'urcrnrlat':80,
                'rsphere':(6378137.00,6356752.3142),
                'resolution':'l', 'area_thresh':1000., 'projection':'merc'}

# GLACIER AREAS
class GlacierBounds(object):
    #bar = [(-.78, 85.07), (80.07, 85.07), (80.7, 74.7), (75.8, 74.7), 
    #       (47.7, 63.87), (39.6, 68.11), (29.52, 70.9), (23.78, 71.18),
    #       (17.63, 70.29), (10.47, 68.57), (-0.78, 68.57)]
    
    #bar = [(-1.0, 81.33), (12.5, 75.7), (24, 75), (36, 76.65), (36, 85.07),
    #       (43, 82.4), (43, 79.9), (50, 78), (66, 78.9), (50.93, 72.42), 
    #       (50.93, 71.01), (57.5, 69.06), (59.06, 72.11), (69.38, 76.01), (80.7, 74.7), 
    #       (75.8, 74.7), (47.7, 63.87), (39.6, 68.11), (29.52, 70.9), (23.78, 71.18), 
    #       (17.63, 70.29), (10.47, 68.57), (-0.78, 68.57)]
    
    bar = [(-1.0, 81.33), (12.5, 75.7), (24, 75), (36, 85.07), (43, 82.4), 
           (43, 79.9), (50, 78), (66, 78.9), (50.93, 72.42), (50.93, 71.01), 
           (57.5, 69.06), (59.06, 72.11), (69.38, 76.01), (80.7, 74.7), (75.8, 74.7), 
           (47.7, 63.87), (39.6, 68.11), (29.52, 70.9), (23.78, 71.18), (17.63, 70.29), 
           (10.47, 68.57), (-0.78, 68.57)]
    
    sval = [(7.18, 81.33), (36, 81.33), (36, 76.64), (24, 75), (12.5, 57.7)]

    def __repr__(self):
        return "An object containing lon/lat pairs on the boundaries of areas"
