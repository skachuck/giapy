"""Module data define regular areas for plots"""

from mpl_toolkits.basemap import Basemap

# SQUARE AREAS
class SquareAreas(object):
    """An object with square areas defined"""
    sval = {'latmin':76, 'latmax':81, 'lonmin':10, 'lonmax':30}
    nam = {'latmin':-90, 'latmax':90, 'lonmin':-180, 'lonmax':180}
    namc = {'latmin':30, 'latmax':50, 'lonmin':-80, 'lonmax':-50}
    eur = {'latmax': 82, 'latmin': 49, 'lonmax': 75, 'lonmin': -15}
    bar = {'latmax': 85, 'latmin': 68, 'lonmax': 90, 'lonmin': 15}
    svfj = {'latmin':76, 'latmax':81, 'lonmin':10, 'lonmax':62}

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
    bar = {'llcrnrlon':15,'llcrnrlat':68,'urcrnrlon':89,'urcrnrlat':80,\
                'rsphere':(6378137.00,6356752.3142),\
                'resolution':'i','area_thresh':250.,'projection':'lcc',\
                'lat_0':78,'lon_0':38.}
    svalfjl = {'llcrnrlon':17,'llcrnrlat':75,'urcrnrlon':85,'urcrnrlat':83,\
                'rsphere':(6378137.00,6356752.3142),\
                'resolution':'i','area_thresh':250.,'projection':'lcc',\
                'lat_1':70.,'lat_2':83,'lat_0':75.5,'lon_0':57}
    namc = {'llcrnrlon':-80, 'llcrnrlat':30, 'urcrnrlon':-50, 'urcrnrlat':50,
                'rsphere':(6378137.00,6356752.3142),
                'resolution':'l', 'area_thresh':1000., 'projection':'merc'}
    nam = {'llcrnrlon':-110, 'llcrnrlat':20, 'urcrnrlon':-50, 'urcrnrlat':60,
                'rsphere':(6378137.00,6356752.3142),
                'resolution':'l', 'area_thresh':1000., 'projection':'merc'}
    glob = {'llcrnrlon':-180, 'llcrnrlat':-90, 
                'urcrnrlon':180, 'urcrnrlat':90,
                'rsphere':(6378137.00,6356752.3142),
                'resolution':'l', 'area_thresh':1000., 'projection':'merc'}


