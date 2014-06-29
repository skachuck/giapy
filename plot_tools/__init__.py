"""Module data define regular areas for plots"""

from mpl_toolkits.basemap import Basemap

sval = {'latmin':76, 'latmax':81, 'lonmin':10, 'lonmax':30}
nam = {'latmin':-90, 'latmax':90, 'lonmin':-180, 'lonmax':180}
namc = {'latmin':30, 'latmax':50, 'lonmin':-80, 'lonmax':-50}
eur = {'latmax': 82, 'latmin': 49, 'lonmax': 75, 'lonmin': -15}

# m = basemap.Basemap(**param_dict)
eur_map_param = {'width':4900000,'height':4700000,
            'rsphere':(6378137.00,6356752.3142),\
            'resolution':'l','area_thresh':1000.,'projection':'lcc',\
            'lat_1':50.,'lat_2':89.9,'lat_0':72,'lon_0':20.}
scan_map_param = {'llcrnrlon':-10,'llcrnrlat':45,'urcrnrlon':60,'urcrnrlat':69,
            'rsphere':(6378137.00,6356752.3142),
            'resolution':'l','area_thresh':1000.,'projection':'lcc',
            'lat_1':45.,'lat_2':69,'lat_0':57,'lon_0':25.}
sval_map_param = {'llcrnrlon':10,'llcrnrlat':76,'urcrnrlon':30,'urcrnrlat':81,\
            'rsphere':(6378137.00,6356752.3142),\
            'resolution':'i','area_thresh':1000.,'projection':'lcc',\
            'lat_1':76.,'lat_2':81,'lat_0':78.5,'lon_0':20.}
fjl_map_param = {'llcrnrlon':46,'llcrnrlat':79,'urcrnrlon':68,'urcrnrlat':82,\
            'rsphere':(6378137.00,6356752.3142),\
            'resolution':'i','area_thresh':250.,'projection':'lcc',\
            'lat_0':80.5,'lon_0':57.}
namc_map_param = {'llcrnrlon':-80, 'llcrnrlat':30, 'urcrnrlon':-50, 'urcrnrlat':50,
            'rsphere':(6378137.00,6356752.3142),
            'resolution':'l', 'area_thresh':1000., 'projection':'merc'}
nam_map_param = {'llcrnrlon':-110, 'llcrnrlat':20, 'urcrnrlon':-50, 'urcrnrlat':60,
            'rsphere':(6378137.00,6356752.3142),
            'resolution':'l', 'area_thresh':1000., 'projection':'merc'}
glob_map_param = {'llcrnrlon':-180, 'llcrnrlat':-70, 
            'urcrnrlon':180, 'urcrnrlat':80,
            'rsphere':(6378137.00,6356752.3142),
            'resolution':'l', 'area_thresh':1000., 'projection':'merc'}
