"""Module data define regular areas for filtering data"""

sval = {'latmin':76, 'latmax':81, 'lonmin':10, 'lonmax':30}
nam = {'latmin':-90, 'latmax':90, 'lonmin':-180, 'lonmax':180}
eur = {'latmax': 82, 'latmin': 49, 'lonmax': 75, 'lonmin': -15}

# Location of c14 correction table
#C14TABLE = u'./Data/marine09.14c.txt'
C14TABLE = u'./Data/intcal13.14c'

import emergedata as emerge
import gravdata as grav
