"""
__init__.py

Author: Samuel B. Kachuck

Import giapy submodules into convenient namespace and provide useful data to be
used across the package.

Data
----
MODPATH: the path to the package
GITVERSION: the hashed version of the git (for recording state of code along
            with computations)
Methods
-------
timestamp: fancy string of the current date and time
load : filename (str)
    Convenience function for unpickling an object
"""


import os, sys
from datetime import datetime
from subprocess import check_output, call
if sys.version_info < (3,):
    import cPickle as pickle
else:
    import _pickle as pickle


# Obtain the github hash of the current version
command = 'git log -n 1 | grep commit | sed s/commit\ //'
MODPATH = os.path.abspath(os.path.dirname(__file__))
command = 'cd ' + MODPATH + ' && ' + command
GITVERSION = check_output(command, shell=True)[:10]
del command

def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def load(filename):
    return pickle.load(open(filename, 'r'))

#import giapy.earth_tools
#import giapy.earth_tools as earth
#import giapy.data_tools
#import giapy.data_tools as data
#import giapy.icehistory as ice
#import giapy.map_tools as maps

#import giapy.sle
# For internal backwards compatibility for now
#import giapy.sle as sim
#import giapy.sle as giasim

#import giapy.plot_tools as plot

