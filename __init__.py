#import giapy.giasim
#import giapy.earth_tools.earth_two_d
#import giapy.ice_tools.icehistory
#import giapy.plot_tools
#import giapy.map_tools

#from giapy.data_tools.meltwater import gen_eustatic
#import giapy.data_tools.emergedata

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

import code.earth_tools as earth
import code.data_tools as data
import code.icehistory as ice
import code.map_tools as maps
import code.giasim as sim
import code.giamc as mc
import code.plot_tools as plot

