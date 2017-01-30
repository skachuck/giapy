#import giapy.giasim
#import giapy.earth_tools.earth_two_d
#import giapy.ice_tools.icehistory
#import giapy.plot_tools
#import giapy.map_tools

#from giapy.data_tools.meltwater import gen_eustatic
#import giapy.data_tools.emergedata

import os
from datetime import datetime
from subprocess import check_output
import cPickle as pickle


# Obtain the github hash of the current version
command = 'git log -n 1 | grep commit | sed s/commit\ //'
script_loc = os.path.abspath(os.path.dirname(__file__))
command = 'cd ' + script_loc + ' && ' + command
GITVERSION = check_output(command, shell=True)[:10]
del command, script_loc

def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def load(filename):
    return pickle.load(open(filename, 'r'))

import earth_tools as earth
import data_tools as data
import ice_tools as ice
import map_tools as maps
import giasim as sim
import giamc as mc
import plot_tools as plot

