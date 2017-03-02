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

import earth_tools as earth
import data_tools as data
import icehistory as ice
import map_tools as maps
import giasim as sim
import giamc as mc
import plot_tools as plot

