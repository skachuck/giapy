"""
This module provides a class and class methods for accessing large data sets
that are four dimensional where one dimension (e.g., time) is 
"""

import numpy as np
import cPickle as pickle

class Abs_xyzt_data(object):
    
    def __init__(self):
        self.filelist = []
        self.fileref = []
        self.desc = ''
        
    def __getitem__(self, key):
        if self.data == None:
            self.load_data()
        return self.data.__getitem__(key)
        
    def __iter__(self):
        if self.data == None:
            self.load_data()
        return self.data.__iter__()
        
    def __repr__(self):
        return self.desc
        
    def load_one(self, one):
        pass
    def load_all(self):
        pass
        
    def save(self, filename):
        """Save the EmergeData interface object, with empty data list
        
        Parameters
        ----------
        filename (str) - path to the file to save
        """
        
        if self.data_filename == None:
            raise NameError('self.data_filename undefined')
        
        self.save_data(self.data_filename)
        self.data = None
        pickle.dump(self, open(filename, "wb"))
        
    def save_data(self, filename):
        """Save the data list and store the path in the interface object for
        later loading.
        
        Parameters
        ----------
        filename (str) - path to the file to save
        """
        self.data_filename = filename
        pickle.dump(self.data, open(filename, "wb"))
        
    def load_data(self):
        """Load the data from self.data_filename"""
        if self.data_filename == None:
            raise NameError('self.data_filename undefined')
        self.data = pickle.load(open(self.data_filename, "rb"))
        
    def set_desc(self, string):
        self.desc = string