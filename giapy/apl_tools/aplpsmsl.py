"""
aplpsmsl.py
Author: Samuel B. Kachuck
Date: August 30, 2017

    Function to download the PSMSL database into the format expected by APL
    GlacialRebound program. Call
    create_rsl_directory(desired/path/to/download/)
"""

import numpy as np
from giapy.data_tools.rsldata import RLR

def format_basefile_str(rlr):
    s = '\t'.join(['{}']*10)+'\n'
    rlrerror = np.sqrt(np.mean((rlr[rlr.inds, 1] - rlr.trend()[rlr.inds])**2))

    bs = s.format(rlr.metadata['sitename'],
                  rlr.metadata['stid'],
                  rlr.metadata['lon'],
                  rlr.metadata['lat'],
                  0,
                  rlr.metadata['yrmin'],
                  rlr.metadata['yrmax'],
                  len(rlr),
                  float(rlr.trend(coeffs=True)[0]),
                  float(rlrerror))
    return bs

def create_rsl_directory(nmax=2400, drctry=''):
    basefile = ''
    for n in range(1, nmax):
        try:
            rlr = RLR(n, typ='annual')
        except:
            continue
        basefile += format_basefile_str(rlr)
        with open(drctry+'{}.txt'.format(rlr.metadata['stid']), 'w') as f:
            np.savetxt(f, rlr[rlr.inds,:2], fmt='%d')
    with open(drctry+'rlr_download_basefile.txt', 'w') as f:
        f.write(basefile)


