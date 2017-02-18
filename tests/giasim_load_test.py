"""
giasim_load_test.py
Author: Samuel B. Kachuck
Date: 16 02 2017

Test the loading of a giasim object and the downloading of relevant inputs.
"""

import giapy

configdict = {'ice': 'AA2_Tail_nochange5_hightres_Pers_288_square',
              'earth': '75km0p04Asth_4e23Lith',
              'topo': 'sstopo288'}

sim = giapy.sim.configure_giasim(configdict)
print('Inputs loaded and GiaSim object configured')

result = sim.performConvolution(out_times=[12, 10, 8, 0.1, 0, -0.1])
print('Glacial isostatic adjustment computed')

print('To test plotting, in an iPython session, run this script and type')
print("result.inputs.grid.pcolormesh(result['sstopo'][0])")
