import numpy as np
import pickle
from giapy.earth_tools.earthParams import EarthParams
from giapy.earth_tools.earthSpherical import SphericalEarth

def read_params(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()

    lines = [line.split(',') for line in lines[0].split('\r')]

    fr23 = float(lines[0][1])

    zs = np.array([float(line[2]) for line in lines[2:]][::-1])
    vs = np.array([float(line[3]) for line in lines[2:]][::-1])
    nas = np.array([float(line[4]) for line in lines[2:]][::-1])

    params = EarthParams()
    params.addLithosphere(D=fr23*1e23)
    params.addViscosity(np.vstack([zs/params.norms['r']*1000, vs*1e21]))
    params.addNonadiabatic(np.vstack([zs/params.norms['r']*1000, nas]))

    return params

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute an earth relaxation')
    parser.add_argument('paramfname', type=str)
    parser.add_argument('efname', type=str)

    comargs = parser.parse_args()

    params = read_params(comargs.paramfname)

    earth = SphericalEarth(params)
    earth.calcResponse(np.linspace(params.rCore, 1), nmax=288)

    pickle.dump(earth, open(comargs.efname, 'w'))
