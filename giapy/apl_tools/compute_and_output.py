import numpy as np
import giapy
import giapy.apl_tools.t_files
import giapy.data_tools.gpsdata
import giapy.data_tools.tiltdata


def read_ice_assignments(fname):
    with open(fname, 'r') as f:
        nglaciers = int(f.readline().split(',')[0])
        alterdict = {}
        alternum = {}

        for i in range(14):
            l = f.readline().split(',')
            num = int(l[0])
            name = l[4]
            toplonlat = []
            for item in f.readline().split(','):
                try:
                    toplonlat.append(float(item))
                except:
                    break
            botlonlat = []
            for item in f.readline().split(','):
                try:
                    botlonlat.append(float(item))
                except:
                    break

            lonlat = np.hstack([np.vstack([toplonlat[::2], toplonlat[1::2]]),
                                np.vstack([botlonlat[::2], botlonlat[1::2]])[:,::-1]])

            alterdict[num] = lonlat.T 
            alternum[name] = num
    return alterdict, alternum

def load_ice_modifications(propfname, glacfname, ice, grid):

    #larry_output_order = ['cor', 'laur', 'naf', 'inu', 'grn', 'ice', 'fen',
    #'want', 'eant', 'bar', 'sval', 'fjl', 'nz', 'eng']

    alterdict, alternum = read_ice_assignments(glacfname)

    with open(propfname, 'r') as f:
        text = f.readlines()

    #times = [float(w.strip())/1000 for w in text[1].split(',')[1:]]

    #newtimes = [10.5, 9.5, 8.5, 7.2, 6., 5.25, 5., 4.75]

    #sortinds = np.argsort(np.r_[times, newtimes])

    props = {}
    for line in text[5:]:
        if ',' in line:
            data = [entry.strip() for entry in line.split(',')]
        elif '\t' in line:
            data = [entry.strip() for entry in line.split('\t')]
            
        prop = [1+float(num) for num in data[1:]]
        name = data[0]
        num = alternum[name]
        #newprop = np.interp(newtimes, times, prop)
        #prop = (np.r_[prop, newprop])[sortinds][::-1]
        # append 0
        prop = np.r_[prop[::-1], [1]]


        props[num] = prop

    ice.createAlterationAreas(grid, props.values(), alterdict.keys(), alterdict)

    return ice.applyAlteration()




if __name__ == '__main__':
    import sys, subprocess,os
    import argparse

    parser = argparse.ArgumentParser(description='Compute and output GIA for '
                                                  'APL GlacialRebound program')
    parser.add_argument('casename', type=str)
    parser.add_argument('alterfile', type=str)
    parser.add_argument('glacfile', type=str)
    parser.add_argument('tnochange', type=float)
    parser.add_argument('--earth', type=str, default=None)
    parser.add_argument('--tfiles', default=False,
                            action='store_const', const=True,
                            dest='tfiles')

    comargs = parser.parse_args()

    casename, alterfile = comargs.casename, comargs.alterfile
    glacfile, tnochange = comargs.glacfile, comargs.tnochange 

    earth = comargs.earth
    tfileflag = comargs.tfiles

    configdict = {'ice': 'aa2_base_pers_288',
                  'earth': '75km0p04Asth_4e23Lith',
                  'topo': 'sstopo288'}

    sim = giapy.giasim.configure_giasim(configdict)

    if earth is not None:
        print('Loading earth model: {}'.format(earth))
        earth = np.load(open(earth, 'r'))
        sim.earth = earth
        assert earth.nmax >= 288, 'earth must be at least 288 resolution'
    
    print('Inputs loaded\r')

    sim.ice.stageOrder = np.array(sim.ice.stageOrder)
    sim.ice.stageOrder[sim.ice.times <= tnochange] = sim.ice.stageOrder[-1]

    sim.ice = load_ice_modifications(alterfile, glacfile, sim.ice, sim.grid) 


    print('Ice load modified\r')

    result = sim.performConvolution(out_times=sim.ice.times)

    print('Result computed, writing out case files\r')

    emergedatafile = giapy.MODPATH+'/data/obs/Emergence_Data_seqnr_2014.txt'
    emergedata = giapy.data_tools.emergedata.importEmergeDataFromFile(emergedatafile)
    rsldata = giapy.load(giapy.MODPATH+'/data/obs/psmsl_download_02082017.p')
    gpsdata = np.load(giapy.MODPATH+'/data/obs/gps_obs.p')
    tiltdata = giapy.data_tools.tiltdata.TiltData()

    giapy.apl_tools.t_files.write_case_files(casename, result,
                                                tfileflag=tfileflag)
    print('Result computed, writing out data files\r')
    giapy.apl_tools.t_files.write_data_files(casename, result,
                        emergedata=emergedata, rsldata=rsldata,
                        gpsdata=gpsdata, tiltdata=tiltdata)

    print os.path.abspath(os.path.curdir)
    command = 'cp {0} ./{1} && cp {2} ./{1}'.format(alterfile,
                                                        casename,
                                                        glacfile)
    print command
    subprocess.call(command, shell=True)
