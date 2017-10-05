import numpy as np
import giapy
import giapy.apl_tools.t_files
import giapy.data_tools.gpsdata
import giapy.data_tools.tiltdata



def load_ice_modifications(fname, ice, grid):

    larry_output_order = ['cor', 'laur', 'naf', 'inu', 'grn', 'ice', 'fen',
    'want', 'eant', 'bar', 'sval', 'fjl', 'nz', 'eng']

    with open(fname, 'r') as f:
        text = f.readlines()

    times = [float(w.strip())/1000 for w in text[1].split(',')[1:]]

    newtimes = [10.5, 9.5, 8.5, 7.2, 6., 5.25, 5., 4.75]

    sortinds = np.argsort(np.r_[times, newtimes])

    props = []
    for line in text[5:]:
        data = [entry.strip() for entry in line.split(',')]
        prop = [1+float(num) for num in data[1:]]
        newprop = np.interp(newtimes, times, prop)
        prop = (np.r_[prop, newprop])[sortinds][::-1]
        # append 0
        prop = np.r_[prop, [1]]


        props.append(prop)

    ice.createAlterationAreas(grid, props, larry_output_order)

    return ice.applyAlteration()




if __name__ == '__main__':
    import sys
    casename, alterfile = sys.argv[1:3]


    configdict = {'ice': 'AA2_Tail_nochange5_hightres_Pers_288_square',
                  'earth': '75km0p04Asth_4e23Lith',
                  'topo': 'sstopo288'}

    sim = giapy.giasim.configure_giasim(configdict)
    
    print('Inputs loaded\r')

    sim.ice = load_ice_modifications(alterfile, sim.ice, sim.grid) 


    print('Ice load modified\r')

    result = sim.performConvolution(out_times=sim.ice.times)

    print('Result computed, writing out case files\r')

    emergedatafile = giapy.MODPATH+'/data/obs/Emergence_Data_seqnr_2014.txt'
    emergedata = giapy.data_tools.emergedata.importEmergeDataFromFile(emergedatafile)
    rsldata = giapy.load(giapy.MODPATH+'/data/obs/psmsl_download_02082017.p')
    gpsdata = np.load(giapy.MODPATH+'/data/obs/gps_obs.p')
    tiltdata = giapy.data_tools.tiltdata.TiltData()

    giapy.apl_tools.t_files.write_case_files(casename, result)
    print('Result computed, writing out data files\r')
    giapy.apl_tools.t_files.write_data_files(casename, result,
                        emergedata=emergedata, rsldata=rsldata,
                        gpsdata=gpsdata, tiltdata=tiltdata)

    
