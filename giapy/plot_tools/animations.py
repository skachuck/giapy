import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RectBivariateSpline

from mpl_toolkits.basemap import Basemap

from giapy.apl_tools import read_t_files

"""
m = Basemap(width=4900000,height=4700000,\
              rsphere=(6378137.00,6356752.3142),\
              resolution='l',area_thresh=1000.,projection='lcc',\
              lat_1=50.,lat_2=89.9,lat_0=72,lon_0=20.)

Lat1152, Lon1152, topo1152 = read_t_files('./',['Surf0Load_m_20p_Eu_polyc.txt'], 4)
Lat576, Lon576, emerg = read_t_files('./',['SURFtxt_20_Eu100000010Polyct_file_cycl_1_42.txt'], 2)
Lat576, Lon576, load = read_t_files('./',['SURFtxt_20_Eu000010010Polyct_file_cycl_1_42.txt'], 3)

lat576 = Lat576[:,0]
lon576 = Lon576[0,:]

lon1152 = Lon1152[0,:]
lat1152 = Lat1152[:,0]

emerg_int = RectBivariateSpline(lon576, lat576, emerg[0,:,:].T)
load_int = RectBivariateSpline(lon576, lat576, load[0,:,:].T)
topo_int = RectBivariateSpline(lon1152, lat1152, topo1152[0,:,:].T)
"""


def anim_transects(emerg_func, load_func, topo_func, paths, filename):
    """Animate the progression of ice stages
    
    anim_emerg(emerg_int, load_int, topo_int, 50, 83, 0.1)
    """
    
    frames = len(paths)

    fig = plt.figure(figsize=(12, 9), dpi=240)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_ylim((-1500, 2000))
    ax.set_xlim((-.2, 1.4))
    plt.axis('off')
    
    lonvalues = np.linspace(-.18, 1.4, 1000)

    emvalues = -1*emerg_func.ev(paths[0].xs, paths[0].ys)
    loadvalues = load_func.ev(paths[0].xs, paths[0].ys)
    topovalues = topo_func.ev(paths[0].xs, paths[0].ys)
    sea_level = -106.2*np.ones(len(topovalues))
    
    ax.plot([-0.2, 1.4], [-106.2, -106.2], color='blue', lw=1.5, ls='-', alpha=0.25)
    ax.plot([-0.2, 1.4], [0, 0], color='blue', lw=1.5, ls='-.', alpha=0.25)
    
    toponow_line, = ax.plot(paths[0].ds, topovalues, color='#B88A00', ls='--', lw=2, zorder=1)
    topothe_line, = ax.plot(paths[0].ds, topovalues-emvalues, color='#B88A00', ls='-', lw=2.3, zorder=3)
    icethen_line, = ax.plot(paths[0].ds, loadvalues+(topovalues-emvalues), color='#33FFCC', ls='-', lw=2, zorder=2)
    
    curr_sl_label = ax.text(1.2, -70, 'Current Sea Level', color='blue', transform=ax.transData)
    past_sl_label = ax.text(1.2, -180, 'Past Sea Level: -106.2 m', color='blue', transform=ax.transData)
    
    # The inset map
    insetax = fig.add_axes([0.6, 0.6, 0.25, 0.25])
    m.drawcoastlines(linewidth = 0.7)
    pathline = insetax.plot(paths[0].xs, paths[0].ys)

    def init():
        toponow_line.set_data([], [])
        topothe_line.set_data([], [])
        icethen_line.set_data([], [])
        pathline.set_data([], [])
        for coll in (ax.collections):
            ax.collections.remove(coll)
        return icethen_line, toponow_line, topothe_line, pathline

    def animate(i):
        path = paths[i]
        emvalues = -1*emerg_func.ev(path.xs, path.ys)
        loadvalues = load_func.ev(path.xs, path.ys)
        topovalues = topo_func.ev(path.xs, path.ys)
        
        icethen_line.set_data(lonvalues, loadvalues+(topovalues-emvalues))
        toponow_line.set_data(lonvalues, topovalues)
        topothe_line.set_data(lonvalues, topovalues-emvalues)
        pathline.set_data(path.xs, path.ys)
        
        for coll in (ax.collections):
            ax.collections.remove(coll)
            
        ax.fill_between(lonvalues, sea_level, topovalues-emvalues, 
                    where=(np.logical_and(sea_level>(topovalues-emvalues), 
                            sea_level>loadvalues+(topovalues-emvalues))), 
                    facecolor='blue', alpha=0.5)
        
        return icethen_line, toponow_line, topothe_line, pathline
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(frames), interval=30, blit=True)

    anim.save(filename, fps=30)
