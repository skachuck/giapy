import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RectBivariateSpline, interp1d

from mpl_toolkits.basemap import Basemap

import giapy.ice_tools.icehistory as icehistory
import giapy.earth_tools.earth_two_d as earth_two_d
from giapy.apl_tools.t_files import read_t_files
from giapy.plot_tools.interp_path import *
from giapy.rebound_twoD import perform_convolution


m = Basemap(width=4900000,height=4700000,\
              rsphere=(6378137.00,6356752.3142),\
              resolution='l',area_thresh=1000.,projection='lcc',\
              lat_1=50.,lat_2=89.9,lat_0=72,lon_0=20.)
            
ice = icehistory.load(u'/Users/skachuck/Documents/Work Documents/GIA_Modeling/ProgramRewrite/IceModels/eur_ice.p')
earth = earth_two_d.load(u'/Users/skachuck/Documents/Work Documents/GIA_Modeling/ProgramRewrite/EarthModels/earth64.p')
earth.calc_taus_from_earth(512)


out_times = np.arange(20,0, -1)
uplift = perform_convolution(earth.taus, earth, ice, out_times)
uplift = uplift-uplift[-1]

Lat1152, Lon1152, topo1152 = read_t_files('', [u'/Users/skachuck/Documents/Work Documents/GIA_Modeling/ice_model/Surf0Load_m_20p_Eu_polyc.txt'], 4)
lon1152 = Lon1152[0,:]*180/np.pi
lat1152 = Lat1152[:,0]*180/np.pi

n = 100
path = map_path([-10, 48], [60, 65], m, n)
path_lonlat = map_path([-10, 48], [60, 65], m, n, lonlat=True)


ices_along_path = np.array([off_grid_profile(ice.lon, ice.lat, ice_heights, path) for ice_heights in ice.heights])
uplifts_along_path = np.array([off_grid_profile(np.linspace(0, 4900000, min(earth.N, 491), endpoint=False), np.linspace(0, 4700000, min(earth.N, 471), endpoint=False), u, path) for u in uplift])
topo_along_path = off_grid_profile(lon1152, lat1152, topo1152, path_lonlat)

ices_interp = RectBivariateSpline(ice.times[::-1], path.ds, ices_along_path[::-1,:])
uplifts_interp = RectBivariateSpline(out_times[::-1], path.ds, uplifts_along_path[::-1,:])

# Meltwater curve
class MyObject(object):
    pass
sealevel_curve = MyObject()
times = np.array([ 0,  6,  7,  8,  
                   9, 10, 11, 12,
                  13, 14, 15, 16, 
                  17, 18, 19, 20])

meters  = np.array([   0,    0,     -4, -14.2,
                     -26,  -44,    -61,   -63,
                     -78,  -85,   -110,  -113,
                    -117, -120, -122.5,  -125])

sealevel_curve = interp1d(times, meters)


def anim_transect(emerg_func, load_func, sl_curve, topo, path, ts, m, filename):
    """Animate the progression of ice stages
    
    anim_emerg(emerg_int, load_int, topo_int, 50, 83, 0.1)
    """
    
    frames = len(ts)

    fig = plt.figure(figsize=(12, 9), dpi=240)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_ylim((-1000, 2500))
    ax.set_xlim((0, path.ds.max()))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    #ax.get_yaxis().tick_left()
    ax.tick_params(axis='both', direction='out')
    ax.get_xaxis().set_ticks([])   # remove unneeded ticks 
    ax.get_yaxis().tick_left()

    ax.set_ylabel('meters above presnt sea level')
    ax.set_xlabel('km along transect')
    
    ice_t = load_func.ev(ts[0]*np.ones(path.n), path.ds)
    u_t = emerg_func.ev(ts[0]*np.ones(path.n), path.ds)
    
    sl_line = ax.axhline(sl_curve.interp(ts[0]))
    sl_line_now = ax.axhline(sl_curve.interp(ts[-1]), color='blue', ls = '--')
    
    ind = [np.argmin(np.abs(path.ds-i*path.ds[-1]/5)) for i in range(1,6)]
    for x in path.ds[ind]:
        ax.axvline(x, color='white')
    
    topo_tdy, = ax.plot(path.ds, topo, color='#B88A00', ls='--', lw=1)
    topo_line, = ax.plot(path.ds, topo+u_t, color='#B88A00', ls='-', lw=1)
    ice_line, = ax.plot(path.ds, topo+u_t+ice_t, color='#33FFCC', ls='-', lw=1)
    
    t_text = ax.text(0.12, 0.97, '', transform=ax.transAxes)
    
    # The inset map
    insetax = fig.add_axes([0.65, 0.65, 0.25, 0.25])
    m.drawcoastlines(linewidth = 0.7)
    pathline = insetax.plot(path.xs, path.ys, color='k', ls='-')
    insetax.plot(path.xs[ind], path.ys[ind], 'ko')

    def init():
        topo_line.set_data([], [])
        ice_line.set_data([], [])
        sl_line.set_data([],[])
        for coll in (ax.collections):
            ax.collections.remove(coll)
        t_text.set_text('')
        return ice_line, topo_line, sl_line, t_text

    def animate(i):
        t = ts[i]
        
        ice_t = load_func.ev(t*np.ones(path.n), path.ds)
        u_t = emerg_func.ev(t*np.ones(path.n), path.ds)
        sea_level = sl_curve.interp(t)
        
        ice_line.set_data(path.ds, topo+u_t+np.maximum(ice_t, 0))
        topo_line.set_data(path.ds, topo+u_t)
        sl_line.set_data([0, 1],[sea_level, sea_level])
        t_text.set_text('t = %1.f ka bp' % t)
        
        for coll in (ax.collections):
            ax.collections.remove(coll)
            
        ax.fill_between(path.ds, sea_level, np.maximum(topo+u_t, topo+u_t+ice_t), 
                    where=(np.logical_and(sea_level>topo+u_t, 
                            sea_level>topo+u_t+ice_t)), 
                    facecolor='blue', alpha=0.5)
                    
        ax.fill_between(path.ds, topo+u_t, -1000, color='#B88A00', alpha = 0.5)
        
        return ice_line, topo_line, sl_line, t_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(frames), interval=30, blit=True)

    anim.save(filename, writer='ffmpeg', fps=30, extra_args=['-vcodec', 'libx264'])


anim_transect(uplifts_interp, ices_interp, sealevel_curve, topo_along_path,
path, np.linspace(20, 0, 300), m, 'transects.gif')



