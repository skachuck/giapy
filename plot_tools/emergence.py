import numpy as np
import matplotlib.pyplot as plt

#class CompEmergTable(object):
    #    def __init__(self, ):
    #    self.times
    #    self.data = np.loadtxt(datapath+model+str(loc['recnbr'])+'.txt', delimiter=',')
    #    self.data = data[-len(self.times):,:]

dcolor1 = '#4d4e53'     # cornell grey
mcolor1 = '#a2998b'     # light brown
accent1 = '#ef4035'     # red
accent2 = '#0093d0'     # blue

modelplot = {'ls':'-', 'lw':2.5, 'alpha':0.7}
pointplot = {'marker':'+', 'ls':'None', 'mfc':'None', 'ms':12, 'mew':1.2}
colors = ['r', 'purple', 'orange', 'green']

def comp_emerg_locs(models, colors=None, insetmap=None):
    # axis configuration
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(-15,0)
    ax.set_xticks([-15, -10, -5, 0])
    ax.set_xticklabels([15, 10, 5, 0])
    ax.set_xlabel('ka cal bp', fontsize=16)
    ax.set_ylim(-5, 150)
    ax.set_ylabel('Emergence (m)', fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
    ax.get_yaxis().tick_left()
    ax.axhline(0, color='k', ls='-')
    
    if insetmap is not None:
        inax = fig.add_axes([0.5, 0.5, 0.4, 0.4])
        inax.axis('off')
        insetmap.drawcoastlines(linewidth = 0.7)
    
    for loc, color in zip(sval_data, colors):
        # import data
        data = np.loadtxt(model_path+str(loc['recnbr'])+'.txt', 
                            delimiter=',')
        data = data[-len(times):,:]
        # plot
        ax.plot(times[::-1], data[:,0], c=color, **modelplot)
        ax.plot(-np.array(loc['data_dict']['times']), 
                          loc['data_dict']['emerg'],
                            color=color, **pointplot)
        # plot point in inset
        if insetmap is not None:
            x, y = insetmap(loc['lon'], loc['lat'])
            inax.plot(x, y, marker='o', ms=8, color=color)
            
    return fig

def plotEmergeGrid(data, models, insetmap=None):
    raise NotImplementedError()
    
def comp_emerg_model(locs, colors=None, insetmap=None):
    
    for model,color in zip(models, ['r','purple']):
        for loc in sval_emerge_data:
            fig, ax = plt.subplots(1,1)
            # axis configuration
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlim(-15,0)
            ax.set_xticks([-15, -10, -5, 0])
            ax.set_xticklabels([15, 10, 5, 0])
            ax.set_xlabel('ka cal bp', fontsize=16)
            ax.set_ylim(-5, 250)
            ax.set_ylabel('Emergence (m)', fontsize=16)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
            ax.get_yaxis().tick_left()
            # plot
            ax.plot(times, data[:,0], c=color, **modelplot)
            ax.plot(-np.array(loc['data_dict']['times']), 
                              loc['data_dict']['emerg'], **pointplot)
            # Legend fix
            auth = loc['auth']
            auth = auth.replace('\xf8', 'o')
            ax.text(0.3, 0.9, '+ from '+auth, transform=ax.transAxes, 
                        fontsize=16)
            place = loc['desc'][1:-1]
            place = place.replace('\xf8', 'o')
            
            ax.text(0.3, 0.82, place+' '+str(loc['recnbr']), 
                        transform=ax.transAxes, fontsize=16)

            fig.savefig(model+str(loc['recnbr'])+'.jpg')

def plotEmergenceLoc(emergeDatum, simobject):
    # Extract attributes from simobject
    uplift = simobject.uplift
    out_times = simobject.out_times
    grid = simobject.grid
    
    # Use map coordinates from sim.grid.basemap for interpolation
    x, y = grid.basemap(emergeDatum['lon'], emergeDatum['lat'])

    # interp_data will be an array of size (N_output_times, N_locations)
    # for use in interpolating the calculated emergence to the locations
    # and times at which there are data in data
    interp_data = []
    # Interpolate the calculated uplift at each time on the Lat-Lon grid
    # to the data locations.
    for uplift_at_a_time in uplift:
        interp_func = grid.create_interper(uplift_at_a_time.T)
        interp_data.append(interp_func.ev(x, y))
    
    # A whole lot of set up
    fig, ax = plt.subplots(1, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_smart_bounds(True)
    ax.spines['left'].set_smart_bounds(True)
    ax.get_xaxis().tick_bottom()   # remove unneeded ticks
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='x', colors=dcolor1)
    ax.tick_params(axis='y', colors=dcolor1)
    ax.spines['left'].set_color(dcolor1)
    ax.spines['bottom'].set_color(dcolor1)
    ax.yaxis.label.set_color(dcolor1)
    ax.xaxis.label.set_color(dcolor1)

    
    #ax.set_xlim(-15,0)
    #ax.set_xticks([-15, -10, -5, 0])
    #ax.set_xticklabels([15, 10, 5, 0])
    #ax.set_ylim(-25, 150)
    #ax.set_yticklabels([-25, 0, 50, 100, 150])
         
    ax.plot(-np.array(emergeDatum['data_dict']['times']), 
            emergeDatum['data_dict']['emerg'], 
            **pointplot)
    
    ax.plot(-out_times, interp_data)

    ax.text(0.9, 0.9, unicode(emergeDatum['desc'], 'utf-8', 'ignore'),
            ha='right', transform=ax.transAxes, fontsize=14, color=dcolor1)

    return plt.gca()
