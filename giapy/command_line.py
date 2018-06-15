from argparse import ArgumentParser, FileType
import sys
import numpy as np

from giapy.earth_tools.elasticlove import compute_love_numbers, hLK_asymptotic
from giapy.earth_tools.viscellove import compute_viscel_numbers
from giapy.earth_tools.earthParams import EarthParams

def ellove():
    """useage: giapy-ellove [-h] [--lstart LSTART] [--params PARAMS]
                            [--nlayers NLAYERS]
                            lmax [outfile]

        Compute the elastic surface load love numbers

        positional arguments:
            lmax               maximum order number to compute
            outfile            file to save out

        optional arguments:
            -h, --help         show this help message and exit
            --lstart LSTART    starting order number (default: 1). Cannot be less than
                               1.
            --params PARAMS    material parameter table
            --nlayers NLAYERS  number of layers (default: 100)
            --incomp           flag for incompressibility (default: False) 
            --conv [CONV]      perform convergence check for asymptotic love
                               number at supplied (very large) l (if flag
                               present, defaults to l=50000).
    """
    # Read the command line arguments.
    parser = ArgumentParser(description='Compute the elastic surface load love numbers')
    parser.add_argument('-l', '--lstart', type=int, default=1,
                        help='starting order number (default: %(default)s). Cannot be less than 1.')
    parser.add_argument('lmax', type=int, 
                        help='maximum order number to compute')
    parser.add_argument('--params', default=None,
                        help="""material parameter table with columns: r (km) 
density (kg/m^3) bulk mod (GPa) shear mod (GPa) g (m/2^2) (default: PREM)""") 
    parser.add_argument('-n', '--nlayers', type=int, default=100,
                        help='number of layers (default: %(default)s)')
    parser.add_argument('outfile', nargs='?', type=FileType('w'),
                        default=sys.stdout,
                        help='file to save out')
    parser.add_argument('--conv', nargs='?', const=50000, default=False,
                        help='''perform convergence check for asymptotic love
number at supplied (very large) l (if present, defaults to l=50000)''')
    parser.add_argument('--incomp', default=False, action='store_const',
                        const=True, help='impose incompressibility')
    args = parser.parse_args()

   
    
    # Set up the order number range
    assert args.lstart >= 1, 'lstart must be 1 or greater.'
    ls = range(args.lstart, args.lmax+1)

    # Load the parameters
    params = EarthParams(modelpath=args.params, normmode='love')

    # If convergence check requested, append to ls.
    if args.conv:
        ls = np.r_[ls, args.conv]

    zarray = np.linspace(params.rCore, 1., args.n)

    # Compute the love numbers.
    hLks = compute_love_numbers(ls, zarray, params, err=1e-14, Q=2,
                                it_counts=False, comp=not args.incomp,
                                scaled=True)

    if args.conv:
        hLk_conv = hLks[:,-1]
        hLk_conv[-1] = args.conv*(1+hLk_conv[-1])
        hLks = hLks[:,:-1]
        

    # Write them out.
    fmt = '{0:'+'{0:.0f}'.format(1+np.floor(np.log10(args.lmax)))+'d}\t{1}\t{2}\t{3}\n'
    # Write out header
    args.outfile.write("n\th'\tl'\tk'\n")
    for l, hLk in zip(ls, hLks.T):
        args.outfile.write(fmt.format(l, hLk[0], hLk[1]/l, -(1+hLk[2])))

    if args.conv:
        hLk_inf = np.array(hLK_asymptotic(params))
        errs = np.abs(hLk_conv - hLk_inf)
        sys.stdout.write('''Difference of computed love numbers at {} from 
analytic value (if too large, consider increasing
layers with '--nlayers'):\n'''.format(args.conv))
        for tag, err in zip('hLK', errs):
            sys.stdout.write('\t{} : {}\n'.format(tag, err))
            
#def velove():
if __name__ == '__main__':
    """useage: giapy-velove [-h] [--lstart LSTART] [--params PARAMS]
                            [--nlayers NLAYERS]
                            lmax [outfile]

        Compute the viscoelastic surface load love numbers

        positional arguments:
            lmax               maximum order number to compute
            outfile            file to save out

        optional arguments:
            -h, --help         show this help message and exit
            --lstart LSTART    starting order number (default: 1). Cannot be less than
                               1.
            --params PARAMS    material parameter table
            --nlayers NLAYERS  number of layers (default: 100)
            --incomp           flag for incompressibility (default: False) 
            --conv [CONV]      perform convergence check for asymptotic love
                               number at supplied (very large) l (if flag
                               present, defaults to l=50000).
    """
    # Read the command line arguments.
    parser = ArgumentParser(description='Compute the elastic surface load love numbers')
    parser.add_argument('-l', '--lstart', type=int, default=1,
                        help='starting order number (default: %(default)s). Cannot be less than 1.')
    parser.add_argument('lmax', type=int, 
                        help='maximum order number to compute')
    parser.add_argument('--params', default=None,
                        help="""material parameter table with columns: r (km) 
density (kg/m^3) bulk mod (GPa) shear mod (GPa) g (m/2^2) (default: PREM)""") 
    parser.add_argument('-n', '--nlayers', type=int, default=1000,
                        help='number of layers (default: %(default)s)')
    parser.add_argument('outfile', nargs='?', type=FileType('w'),
                        default=sys.stdout,
                        help='file to save out')
    parser.add_argument('--conv', nargs='?', const=50000, default=False,
                        help='''perform convergence check for asymptotic love
number at supplied (very large) l (if present, defaults to l=50000)''')
    parser.add_argument('--incomp', default=False, action='store_const',
                        const=True, help='impose incompressibility')
    parser.add_argument('--lith', '-D', type=float, default=0., dest='lith',
                        help='''The flexural rigidit of the lithosphere, in units
of 1e23 N m (overrides parameter table, if set)''')
    args = parser.parse_args()

   
    
    # Set up the order number range
    assert args.lstart >= 1, 'lstart must be 1 or greater.'
    ls = range(args.lstart, args.lmax+1)

    paramname = args.params or 'prem'
    # Load the parameters with no crust for viscoelastic response
    params = EarthParams(model=paramname+'_nocrust')
    # Check for lithospheric override
    if args.lith:
        params.addLithosphere(D=args.lith)

    zarray = np.linspace(params.rCore, 1., args.nlayers)
    times = np.logspace(-3,2,20)
    # Compute the viscoelastic Love numbers.
    hLkf = compute_viscel_numbers(ls, times, zarray, params,
                                comp=not args.incomp, scaled=True)


    # Load the parameters with crust for elastic response.
    params_crust = EarthParams(model=paramname)
    # Compute the elastic response for lithosphere correction.
    hLke = compute_love_numbers(ls, zarray, params_crust, err=1e-14, Q=2,
                                it_counts=False, comp=not args.incomp,
                                scaled=True).T

    # Incorporate the lithosphere correction.
    a = (1. - 1./ params.getLithFilter(n=np.asarray(ls)))
    hLkf += a[:,None,None]*hLke[:,:,None]
    hLkf[:,2,:] = -1 - hLkf[:,2,:]

    # Add t=0 elastic response
    hLkf = np.dstack([hLke[:,:,None], hLkf])
    

    # Write them out.
    fmt = '{0:'+'{0:.0f}'.format(1+np.floor(np.log10(args.lmax)))+'d}\t{1}\t{2}\t{3}\n'
    fmt = '{0}\t{1}\t{2}\t{3}\n'
    # Write out header
    args.outfile.write("t\th'\tl'\tk'\n")
    for l, hLkl in zip(ls, hLkf):
        args.outfile.write('# l={}\n'.format(l))
        for t, hLk in zip(np.r_[0,times], hLkl.T):
            args.outfile.write(fmt.format(t, hLk[0], hLk[1]/l, -(1+hLk[2])))
