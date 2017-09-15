from argparse import ArgumentParser, FileType
import sys
import numpy as np

from giapy.earth_tools.elasticlove import compute_love_numbers, exp_pt_density
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
            --nlayers NLAYERS  number of layers (default: 50)
    """
    # Read the command line arguments.
    parser = ArgumentParser(description='Compute the elastic surface load love numbers')
    parser.add_argument('--lstart', type=int, default=1,
                        help='starting order number (default: %(default)s). Cannot be less than 1.')
    parser.add_argument('lmax', type=int, 
                        help='maximum order number to compute')
    parser.add_argument('--params', default=None,
                        help="""material parameter table with columns: r (km) 
density (kg/m^3) bulk mod (GPa) shear mod (GPa) g (m/2^2) (default: PREM)""") 
    parser.add_argument('--nlayers', type=int, default=50,
                        help='number of layers (default: %(default)s)')
    parser.add_argument('outfile', nargs='?', type=FileType('w'),
                        default=sys.stdout,
                        help='file to save out')
    args = parser.parse_args()

   
    
    # Set up the order number range
    assert args.lstart >= 1, 'lstart must be 1 or greater.'
    ls = range(args.lstart, args.lmax+1)

    # Load the parameters
    params = EarthParams(modelpath=args.params) 

    # Generate the l-dependent exponential layer depth function.
    def kbase(l, args, rCore):
        #TODO: address this semi-empircal kluge
        l = max(min(l, 5000), 2)
        return exp_pt_density(args.nlayers, 2*np.log(l)/(2.*l+1.), rCore, 1.)

    # Compute the love numbers.
    hLks = compute_love_numbers(ls, kbase, params, err=1e-14, Q=2,
                                it_counts=False, zgen=True,
                                args=[args,params.rCore])

    # Write them out.
    fmt = '{0:'+'{0:.0f}'.format(1+np.floor(np.log10(args.lmax)))+'d}\t{1}\t{2}\t{3}\n'
    for l, hLk in zip(ls, hLks.T):
        args.outfile.write(fmt.format(l, hLk[0], hLk[1], l*(1+hLk[2]))) 

if __name__ == '__main__':
    ellove()
