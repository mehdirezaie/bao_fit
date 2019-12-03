#!/Users/ding/miniconda3/bin/python

from argparse import ArgumentParser as ap

from mcmc_funs import fit_BAO

if __name__ == '__main__': 
    parser = ap(description='Use mcmc routine to get the BAO peak stretching parameter alpha'\
                           +', damping parameter A, amplitude parameter B.')
    parser.add_argument("--kmin", help = 'kmin fit boundary.', required=True)
    parser.add_argument("--kmax", help = 'kmax fit boundary.', required=True)
    parser.add_argument("--params_str", help = 'Set fitting parameters. 1: free; 0: fixed.', required=True)
    parser.add_argument("--Pk_type", help = "The type of P(k) to be fitted. Pwig: wiggled P(k)"\
                                            +"with BAO; (Pwnw: Pwig-Pnow? Maybe it's not necessary.", required=True)
    args = parser.parse_args()
    fit_BAO(args)
