"""
    Copy the code mcmc_fit_nonlinear_BAO from /Users/ding/Documents/playground/shear_ps/SVD_ps/mcmc_fit
    modify it to fit observed power spectrum Pwig with damped BAO.
    Modify the function cal_pk_model in the lnprob module for the fitting model:
    Pwig(k') = A*((Plin(k) - Psm(k))*exp(-k^2*Sigma^2/2) + Psm(k)) + B; --09/25/2017
"""
import emcee
from emcee.utils import MPIPool
from mpi4py import MPI
import time
import numpy as np
import scipy.optimize as op
from scipy import linalg
from scipy.interpolate import InterpolatedUnivariateSpline
from functools import reduce
import os, sys
from lnprob_nonlinear import match_params, cal_pk_model, lnprior
import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
import argparse
from growth_fun import growth_factor





















# Fit extracted power spectrum from shear power spectrum. We use Pwig/Pnow as the observable.
def fit_BAO(args):

    kmin = float(args.kmin)
    kmax = float(args.kmax)
    params_str = args.params_str
    Pk_type = args.Pk_type
    params_indices = [int(i) for i in params_str]

    old_stdout = sys.stdout
    odir = './fit_kmin{}_kmax{}_{}/'.format(kmin, kmax, Pk_type)
    if not os.path.exists(odir):
        os.makedirs(odir)

    ofile = odir + "mcmc_fit_params{}.log".format(params_str)
    log_file = open(ofile, "w")
    sys.stdout = log_file
    print('Arguments for the fitting: ', args)

    ifile = '/Users/mehdi/work/quicksurvey/ELG/run8/planck_camb_56106182_matterpower_z0.dat'
    klin, Pk_linw = np.loadtxt(ifile, dtype='f8', comments='#', unpack=True)
    Pwig_spl = InterpolatedUnivariateSpline(klin, Pk_linw)

    ifile = '/Users/mehdi/work/quicksurvey/ELG/run8/planck_camb_56106182_matterpower_smooth_z0.dat'
    klin, Pk_sm = np.loadtxt(ifile, dtype='f8', comments='#', unpack=True)
    Psm_spl = InterpolatedUnivariateSpline(klin, Pk_sm)

    norm_gf = 1.0
    N_walkers = 40
    ##params_indices = [1, 0, 0]  # 1: free parameter; 0: fixed parameter

    all_param_names = 'alpha', 'Sigma2_xy', 'A', 'B'
    all_temperature = 0.01, 1.0, 0.1, 0.1
    Omega_m = 0.3075              # matter density
    G_0 = growth_factor(0.0, Omega_m)
    Sigma_0 = 7.7840              # This is exactly calculated from theoretical prediction with q_{BAO}=110 Mpc/h.

    z_list = [0.625]#, 0.875, 1.125, 1.375]   # z list for the data files
    cut_list = ['F']#, 'T']
    # initial guess for fitting, Sigma2_xy=31.176 at z=0.65 is from theory prediction
    alpha, A, B = 1.0, 1.0, 10.0
    idir = '../kp0kp2knmodes/surveyscaled-nmodes/'
    odir = './mcmc_fit_params_{}/kmin{}_kmax{}/'.format(Pk_type, kmin, kmax)


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if not os.path.exists(odir):
            os.makedirs(odir)

    pool = MPIPool(loadbalance=True)
    for z_value in z_list:
        norm_gf = growth_factor(z_value, Omega_m)/G_0
        Sigma2_xy = (Sigma_0* norm_gf)**2.0
        print('z, Sigma2_xy: ', z_value, Sigma2_xy)
        all_params = alpha, Sigma2_xy, A, B
        N_params, theta, fix_params, params_T, params_name = set_params(all_params, params_indices, all_param_names, all_temperature)
        for cut_type in cut_list:
            ifile = idir + 'kp0kp2knmodes_z{}RADECcut{}.dat'.format(z_value, cut_type)
            print(ifile)
            data_m = np.loadtxt(ifile, dtype='f8', comments='#') # k, P0(k), P2(k), N_modes
            indices = np.argwhere((data_m[:,0] >= kmin) & (data_m[:,0] <= kmax)).flatten()
            N_fitbin = len(indices)
            k_obs, Pk_wig_obs, N_modes = data_m[indices, 0], data_m[indices, 1], data_m[indices, 3]
            ivar_Pk_wig = N_modes/(2.0 * Pk_wig_obs**2.0)
            #print('ivar_Pk_wig', ivar_Pk_wig)

            params_mcmc = mcmc_routine(N_params, N_walkers, theta, params_T,\
                                       params_indices, fix_params, k_obs, Pk_wig_obs,\
                                       ivar_Pk_wig, Pwig_spl, Psm_spl, norm_gf, params_name, pool)
            print(params_mcmc)
            chi_square = chi2(params_mcmc[:, 0], params_indices, fix_params,\
                              k_obs, Pk_wig_obs, ivar_Pk_wig, Pwig_spl, Psm_spl, norm_gf)
            reduced_chi2 = chi_square/(N_fitbin-N_params)
            print("chi^2/dof: ", reduced_chi2, "\n")
            ofile_params = odir + 'fit_p0_z{}RADECcut{}_params{}.dat'.format(z_value, cut_type, params_str)
            write_params(ofile_params, params_mcmc, params_name, reduced_chi2)

    pool.close()
    sys.stdout = old_stdout
    log_file.close()
