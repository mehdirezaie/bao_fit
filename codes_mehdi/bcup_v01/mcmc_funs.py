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


def gelman_rubin_convergence(withinchainvar, meanchain, n, Nchains, ndim):
    """
    Gelman&Rubin convergence criterion, 
    referenced from Florian's code RSDfit_challenge_hex_steps_fc_hoppper.py

     Calculate Gelman & Rubin diagnostic
     1. Remove the first half of the current chains
     2. Calculate the within chain and between chain variances
     3. estimate your variance from the within chain and between chain variance
     4. Calculate the potential scale reduction parameter
    """
    meanall = np.mean(meanchain, axis=0)
    W       = np.mean(withinchainvar, axis=0)
    B       = np.zeros(ndim, dtype=np.float)
    for jj in range(0, Nchains):
        B = B + n*(meanall - meanchain[jj])**2/(Nchains-1.)
    estvar = (1. - 1./n)*W + B/n
    scalereduction = np.sqrt(estvar/W)
    return scalereduction



def write_params(filename, params_mcmc, params_name, reduced_chi2):
    """
      write parameters fitted in files
    """
    header_line = '# The fitted parameters {} (by row) with their upward'\
                + ' and downward one sigma error, and the reduced \chi^2.\n'\
                .format(params_name[:])

    with open(filename, 'w') as fwriter:
        fwriter.write(header_line)
        for i in range(len(params_name)):
            fwriter.write("#{0}: {1:.7f} {2:.7f} {3:.7f}\n"\
                   .format(params_name[i], params_mcmc[i][0],
                           params_mcmc[i][1], params_mcmc[i][2]))
        fwriter.write("#reduced_chi2: {0:.7f}".format(reduced_chi2))



def set_params(all_params, params_indices, all_names, all_temperature):
    """
     Define a function to set parameters which are free and which are fixed.
    """
    fix_params = np.array([], dtype=np.float)
    theta = np.array([], dtype=np.float)
    params_T = np.array([], dtype=np.float)
    params_name = []
    N_params = 0
    count = 0
    for i in params_indices:
        if i == 1:
            fix_params = np.append(fix_params, 0.)
            theta = np.append(theta, all_params[count])
            params_T = np.append(params_T, all_temperature[count])
            params_name.append(all_names[count])
            N_params += 1
        else:
            fix_params = np.append(fix_params, all_params[count])
        count += 1
    print(theta, params_name, N_params)
    print("fixed params: ", fix_params)
    return N_params, theta, fix_params, params_T, params_name


"""
 Input observed Pk_obs(k') = Pwig(k').
 If the observed and input power spectrum have different redshifts,
 we need norm_gf as the normalized growth factor D(z). Here since we introduce the parameter A,
 we don't need growth function. But still keep the parameter in the code now. -- 09/25/2017
"""
def lnlike(theta, params_indices, fix_params, k_p, Pk_obs, ivar, Pwig_spl, Psm_spl, norm_gf):
    alpha, Sigma2_xy, A, B = match_params(theta, params_indices, fix_params)
    coeff = 1.0/alpha
    k_t = k_p*coeff        # k=k'/alpha
    Pk_linw = Pwig_spl(k_t)
    Pk_sm = Psm_spl(k_t)
    Pk_model = cal_pk_model(Pk_linw, Pk_sm, k_t, Sigma2_xy, A, B)
    diff = Pk_model - Pk_obs
    return -0.5* np.dot(diff**2.0, ivar)

def lnprob(theta, params_indices, fix_params, k_p, Pk_obs, ivar, Pwig_spl, Psm_spl, norm_gf):
    lp = lnprior(theta, params_indices, fix_params)
    if (lp < -1.e20):
        return -np.inf
    return lp + lnlike(theta, params_indices, fix_params, k_p,
                       Pk_obs, ivar, Pwig_spl, Psm_spl, norm_gf)

# Find the maximum likelihood value.
chi2 = lambda *args: -2 * lnlike(*args)


# MCMC routine
def mcmc_routine(ndim, N_walkers, theta, params_T, params_indices, 
                 fix_params, k_range, Pk_obs, ivar_Pk_obs, Pwig_spl, 
                 Psm_spl, norm_gf, params_name, pool):
    ti = time.time()

    Nchains = 4
    minlength = 800
    epsilon = 0.01
    ichaincheck = 50
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    result = op.minimize(chi2, theta, args=(params_indices, fix_params, k_range,\
                         Pk_obs, ivar_Pk_obs, Pwig_spl, Psm_spl, norm_gf),\
                         method='Powell')
    theta_optimize = result["x"]

    # only output parameters which are free to change
    print("Parameters from Powell optimization: ", theta_optimize)
    theta_optimize = theta
    print("Initial parameters for MCMC: ", theta_optimize)

    pos = []
    sampler = []
    rstate = np.random.get_state()
    # Set up the sampler.
    for jj in range(Nchains):
        pos.append([theta_optimize + params_T*np.random.uniform(-1.0, 1.0, ndim) for i in range(N_walkers)])
        sampler.append(emcee.EnsembleSampler(N_walkers, ndim, lnprob, a=2.0,\
                args=(params_indices, fix_params, k_range, Pk_obs, ivar_Pk_obs,\
                      Pwig_spl, Psm_spl, norm_gf), pool=pool))
    print(type(sampler))

    # Clear and run the production chain.
    print("Running MCMC...")

    withinchainvar = np.zeros((Nchains,ndim))
    meanchain = np.zeros((Nchains,ndim))
    scalereduction = 2.*np.ones(ndim,dtype=np.float)
    itercounter = 0
    chainstep = minlength
    loopcriteria = 1
    num_iteration = 1
    while loopcriteria and num_iteration<50:
        itercounter = itercounter + chainstep
        print("chain length =",itercounter," minlength =",minlength)

        for jj in range(Nchains):
            # Since we write the chain to a file we could put storechain=False, but in that case
            # the function sampler.get_autocorr_time() below will give an error
            for result in sampler[jj].sample(pos[jj], iterations=chainstep,\
                                             rstate0=np.random.get_state(),\
                                             storechain=True, thin=1):
                pos[jj] = result[0]
                #print(pos)
                chainchi2 = -2.*result[1]
                rstate = result[2]

            # we do the convergence test on the second half of the current chain (itercounter/2)
            chainsamples = sampler[jj].chain[:, itercounter//2:, :].reshape((-1, ndim))
            #print("len chain = ", chainsamples.shape)
            withinchainvar[jj] = np.var(chainsamples, axis=0)
            meanchain[jj] = np.mean(chainsamples, axis=0)

        scalereduction = gelman_rubin_convergence(withinchainvar, meanchain,\
                                                  itercounter//2, Nchains, ndim)
        print("scalereduction = ", scalereduction)
        loopcriteria = 0
        for jj in range(0, ndim):
            if np.absolute(1.0-scalereduction[jj]) > epsilon:
                loopcriteria = 1
        chainstep = ichaincheck
        num_iteration = num_iteration + 1
    print("Done.")

    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker so, in this case, it is a 250-dimensional vector.
    for jj in range(0, Nchains):
        print("Mean acceptance fraction for chain ", jj,": ", np.mean(sampler[jj].acceptance_fraction))
    # Estimate the integrated autocorrelation time for the time series in each parameter.
    #for jj in range(0, Nchains):
    #    print("Autocorrelation time for chain ", jj,": ", sampler[jj].get_autocorr_time())
    ###################################
    ## Compute the quantiles ##########
    ###################################

    #samples=[]
    mergedsamples=[]

    for jj in range(0, Nchains):
        #samples.append(sampler[jj].chain[:, itercounter/2:, :].reshape((-1, ndim)))
        mergedsamples.extend(sampler[jj].chain[:, itercounter//2:, :].reshape((-1, ndim)))
    print("length of merged chain = ", sum(map(len,mergedsamples))//ndim)

    theta_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
                     zip(*np.percentile(mergedsamples, [15.86555, 50, 84.13445], axis=0)))
    theta_mcmc = list(theta_mcmc)

    print("MCMC result: ")
    for i in range(len(theta)):
        print("{0}={1[0]}+{1[1]}-{1[2]}".format(params_name[i], theta_mcmc[i]))

    del sampler
    tf = time.time()
    print("One mcmc running set time: ", tf-ti)
    return np.array(theta_mcmc)




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

            params_mcmc = mcmc_routine(N_params, N_walkers, theta, params_T, params_indices, fix_params, k_obs, Pk_wig_obs, ivar_Pk_wig, Pwig_spl, Psm_spl, norm_gf, params_name, pool)
            print(params_mcmc)
            chi_square = chi2(params_mcmc[:, 0], params_indices, fix_params, k_obs, Pk_wig_obs, ivar_Pk_wig, Pwig_spl, Psm_spl, norm_gf)
            reduced_chi2 = chi_square/(N_fitbin-N_params)
            print("chi^2/dof: ", reduced_chi2, "\n")
            ofile_params = odir + 'fit_p0_z{}RADECcut{}_params{}.dat'.format(z_value, cut_type, params_str)
            write_params(ofile_params, params_mcmc, params_name, reduced_chi2)

    pool.close()

    sys.stdout = old_stdout
    log_file.close()

