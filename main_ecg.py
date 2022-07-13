# %%
import time

from numpy import empty_like
from Linear_sysmdl import System_Model_nti
from ecg_tester import test_local, test_KS, test_LFKS
from RTS_NUV import RTS_UV_nti
import torch
from local_fit import LocalFitter, ExpWin_DS, moving_avg, LCR, merge
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
from gen_ecg import range_prec_r_dB
from multiprocessing import Pool
import time
import os
import itertools

# %%
# number of traj
NB = 1000

# load data to be filtered
ppath = 'Sim_ecg/' # parent path
traj_path = ppath + 'traj/' # traj path
Y_ALL = torch.load(traj_path+'ECG_noisy.pt')[:, 0:NB, :]
ecg_gt = torch.load(traj_path+'ECG_gt.pt')

T_ecg = ecg_gt.size(0)

# optimal window parameter
gamma_optm = [0.85, 0.7, 0.45, 0.2, 0.1]

# MSEs of pure LF
MSEs_LF = torch.empty(size=[len(range_prec_r_dB)])
std_MSE_LF = torch.empty(size=[len(range_prec_r_dB)])
# MSEs of pure KS
MSEs_KS = torch.empty(size=[len(range_prec_r_dB)])
std_MSE_KS = torch.empty(size=[len(range_prec_r_dB)])
# MSEs of LF + KS
MSEs_LFKS = torch.empty(size=[len(range_prec_r_dB)])
std_MSE_LFKS = torch.empty(size=[len(range_prec_r_dB)])

# Smooting result of pure LF
Y_LF = torch.empty_like(Y_ALL)
# Smooting result of pure KS
Y_KS = torch.empty_like(Y_ALL)
# Smooting result of LF + KS
Y_LFKS = torch.empty_like(Y_ALL)

# %%
def worker_LFKS(prec_r_dB):
    # get index
    idx_r = range_prec_r_dB.index(prec_r_dB)
    [ # test the current batch
        MSE, 
        std, 
        Y_smth
    ] = test_LFKS(
                NB, 
                test_input=Y_ALL[idx_r, :, :], 
                test_target=ecg_gt, 
                gamma=gamma_optm[idx_r]
        )
    return [idx_r, MSE, std, Y_smth] 


# Worker process for pure KS
def worker_KS(prec_r_dB):
    # get index
    idx_r = range_prec_r_dB.index(prec_r_dB)
    [ # test the current batch
        MSE, 
        std, 
        Y_smth
    ] = test_KS(
                NB, 
                test_input=Y_ALL[idx_r, :, :], 
                test_target=ecg_gt, 
        )
    return [idx_r, MSE, std, Y_smth] 


# Worker process for pure local fitting
def worker_LF(prec_r_dB):
    # get index
    idx_r = range_prec_r_dB.index(prec_r_dB)
    
    # split linear model
    A = torch.tensor([[1.0, 1.0],
                    [0.0, 1.0]])
    C = torch.tensor([1.0, 0.0]).unsqueeze(0)
    split_linear_model = LocalFitter(A, C, ExpWin_DS(), A, C)

    pm_conti_y = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])    
    
    [ # test the current batch
        MSE, 
        std, 
        Y_smth
    ] = test_local(
                NB, 
                test_input = Y_ALL[idx_r, :, :], 
                test_target = ecg_gt,
                fitter = split_linear_model, 
                win_param_fwd = gamma_optm[idx_r], 
                win_param_bwd = gamma_optm[idx_r], 
                post_mult = pm_conti_y
        )
    return [idx_r, MSE, std, Y_smth]

# %%
################################
###### main for LF + KS ########
################################
def main_LFKS(opath=ppath+'LFKS/'):
    start = time.time()
    # PC
    core_count = os.cpu_count()
    try:
        # Euler HPC
        core_count = int(os.environ['LSB_DJOB_NUMPROC'])
    except:
        pass

    # parrallel computing for each r
    pool = Pool(core_count)
    results = pool.map(worker_LFKS, range_prec_r_dB)
    pool.close()
    pool.join()

    # get results
    for res in results:
        idx_r = res[0]
        [
            MSEs_LF[idx_r],
            std_MSE_LF[idx_r],
            Y_LF[idx_r]
        ] = res[1:]
    print("Total Time - multi processing:", time.time()-start)
    save_data(Y_LF, MSEs_LF, std_MSE_LF, opath)
    save_plot(lf_path=ppath+'LF/', lfks_path=ppath+'LFKS/')

################################
######### main for KS ##########
################################
def main_KS(opath='Sim_ecg/KS/'):
    start = time.time()
    # PC
    core_count = os.cpu_count()
    try:
        # Euler HPC
        core_count = int(os.environ['LSB_DJOB_NUMPROC'])
    except:
        pass

    # parrallel computing for each r
    pool = Pool(core_count)
    results = pool.map(worker_KS, range_prec_r_dB)
    pool.close()
    pool.join()

    # get results
    for res in results:
        idx_r = res[0]
        [
            MSEs_LF[idx_r],
            std_MSE_LF[idx_r],
            Y_LF[idx_r]
        ] = res[1:]
    print("Total Time - multi processing:", time.time()-start)
    save_data(Y_LF, MSEs_LF, std_MSE_LF, opath)
    save_plot(lf_path=ppath+'LF/', lfks_path=ppath+'LFKS/')

################################
######### main for LF ##########
################################
def main_LF(opath='Sim_ecg/LF/'):
    start = time.time()
    # PC
    core_count = os.cpu_count()
    try:
        # Euler HPC
        core_count = int(os.environ['LSB_DJOB_NUMPROC'])
    except:
        pass

    # parrallel computing for each r
    pool = Pool(core_count)
    results = pool.map(worker_LF, range_prec_r_dB)
    pool.close()
    pool.join()

    # get results
    for res in results:
        idx_r = res[0]
        [
            MSEs_LF[idx_r],
            std_MSE_LF[idx_r],
            Y_LF[idx_r]
        ] = res[1:]
    print("Total Time - multi processing:", time.time()-start)
    save_data(Y_LF, MSEs_LF, std_MSE_LF, opath)
    save_plot(lf_path=ppath+'LF/', lfks_path=ppath+'LFKS/')


def main_LF_sp(opath='Sim_ecg/LF/'):
    A = torch.tensor([[1.0, 1.0],
                    [0.0, 1.0]])
    C = torch.tensor([1.0, 0.0]).unsqueeze(0)
    split_linear_model = LocalFitter(A, C, ExpWin_DS(), A, C)
    gamma = 0.85

    pm_conti_y = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    for idx_r, prec_r_dB in enumerate(range_prec_r_dB):
        [
            MSEs_LF[idx_r], std_MSE_LF[idx_r], Y_LF[idx_r, :, :]
        ] = test_local(
            NB, test_input = Y_ALL[idx_r, :, :], test_target = ecg_gt,
            fitter = split_linear_model, 
            win_param_fwd = gamma, win_param_bwd = gamma, post_mult = pm_conti_y
        )
    save_data(Y_LF, MSEs_LF, std_MSE_LF, opath)


def save_data(Y_smooth, MSEs, std_MSE, opath):
    torch.save(Y_smooth, opath+'Y_smth.pt')
    torch.save(MSEs, opath+'MSEs.pt')
    torch.save(std_MSE, opath+'std_MSE.pt')


def save_plot(lf_path=ppath+'KS/', ks_path=ppath+'KS/', lfks_path=ppath+'LFKS/'):
    plt.figure()
    plt.title('Evaluation of Smoothing ECG Signals')
    plt.xlabel(r'$\frac{1}{r^2}$ in dB')
    plt.ylabel('MSE of Smoothing Algorithms [dB]')
    # Noise floor
    plt.plot(range_prec_r_dB, [ -x for x in range_prec_r_dB], '--', c='r', linewidth=0.75, label='Noise floor')
    # Plot 
    MSEs_LF = torch.load(lf_path+'MSEs.pt')
    MSEs_KS = torch.load(ks_path+'MSEs.pt')
    MSEs_LFKS = torch.load(lfks_path+'MSEs.pt')
    plt.plot(range_prec_r_dB, MSEs_LF, '*-', c='orange', linewidth=0.75, label='LF')
    plt.plot(range_prec_r_dB, MSEs_KS, 'o-', c='c', linewidth=0.75, label='KS')
    plt.plot(range_prec_r_dB, MSEs_LFKS, '^-', c='b', linewidth=0.75, label='LF+KS')
    
    plt.legend()
    plt.grid()
    plt.savefig(ppath+'MSE.pdf')
    plt.close()

# %%
if __name__ == '__main__':
    # main_LF()
    main_KS()
    main_LFKS()






