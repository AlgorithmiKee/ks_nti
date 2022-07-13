# This py file runs input estimation (base line) for different set of parameters.
# For each set of nu(q^2/r^2) and r, the RTS smoother 
#   estimates an input trajecoty (100 time steps) 
#   NB=1000 times
#   assuming that RTS smoother knows the true q
# To the model, we assume that
#   G=H=I_2
#   Q=q^2*I_2, R=r^2*I_2
import io
import torch
import matplotlib.pyplot as plt
from Linear_sysmdl import System_Model_NE
from RTS_Smoother_test import RTS_Tester_NE
from gen_traj_sin_Q import periodic_Q
from gen_traj_sin_R import periodic_R
from param_lin import *
from math import pow
from multiprocessing import Pool
import os
import itertools
from functools import partial
import time
from export_fig import save_ksplot



# number of traj
NB = 1000

# load data and ground truth
# traj_path = 'Sim_sin_R/Pd100/traj/'
traj_path = 'Sim_sin_Q/Pd100/traj/'

Y_ALL = torch.load(traj_path+'Y_ALL.pt')[:, :, 0:NB, :, :]
U_ALL = torch.load(traj_path+'U_ALL.pt')[:, :, 0:NB, :, :]
X_ALL = torch.load(traj_path+'X_ALL.pt')[:, :, 0:NB, :, :]

# save mean MSE and std MSE
MSEs_state_rts = torch.empty((len(range_nu_db), len(range_prec_r_db)))
MSEs_input_rts = torch.empty((len(range_nu_db), len(range_prec_r_db)))
std_MSE_state_rts = torch.empty((len(range_nu_db), len(range_prec_r_db)))
std_MSE_input_rts = torch.empty((len(range_nu_db), len(range_prec_r_db)))

# save estimated traj
X_rts = torch.empty_like(X_ALL)
Sigma_rts = torch.empty(size=[len(range_nu_db), len(range_prec_r_db), NB, dim_x, dim_x, T])
U_rts, Q_rts = torch.empty_like(X_rts), torch.empty_like(Sigma_rts)


# helper for worker process
def true_Q(nu_db, prec_r_db):
    q = pow(10, (nu_db-prec_r_db)/20)
    Q = q * q * torch.eye(dim_x)
    return Q

# helper for worker process
def true_R(prec_r_db):
    r = pow(10, -prec_r_db/20)
    R = r*r*torch.eye(dim_y)
    return R

# Worker Process
# nu_r: list of [nu, r] in dB
# q_pattern: 'const': Q(t)=q2*I  (time-invariant)
#            'sin': Q(t)=q2(t)*I, q2(t)=q^2[1+0.5sin(\omgea t)] (sinusoidal)
# r_pattern: 'const': R(t)=r2*I  (time-invariant)
#            'sin': R(t)=r2(t)*I, r2(t)=r^2[1+0.5sin(\omgea t)] (sinusoidal)
def test_nebatch(nu_r, q_pattern, r_pattern, Pdq, Pdr):
    nu_db, prec_r_db = nu_r[0], nu_r[1]
    # get index
    idx_nu = range_nu_db.index(nu_db)
    idx_r = range_prec_r_db.index(prec_r_db)

    # convert dB to linear scaling
    r = pow(10, -prec_r_db/20)
    q = pow(10, (nu_db-prec_r_db)/20)  

    # get the noise evolution
    Q_true, R_true = None, None
    if q_pattern == 'const':
        Q_true = q*q*torch.eye(dim_x).unsqueeze(2).repeat(1, 1, T)
    elif q_pattern == 'sin':
        Q_true = periodic_Q(nu_db, prec_r_db, Pdq, T)

    if r_pattern == 'const':
        R_true = r*r*torch.eye(dim_y).unsqueeze(2).repeat(1, 1, T)
    elif r_pattern == 'sin':
        R_true = periodic_R(prec_r_db, Pdr, T)

    # set the model
    lin_model = System_Model_NE(F, H, T, Q_true, R_true)
    lin_model.init_sequence(m1x_0, m2x_0)
    u_prior = torch.zeros(dim_x, T)
    Q_prior = true_Q(nu_db, prec_r_db).unsqueeze(2).repeat(1, 1, T)
    
    # tester
    test_input = Y_ALL[idx_nu, idx_r, :, :, :]
    U_target = U_ALL[idx_nu, idx_r, :, :, :]
    X_target = X_ALL[idx_nu, idx_r, :, :, :]
    tester = RTS_Tester_NE(lin_model)
    # idx for trace back
    idices = [idx_nu, idx_r]
    # MSE_and_std = [MSEs_state_rts, std_MSE_state_rts, MSEs_input_rts, std_MSE_input_rts]
    MSE_and_std = tester.test(NB, test_input, X_target, U_target, u_prior, Q_prior)
    # save estimates
    estimates = [tester.x_rts, tester.sigma_rts, tester.u_rts, tester.post_Q_rts ]
    return idices + MSE_and_std + estimates
 


################################################
################# Main Programs ################
################################################
# main(), main2() and main_sp() do the same job, 
# the only difference is how the CPU tries to process the job
# on my own computer, main() works fastest


## main function with multi processing enabled
# wpath: working directory
def main(wpath, q_pattern, r_pattern, Pdq=None, Pdr=None):
    start = time.time()
    # PC
    core_count = os.cpu_count()
    try:
        # Euler HPC
        core_count = int(os.environ['LSB_DJOB_NUMPROC'])
    except:
        pass
    
    # parrallel computing for each [nu, r]
    pool = Pool(core_count)

    # parameters
    opath = wpath + 'baseline/'
    nus_and_rs = list(itertools.product(range_nu_db, range_prec_r_db))
    work = partial(test_nebatch, q_pattern=q_pattern, r_pattern=r_pattern, Pdr=Pdr, Pdq=Pdq)
    # map
    results = pool.map(work, nus_and_rs)
    pool.close()
    pool.join()

    # save simulation results
    for result in results:
        idx_nu = result[0]
        idx_r = result[1]
        [   # MSEs and std
            MSEs_state_rts[idx_nu, idx_r],
            std_MSE_state_rts[idx_nu, idx_r],
            MSEs_input_rts[idx_nu, idx_r],
            std_MSE_input_rts[idx_nu, idx_r],
            # State and Input estimates
            X_rts[idx_nu, idx_r, :, :, :], 
            Sigma_rts[idx_nu, idx_r, :, :, :, :],
            U_rts[idx_nu, idx_r, :, :, :], 
            Q_rts[idx_nu, idx_r, :, :, :, :]
        ] = result[2:]

    print("Total Time - multi processing:", time.time()-start)

    # save file
    torch.save(MSEs_state_rts,    opath+'MSE_state_rts.pt')
    torch.save(std_MSE_state_rts, opath+'std_MSE_state_rts.pt')
    torch.save(MSEs_input_rts,    opath+'MSE_input_rts.pt')
    torch.save(std_MSE_input_rts, opath+'std_MSE_input_rts.pt')

    torch.save(X_rts, opath+'X_rts.pt')
    torch.save(Sigma_rts, opath+'Sigma_rts.pt')
    torch.save(U_rts, opath+'U_rts.pt')
    torch.save(Q_rts, opath+'Q_rts.pt')



if __name__ == '__main__':
    tpath_to_param = {
        # traj_path: [working_path, q_pattern, r_pattern, Pdq, Pdr]
        'Sim_sin_Q/Pd100/traj/': ['Sim_sin_Q/Pd100/', 'sin', 'const', 100, 0],
        'Sim_sin_R/Pd100/traj/': ['Sim_sin_R/Pd100/', 'const', 'sin', 100, 0]
    }
    wpath, q_pattern, r_pattern, pdq, pdr = tpath_to_param[traj_path]
    main(wpath, q_pattern, r_pattern, pdq, pdr)

