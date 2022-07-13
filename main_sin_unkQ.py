from Linear_sysmdl import System_Model_NE
import torch
from NUV_test import Unknown_Q_Tester
from math import pow
import torch.nn as nn
from export_fig import save_ksplot
from param_lin import range_nu_db, range_prec_r_db, F, H, dim_x, dim_y, m1x_0, m2x_0, T
from multiprocessing import Pool
import time
import os
import itertools

# number of traj
NB = 1000

# load data to be filtered
traj_path = 'Sim_sin_Q/Pd100/traj/'
Y_ALL = torch.load(traj_path+'Y_ALL.pt')[:, :, 0:NB, :, :]
U_ALL = torch.load(traj_path+'U_ALL.pt')[:, :, 0:NB, :, :]
X_ALL = torch.load(traj_path+'X_ALL.pt')[:, :, 0:NB, :, :]


# Simple NUV MSE for each pair of [nv, 1/r^2]
MSEs_state_unknownQ = torch.empty((len(range_nu_db), len(range_prec_r_db)))
std_MSE_state_unknownQ = torch.empty((len(range_nu_db), len(range_prec_r_db)))
MSEs_input_unknownQ = torch.empty((len(range_nu_db), len(range_prec_r_db)))
std_MSE_input_unknownQ = torch.empty((len(range_nu_db), len(range_prec_r_db)))

# save estimated traj
X_unknownQ = torch.empty_like(X_ALL)
Sigma_unknownQ = torch.empty(size=[len(range_nu_db), len(range_prec_r_db), NB, dim_x, dim_x, T])
U_unknownQ, Q_unknownQ = torch.empty_like(X_unknownQ), torch.empty_like(Sigma_unknownQ)

# Worker Process
def test_batch(nu_r):
    nu_db, prec_r_db = nu_r[0], nu_r[1]
    # get index
    idx_nu = range_nu_db.index(nu_db)
    idx_r = range_prec_r_db.index(prec_r_db)

    # convert dB to linear scaling
    r, q = pow(10, -prec_r_db/20), pow(10, (nu_db-prec_r_db)/20)
    Q = q*q*torch.eye(dim_x)
    Q = Q.unsqueeze(2).repeat(1, 1, T)
    R = r*r*torch.eye(dim_y)
    R = R.unsqueeze(2).repeat(1, 1, T) 
    # set the model
    lin_model = System_Model_NE(F, H, T, Q, R)
    lin_model.init_sequence(m1x_0, m2x_0)
    
    ################################################
    # Evaluate KS over NB batches #
    ################################################
    test_input = Y_ALL[idx_nu, idx_r, :, :, :]
    U_target = U_ALL[idx_nu, idx_r, :, :, :]
    X_target = X_ALL[idx_nu, idx_r, :, :, :]
    tester = Unknown_Q_Tester(lin_model)
    # idx for trace back
    idices = [idx_nu, idx_r]
    # stable config: 
    # itr=40, win=15, forget=0.6
    # itr=40, win=None, forget=0.9
    # MSE_and_std = [MSEs_state_rts, std_MSE_state_rts, MSEs_input_rts, std_MSE_input_rts]
    MSE_and_std = tester.test(NB, test_input, X_target, U_target, init_unknown=q, itr=80, win=5, forget_itr=0.0)     
    # save estimates
    estimates = [tester.x_rts, tester.sigma_rts, tester.u_rts, tester.post_Q_rts ]
    return idices + MSE_and_std + estimates


#######################################################################
#################### Simple NUV for unknown Q ######################
#######################################################################
def main(opath, bpath):
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
    nus_and_rs = list(itertools.product(range_nu_db, range_prec_r_db))
    results = pool.map(test_batch, nus_and_rs)
    pool.close()
    pool.join()

    # save simulation results
    for result in results:
        idx_nu = result[0]
        idx_r = result[1]
        [   # MSEs and std
            MSEs_state_unknownQ[idx_nu, idx_r],
            std_MSE_state_unknownQ[idx_nu, idx_r],
            MSEs_input_unknownQ[idx_nu, idx_r],
            std_MSE_input_unknownQ[idx_nu, idx_r],
            # State and Input estimates
            X_unknownQ[idx_nu, idx_r, :, :, :], 
            Sigma_unknownQ[idx_nu, idx_r, :, :, :, :],
            U_unknownQ[idx_nu, idx_r, :, :, :], 
            Q_unknownQ[idx_nu, idx_r, :, :, :, :]
        ] = result[2:]
    print("Total Time - multi processing:", time.time()-start)
    save_data(opath)
    ############################
    #### unknown sin Q (LF) ####
    ############################
    bpath = 'Sim_sin_Q/Pd100/baseline/MSE_state_rts.pt'
    ipath = 'Sim_sin_Q/Pd100/MA/MSE_state_unknownQ.pt'
    opath = 'Sim_sin_Q/Pd100/MA/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+LF, unknown $Q$', blabel=r'KS, known $Q$')

    bpath = 'Sim_sin_Q/Pd100/baseline/MSE_input_rts.pt'
    ipath = 'Sim_sin_Q/Pd100/MA/MSE_input_unknownQ.pt'
    opath = 'Sim_sin_Q/Pd100/MA/' 
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+LF, unknown $Q$', blabel=r'KS, known $Q$')

def main_sp(opath, bpath):
    # # Start estimating for each pair [nu, r]
    for idx_nu, nu_db in enumerate(range_nu_db): 
        for idx_r, prec_r_db in enumerate(range_prec_r_db): 
            # specify the model parameter
            r = pow(10, -prec_r_db/20)
            q = pow(10, (nu_db-prec_r_db)/20)

            Q = q*q*torch.eye(dim_x)
            Q = Q.unsqueeze(2).repeat(1, 1, T)
            R = r*r*torch.eye(dim_y)
            R = R.unsqueeze(2).repeat(1, 1, T)        
            
            lin_model = System_Model_NE(F, H, T, Q, R)
            lin_model.init_sequence(m1x_0, m2x_0)

            ########################################################################
            # case 1: Baseline
            ########################################################################
            # Done

            ######################################################################       
            # case 2: asuming that our model don't know the true R (but true Q)  #
            ######################################################################     
        
            # Test MSE of state for unknown R
            test_input, X_target, U_target = Y_ALL[idx_nu, idx_r, :, :, :], X_ALL[idx_nu, idx_r, :, :, :], U_ALL[idx_nu, idx_r, :, :, :]
            tester = Unknown_Q_Tester(lin_model)
            # test on NB traj
            [
                MSEs_state_unknownQ[idx_nu, idx_r],
                std_MSE_state_unknownQ[idx_nu, idx_r],
                MSEs_input_unknownQ[idx_nu, idx_r],
                std_MSE_input_unknownQ[idx_nu, idx_r]
            ] = tester.test(NB, test_input, X_target, U_target, init_unknown=1.0, itr=20, win=15, forget_itr=0.0)             
            # save estimated traj
            X_unknownQ[idx_nu, idx_r, :, :, :], Sigma_unknownQ[idx_nu, idx_r, :, :, :, :] = tester.x_rts, tester.sigma_rts
            U_unknownQ[idx_nu, idx_r, :, :, :], Q_unknownQ[idx_nu, idx_r, :, :, :, :] = tester.u_rts, tester.post_Q_rts
    save_data(opath)
    ############################
    #### unknown sin Q (LF) ####
    ############################
    bpath = 'Sim_sin_Q/Pd100/baseline/MSE_state_rts.pt'
    ipath = 'Sim_sin_Q/Pd100/MA/MSE_state_unknownQ.pt'
    opath = 'Sim_sin_Q/Pd100/MA/'
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+LF, unknown $Q$', blabel=r'KS, known $Q$')

    bpath = 'Sim_sin_Q/Pd100/baseline/MSE_input_rts.pt'
    ipath = 'Sim_sin_Q/Pd100/MA/MSE_input_unknownQ.pt'
    opath = 'Sim_sin_Q/Pd100/MA/' 
    save_ksplot(ipath, opath,  bpath, ilabel=r'KS+LF, unknown $Q$', blabel=r'KS, known $Q$')


def save_data(opath):
    # save MSE and std
    torch.save(MSEs_state_unknownQ,     opath+'MSE_state_unknownQ.pt')
    torch.save(std_MSE_state_unknownQ,  opath+'std_MSE_state_unknownQ.pt')
    torch.save(MSEs_input_unknownQ,     opath+'MSE_input_unknownQ.pt')
    torch.save(std_MSE_input_unknownQ,  opath+'/std_MSE_input_unknownQ.pt')
    # save estimated traj
    torch.save(X_unknownQ,     opath+'X_unknownQ.pt')
    torch.save(Sigma_unknownQ, opath+'Sigma_unknownQ.pt')
    torch.save(U_unknownQ,     opath+'U_unknownQ.pt')
    torch.save(Q_unknownQ,     opath+'Q_unknownQ.pt')

if __name__ == '__main__':
    main(opath='Sim_sin_Q/Pd100/MA/', bpath='Sim_baseline/KS/')
