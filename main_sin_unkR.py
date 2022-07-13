import torch
from math import pow
from NUV_test import Unknown_R_Tester
import matplotlib.pyplot as plt
from param_lin import range_nu_db, range_prec_r_db, F, H, dim_x, dim_y, m1x_0, m2x_0, T
from Linear_sysmdl import System_Model_NE
from multiprocessing import Pool
import time
import os
import itertools

# number of traj
NB = 1000

# load data to be filtered
traj_path = 'Sim_sin_R/Pd100/traj/'
Y_ALL = torch.load(traj_path+'Y_ALL.pt')[:, :, 0:NB, :, :]
U_ALL = torch.load(traj_path+'U_ALL.pt')[:, :, 0:NB, :, :]
X_ALL = torch.load(traj_path+'X_ALL.pt')[:, :, 0:NB, :, :]

# Simple NUV MSE for each pair of [nv, 1/r^2]
MSEs_state_unknownR = torch.empty((len(range_nu_db), len(range_prec_r_db)))
std_MSE_state_unknownR = torch.empty((len(range_nu_db), len(range_prec_r_db)))
MSEs_input_unknownR = torch.empty((len(range_nu_db), len(range_prec_r_db)))
std_MSE_input_unknownR = torch.empty((len(range_nu_db), len(range_prec_r_db)))

# save estimated traj
X_unknownR = torch.empty_like(X_ALL)
Sigma_unknownR = torch.empty(size=[len(range_nu_db), len(range_prec_r_db), NB, dim_x, dim_x, T])
U_unknownR, Q_unknownR = torch.empty_like(X_unknownR), torch.empty_like(Sigma_unknownR)

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
    tester = Unknown_R_Tester(lin_model)
    # idx for trace back
    idices = [idx_nu, idx_r]
    # stable config: 
    # itr=40, win=15, forget=0.6
    # itr=40, win=None, forget=0.9
    # MSE_and_std = [MSEs_state_rts, std_MSE_state_rts, MSEs_input_rts, std_MSE_input_rts]
    MSE_and_std = tester.test(NB, test_input, X_target, U_target, init_unknown=r, itr=20, win=5, forget_itr=0.0)     
    # save estimates
    estimates = [tester.x_rts, tester.sigma_rts, tester.u_rts, tester.post_Q_rts ]
    return idices + MSE_and_std + estimates

#######################################################################
#################### Simple NUV for unknown R ######################
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
            MSEs_state_unknownR[idx_nu, idx_r],
            std_MSE_state_unknownR[idx_nu, idx_r],
            MSEs_input_unknownR[idx_nu, idx_r],
            std_MSE_input_unknownR[idx_nu, idx_r],
            # State and Input estimates
            X_unknownR[idx_nu, idx_r, :, :, :], 
            Sigma_unknownR[idx_nu, idx_r, :, :, :, :],
            U_unknownR[idx_nu, idx_r, :, :, :], 
            Q_unknownR[idx_nu, idx_r, :, :, :, :]
        ] = result[2:]
    print("Total Time - multi processing:", time.time()-start)
    save_data(opath)
    save_plot(opath, bpath)


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
            test_input, X_target, U_target = Y_ALL[idx_nu, idx_r, :, :, :], X_ALL[idx_nu, idx_r, :, :, :], U_ALL[idx_nu, idx_r, :, :, :]
            tester = Unknown_R_Tester(lin_model)
            # test on NB traj
            [
                MSEs_state_unknownR[idx_nu, idx_r],
                std_MSE_state_unknownR[idx_nu, idx_r],
                MSEs_input_unknownR[idx_nu, idx_r],
                std_MSE_input_unknownR[idx_nu, idx_r]
            ] = tester.test(NB, test_input, X_target, U_target, init_unknown=r, itr=20, win=15, forget_itr=0.0)     
            
            # save estimated traj
            X_unknownR[idx_nu, idx_r, :, :, :], Sigma_unknownR[idx_nu, idx_r, :, :, :, :] = tester.x_rts, tester.sigma_rts
            U_unknownR[idx_nu, idx_r, :, :, :], Q_unknownR[idx_nu, idx_r, :, :, :, :] = tester.u_rts, tester.post_Q_rts
    save_data(opath)
    save_plot(opath, bpath)


def save_data(opath):
    # save MSE and std
    torch.save(MSEs_state_unknownR,     opath+'MSE_state_unknownR.pt')
    torch.save(std_MSE_state_unknownR,  opath+'std_MSE_state_unknownR.pt')
    torch.save(MSEs_input_unknownR,     opath+'MSE_input_unknownR.pt')
    torch.save(std_MSE_input_unknownR,  opath+'/std_MSE_input_unknownR.pt')
    # save estimated traj
    torch.save(X_unknownR,     opath+'X_unknownR.pt')
    torch.save(Sigma_unknownR, opath+'Sigma_unknownR.pt')
    torch.save(U_unknownR,     opath+'U_unknownR.pt')
    torch.save(Q_unknownR,     opath+'Q_unknownR.pt')


def save_plot(opath, bpath):
    ##########################################
    #### Plot MSE of state for periodic R ####
    ##########################################
    plt.figure()
    plt.title('State estimation (unknown periodic R)')
    plt.xlabel(r'$\frac{1}{r^2}$ in dB')
    plt.ylabel('MSE of state estimation in dB')
    # Noise floor
    plt.plot(range_prec_r_db, [ -x for x in range_prec_r_db], '--', c='r', linewidth=0.75, label='Noise floor')
    # load & plot MSE for base line
    MSEs_state_baseline = torch.load(bpath+'MSE_state_rts.pt')
    plt.plot(range_prec_r_db, MSEs_state_baseline[0, :], '--', c='c', linewidth=0.75, label=r'$\nu$ = 0 dB, baseline')
    plt.plot(range_prec_r_db, MSEs_state_baseline[1, :], '--', c='b', linewidth=0.75, label=r'$\nu$ = -10 dB, baseline')
    plt.plot(range_prec_r_db, MSEs_state_baseline[2, :], '--', c='g', linewidth=0.75, label=r'$\nu$ = -20 dB, baseline')
    # plot MSE for simple nuv
    plt.plot(range_prec_r_db, MSEs_state_unknownR[0, :], '*-', c='c', linewidth=0.75, label=r'$\nu$ =   0 dB, unknown $R$')
    plt.plot(range_prec_r_db, MSEs_state_unknownR[1, :], 'o-', c='b', linewidth=0.75, label=r'$\nu$ = -10 dB, unknown $R$')
    plt.plot(range_prec_r_db, MSEs_state_unknownR[2, :], '^-', c='g', linewidth=0.75, label=r'$\nu$ = -20 dB, unknown $R$')

    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(opath+'MSE_state_periodR.pdf')

    ##########################################
    #### Plot MSE of input for periodic R ####
    ##########################################
    plt.figure()
    plt.title('Input estimation (unknown periodic R)')
    plt.xlabel(r'$\frac{1}{r^2}$ in dB')
    plt.ylabel('MSE of input estimation in dB')
    # Noise floor
    plt.plot(range_prec_r_db, [ -x for x in range_prec_r_db], '--', c='r', linewidth=0.75, label='Noise floor')
    # load & plot MSE for base line
    MSEs_input_baseline = torch.load(bpath+'MSE_input_rts.pt')
    plt.plot(range_prec_r_db, MSEs_input_baseline[0, :], '--', c='c', linewidth=0.75, label=r'$\nu$ = 0 dB, baseline')
    plt.plot(range_prec_r_db, MSEs_input_baseline[1, :], '--', c='b', linewidth=0.75, label=r'$\nu$ = -10 dB, baseline')
    plt.plot(range_prec_r_db, MSEs_input_baseline[2, :], '--', c='g', linewidth=0.75, label=r'$\nu$ = -20 dB, baseline')
    # plot MSE for simple nuv
    plt.plot(range_prec_r_db, MSEs_input_unknownR[0, :], '*-', c='c', linewidth=0.75, label=r'$\nu$ =   0 dB, unknown $R$')
    plt.plot(range_prec_r_db, MSEs_input_unknownR[1, :], 'o-', c='b', linewidth=0.75, label=r'$\nu$ = -10 dB, unknown $R$')
    plt.plot(range_prec_r_db, MSEs_input_unknownR[2, :], '^-', c='g', linewidth=0.75, label=r'$\nu$ = -20 dB, unknown $R$')

    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(opath+'MSE_input_periodR.pdf')
    
    plt.show()

if __name__ == '__main__':
    main(opath='Sim_sin_R/Pd100/MA/', bpath='Sim_baseline/KS/')
