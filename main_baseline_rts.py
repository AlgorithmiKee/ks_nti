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
from Linear_sysmdl import SystemModel
from RTS_Smoother_test import RTS_Tester
from param_lin import *
from math import pow
from multiprocessing import Pool
import os
import itertools
import time

# number of traj
NB = 1000

# load data and ground truth
traj_path = 'Sim_baseline/traj/'
Y_ALL = torch.load(traj_path+'Y_obs.pt')[:, :, 0:NB, :, :]
U_ALL = torch.load(traj_path+'U_gt.pt')[:, :, 0:NB, :, :]
X_ALL = torch.load(traj_path+'X_gt.pt')[:, :, 0:NB, :, :]

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
def test_batch(nu_r):
    nu_db, prec_r_db = nu_r[0], nu_r[1]
    # get index
    idx_nu = range_nu_db.index(nu_db)
    idx_r = range_prec_r_db.index(prec_r_db)

    # convert dB to linear scaling
    r, q = pow(10, -prec_r_db/20), pow(10, (nu_db-prec_r_db)/20)

    # set the model
    lin_model = SystemModel(F, q, H, r, T, T_test)
    lin_model.InitSequence(m1x_0, m2x_0)
    u_prior = torch.zeros(dim_x, T)
    Q_prior = true_Q(nu_db, prec_r_db).unsqueeze(2).repeat(1, 1, T)
    
    ################################################
    # Evaluate KS over NB batches #
    ################################################
    test_input = Y_ALL[idx_nu, idx_r, :, :, :]
    U_target = U_ALL[idx_nu, idx_r, :, :, :]
    X_target = X_ALL[idx_nu, idx_r, :, :, :]
    tester = RTS_Tester(lin_model)
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


## main function (with multi processing enabled)
# opath: path for output
# bpath: path for comparison
def main(opath, bpath=None):
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

    # plot
    plt.figure()
    plt.title('State estimation')
    plt.xlabel(r'$\frac{1}{r^2}$ in dB')
    plt.ylabel('MSE in dB')
    plt.plot(range_prec_r_db, [ -x for x in range_prec_r_db], '--', c='r', linewidth=0.75, label='Noise floor')
    MSEs_state_kf = torch.load(bpath+'MSEs_state_baseline.pt')
    plt.plot(range_prec_r_db, MSEs_state_kf[0, :], '*--', c='c', linewidth=0.75, label=r'KF, $\nu$ = 0 dB')
    plt.plot(range_prec_r_db, MSEs_state_kf[1, :], 'o--', c='b', linewidth=0.75, label=r'KF, $\nu$ = -10 dB')
    plt.plot(range_prec_r_db, MSEs_state_kf[2, :], '^--', c='g', linewidth=0.75, label=r'KF, $\nu$ = -20 dB')    
    plt.plot(range_prec_r_db, MSEs_state_rts[0, :], '*-', c='c', linewidth=0.75, label=r'KS, $\nu$ = 0 dB')
    plt.plot(range_prec_r_db, MSEs_state_rts[1, :], 'o-', c='b', linewidth=0.75, label=r'KS, $\nu$ = -10 dB')
    plt.plot(range_prec_r_db, MSEs_state_rts[2, :], '^-', c='g', linewidth=0.75, label=r'KS, $\nu$ = -20 dB')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(opath+'MSE state rts.pdf')

    plt.figure()
    plt.title('Input estimation')
    plt.xlabel(r'$\frac{1}{r^2}$ in dB')
    plt.ylabel('MSE in dB')
    plt.plot(range_prec_r_db, [ -x for x in range_prec_r_db], '--', c='r', linewidth=0.75, label='Noise floor')
    plt.plot(range_prec_r_db, MSEs_input_rts[0, :], '*-', c='c', linewidth=0.75, label=r'$\nu$ = 0 dB')
    plt.plot(range_prec_r_db, MSEs_input_rts[1, :], 'o-', c='b', linewidth=0.75, label=r'$\nu$ = -10 dB')
    plt.plot(range_prec_r_db, MSEs_input_rts[2, :], '^-', c='g', linewidth=0.75, label=r'$\nu$ = -20 dB')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(opath+'MSE input rts.pdf')

    plt.show()


## main function (with multi processing enabled)
def main2(opath, bpath=None):
    start = time.time()
    for idx_nu, nu_db in enumerate(range_nu_db): 
        for idx_r, prec_r_db in enumerate(range_prec_r_db): 
            # convert dB to linear scaling
            r, q = pow(10, -prec_r_db/20), pow(10, (nu_db-prec_r_db)/20)

            # set the model
            lin_model = SystemModel(F, q, H, r, T, T_test)
            lin_model.InitSequence(m1x_0, m2x_0)
            u_prior = torch.zeros(dim_x, T)
            Q_prior = true_Q(nu_db, prec_r_db).unsqueeze(2).repeat(1, 1, T)
            
            ################################################
            # Evaluate KS over NB batches #
            ################################################
            test_input = Y_ALL[idx_nu, idx_r, :, :, :]
            U_target = U_ALL[idx_nu, idx_r, :, :, :]
            X_target = X_ALL[idx_nu, idx_r, :, :, :]
            tester = RTS_Tester(lin_model)
            # test on NB traj
            [
                MSEs_state_rts[idx_nu, idx_r],
                std_MSE_state_rts[idx_nu, idx_r],
                MSEs_input_rts[idx_nu, idx_r],
                std_MSE_input_rts[idx_nu, idx_r]
            ]= tester.test_mp(NB, test_input, X_target, U_target, u_prior, Q_prior)

            # save estimates
            X_rts[idx_nu, idx_r, :, :, :], Sigma_rts[idx_nu, idx_r, :, :, :, :] = tester.x_rts, tester.sigma_rts
            U_rts[idx_nu, idx_r, :, :, :], Q_rts[idx_nu, idx_r, :, :, :, :] = tester.u_rts, tester.post_Q_rts

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

    # plot
    plt.figure()
    plt.title('State estimation')
    plt.xlabel(r'$\frac{1}{r^2}$ in dB')
    plt.ylabel('MSE in dB')
    plt.plot(range_prec_r_db, [ -x for x in range_prec_r_db], '--', c='r', linewidth=0.75, label='Noise floor')
    MSEs_state_kf = torch.load(bpath+'MSEs_state_baseline.pt')
    plt.plot(range_prec_r_db, MSEs_state_kf[0, :], '*--', c='c', linewidth=0.75, label=r'KF, $\nu$ = 0 dB')
    plt.plot(range_prec_r_db, MSEs_state_kf[1, :], 'o--', c='b', linewidth=0.75, label=r'KF, $\nu$ = -10 dB')
    plt.plot(range_prec_r_db, MSEs_state_kf[2, :], '^--', c='g', linewidth=0.75, label=r'KF, $\nu$ = -20 dB')    
    plt.plot(range_prec_r_db, MSEs_state_rts[0, :], '*-', c='c', linewidth=0.75, label=r'KS, $\nu$ = 0 dB')
    plt.plot(range_prec_r_db, MSEs_state_rts[1, :], 'o-', c='b', linewidth=0.75, label=r'KS, $\nu$ = -10 dB')
    plt.plot(range_prec_r_db, MSEs_state_rts[2, :], '^-', c='g', linewidth=0.75, label=r'KS, $\nu$ = -20 dB')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(opath+'MSE state rts.pdf')

    plt.figure()
    plt.title('Input estimation')
    plt.xlabel(r'$\frac{1}{r^2}$ in dB')
    plt.ylabel('MSE in dB')
    plt.plot(range_prec_r_db, [ -x for x in range_prec_r_db], '--', c='r', linewidth=0.75, label='Noise floor')
    plt.plot(range_prec_r_db, MSEs_input_rts[0, :], '*-', c='c', linewidth=0.75, label=r'$\nu$ = 0 dB')
    plt.plot(range_prec_r_db, MSEs_input_rts[1, :], 'o-', c='b', linewidth=0.75, label=r'$\nu$ = -10 dB')
    plt.plot(range_prec_r_db, MSEs_input_rts[2, :], '^-', c='g', linewidth=0.75, label=r'$\nu$ = -20 dB')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(opath+'MSE input rts.pdf')

    plt.show()


# main function (using single processing)
def main_sp(opath, bpath=None):
    start = time.time()
    for idx_nu, nu_db in enumerate(range_nu_db): 
        for idx_r, prec_r_db in enumerate(range_prec_r_db): 
            # convert dB to linear scaling
            r, q = pow(10, -prec_r_db/20), pow(10, (nu_db-prec_r_db)/20)

            # set the model
            lin_model = SystemModel(F, q, H, r, T, T_test)
            lin_model.InitSequence(m1x_0, m2x_0)
            u_prior = torch.zeros(dim_x, T)
            Q_prior = true_Q(nu_db, prec_r_db).unsqueeze(2).repeat(1, 1, T)
            
            ################################################
            # Evaluate KS over NB batches #
            ################################################
            test_input = Y_ALL[idx_nu, idx_r, :, :, :]
            U_target = U_ALL[idx_nu, idx_r, :, :, :]
            X_target = X_ALL[idx_nu, idx_r, :, :, :]
            tester = RTS_Tester(lin_model)
            # test on NB traj
            [
                MSEs_state_rts[idx_nu, idx_r],
                std_MSE_state_rts[idx_nu, idx_r],
                MSEs_input_rts[idx_nu, idx_r],
                std_MSE_input_rts[idx_nu, idx_r]
            ]= tester.test(NB, test_input, X_target, U_target, u_prior, Q_prior)

            # save estimates
            X_rts[idx_nu, idx_r, :, :, :], Sigma_rts[idx_nu, idx_r, :, :, :, :] = tester.x_rts, tester.sigma_rts
            U_rts[idx_nu, idx_r, :, :, :], Q_rts[idx_nu, idx_r, :, :, :, :] = tester.u_rts, tester.post_Q_rts

    print("Total time - single processing: ", time.time()-start, " s")

    torch.save(MSEs_state_rts,    opath+'MSE_state_rts.pt')
    torch.save(std_MSE_state_rts, opath+'std_MSE_state_rts.pt')
    torch.save(MSEs_input_rts,    opath+'MSE_input_rts.pt')
    torch.save(std_MSE_input_rts, opath+'std_MSE_input_rts.pt')

    torch.save(X_rts, opath+'X_rts.pt')
    torch.save(Sigma_rts, opath+'Sigma_rts.pt')
    torch.save(U_rts, opath+'U_rts.pt')
    torch.save(Q_rts, opath+'Q_rts.pt')

    MSEs_state_kf = torch.load(bpath+'MSEs_state_baseline.pt')
    plt.figure()
    plt.title('State estimation (Kalman Smoothing)')
    plt.xlabel(r'$\frac{1}{r^2}$ in dB')
    plt.ylabel('MSE in dB')
    plt.plot(range_prec_r_db, [ -x for x in range_prec_r_db], '--', c='r', linewidth=0.75, label='Noise floor')
    plt.plot(range_prec_r_db, MSEs_state_kf[0, :], '*--', c='c', linewidth=0.75, label=r'$\nu$ = 0 dB')
    plt.plot(range_prec_r_db, MSEs_state_kf[1, :], 'o--', c='b', linewidth=0.75, label=r'$\nu$ = -10 dB')
    plt.plot(range_prec_r_db, MSEs_state_kf[2, :], '^--', c='g', linewidth=0.75, label=r'$\nu$ = -20 dB')
    plt.plot(range_prec_r_db, MSEs_state_rts[0, :], '*-', c='c', linewidth=0.75, label=r'$\nu$ = 0 dB')
    plt.plot(range_prec_r_db, MSEs_state_rts[1, :], 'o-', c='b', linewidth=0.75, label=r'$\nu$ = -10 dB')
    plt.plot(range_prec_r_db, MSEs_state_rts[2, :], '^-', c='g', linewidth=0.75, label=r'$\nu$ = -20 dB')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(opath+'MSE state rts.pdf')

    plt.figure()
    plt.title('Input estimation')
    plt.xlabel(r'$\frac{1}{r^2}$ in dB')
    plt.ylabel('MSE in dB')
    plt.plot(range_prec_r_db, [ -x for x in range_prec_r_db], '--', c='r', linewidth=0.75, label='Noise floor')
    plt.plot(range_prec_r_db, MSEs_input_rts[0, :], '*-', c='c', linewidth=0.75, label=r'$\nu$ = 0 dB')
    plt.plot(range_prec_r_db, MSEs_input_rts[1, :], 'o-', c='b', linewidth=0.75, label=r'$\nu$ = -10 dB')
    plt.plot(range_prec_r_db, MSEs_input_rts[2, :], '^-', c='g', linewidth=0.75, label=r'$\nu$ = -20 dB')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(opath+'MSE input rts.pdf')

    plt.show()


if __name__ == '__main__':
    main(opath = 'Sim_baseline/KS/', bpath = 'Sim_baseline/KF/')
