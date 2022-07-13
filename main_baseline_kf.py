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
from RTS_Smoother_test import KF_Tester
from param_lin import *
from math import pow

# number of traj
NB = 1000

traj_path = 'Sim_baseline/traj/'
# load data to be filtered
Y_ALL = torch.load(traj_path+'Y_obs.pt')[:, :, 0:NB, :, :]
# load ground truth
U_ALL = torch.load(traj_path+'U_gt.pt')[:, :, 0:NB, :, :]
X_ALL = torch.load(traj_path+'X_gt.pt')[:, :, 0:NB, :, :]

# save mean MSE and std MSE
MSEs_state_baseline = torch.empty((len(range_nu_db), len(range_prec_r_db)))
std_MSE_state_baseline = torch.empty((len(range_nu_db), len(range_prec_r_db)))

# save estimated traj
X_kf = torch.empty_like(X_ALL)
Sigma_kf = torch.empty(size=[len(range_nu_db), len(range_prec_r_db), NB, dim_x, dim_x, T])


for idx_nu, nu_db in enumerate(range_nu_db): 
    for idx_r, prec_r_db in enumerate(range_prec_r_db): 
        # convert dB to linear scaling
        r = pow(10, -prec_r_db/20)
        q = pow(10, (nu_db-prec_r_db)/20)

        # set the model
        lin_model = SystemModel(F, q, H, r, T, T_test)
        lin_model.InitSequence(m1x_0, m2x_0)
        
        ################################################
        # Evaluate the state estimator over NB batches #
        ################################################
        test_input = Y_ALL[idx_nu, idx_r, :, :, :]
        test_target = X_ALL[idx_nu, idx_r, :, :, :]
        tester = KF_Tester(lin_model)
        MSEs_state_baseline[idx_nu, idx_r], std_MSE_state_baseline[idx_nu, idx_r] = tester.test(NB, test_input, test_target)

        # save state estm
        X_kf[idx_nu, idx_r, :, :, :], Sigma_kf[idx_nu, idx_r, :, :, :, :] = tester.x_kf, tester.sigma_kf

print(MSEs_state_baseline)
print(std_MSE_state_baseline)

fpath = 'Sim_baseline/KF/'
torch.save(MSEs_state_baseline,    fpath+'MSEs_state_baseline.pt')
torch.save(std_MSE_state_baseline, fpath+'std_MSE_state_baseline.pt')

torch.save(X_kf,     fpath+'X_kf.pt')
torch.save(Sigma_kf, fpath+'Sigma_kf.pt')

plt.figure()
plt.title('State estimation')
plt.xlabel(r'$\frac{1}{r^2}$ in dB')
plt.ylabel('MSE in dB')
plt.plot(range_prec_r_db, [ -x for x in range_prec_r_db], '--', c='r', linewidth=0.75, label='Noise floor')
plt.plot(range_prec_r_db, MSEs_state_baseline[0, :], '*--', c='c', linewidth=0.75, label=r'$\nu$ = 0 dB')
plt.plot(range_prec_r_db, MSEs_state_baseline[1, :], 'o--', c='b', linewidth=0.75, label=r'$\nu$ = -10 dB')
plt.plot(range_prec_r_db, MSEs_state_baseline[2, :], '^--', c='g', linewidth=0.75, label=r'$\nu$ = -20 dB')
plt.legend(loc="upper right")
plt.grid()
plt.savefig(fpath+'MSE state baseline.pdf')

plt.show()
