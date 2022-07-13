# Simulate EM algorithm for estimating Q on NB trajectories
# plot the averaged MSE in every iterations
from numpy import size
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from param_lin import *
from copy import copy
from Linear_sysmdl import SystemModel
from Simple_NUV_test import *

# load data to be filtered
Y_ALL = torch.load('Data_simple_nuv/Y_lin.pt')

# load ground truth
U_ALL = torch.load('Data_simple_nuv/U_lin.pt')
X_ALL = torch.load('Data_simple_nuv/X_lin.pt')

# number of iterations for EM
itr = 20

# save the averaged MSE of every iteration over NB trajectoires
# MSE_state[idx_nu, idx_r, idx_batch, idx_itr] saves the MSE for
#   specific pair [nu, r] at iteration idx_itr for idx_batch
MSE_state = torch.empty((len(range_nu_db), len(range_prec_r_db), NB, itr))

for idx_nu, nu_db in enumerate(range_nu_db): 
    for idx_r, prec_r_db in enumerate(range_prec_r_db): 
        # specify model parameters
        r = pow(10, -prec_r_db/20)
        q = pow(10, (nu_db-prec_r_db)/20)

        lin_model = SystemModel(F, q, H, r, T, T_test)
        lin_model.InitSequence(m1x_0, m2x_0)

        # assumed model: R is now unknown
        assumed_model = copy(lin_model)
        assumed_r = 1.0
        assumed_model.UpdateCovariance_Gain(lin_model.q, assumed_r)
        
        # Track the MSE fo state estimate for unknown R
        test_input = Y_ALL[idx_nu, idx_r, :, :, :]
        test_target = X_ALL[idx_nu, idx_r, :, :, :]
        tracker = Unknown_R_State_Tracker(assumed_model)
        MSE_state[idx_nu, idx_r, :, :] = tracker.track(NB, test_input, test_target, itr)

# save MSE_state for all trjactories and all iterations
torch.save(MSE_state, 'Data_simple_nuv/MSE_state_all_itr_unknR.pt')

# average over all trajectories
MSE_state = torch.mean(MSE_state, dim=2)
MSE_state = 10*torch.log10(MSE_state)

# save MSE_state for all iterations averaged on all traj.
torch.save(MSE_state, 'Data_simple_nuv/MSE_state_track_unknR.pt')

### show the convergence of MSEs
max_itr = 15
itr_axis = torch.arange(0, max_itr)

fig_mse = plt.figure()
idx_r = 1   # prec_r = 0 dB
plt.title('MSE state estimation (Unknown R, 1/r^2 = 0dB)')
plt.xlabel('iteration')
plt.ylabel('MSE of state estimation [dB]')
plt.plot(itr_axis, MSE_state[0, idx_r, 0:max_itr], 'o-c', linewidth=0.75, label=r'$\nu$ =   0 dB')
plt.plot(itr_axis, MSE_state[1, idx_r, 0:max_itr], 'o-g', linewidth=0.75, label=r'$\nu$ = -10 dB')
plt.plot(itr_axis, MSE_state[2, idx_r, 0:max_itr], 'o-r', linewidth=0.75, label=r'$\nu$ = -20 dB')
plt.grid()
plt.legend()

fig_mse = plt.figure()
idx_r = 2   # prec_r = 10 dB
plt.title('MSE state estimation (Unknown R, 1/r^2 = 10dB)')
plt.xlabel('iteration')
plt.ylabel('MSE of state estimation [dB]')
plt.plot(itr_axis, MSE_state[0, idx_r, 0:max_itr], 'o-c', linewidth=0.75, label=r'$\nu$ =   0 dB')
plt.plot(itr_axis, MSE_state[1, idx_r, 0:max_itr], 'o-g', linewidth=0.75, label=r'$\nu$ = -10 dB')
plt.plot(itr_axis, MSE_state[2, idx_r, 0:max_itr], 'o-r', linewidth=0.75, label=r'$\nu$ = -20 dB')
plt.grid()
plt.legend()

fig_mse = plt.figure()
idx_r = 3   # prec_r = 20 dB
plt.title('MSE state estimation (Unknown R, 1/r^2 = 20dB)')
plt.xlabel('iteration')
plt.ylabel('MSE of state estimation [dB]')
plt.plot(itr_axis, MSE_state[0, idx_r, 0:max_itr], 'o-c', linewidth=0.75, label=r'$\nu$ =   0 dB')
plt.plot(itr_axis, MSE_state[1, idx_r, 0:max_itr], 'o-g', linewidth=0.75, label=r'$\nu$ = -10 dB')
plt.plot(itr_axis, MSE_state[2, idx_r, 0:max_itr], 'o-r', linewidth=0.75, label=r'$\nu$ = -20 dB')
plt.grid()
plt.legend()


plt.show()