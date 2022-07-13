from Linear_sysmdl import System_Model_NE
from torch.distributions.normal import Normal
import torch
from math import pow
from RTS_NUV import RTS_NUV
import torch.nn as nn
from RTS_Smoother_test import RTS_State_Tester_NE
from NUV_test import Unknown_R_State_Tester
from param_lin import F, H, dim_x, dim_y, m1x_0, m2x_0

T=100
NB =10

# Fixed nu and mean of 1/r^2 [dB]
nu_dB = 0.0
prec_r_dB_mean = 0.0

# Randomized Rt and Qt
Q_rand = torch.empty(size=[dim_x, dim_x, T])
R_rand = torch.empty(size=[dim_y, dim_y, T])
distr_r_prec = Normal(torch.tensor([prec_r_dB_mean]), torch.tensor([3]))
for t in range(0, T):
    prec_r_dB = distr_r_prec.rsample()
    r = pow(10, -prec_r_dB/20)
    q = pow(10, (nu_dB - prec_r_dB)/20)

    r = r*r*torch.ones(dim_y)
    q = q*q*torch.ones(dim_x)
    R_rand[:, :, t] = torch.diag(r)
    Q_rand[:, :, t] = torch.diag(q)

# generate trajectory
lin_model = System_Model_NE(F, H, T, Q_rand, Q_rand)
lin_model.init_sequence(m1x_0, m2x_0)
lin_model.generate_nuv_batch(NB, T)

# save traj
# torch.save(lin_model.Y, 'Data_rand_nuv/Y_lin.pt')
# torch.save(lin_model.X, 'Data_rand_nuv/X_lin.pt')
# torch.save(lin_model.U, 'Data_rand_nuv/U_lin.pt')

# Load trajectories
Y = lin_model.Y
X = lin_model.X
U = lin_model.U

###################################
###########  Baseline  ############
###################################
tester = RTS_State_Tester_NE(lin_model)
u_prior = torch.zeros(dim_x, T)
Q_prior = lin_model.Q_evo
test_target=X
test_input=Y
MSE_baseline_dB = tester.test(NB, test_input, test_target, u_prior, Q_prior)[2]
print('baseline, MSE=', MSE_baseline_dB, '[dB]')

###################################
### Unknown R, start testing EM ###
###################################
tester = Unknown_R_State_Tester(lin_model)
MSE_mean_dB = tester.test(NB, test_input=Y, test_target=X, init_unknown=1.0, itr=30, win=20, forget_itr=0.0)[2]
