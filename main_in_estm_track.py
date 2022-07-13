# This py file runs input estimation (base line) for a fixed set of parameters
# and compares the true input trajectory and the estimated input trajecory

import torch
import matplotlib.pyplot as plt
from math import pow
from Linear_sysmdl import SystemModel
from Linear_KF import KalmanFilter
from RTS_Smoother import rts_smoother, rts_smoother_in


# define system parameters
F = torch.tensor([[1.0, 1.0],[0.0, 1.0]])
H = torch.eye(2)
T = 100
T_test = T

# specify the model parameter
prec_r_db = 20 # 1/r^2 in dB
nu_db = 0 # q^2/r^2 in dB
r = pow(10, -prec_r_db/20)
q = pow(10, (nu_db-prec_r_db)/20)
lin_model = SystemModel(F, q, H, r, T, T_test)

# generate data
m1x_0 = torch.tensor([1.0, 0.0])
m2x_0 = torch.eye(lin_model.m)
lin_model.InitSequence(m1x_0, m2x_0)
lin_model.GenerateSequence(lin_model.Q, lin_model.R, T)

# prepare the filter and smoother
KF = KalmanFilter(lin_model)
KF.InitSequence(lin_model.m1x_0, lin_model.m2x_0)
RTS = rts_smoother_in(lin_model)

# asuming that our model has full knowledge of the model error
u_prior = torch.zeros(lin_model.m, T)
Q_prior = lin_model.Q.unsqueeze(2).repeat(1, 1, T)

# fwd and bwd recursion
KF.GenerateSequence(lin_model.y, T)
RTS.GenerateSequence_in(KF.x, KF.sigma, u_prior, Q_prior, RTS.T_test)

plt.title(rf'Input estimation, $\nu$={nu_db} dB, $\frac{{1}}{{r^2}}$={prec_r_db} dB')
plt.xlabel(r'$t$')
plt.ylabel('input')
plt.plot(torch.arange(1, T+1), lin_model.u[0, :], 'g--', label=r'$u_1$')
plt.plot(torch.arange(1, T+1), RTS.s_u[0, :], 'b-', label=r'$\hat{u}_1$')
plt.legend(loc="upper right")
plt.grid()
plt.show()
