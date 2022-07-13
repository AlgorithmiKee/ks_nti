import torch
from math import pi, cos
#########################
### System parameters ###
#########################
# system matrices
F = torch.tensor([[1.0, 1.0],[0.0, 1.0]])
H = torch.eye(2)

# dimensions
dim_x = F.size()[0]
dim_y = H.size()[0]

# length of trajectories
T = 100
T_test = T

# fixed initial condition
m1x_0 = torch.tensor([1.0, 0.0])
m2x_0 = torch.zeros(size=[dim_x, dim_x])

#############################
### Simulation parameters ###
#############################

# q^2/r^2 in dB
range_nu_db = [0.0, -10.0, -20.0]

# 1/r^2 in dB
range_prec_r_db = [-10.0, 0.0, 10.0, 20.0, 30.0]

# number of trajectories per [nu, r], change to 1000 later
NB = 1000
