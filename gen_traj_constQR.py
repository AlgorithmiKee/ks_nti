## Generate trajectories for static Q & R (Linear model)
import torch
from Linear_sysmdl import SystemModel
from param_lin import *

# For [nu, r] each pair, generates 1000 trajectories 
# where Q and R matrices are time-invariant
# F: F in LSSM
# H: H in LSSM
# T: length of each trajectory
def generate_traj(F, H, T) -> None:
    nus = len(range_nu_db)
    rs = len(range_prec_r_db)
    
    # stores Y-trajectories for each pair [nu, r]
    # Y_ALL[i, j, :, :, :] contains trajectories for [nu_i, r_j]
    Y_ALL = torch.empty(size=[nus, rs, NB, dim_y, T])

    # stores U-trajectories for each pair [nu, r]
    U_ALL = torch.empty(size=[nus, rs, NB, dim_x, T])

    # stores X-trajectories for each pair [nu, r]
    X_ALL = torch.empty(size=[nus, rs, NB, dim_x, T])

    for idx_nu, nu_db in enumerate(range_nu_db): 
        for idx_r, prec_r_db in enumerate(range_prec_r_db):
            # convert dB to linear scaling
            r = pow(10, -prec_r_db/20)
            q = pow(10, (nu_db-prec_r_db)/20)
            lin_model = SystemModel(F, q, H, r, T, T_test=T)

            # generate trajectories for current pair [nu, r]
            lin_model.InitSequence(m1x_0, m2x_0)
            lin_model.GenerateBatch(NB, T)

            # save trajectories for current pair [nu, r]
            Y_ALL[idx_nu, idx_r, :, :, :] = lin_model.Observed
            U_ALL[idx_nu, idx_r, :, :, :] = lin_model.Input_Target
            X_ALL[idx_nu, idx_r, :, :, :] = lin_model.State_Target
    # save files
    torch.save(Y_ALL, 'Sim_baseline/traj/Y_obs.pt')
    torch.save(X_ALL, 'Sim_baseline/traj/X_gt.pt')
    torch.save(U_ALL, 'Sim_baseline/traj/U_gt.pt')


# generate and save trajectories
generate_traj(F, H, T)