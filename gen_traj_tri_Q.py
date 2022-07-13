import torch
from math import ceil, pi
from Linear_sysmdl import System_Model_NE
from param_lin import range_nu_db, range_prec_r_db, F, H, NB, dim_x, dim_y, m1x_0, m2x_0, T
import matplotlib.pyplot as plt

# generate a tri wave
def gen_tri_wave(low, high, Pd, T):
    tri_wave_left = torch.linspace(start=low, end=high, steps=int(Pd/2))
    tri_wave_right = torch.linspace(start=high, end=low, steps=int(Pd/2))
    tri_wave = torch.cat([tri_wave_left, tri_wave_right]).repeat(ceil(T/Pd))
    tri_wave = tri_wave[0:T]
    return tri_wave

# generate of sequence of Q(t)
def gen_tri_Q(nu_dB, prec_r_dB, period_Q, T):
    # convert dB to linear scaling
    q_square_mean = pow(10.0, (nu_dB-prec_r_dB)/10)
    low = q_square_mean*0.5
    high = q_square_mean*1.5
    
    # Sequence of Q(t)=r(t)*I
    Q_tri = torch.empty(size=[dim_y, dim_y, T])
    r_square = gen_tri_wave(low, high, period_Q, T) 
    for t in range(0, T):
        Q_tri[:,:, t] = r_square[t]*torch.eye(2)
    return Q_tri


def generate_traj(NB, F, H, period_Q, T, opath) -> None:
    nus = len(range_nu_db)
    rs = len(range_prec_r_db)
    
    # stores Y-trajectories for each pair [nu, r]
    # Y_ALL[i, j, :, :, :] contains trajectories for [nu_i, r_j]
    Y_ALL = torch.empty(size=[nus, rs, NB, dim_y, T])

    # stores U-trajectories for each pair [nu, r]
    U_ALL = torch.empty(size=[nus, rs, NB, dim_x, T])

    # stores X-trajectories for each pair [nu, r]
    X_ALL = torch.empty(size=[nus, rs, NB, dim_x, T])

    # stores q(t) for each pair [nu, r]
    Q_ALL = torch.empty(size=[nus, rs, NB, T])

    for idx_nu, nu_db in enumerate(range_nu_db): 
        for idx_r, prec_r_db in enumerate(range_prec_r_db):
            # convert dB to linear scaling
            r = pow(10, -prec_r_db/20)
            q = pow(10, (nu_db-prec_r_db)/20)

            # periodic Q
            Q = gen_tri_Q(nu_db, prec_r_db, period_Q, T)
            # const R
            R = r*r*torch.eye(dim_y)
            R = R.unsqueeze(2).repeat(1, 1, T)
            # model
            lin_model = System_Model_NE(F, H, T, Q, R)

            # generate trajectories for current pair [nu, r]
            lin_model.init_sequence(m1x_0, m2x_0)
            lin_model.generate_nuv_batch(NB, T)

            # save trajectories for current pair [nu, r]
            Y_ALL[idx_nu, idx_r, :, :, :] = lin_model.Y
            U_ALL[idx_nu, idx_r, :, :, :] = lin_model.U
            X_ALL[idx_nu, idx_r, :, :, :] = lin_model.X
    # save files
    torch.save(Y_ALL, opath+'Y_ALL.pt')
    torch.save(X_ALL, opath+'X_ALL.pt')
    torch.save(U_ALL, opath+'U_ALL.pt')
    torch.save(Q_ALL, opath+'Q_ALL.pt')
    

# generate and save trajectories
if __name__ == '__main__':
    generate_traj(1000, F, H, period_Q=20, T=100, opath='Sim_tri_Q/Pd20/traj/')
    generate_traj(1000, F, H, period_Q=50, T=100, opath='Sim_tri_Q/Pd50/traj/')
    generate_traj(1000, F, H, period_Q=100, T=100, opath='Sim_tri_Q/Pd100/traj/')
    

