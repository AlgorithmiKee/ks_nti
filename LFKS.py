import torch
from local_fit import LocalFitter, ExpWin_DS
from RTS_NUV import RTS_NUV, RTS_UV_nti
from Linear_sysmdl import System_Model_nti, System_Model_NE
from Linear_sysmdl import System_Model_NE
from Linear_KF import KalmanFilter_NE
from RTS_Smoother import RTS_Smoother_NE


def get_init_ecg():
    ecg_gt = torch.load('Sim_ecg/traj/ECG_gt.pt')
    return ecg_gt[0].unsqueeze(0)

# y_ecg_noisy is 1D tensor!
def extract_slopes(y_ecg_noisy, gamma):
    A = torch.tensor([[1.0, 1.0],
                      [0.0, 1.0]])
    C = torch.tensor([1.0, 0.0]).unsqueeze(0)
    split_linear_model = LocalFitter(A, C, ExpWin_DS(), A, C)

    pm_conti_y = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    # fit the split model (fwd and bwd recursion)
    split_linear_model.fit(y_ecg_noisy.unsqueeze(0), 
                        win_param_fwd=gamma, 
                        win_param_bwd=gamma, 
                        post_mult=pm_conti_y)             
    
    # left and right sided local slopes
    slope_l = -1.0*split_linear_model.x_optimal[1, :]
    slope_r = split_linear_model.x_optimal[3, :]
    # averge both slopes
    slope = (slope_l + slope_r)/2.0
    slope = slope.unsqueeze(0)

    return slope



# y_ecg_noisy is 1D tensor
def smooth_ecg_LFKS(y_ecg_noisy, gamma,  
                    itr_r=60,   itr_q=80, 
                    r_init=1.0, q_init=1.0, 
                    win_q=2, forget_q=0.0):
    T_ecg = y_ecg_noisy.size()[0]

    # slopes are interpreted as inputs fed into sytem model
    u_in = extract_slopes(y_ecg_noisy, gamma)

    # define the system model for ECG
    F = 1.0*torch.ones(size=[1, 1, T_ecg])
    H = 1.0*torch.ones(size=[1, 1, T_ecg])
    Q = 1.0*torch.ones(size=[1, 1, T_ecg])
    R = 1.0*torch.ones(size=[1, 1, T_ecg])

    m1x_0 = get_init_ecg()
    m2x_0 = torch.tensor([[0.0]])

    ecg_model = System_Model_nti(F, H, T_ecg, Q, R)
    ecg_model.init_sequence(m1x_0, m2x_0)

    # set up the EM algorithm
    ks_uv = RTS_UV_nti(ecg_model)
    ks_uv.init_KF(m1x_0, m2x_0)

    # run EM 
    ks_uv.smooth_ecg(y_ecg_noisy.unsqueeze(0), u_in, 
                     itr_r,   itr_q, 
                     r_init, q_init, 
                     win_q, forget_q
    )
    return ks_uv.RTS.s_x



# y_ecg_noisy is 1D tensor
def smooth_ecg_KS(y_ecg_noisy,  
                    itr_r=60,   itr_q=80, 
                    r_init=1.0, q_init=1.0, 
                    win_q=2, forget_q=0.0):
    T_ecg = y_ecg_noisy.size()[0]

    # slopes are interpreted as inputs fed into sytem model
    u_in = torch.zeros(size=[1, T_ecg])

    # define the system model for ECG
    F = 1.0*torch.ones(size=[1, 1, T_ecg])
    H = 1.0*torch.ones(size=[1, 1, T_ecg])
    Q = 1.0*torch.ones(size=[1, 1, T_ecg])
    R = 1.0*torch.ones(size=[1, 1, T_ecg])

    m1x_0 = get_init_ecg()
    m2x_0 = torch.tensor([[0.0]])

    ecg_model = System_Model_nti(F, H, T_ecg, Q, R)
    ecg_model.init_sequence(m1x_0, m2x_0)

    # set up the EM algorithm
    ks_uv = RTS_UV_nti(ecg_model)
    ks_uv.init_KF(m1x_0, m2x_0)

    # run EM 
    ks_uv.smooth_ecg(y_ecg_noisy.unsqueeze(0), u_in, 
                     itr_r,   itr_q, 
                     r_init, q_init, 
                     win_q, forget_q
    )
    return ks_uv.RTS.s_x


###########################################################
#######$$#### 2D modeling of ECG state vector ###$$########
###########################################################
def get_init_dervt1_ecg():
    ecg_gt = torch.load('Sim_ecg/traj/ECG_gt.pt')
    return (ecg_gt[1] - ecg_gt[0]).unsqueeze(0)


def smooth_ecg_KS_2D(y_ecg_noisy, R_init, Q_init, itr=80, win_q=5, forget_q=0.0, forget_r=0.0):
    T_ecg = y_ecg_noisy.size()[0]

    ######## define the system model for ECG ########
    F = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    H = torch.tensor([[1.0, 0.0]])
    Q = Q_init
    R = R_init

    # initial condition captures the amplitude and first order derivative
    m1x_0 = torch.zeros(size=[2])
    m1x_0[0], m1x_0[1] = get_init_ecg(), get_init_dervt1_ecg()
    m2x_0 = torch.zeros(size=[2, 2])

    ecg_model = System_Model_NE(F, H, T_ecg, Q, R)
    ecg_model.init_sequence(m1x_0, m2x_0)

    # set up the EM algorithm
    ks_uv2 = RTS_NUV(ecg_model)
    ks_uv2.init_KF(m1x_0, m2x_0)

    # smooth the ECG signal
    ks_uv2.smooth_unkQR(y_ecg_noisy.unsqueeze(0), R_init, Q_init, itr, win_q, forget_q, forget_r)
    return ks_uv2.RTS.s_x


# Helper for smooth_ecg_LFKS_2D
# y_ecg_noisy is 1D tensor!
def get_x_init_ecg(y_ecg_noisy, gamma):
    dervt0 = y_ecg_noisy.unsqueeze(0)
    dervt1 = extract_slopes(y_ecg_noisy, gamma)
    x_init = torch.cat([dervt0, dervt1], 0)
    return x_init


# Helper for smooth_ecg_LFKS_2D
# y_ecg_noisy is 1D tensor!
def get_sigma_init_ecg(y_ecg_noisy, ecg_model):
    T = ecg_model.T
    kf = KalmanFilter_NE(ecg_model)
    ks = RTS_Smoother_NE(ecg_model)
    kf.init_sequence(ecg_model.m1x_0, ecg_model.m2x_0)
    kf.generate_sequence(y_ecg_noisy.unsqueeze(0), T)
    ks.generate_sequence_cross(kf.x, kf.sigma, kf.KG, ecg_model.Q_evo, T)
    return ks.s_sigma, ks.sigma_cross


# y_ecg_noisy is 1D tensor!
def smooth_ecg_LFKS_2D(y_ecg_noisy, gamma, r0=1.0, q0=1.0, itr=80, win_q=5, forget_q=0.0, forget_r=0.0):
    T_ecg = y_ecg_noisy.size()[0]

    ######## define the system model for ECG ########
    F = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    H = torch.tensor([[1.0, 0.0]])
    Q = q0*q0*torch.tensor([[1/3, 1/2], [1/2, 1.0]]).unsqueeze(2).repeat(1, 1, T_ecg)
    R = r0*r0*torch.eye(1).unsqueeze(2).repeat(1, 1, T_ecg)

    # initial condition captures the amplitude and first order derivative
    m1x_0 = torch.zeros(size=[2])
    m1x_0[0], m1x_0[1] = get_init_ecg(), get_init_dervt1_ecg()
    m2x_0 = torch.zeros(size=[2, 2])

    ecg_model = System_Model_NE(F, H, T_ecg, Q, R)
    ecg_model.init_sequence(m1x_0, m2x_0)

    # set up the EM algorithm
    ks_uv2 = RTS_NUV(ecg_model)
    ks_uv2.init_KF(m1x_0, m2x_0)
    
    ######## init EM #######
    # -> get initial approximation of x-traj
    ks_uv2.RTS.s_x = get_x_init_ecg(y_ecg_noisy, gamma)
    # -> get initial approximation of Sigma-traj 
    ks_uv2.RTS.s_sigma, ks_uv2.RTS.sigma_cross = get_sigma_init_ecg(y_ecg_noisy, ecg_model)
    # -> get initial Q/R
    ks_uv2.update_R(y_ecg_noisy.unsqueeze(0), forget_itr=0.0, total_avg=True)
    ks_uv2.update_Q(forget_itr=0.0, total_avg=False)
    R_init, Q_init = ks_uv2.R, ks_uv2.Q

    ######## EM main loop (unknown const R, unknown var Q) ########
    ks_uv2.smooth_unkQR(y_ecg_noisy.unsqueeze(0), R_init, Q_init, itr, win_q, forget_q, forget_r)
    return ks_uv2.RTS.s_x