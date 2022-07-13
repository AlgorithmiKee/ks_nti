import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from Linear_KF import KalmanFilter, KalmanFilter_NE, KalmanFilter_nti
from RTS_Smoother import RTS_Smoother_NE, rts_smoother_in, RTS_Smoother_nti
from numpy import inf
import torch.nn as nn
from local_fit import LocalFitter, ExpWin_DS

class RTS_NUV:
    def __init__(self, assumed_model):
        self.KF = KalmanFilter_NE(assumed_model)
        self.RTS = RTS_Smoother_NE(assumed_model)

        self.F = assumed_model.F
        self.F_T = torch.transpose(self.F, 0, 1)
        self.m = assumed_model.m

        # Q[:,:,t] is time-variant!
        self.Q = assumed_model.Q_evo

        self.H = assumed_model.H
        self.H_T = torch.transpose(self.H, 0, 1)
        self.n = assumed_model.n
        
        # R[:,:,t] is time-variant!
        self.R = assumed_model.R_evo

        self.T = assumed_model.T

    def init_KF(self, m1x_0, m2x_0):
        self.KF.init_sequence(m1x_0, m2x_0)

    def set_R(self, R_new):
        self.R = R_new
        self.KF.R = R_new
        self.RTS.R = R_new

    def set_Q(self, Q_new):
        self.Q = Q_new
        self.KF.Q = Q_new
        self.RTS.Q = Q_new

    def smooth_unknownR(self, y_obs, itr = 20, r_init=1.0, win=None, forget_itr=0.0): 
        T = self.T
        loss_fun = nn.MSELoss(reduction='mean')
        # initialization
        R_init = r_init*r_init*torch.eye(self.n)
        R_init = R_init.unsqueeze(2).repeat(1, 1, T)
        self.set_R(R_init)

        # param for stopping condition
        s_x_prev = float('inf')*torch.ones(size=[self.m, self.T])
        MSE_prev = float('inf')

        for i in range(itr):
            # E-step
            self.fwd_bwd_filter_unknownR(y_obs)
            # M-step
            self.update_R(y_obs, forget_itr)
            # Add moving average if required so
            if(win is not None):
                R_ma = self.moving_avg(self.R, win)
                self.set_R(R_ma)
            # stopping condition
            s_x_curr = self.RTS.s_x
            MSE_curr = loss_fun(s_x_prev, s_x_curr)
            if(MSE_prev - MSE_curr < 1e-6):
                break
            else:
                s_x_prev = s_x_curr
                MSE_prev = MSE_curr

    def fwd_bwd_filter_unknownR(self, y_obs):
        T = self.T
        u_prior = torch.zeros(self.m, T)
        Q_prior = self.Q
        # fwd recursion
        self.KF.generate_sequence(y_obs, T)        
        # bwd recursion
        self.RTS.generate_sequence_in(self.KF.x, self.KF.sigma, u_prior, Q_prior, T)
    
    def fwd_bwd_filter_unknownQ(self, y_obs):
        T = self.T
        u_prior = torch.zeros(self.m, T)
        Q_prior = self.Q
        # fwd recursion
        self.KF.generate_sequence_plus(y_obs, T)        
        # bwd recursion
        self.RTS.generate_sequence_in(self.KF.x, self.KF.sigma, u_prior, Q_prior, T, cross_x1x0=True, KG_end=self.KF.KG_all[:, :, -1])

    def moving_avg(self, x, win):
        x_ma = torch.empty_like(x)
        T = x.size()[-1]
        for t in range(0, win+1):
            x_ma[:, :, t] = torch.mean(x[:, :, 0:t+win+1], dim=2)
        for t in range(win+1, T-win-1):
            x_ma[:, :, t] = x_ma[:, :, t-1] + (x[:, :, t+win]-x[:, :, t-1-win])/(2*win+1)
        for t in range(T-win-1, T):
            x_ma[:, :, t] = torch.mean(x[:, :, t-win:], dim=2)
        return x_ma    
    
    def update_R(self, y_obs, forget_itr=0.0, total_avg=False):
        T = y_obs.size()[1]
        R_new = torch.empty(size=[self.n, self.n, T])
        for t in range(T):
            yt = y_obs[:, t]
            xt = self.RTS.s_x[:, t]
            sigmat = self.RTS.s_sigma[:, :, t]
            yt = yt.unsqueeze(1)
            xt = xt.unsqueeze(1)

            Uyy = self.uyy(yt)
            Uxy = self.uxy(xt, yt)
            Uyx = torch.transpose(Uxy, 0, 1)
            Uxx = self.uxx(xt, sigmat)

            Rt = torch.matmul(self.H, Uxx)
            Rt = torch.matmul(Rt, self.H_T)
            Rt = Rt + Uyy - torch.matmul(self.H, Uxy) - torch.matmul(Uyx, self.H_T)
            # Rt = self.enf_symm(Rt)
            R_new[:, :, t] = (1-forget_itr)*Rt + forget_itr*self.R[:, :, t]
        ### assuming const Rt:
        if total_avg:
            R_avg = torch.mean(R_new, dim=2)
            R_new = R_avg.unsqueeze(2).repeat(1, 1, T)
        self.set_R(R_new)
    
    # helper for estimating Rt
    def uyy(self, yt):
        yt_T = torch.transpose(yt, 0, 1)
        return torch.matmul(yt, yt_T)
    
    # helper for estimating Rt
    def uxy(self, xt, yt):
        yt_T = torch.transpose(yt, 0, 1)
        return torch.matmul(xt, yt_T)
    
    # computes E[xt*xt^T]
    def uxx(self, xt, sigmat):
        xt_T = torch.transpose(xt, 0, 1)
        return torch.matmul(xt, xt_T) + sigmat
    
    def smooth_unknownQ(self, y_obs, itr=20, q_init=1.0, win=None, forget_itr=0.0): 
        T = self.T
        loss_fun = nn.MSELoss(reduction='mean')

        # initialization
        Q_init = q_init*q_init*torch.eye(self.n)
        Q_init = Q_init.unsqueeze(2).repeat(1, 1, T)
        self.set_Q(Q_init)

        # param for stopping condition
        s_x_prev = float('inf')*torch.ones(size=[self.m, self.T])
        MSE_prev = float('inf')

        # EM-Loop
        for i in range(itr):
            # E-step
            self.fwd_bwd_filter_unknownQ(y_obs)
            # M-step
            self.update_Q(forget_itr)
            # Add moving average if required so
            if(win is not None):
                Q_ma = self.moving_avg(self.Q, win)
                self.set_Q(Q_ma)
            # stopping condition
            s_x_curr = self.RTS.s_x
            MSE_curr = loss_fun(s_x_prev, s_x_curr)
            if(MSE_prev - MSE_curr < 1e-9):
                break
            else:
                s_x_prev = s_x_curr
                MSE_prev = MSE_curr
    
    # enforce symmetry
    def enf_symm(self, Q):
        if (Q==torch.transpose(Q, 0, 1)).all():
            return Q
        else:
            off_diag = (Q[0, 1]+Q[1, 0])/2
            Q[0, 1] = off_diag
            Q[1, 0] = off_diag
            return Q

    def update_Q(self, forget_itr=0.0, total_avg=False):
        T = self.T
        Q_new = torch.empty(size=[self.m, self.m, T])
        Q_new[:,:,-1] = self.Q[:,:,-1]
        for t in range(0, T-1):
            xt = self.RTS.s_x[:, t]
            xt = xt.unsqueeze(1)
            xt_T = torch.transpose(xt, 0, 1)       
            sigmat = self.RTS.s_sigma[:, :, t]

            x1t = self.RTS.s_x[:, t+1]
            x1t = x1t.unsqueeze(1)
            sigma1t = self.RTS.s_sigma[:, :, t+1]

            V11 = self.uxx(x1t, sigma1t)
            V00 = self.uxx(xt, sigmat)
            V10 = self.RTS.sigma_cross[:, :, t] + torch.matmul(x1t, xt_T)
            V01 = torch.transpose(V10, 0, 1)

            Qt = torch.matmul(self.F, V00)
            Qt = torch.matmul(Qt, self.F_T)
            Qt += V11 - torch.matmul(self.F, V01) - torch.matmul(V10, self.F_T)
            # enforce the symmetry of Qt
            Qt = self.enf_symm(Qt)
            Q_new[:, :, t] = (1-forget_itr)*Qt + forget_itr*self.Q[:, :, t]
        ### for const Qt ONLY:
        # Q_avg = torch.mean(Q_new, dim=2)
        # Q_new = Q_avg.unsqueeze(2).repeat(1, 1, T)
        ### end cost Qt ONLY
        self.set_Q(Q_new)

    
    ################################################################
    ###################### Embed Local Method ######################
    ################################################################
    
    # noise_var is 1D, has the dimension of T
    def smooth_local_split_linear(self, noise_var, gamma=0.945):
        # define split linear model
        A = torch.tensor([[1.0, 1.0],
                          [0.0, 1.0]])
        C = torch.tensor([1.0, 0.0]).unsqueeze(0)
        split_linear_model = LocalFitter(A, C, ExpWin_DS(), A, C)

        # post multiplier for local fit
        pm_conti_y = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        # fit split linear model to noise variance
        split_linear_model.fit(noise_var.unsqueeze(0), gamma, gamma, post_mult=pm_conti_y)
        noise_var_smth = split_linear_model.generate_signal().squeeze()
        return noise_var_smth      

         
    def smooth_unknownQ_LF(self, y_obs, itr=80, q_init=1.0, forget_itr=0.0): 
        T = self.T
        loss_fun = nn.MSELoss(reduction='mean')

        # initialization
        Q_init = q_init*q_init*torch.eye(self.n)
        Q_init = Q_init.unsqueeze(2).repeat(1, 1, T)
        self.set_Q(Q_init)
        
        # param for stopping condition
        s_x_prev = float('inf')*torch.ones(size=[self.m, self.T])
        MSE_prev = float('inf')        
        
        for i in range(itr):
            # E-step
            self.fwd_bwd_filter_unknownQ(y_obs)
            # M-step
            self.update_Q(forget_itr)
            
            ##############################
            ### Apply Local Model Fit ####
            ##############################
            # prepare for LF
            diag_Q = torch.diagonal(self.Q)
            q_square = torch.mean(diag_Q, dim=1)
            # LF: smooth q_square
            q_square = self.smooth_local_split_linear(q_square, gamma=0.98)
            Q_smooth = torch.zeros_like(self.Q)
            Q_smooth[0,0,:] = q_square
            Q_smooth[1,1,:] = q_square
            self.set_Q(Q_smooth)

            # stopping condition
            s_x_curr = self.RTS.s_x
            MSE_curr = loss_fun(s_x_prev, s_x_curr)
            if(i>2 and MSE_prev - MSE_curr < 1e-9):
                break
            else:
                s_x_prev = s_x_curr
                MSE_prev = MSE_curr
    
    
    def smooth_unknownR_LF(self, y_obs, itr = 20, r_init=1.0, forget_itr=0.0): 
        T = self.T
        loss_fun = nn.MSELoss(reduction='mean')
        # initialization
        R_init = r_init*r_init*torch.eye(self.n)
        R_init = R_init.unsqueeze(2).repeat(1, 1, T)
        self.set_R(R_init)

        # param for stopping condition
        s_x_prev = float('inf')*torch.ones(size=[self.m, self.T])
        MSE_prev = float('inf')

        for i in range(itr):
            # E-step
            self.fwd_bwd_filter_unknownR(y_obs)
            # M-step
            self.update_R(y_obs, forget_itr)
            
            ##############################
            ### Apply Local Model Fit ####
            ##############################
            # prepare for LF
            diag_R = torch.diagonal(self.R)
            r_square = torch.mean(diag_R, dim=1)
            # LF: smooth r_square
            r_square = self.smooth_local_split_linear(r_square)
            R_smooth = torch.zeros_like(self.R)
            R_smooth[0,0,:] = r_square
            R_smooth[1,1,:] = r_square
            self.set_R(R_smooth)

            # stopping condition
            s_x_curr = self.RTS.s_x
            MSE_curr = loss_fun(s_x_prev, s_x_curr)
            if(MSE_prev - MSE_curr < 1e-9):
                break
            else:
                s_x_prev = s_x_curr
                MSE_prev = MSE_curr


    def track_unknownQ_LF(self, y_obs, itr=80, q_init=1.0, forget_itr=0.0, x_gt=None): 

        T = self.T
        loss_fun = nn.MSELoss(reduction='mean')

        # show q(t)
        q2 = torch.zeros(size=[itr, T])
        s_q2 = torch.zeros(size=[itr, T])
        # show MSE
        MSEs_track = torch.zeros(size=[itr])


        # initialization
        Q_init = q_init*q_init*torch.eye(self.n)
        Q_init = Q_init.unsqueeze(2).repeat(1, 1, T)
        self.set_Q(Q_init)
        
        # param for stopping condition
        s_x_prev = float('inf')*torch.ones(size=[self.m, self.T])
        MSE_prev = float('inf')        
        
        for i in range(itr):
            # E-step
            self.fwd_bwd_filter_unknownQ(y_obs)
            # M-step
            self.update_Q(forget_itr)
            
            ##############################
            ### Apply Local Model Fit ####
            ##############################
            # prepare for LF
            diag_Q = torch.diagonal(self.Q)
            q_square = torch.mean(diag_Q, dim=1)

            ####### track raw q2 ######
            q2[i, :] = q_square

            # LF: smooth q_square
            q_square = self.smooth_local_split_linear(q_square, gamma=0.98)

            ####### track smoothed q2 ######
            s_q2[i, :] = q_square

            Q_smooth = torch.zeros_like(self.Q)
            Q_smooth[0,0,:] = q_square
            Q_smooth[1,1,:] = q_square
            self.set_Q(Q_smooth)

            ####### track MSE  ######
            MSEs_track[i] = loss_fun(x_gt, self.RTS.s_x)

            # stopping condition
            s_x_curr = self.RTS.s_x
            MSE_curr = loss_fun(s_x_prev, s_x_curr)
            if(i>2 and MSE_prev - MSE_curr < 1e-9):
                break
            else:
                s_x_prev = s_x_curr
                MSE_prev = MSE_curr

        return q2[0:i, :], s_q2[0:i, :], 10*torch.log10(MSEs_track[0:i]), i

    ######################################################
    ################# smooth ecg signal ##################
    ######################################################
    # Unknown const R using total average
    # Unknown varying Q using moving average
    def smooth_unkQR(self, y_obs, R_init, Q_init, itr=80, win_q=5, forget_q=0.0, forget_r=0.0): 
        # initialization for R, Q
        self.set_R(R_init)
        self.set_Q(Q_init)

        # EM-loop for R and Q
        for i in range(itr):
            # E-step
            self.fwd_bwd_filter_unknownQ(y_obs)
            # M-step for R
            self.update_R(y_obs, forget_itr=forget_r, total_avg=True)
            # M-step for Q
            self.update_Q(forget_itr=forget_q)
            # Add moving average if required so
            if(win_q is not None):
                Q_ma = self.moving_avg(self.Q, win_q)
                self.set_Q(Q_ma)
        





class RTS_UV_nti(RTS_NUV):
    def __init__(self, assumed_model):
        self.KF = KalmanFilter_nti(assumed_model)
        self.RTS = RTS_Smoother_nti(assumed_model)

        # F[:,:,t] is time-variant!
        self.F = assumed_model.F
        self.F_T = torch.transpose(self.F, 0, 1)
        self.m = assumed_model.m

        # Q[:,:,t] is time-variant!
        self.Q = assumed_model.Q_evo

        # H[:,:,t] is time-variant!
        self.H = assumed_model.H
        self.H_T = torch.transpose(self.H, 0, 1)
        self.n = assumed_model.n
        
        # R[:,:,t] is time-variant!
        self.R = assumed_model.R_evo

        self.T = assumed_model.T


    def fwd_bwd_filter_unknownR(self, y_obs, u_in):
        T = self.T
        Q_prior = self.Q
        # fwd recursion
        self.KF.generate_sequence(y_obs, u_in, T)        
        # bwd recursion
        self.RTS.generate_sequence_cross(self.KF.x, self.KF.sigma, u_in, Q_prior, T, cross_x1x0=True, KG_end=self.KF.KG)
    
    
    def fwd_bwd_filter_unknownQ(self, y_obs, u_in):
        T = self.T
        Q_prior = self.Q
        # fwd recursion
        self.KF.generate_sequence(y_obs, u_in, T)        
        # bwd recursion (no input estimation)
        self.RTS.generate_sequence_cross(self.KF.x, self.KF.sigma, u_in, Q_prior, T, cross_x1x0=True, KG_end=self.KF.KG)
   
    
    def update_const_R(self, y_obs):
        T = y_obs.size()[1]
        R_new = torch.empty(size=[self.n, self.n, T])
        for t in range(T):
            yt = y_obs[:, t]
            xt = self.RTS.s_x[:, t]
            sigmat = self.RTS.s_sigma[:, :, t]
            yt = yt.unsqueeze(1)
            xt = xt.unsqueeze(1)

            Uyy = self.uyy(yt)
            Uxy = self.uxy(xt, yt)
            Uyx = torch.transpose(Uxy, 0, 1)
            Uxx = self.uxx(xt, sigmat)

            Ht = self.H[:, :, t]
            Ht_T = self.H_T[:, :, t]
            Rt = torch.matmul(Ht, Uxx)
            Rt = torch.matmul(Rt, Ht_T)
            Rt = Rt + Uyy - torch.matmul(Ht, Uxy) - torch.matmul(Uyx, Ht_T)
            R_new[:, :, t] = Rt 
        # average out all R(1),...,R(T)
        R_avg = torch.mean(R_new, dim=2)
        R_new = R_avg.unsqueeze(2).repeat(1, 1, T)
        self.set_R(R_new)


    def update_Q(self, forget_itr):
        T = self.T
        Q_new = torch.empty(size=[self.m, self.m, T])
        Q_new[:,:,-1] = self.Q[:,:,-1]
        for t in range(0, T-1):
            xt = self.RTS.s_x[:, t]
            xt = xt.unsqueeze(1)
            xt_T = torch.transpose(xt, 0, 1)       
            sigmat = self.RTS.s_sigma[:, :, t]

            x1t = self.RTS.s_x[:, t+1]
            x1t = x1t.unsqueeze(1)
            sigma1t = self.RTS.s_sigma[:, :, t+1]

            Ft = self.F[:, :, t]
            Ft_T = self.F_T[:, :, t]

            V11 = self.uxx(x1t, sigma1t)
            V00 = self.uxx(xt, sigmat)
            V10 = self.RTS.sigma_cross[:, :, t] + torch.matmul(x1t, xt_T)
            V01 = torch.transpose(V10, 0, 1)

            Qt = torch.matmul(Ft, V00)
            Qt = torch.matmul(Qt, Ft_T)
            Qt += V11 - torch.matmul(Ft, V01) - torch.matmul(V10, Ft_T)
            # # enforce the symmetry of Qt
            # Qt = self.enf_symm(Qt)
            Q_new[:, :, t] = (1-forget_itr)*Qt + forget_itr*self.Q[:, :, t]
        ### for const Qt ONLY:
        # Q_avg = torch.mean(Q_new, dim=2)
        # Q_new = Q_avg.unsqueeze(2).repeat(1, 1, T)
        ### end cost Qt ONLY
        self.set_Q(Q_new)


    ######################################################
    ################# smooth ecg signal ##################
    ######################################################
    # Unknown const R using total average
    # Unknown varying Q using moving average
    def smooth_ecg(self, y_obs, u_in, itr_r=20, itr_q=80, r_init=1.0, q_init=1.0, win_q=None, forget_q=0.0): 
        T = self.T
        # initialization for R
        R_init = r_init*r_init*torch.eye(self.n)
        R_init = R_init.unsqueeze(2).repeat(1, 1, T)
        self.set_R(R_init)

        # initialization for Q
        Q_init = q_init*q_init*torch.eye(self.n)
        Q_init = Q_init.unsqueeze(2).repeat(1, 1, T)
        self.set_Q(Q_init)
        
        # param for stopping condition
        threshold_MSE = 1e-6
        s_x_prev = float('inf')*torch.ones(size=[self.m, self.T])
        loss_fun = nn.MSELoss(reduction='mean')


        # EM-loop for R and Q
        for i in range(itr_r):
            # E-step
            self.fwd_bwd_filter_unknownQ(y_obs, u_in)
            # M-step for R
            self.update_const_R(y_obs)
            # M-step for Q
            self.update_Q(forget_itr=forget_q)
            # Add moving average if required so
            if(win_q is not None):
                Q_ma = self.moving_avg(self.Q, win_q)
                self.set_Q(Q_ma)
            # stopping condition
            s_x_curr = self.RTS.s_x
            MSE = loss_fun(s_x_prev, s_x_curr)
            if(MSE < threshold_MSE):
                break
            else:
                s_x_prev = s_x_curr
        
        # EM-loop for Q
        for i in range(itr_q - itr_r):
            # E-step
            self.fwd_bwd_filter_unknownQ(y_obs, u_in)
            # M-step for Q
            self.update_Q(forget_itr=forget_q)
            # Add moving average if required so
            if(win_q is not None):
                Q_ma = self.moving_avg(self.Q, win_q)
                self.set_Q(Q_ma)
            # stopping condition
            s_x_curr = self.RTS.s_x
            MSE = loss_fun(s_x_prev, s_x_curr)
            if(MSE < threshold_MSE):
                break
            else:
                s_x_prev = s_x_curr



# For constant Q and Rs
class RTS_Simple_NUV:
    def __init__(self, AssumedModel):
        self.KF = KalmanFilter(AssumedModel)
        self.RTS = rts_smoother_in(AssumedModel)

        self.F = AssumedModel.F
        self.F_T = torch.transpose(self.F, 0, 1)
        self.m = AssumedModel.m

        self.Q = AssumedModel.Q

        self.H = AssumedModel.H
        self.H_T = torch.transpose(self.H, 0, 1)
        self.n = AssumedModel.n

        self.R = AssumedModel.R

        self.T = AssumedModel.T
        self.T_test = AssumedModel.T_test

        self.init_KF(AssumedModel.m1x_0, AssumedModel.m2x_0)
    
    # set initial condition for KF
    # m1x_0: first moment of x0
    # m2x_0: second moment of x0
    def init_KF(self, m1x_0, m2x_0):
        self.KF.InitSequence(m1x_0, m2x_0)
    
    def set_R(self, R_new):
        self.R = R_new
        self.KF.R = R_new
        self.RTS.R = R_new  
    
    def set_Q(self, Q_new):
        self.Q = Q_new
        self.KF.Q = Q_new
        self.RTS.Q = Q_new      
    
    
    #########################################################
    ############# EM algortihm for estimating Q #############
    #########################################################
    def llh(self, mtx):
        det_mtx = torch.det(mtx)
        return torch.log10(det_mtx).item()

    def correct_Q(self, Q):
        if (Q==torch.transpose(Q, 0, 1)).all():
            return Q
        else:
            off_diag = (Q[0, 1]+Q[1, 0])/2
            Q[0, 1] = off_diag
            Q[1, 0] = off_diag
            return Q

    # M-step for estimating Q: Update the estimate of Q given the smoothed data，
    # read the book for reference:
    #   Applied Optimum Signal Processing},
    #   Sophocles J. Orfanidis
    #   pages: 614--677
    def update_Q(self):
        Vxx = self.vxx()
        V11 = self.v11()
        V1x = self.v1x()
        Vx1 = torch.transpose(V1x, 0, 1)
        Q_new = torch.matmul(self.F, Vxx)
        Q_new = torch.matmul(Q_new, self.F_T)
        Q_new += V11 
        Q_new -= torch.matmul(self.F, Vx1)
        Q_new -= torch.matmul(V1x, self.F_T)
        Q_new = self.correct_Q(Q_new)

        # Update model parameters
        self.set_Q(Q_new)
    

    # E-step for estimating Q 
    def fwd_bwd_filter_unkQ(self, y_observed):
        # fwd recursion
        self.KF.GenerateSequence_plus(y_observed, self.T)
        # For unknown Q, Q_prior = variance estimation of current iteration
        # For unknown R, Q_prior = true Q
        u_prior = torch.zeros(self.m, self.T)
        Q_prior = self.Q.unsqueeze(2).repeat(1, 1, self.T)
        # bwd recursion
        self.RTS.GenerateSequence_in(self.KF.x, self.KF.sigma, u_prior, Q_prior, self.T, cross_x1x0=True, KG_end=self.KF.KG_all[:, :, -1])

    # E-step for estimating R
    def fwd_bwd_filter_unkR(self, y_observed):
        # fwd recursion
        self.KF.GenerateSequence(y_observed, self.T)
        # For unknown Q, Q_prior = variance estimation of current iteration
        # For unknown R, Q_prior = true Q
        u_prior = torch.zeros(self.m, self.T)
        Q_prior = self.Q.unsqueeze(2).repeat(1, 1, self.T)
        # bwd recursion
        self.RTS.GenerateSequence_in(self.KF.x, self.KF.sigma, u_prior, Q_prior, self.T, cross_x1x0=False)



    # Run EM algorithm once to estimate Q
    # Return: none
    # itr: run EM algorithm itr times
    # y_observed: observtion trajectory based on true model
    def smooth_unknownQ(self, y_observed, itr, q_init=1.0):
        # initialization
        Q_init = q_init*q_init*torch.eye(self.m)
        self.set_Q(Q_init)

        # EM for estimating Q
        for k in range(itr):
            # E-step
            self.fwd_bwd_filter_unkQ(y_observed)
            # M-step
            self.update_Q()
            
            # debug
            # print('itr: ', k, ', llh=', self.eval_llh())
            # print('Q=\n', self.Q.numpy())


    #########################################################
    ############# EM algortihm for estimating R #############
    #########################################################

    # M-step for estimating R: Update the estimate of Q given the smoothed data，
    # y_obs: oberveration trajectory
    # read the book for reference:
    #   Applied Optimum Signal Processing},
    #   Sophocles J. Orfanidis
    #   pages: 614--677
    def update_R(self, y_obs):
        Uyy = self.uyy(y_obs)
        Uyx = self.uyx(y_obs)
        Uxy = torch.transpose(Uyx, 0, 1)
        Uxx = self.uxx()
        R_new = torch.matmul(self.H, Uxx)
        R_new = torch.matmul(R_new, self.H_T)
        R_new = R_new + Uyy - torch.matmul(self.H, Uxy) - torch.matmul(Uyx, self.H_T)

        self.set_R(R_new)

    # Run EM algorithm once to estimate R
    # Return: none
    # itr: run EM algorithm itr times
    # y_observed: observtion trajectory based on true model
    def smooth_unknownR(self, y_obs, itr=50, r_init=10.0):
        # initialization
        R_init = r_init*r_init*torch.eye(self.n)
        self.R = R_init
        self.KF.R = R_init
        self.RTS.R = R_init

        for i in range(itr):
            # E-step
            self.fwd_bwd_filter_unkR(y_obs)
            # M-step
            self.update_R(y_obs)
    
    ##############################################################################
    ######### evaluators and stoppers
    ##############################################################################
    # compares the smoothed state trajectories at current iteration and the previous iteration
    # if the mse of them < threshold, then stop
    def stop(threshold, itr=300):
        # if itr
        loss = nn.MSELoss(reduction='mean')
        return False


    # evaluate MSE of input
    # target_u: true input trajectory
    def mse_input(self, target_u):
        loss = nn.MSELoss(reduction='mean')
        mse = loss(self.RTS.s_u, target_u).item()
        return mse

    # evaluate MSE of state
    # target_x: true input trajectory
    def mse_state(self, target_x):
        loss = nn.MSELoss(reduction='mean')
        mse = loss(self.RTS.s_x, target_x).item()
        return mse

    # evaluate using log likelihood
    # y_obs: observed output trajectory
    def eval_llh(self):
        llh = -torch.log10(torch.det(self.Q))
        return llh

    ##############################################################################
    ######### Helpers for estimating Q
    ##############################################################################
    # helper function for M-step for estimating Q
    def vxx(self):
        s_x = self.RTS.s_x[:, 0:-1]
        s_sigma = self.RTS.s_sigma[:, :, 0:-1]
        Vxx = torch.matmul(s_x, torch.transpose(s_x, 0, 1)) + torch.sum(s_sigma, 2)
        Vxx /= self.T-1    
        return Vxx    
    
    # helper function for M-step for estimating Q
    def v11(self):
        s_x = self.RTS.s_x[:, 1:]
        s_sigma = self.RTS.s_sigma[:, :, 1:]
        V11 = torch.matmul(s_x, torch.transpose(s_x, 0, 1)) + torch.sum(s_sigma, 2)
        V11 /= self.T-1
        return V11
    
    # helper function for M-step for estimating Q
    def v1x(self):
        s_x1 = self.RTS.s_x[:, 1:]
        s_x0 = self.RTS.s_x[:, 0:-1]
        V1x = torch.matmul(s_x1, torch.transpose(s_x0, 0, 1))/(self.T-1)
        V1x += torch.mean(self.RTS.sigma_cross, dim=2)
        return V1x

    ##############################################################################
    ######### Helpers for estimating R
    ##############################################################################
    def uxx(self):
        s_x = self.RTS.s_x
        s_sigma = self.RTS.s_sigma
        Uxx = torch.matmul(s_x, torch.transpose(s_x, 0, 1)) + torch.sum(s_sigma, 2)
        Uxx /= self.T
        return Uxx
    
    def uyx(self, y_obs):
        s_x = self.RTS.s_x
        Uyx = torch.matmul(y_obs, torch.transpose(s_x, 0, 1))
        Uyx /= self.T
        return Uyx
    
    def uyy(self, y_obs):
        Uyy = torch.matmul(y_obs, torch.transpose(y_obs, 0, 1))
        Uyy /= self.T
        return Uyy

