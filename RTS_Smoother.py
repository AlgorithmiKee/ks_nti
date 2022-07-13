"""# **Class: RTS Smoother**
Theoretical Linear RTS Smoother
"""
import torch


class rts_smoother:

    def __init__(self, SystemModel): 
        self.F = SystemModel.F;
        self.F_T = torch.transpose(self.F, 0, 1);
        self.m = SystemModel.m

        self.Q = SystemModel.Q;

        self.H = SystemModel.H;
        self.H_T = torch.transpose(self.H, 0, 1);
        self.n = SystemModel.n

        self.R = SystemModel.R;

        self.T = SystemModel.T;
        self.T_test = SystemModel.T_test;
    
    # Compute x_prior
    def Predict_x(self, filter_x):
        return torch.matmul(self.F, filter_x)
    
    # Computes sigma_prior
    def Predict_sigma(self, filter_sigma):
        filter_sigma_prior = torch.matmul(self.F, filter_sigma)
        filter_sigma_prior = torch.matmul(filter_sigma_prior, self.F_T) + self.Q    
        return filter_sigma_prior
    
    # Compute the Smoother Gain
    def SGain(self, filter_sigma):
        self.SG = torch.matmul(filter_sigma, self.F_T)
        filter_sigma_prior = self.Predict_sigma(filter_sigma)
        self.SG = torch.matmul(self.SG, torch.inverse(filter_sigma_prior))

    # Innovation for Smoother
    def S_Innovation(self, filter_x, filter_sigma):
        filter_x_prior = self.Predict_x(filter_x)
        filter_sigma_prior = self.Predict_sigma(filter_sigma)
        self.dx = self.s_m1x_nexttime - filter_x_prior
        self.dsigma = filter_sigma_prior - self.s_m2x_nexttime

    # Compute smoothed estimation of x and sigma backwardly in time
    def S_Correct(self, filter_x, filter_sigma):
        # Compute the 1-st moment
        self.s_m1x_nexttime = filter_x + torch.matmul(self.SG, self.dx)

        # Compute the 2-nd moment
        self.s_m2x_nexttime = torch.matmul(self.dsigma, torch.transpose(self.SG, 0, 1))
        self.s_m2x_nexttime = filter_sigma - torch.matmul(self.SG, self.s_m2x_nexttime)

    def S_Update(self, filter_x, filter_sigma):
        self.SGain(filter_sigma)
        self.S_Innovation(filter_x, filter_sigma)
        self.S_Correct(filter_x, filter_sigma)

        return self.s_m1x_nexttime,self.s_m2x_nexttime


    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, filter_x, filter_sigma, T):
        # Pre allocate an array for predicted state and variance
        self.s_x = torch.empty(size=[self.m, T])
        self.s_sigma = torch.empty(size=[self.m, self.m, T])

        self.s_m1x_nexttime = filter_x[:, T-1]
        self.s_m2x_nexttime = filter_sigma[:, :, T-1]
        self.s_x[:, T-1] = torch.squeeze(self.s_m1x_nexttime)
        self.s_sigma[:, :, T-1] = torch.squeeze(self.s_m2x_nexttime)

        for t in range(T-2,-1,-1):
            filter_xt = filter_x[:, t]
            filter_sigmat = filter_sigma[:, :, t]
            s_xt,s_sigmat = self.S_Update(filter_xt, filter_sigmat);
            self.s_x[:, t] = torch.squeeze(s_xt)
            self.s_sigma[:, :, t] = torch.squeeze(s_sigmat)

# linear rts smoother with input estimation capability
class rts_smoother_in(rts_smoother):
    def __init__(self, SystemModel):
        rts_smoother.__init__(self, SystemModel)

    # compute backwards gain for input estimate
    def IGain(self, filter_sigma, filter_Q):
        self.IG = torch.matmul(filter_Q, torch.eye(self.m))
        filter_sigma_prior = self.Predict_sigma(filter_sigma)
        self.IG = torch.matmul(self.IG, torch.inverse(filter_sigma_prior))

    # compute input estimate (both 1. and 2. moments)
    def I_Correct(self, filter_u, filter_Q):
        # compute 1st moment of u
        self.u_post = filter_u + torch.matmul(self.IG, self.dx)
        # compute 2nd moment of u
        self.Q_post = torch.matmul(self.IG, self.dsigma)
        self.Q_post = filter_Q + torch.matmul(self.Q_post, torch.transpose(self.dsigma, 0, 1))

    # this function can only be called AFTER S_Update
    def I_Update(self, filter_sigma, filter_u, filter_Q):
        self.IGain(filter_sigma, filter_Q)
        self.I_Correct(filter_u, filter_Q)
        return self.u_post, self.Q_post
    
    ### compute cross cov[x(t+1), x(t)]
    # sigma: smoothed 2nd moment from KF
    # KG: KG at last time step from KF
    def cross_sigma(self, sigma, KG, T):
        # allocate memory
        self.sigma_cross = torch.empty(size=[self.m, self.m, T-1])

        # initialisation
        sigma_cross_end = torch.eye(self.m) - torch.matmul(KG, self.H)
        sigma_cross_end = torch.matmul(sigma_cross_end, self.F)
        sigma_cross_end = torch.matmul(sigma_cross_end, sigma[:, :, T-2])
        self.sigma_cross[:, :, T-2] = sigma_cross_end

        # recursive computation
        for t in range(T-3, -1, -1):
            SG1 = torch.clone(self.SG_all[:, :, t+1]).detach()
            SG0 = torch.clone(self.SG_all[:, :, t]).detach()
            SG0_T = torch.transpose(SG0, 0, 1)
            d_Exx = self.sigma_cross[:, :, t+1] - torch.matmul(self.F, sigma[:, :, t+1])
            self.sigma_cross[:, :, t] = torch.matmul(SG1, d_Exx)
            self.sigma_cross[:, :, t] = torch.matmul(self.sigma_cross[:, :, t], SG0_T)
            self.sigma_cross[:, :, t] += torch.matmul(sigma[:, :, t+1], SG0_T)


    # Given the results from Kalman filtering,
    # smooth the state estimate and perform input estimation.
    # filter_x, filter_sigma: estimation of x and its covariance from Kalman filtering
    # fitler_u, filter_Q: prior estimate of noise mean and covariance
    # cross_x1x0: compute cross cov or not
    def GenerateSequence_in(self, filter_x, filter_sigma, filter_u, filter_Q, T, cross_x1x0=False, KG_end=None):
        # Pre allocate an array for smoothed state and variance
        self.s_x = torch.empty(size=[self.m, T])
        self.s_sigma = torch.empty(size=[self.m, self.m, T])
        
        # Pre allocate arrays for updated input estimate and its variance
        self.s_u = torch.empty(size=[self.m, T])
        self.s_Q = torch.empty(size=[self.m, self.m, T])
        
        # Pre allocate array for all smoothing gain
        self.SG_all = torch.empty(size=[self.m, self.m, T])

        # Initialisation for smoothing
        self.s_m1x_nexttime = filter_x[:, T-1]
        self.s_m2x_nexttime = filter_sigma[:, :, T-1]
        self.s_x[:, T-1] = torch.squeeze(self.s_m1x_nexttime)
        self.s_sigma[:, :, T-1] = torch.squeeze(self.s_m2x_nexttime)

        for t in range(T-2,-1,-1):
            filter_xt = torch.squeeze(filter_x[:, t])
            filter_sigmat = torch.squeeze(filter_sigma[:, :, t])
            filter_ut = torch.squeeze(filter_u[:, t])
            filter_Qt = torch.squeeze(filter_Q[:, :, t])

            # perform smoothing and input estimation
            s_xt,s_sigmat = self.S_Update(filter_xt, filter_sigmat)
            s_ut, s_Qt = self.I_Update(filter_sigmat, filter_ut, filter_Qt)

            self.s_x[:, t] = torch.squeeze(s_xt)
            self.s_sigma[:, :, t] = torch.squeeze(s_sigmat)
            self.s_u[:, t+1] = torch.squeeze(s_ut)
            self.s_Q[:, :, t+1] = torch.squeeze(s_Qt)
            self.SG_all[:, :, t] = torch.clone(self.SG).detach()
        if cross_x1x0:
            self.cross_sigma(filter_sigma, KG_end, T)


class RTS_Smoother_NE:

    def __init__(self, SystemModel): 
        self.F = SystemModel.F;
        self.F_T = torch.transpose(self.F, 0, 1);
        self.m = SystemModel.m
        self.Q = SystemModel.Q_evo

        self.H = SystemModel.H;
        self.H_T = torch.transpose(self.H, 0, 1);
        self.n = SystemModel.n
        self.R = SystemModel.R_evo

        self.T = SystemModel.T;
    
    ##########################################
    ############# state smoothing ############
    ##########################################

    # Helper for smooth innovation. Compute x_prior
    def predict_x(self, filter_x):
        return torch.matmul(self.F, filter_x)
    
    # Helper for smooth innovation. Computes sigma_prior
    def predict_sigma(self, filter_sigma, Qt):
        filter_sigma_prior = torch.matmul(self.F, filter_sigma)
        filter_sigma_prior = torch.matmul(filter_sigma_prior, self.F_T) + Qt   
        return filter_sigma_prior
    
    # Compute the Smoother Gain
    def smooth_gain(self, filter_sigma, Qt):
        self.SG = torch.matmul(filter_sigma, self.F_T)
        filter_sigma_prior = self.predict_sigma(filter_sigma, Qt)
        self.SG = torch.matmul(self.SG, torch.inverse(filter_sigma_prior))

    # Innovation for Smoother
    def smooth_innovation(self, filter_x, filter_sigma, Qt):
        filter_x_prior = self.predict_x(filter_x)
        filter_sigma_prior = self.predict_sigma(filter_sigma, Qt)
        self.dx = self.s_m1x_nexttime - filter_x_prior
        self.dsigma = filter_sigma_prior - self.s_m2x_nexttime

    # Compute smoothed estimation of x and sigma backwardly in time
    def smooth_correct(self, filter_x, filter_sigma):
        # Compute the 1-st moment
        self.s_m1x_nexttime = filter_x + torch.matmul(self.SG, self.dx)

        # Compute the 2-nd moment
        self.s_m2x_nexttime = torch.matmul(self.dsigma, torch.transpose(self.SG, 0, 1))
        self.s_m2x_nexttime = filter_sigma - torch.matmul(self.SG, self.s_m2x_nexttime)

    # Smooth the data for one time step
    def smooth(self, filter_x, filter_sigma, Qt):
        self.smooth_gain(filter_sigma, Qt)
        self.smooth_innovation(filter_x, filter_sigma, Qt)
        self.smooth_correct(filter_x, filter_sigma)

        return self.s_m1x_nexttime,self.s_m2x_nexttime


    ### Generate Sequence ###
    def generate_sequence(self, filter_x, filter_sigma, prior_Q, T):
        # Pre allocate an array for smoothed x and sigma
        self.s_x = torch.empty(size=[self.m, T])
        self.s_sigma = torch.empty(size=[self.m, self.m, T])

        # initialization for backwards recursion
        self.s_m1x_nexttime = filter_x[:, T-1]
        self.s_m2x_nexttime = filter_sigma[:, :, T-1]
        self.s_x[:, T-1] = torch.squeeze(self.s_m1x_nexttime)
        self.s_sigma[:, :, T-1] = torch.squeeze(self.s_m2x_nexttime)

        for t in range(T-2,-1,-1):
            filter_xt = filter_x[:, t]
            filter_sigmat = filter_sigma[:, :, t]
            Qt = prior_Q[:, :, t]
            s_xt,s_sigmat = self.smooth(filter_xt, filter_sigmat, Qt);
            self.s_x[:, t] = torch.squeeze(s_xt)
            self.s_sigma[:, :, t] = torch.squeeze(s_sigmat)
    
    ############################################################################
    ###### additional feature: compute E[x(t+1)*x(t)^T] for one time step ######
    ############################################################################
    def cross_sigma(self, sigma, KG, T):
        # allocate memory
        self.sigma_cross = torch.empty(size=[self.m, self.m, T-1])

        # initialisation
        sigma_cross_end = torch.eye(self.m) - torch.matmul(KG, self.H)
        sigma_cross_end = torch.matmul(sigma_cross_end, self.F)
        sigma_cross_end = torch.matmul(sigma_cross_end, sigma[:, :, T-2])
        self.sigma_cross[:, :, T-2] = sigma_cross_end

        # recursive computation
        for t in range(T-3, -1, -1):
            SG1 = torch.clone(self.SG_all[:, :, t+1]).detach()
            SG0 = torch.clone(self.SG_all[:, :, t]).detach()
            SG0_T = torch.transpose(SG0, 0, 1)
            d_Exx = self.sigma_cross[:, :, t+1] - torch.matmul(self.F, sigma[:, :, t+1])
            self.sigma_cross[:, :, t] = torch.matmul(SG1, d_Exx)
            self.sigma_cross[:, :, t] = torch.matmul(self.sigma_cross[:, :, t], SG0_T)
            self.sigma_cross[:, :, t] += torch.matmul(sigma[:, :, t+1], SG0_T)

    
    ### Generate Sequence ###
    def generate_sequence_cross(self, filter_x, filter_sigma, KG, prior_Q, T):
        # Pre allocate an array for smoothed x, sigma, smooth gain
        self.s_x = torch.empty(size=[self.m, T])
        self.s_sigma = torch.empty(size=[self.m, self.m, T])
        self.SG_all = torch.empty(size=[self.m, self.m, T])

        # initialization for backwards recursion
        self.s_m1x_nexttime = filter_x[:, T-1]
        self.s_m2x_nexttime = filter_sigma[:, :, T-1]
        self.s_x[:, T-1] = torch.squeeze(self.s_m1x_nexttime)
        self.s_sigma[:, :, T-1] = torch.squeeze(self.s_m2x_nexttime)
        
        # smoothing loop
        for t in range(T-2,-1,-1):
            filter_xt = filter_x[:, t]
            filter_sigmat = filter_sigma[:, :, t]
            Qt = prior_Q[:, :, t]
            s_xt,s_sigmat = self.smooth(filter_xt, filter_sigmat, Qt)
            
            self.s_x[:, t] = torch.squeeze(s_xt)
            self.s_sigma[:, :, t] = torch.squeeze(s_sigmat)
            self.SG_all[:, :, t] = torch.clone(self.SG).detach()
        
        # In addition, compute E[x(t+1)*x(t)^T] for t=0,...,T-2
        self.cross_sigma(filter_sigma, KG, T)
    
    
    ##########################################
    ############ input estimation ############
    ##########################################
    
    # compute backwards gain for input estimate
    def input_gain(self, filter_sigma, filter_Q):
        self.IG = torch.matmul(filter_Q, torch.eye(self.m))
        filter_sigma_prior = self.predict_sigma(filter_sigma, filter_Q)
        self.IG = torch.matmul(self.IG, torch.inverse(filter_sigma_prior))

    # compute input estimate (both 1. and 2. moments)
    def input_correct(self, filter_u, filter_Q):
        # compute 1st moment of u
        self.u_post = filter_u + torch.matmul(self.IG, self.dx)
        # compute 2nd moment of u
        self.Q_post = torch.matmul(self.IG, self.dsigma)
        self.Q_post = filter_Q + torch.matmul(self.Q_post, torch.transpose(self.dsigma, 0, 1))
    # Input estimation
    def input_estimate(self, filter_sigma, filter_u, filter_Q):
        self.input_gain(filter_sigma, filter_Q)
        self.input_correct(filter_u, filter_Q)
    
    # Smooth the data and perform input estimation for one time step
    def smooth_in(self, filter_x, filter_sigma, filter_u, filter_Q):
        s_xt,s_sigmat = self.smooth(filter_x, filter_sigma, Qt=filter_Q)
        self.input_estimate(filter_sigma, filter_u, filter_Q)
        return s_xt,s_sigmat, self.u_post, self.Q_post

    # Given the results from Kalman filtering,
    # smooth the state estimate and perform input estimation.
    # filter_x, filter_sigma: estimation of x and its covariance from Kalman filtering
    # fitler_u, filter_Q: prior estimate of noise mean and covariance
    def generate_sequence_in(self, filter_x, filter_sigma, filter_u, filter_Q, T, cross_x1x0=False, KG_end=None):
        # Pre allocate an array for smoothed state and variance
        self.s_x = torch.empty(size=[self.m, T])
        self.s_sigma = torch.empty(size=[self.m, self.m, T])
        self.SG_all = torch.empty(size=[self.m, self.m, T])

        # Pre allocate arrays for updated input estimate and its variance
        self.s_u = torch.empty(size=[self.m, T])
        self.s_Q = torch.empty(size=[self.m, self.m, T])

        # Initialisation for backwards smoothing
        self.s_m1x_nexttime = filter_x[:, T-1]
        self.s_m2x_nexttime = filter_sigma[:, :, T-1]
        self.s_x[:, T-1] = torch.squeeze(self.s_m1x_nexttime)
        self.s_sigma[:, :, T-1] = torch.squeeze(self.s_m2x_nexttime)

        for t in range(T-2,-1,-1):
            filter_xt = torch.squeeze(filter_x[:, t])
            filter_sigmat = torch.squeeze(filter_sigma[:, :, t])
            filter_ut = torch.squeeze(filter_u[:, t])
            filter_Qt = torch.squeeze(filter_Q[:, :, t])

            # perform smoothing and input estimation
            s_xt,s_sigmat, s_ut, s_Qt = self.smooth_in(filter_xt, filter_sigmat, filter_ut, filter_Qt)

            self.s_x[:, t] = torch.squeeze(s_xt)
            self.s_sigma[:, :, t] = torch.squeeze(s_sigmat)
            self.s_u[:, t+1] = torch.squeeze(s_ut)
            self.s_Q[:, :, t+1] = torch.squeeze(s_Qt)
            self.SG_all[:, :, t] = torch.clone(self.SG).detach()

        self.s_u[:, 0] = filter_u[:, 0]
        self.s_Q[:, :, 0] = filter_Q[:, :, 0]
        if (cross_x1x0 is not None) and (KG_end is not None):
            # In addition, compute E[x(t+1)*x(t)^T] for t=0,...,T-2
            self.cross_sigma(filter_sigma, KG_end, T)   



##################################################
######## Non time invariant version of KS ########
##################################################
class RTS_Smoother_nti(RTS_Smoother_NE):
    # Helper for smooth innovation. Compute x_prior
    def predict_x(self, Ft, filter_x, ut):
        return torch.matmul(Ft, filter_x) + ut
    
    # Helper for smooth innovation. Computes sigma_prior
    def predict_sigma(self, filter_sigma, Qt, Ft, Ft_T):
        filter_sigma_prior = torch.matmul(Ft, filter_sigma)
        filter_sigma_prior = torch.matmul(filter_sigma_prior, Ft_T) + Qt   
        return filter_sigma_prior
    
    # Compute the Smoother Gain
    def smooth_gain(self, filter_sigma, Qt, Ft, Ft_T):
        self.SG = torch.matmul(filter_sigma, Ft_T)
        filter_sigma_prior = self.predict_sigma(filter_sigma, Qt, Ft, Ft_T)
        self.SG = torch.matmul(self.SG, torch.inverse(filter_sigma_prior))

    # Innovation for Smoother
    def smooth_innovation(self, filter_x, filter_sigma, ut, Qt, Ft, Ft_T):
        filter_x_prior = self.predict_x(Ft, filter_x, ut)
        filter_sigma_prior = self.predict_sigma(filter_sigma, Qt, Ft, Ft_T)
        self.dx = self.s_m1x_nexttime - filter_x_prior
        self.dsigma = filter_sigma_prior - self.s_m2x_nexttime

    # Smooth the data for one time step
    def smooth(self, filter_x, filter_sigma, ut, Qt, Ft, Ft_T):
        self.smooth_gain(filter_sigma, Qt, Ft, Ft_T)
        self.smooth_innovation(filter_x, filter_sigma, ut, Qt, Ft, Ft_T)
        self.smooth_correct(filter_x, filter_sigma)

        return self.s_m1x_nexttime,self.s_m2x_nexttime

    ### Generate Sequence ###
    def generate_sequence_cross(self, filter_x, filter_sigma, prior_u, prior_Q, T, cross_x1x0=False, KG_end=None):
        # Pre allocate an array for smoothed x, sigma, smooth gain
        self.s_x = torch.empty(size=[self.m, T])
        self.s_sigma = torch.empty(size=[self.m, self.m, T])
        self.SG_all = torch.empty(size=[self.m, self.m, T])

        # initialization for backwards recursion
        self.s_m1x_nexttime = filter_x[:, T-1]
        self.s_m2x_nexttime = filter_sigma[:, :, T-1]
        self.s_x[:, T-1] = torch.squeeze(self.s_m1x_nexttime)
        self.s_sigma[:, :, T-1] = torch.squeeze(self.s_m2x_nexttime)
        
        # smoothing loop
        for t in range(T-2,-1,-1):
            filter_xt = filter_x[:, t]
            filter_sigmat = filter_sigma[:, :, t]
            Qt = prior_Q[:, :, t]
            ut = prior_u[:, t]
            Ft = self.F[:, :, t]
            Ft_T = self.F_T[:, :, t]

            s_xt,s_sigmat = self.smooth(filter_xt, filter_sigmat, ut, Qt, Ft, Ft_T)
            
            self.s_x[:, t] = torch.squeeze(s_xt)
            self.s_sigma[:, :, t] = torch.squeeze(s_sigmat)
            self.SG_all[:, :, t] = torch.clone(self.SG).detach()
        
        # In addition, compute E[x(t+1)*x(t)^T] for t=0,...,T-2
        if (cross_x1x0):
            self.cross_sigma(filter_sigma, KG_end, T)

    ############################################################################
    ###### additional feature: compute E[x(t+1)*x(t)^T] for one time step ######
    ############################################################################
    def cross_sigma(self, sigma, KG, T):
        # allocate memory
        self.sigma_cross = torch.empty(size=[self.m, self.m, T-1])

        # initialisation
        sigma_cross_end = torch.eye(self.m) - torch.matmul(KG, self.H[:,:,-1])
        sigma_cross_end = torch.matmul(sigma_cross_end, self.F[:, :, -1])
        sigma_cross_end = torch.matmul(sigma_cross_end, sigma[:, :, T-2])
        self.sigma_cross[:, :, T-2] = sigma_cross_end

        # recursive computation
        for t in range(T-3, -1, -1):
            SG1 = torch.clone(self.SG_all[:, :, t+1]).detach()
            SG0 = torch.clone(self.SG_all[:, :, t]).detach()
            SG0_T = torch.transpose(SG0, 0, 1)
            d_Exx = self.sigma_cross[:, :, t+1] - torch.matmul(self.F[:, :, t+2], sigma[:, :, t+1])
            self.sigma_cross[:, :, t] = torch.matmul(SG1, d_Exx)
            self.sigma_cross[:, :, t] = torch.matmul(self.sigma_cross[:, :, t], SG0_T)
            self.sigma_cross[:, :, t] += torch.matmul(sigma[:, :, t+1], SG0_T)