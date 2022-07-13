"""# **Class: Kalman Filter**
Theoretical Linear Kalman
"""
import torch

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

class KalmanFilter:

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
   
    # Predict

    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior);

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior);
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + self.Q;

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior);

        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior);
        self.m2y = torch.matmul(self.m2y, self.H_T) + self.R;

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y;

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy);

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def Update(self, y):
        self.Predict();
        self.KGain();
        self.Innovation(y);
        self.Correct();

        return self.m1x_posterior,self.m2x_posterior;

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    ### Generate Sequence ###
    #########################
    # y: 2D tensor, obervation vectors from time 0 to T
    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T]).to(dev)
        self.sigma = torch.empty(size=[self.m, self.m, T]).to(dev)

        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

        for t in range(0, T):
            yt = y[:, t];
            xt,sigmat = self.Update(yt);
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)

    # perform the save task as GenerateSequence,
    # save additionally m2x_prior, KGs and m2y at each iteration
    def GenerateSequence_plus(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T]).to(dev)
        self.sigma = torch.empty(size=[self.m, self.m, T]).to(dev)

        # save KGs at every iteration
        self.KG_all = torch.empty(size=[self.m, self.n, T]).to(dev)
        # save m2y at every iteration
        self.m2y_all = torch.empty(size=[self.n, self.n, T]).to(dev)
        # save m2x_prior at every iteration
        self.m2x_prior_all = torch.empty(size=[self.m, self.m, T]).to(dev)
        # save all innovations at every iteration
        self.dy_all = torch.empty(size=[self.n, self.T]).to(dev)
        
        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0
        
        for t in range(0, T):
            yt = y[:, t]
            xt,sigmat = self.Update(yt)

            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)

            # save m2x_prior, KG and m2y in addition
            self.KG_all[:, :, t] = self.KG
            self.m2y_all[:, :, t] = self.m2y
            self.m2x_prior_all[:, :, t] = self.m2x_prior
            self.dy_all[:, t] = self.dy


class KalmanFilter_NE:

    def __init__(self, SystemModel):
        self.F = SystemModel.F
        self.F_T = torch.transpose(self.F, 0, 1)
        self.m = SystemModel.m

        self.H = SystemModel.H
        self.H_T = torch.transpose(self.H, 0, 1)
        self.n = SystemModel.n

        # note: Q[:, :, t] is now time variant!
        self.Q = SystemModel.Q_evo

        # note: R[:, :, t] is now time variant!
        self.R = SystemModel.R_evo

        self.T = SystemModel.T
   
    # Predict
    def predict(self, Qt, Rt):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior);

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior);
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F_T) + Qt;

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior);

        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior);
        self.m2y = torch.matmul(self.m2y, self.H_T) + Rt;

    # Compute the Kalman Gain
    def kalman_gain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

    # Innovation
    def innovation(self, y):
        self.dy = y - self.m1y;

    # Compute Posterior
    def correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy);

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

    def estimate_state(self, y, Qt, Rt):
        self.predict(Qt, Rt);
        self.kalman_gain();
        self.innovation(y);
        self.correct();

        return self.m1x_posterior,self.m2x_posterior;

    def init_sequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    ### Generate Sequence ###
    #########################
    # y: 2D tensor, obervation vectors from time 0 to T
    def generate_sequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T]).to(dev)
        self.sigma = torch.empty(size=[self.m, self.m, T]).to(dev)

        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

        for t in range(0, T):
            yt = y[:, t];
            Qt = self.Q[:, :, t]
            Rt = self.R[:, :, t]
            xt, sigmat = self.estimate_state(yt, Qt, Rt);
            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)

    # perform the save task as GenerateSequence,
    # save additionally m2x_prior, KGs and m2y at each iteration
    def generate_sequence_plus(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T]).to(dev)
        self.sigma = torch.empty(size=[self.m, self.m, T]).to(dev)

        # save KGs at every iteration
        self.KG_all = torch.empty(size=[self.m, self.n, T]).to(dev)
        # save m2y at every iteration
        self.m2y_all = torch.empty(size=[self.n, self.n, T]).to(dev)
        # save m2x_prior at every iteration
        self.m2x_prior_all = torch.empty(size=[self.m, self.m, T]).to(dev)
        # save all innovations at every iteration
        self.dy_all = torch.empty(size=[self.n, self.T]).to(dev)
        
        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0
        
        for t in range(0, T):
            yt = y[:, t]
            Qt = self.Q[:, :, t]
            Rt = self.R[:, :, t]            
            xt, sigmat = self.estimate_state(yt, Qt, Rt);

            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)

            # save m2x_prior, KG and m2y in addition
            self.KG_all[:, :, t] = self.KG
            self.m2y_all[:, :, t] = self.m2y
            self.m2x_prior_all[:, :, t] = self.m2x_prior
            self.dy_all[:, t] = self.dy



##################################################
######## Non time invariant version of KF ########
##################################################

class KalmanFilter_nti(KalmanFilter_NE):
    # Predict
    def predict(self, ut, Qt, Rt, Ft, Ht, Ft_T, H_t_T):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.matmul(Ft, self.m1x_posterior) + ut

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(Ft, self.m2x_posterior);
        self.m2x_prior = torch.matmul(self.m2x_prior, Ft_T) + Qt;

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(Ht, self.m1x_prior);

        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(Ht, self.m2x_prior);
        self.m2y = torch.matmul(self.m2y, H_t_T) + Rt;

    # Compute the Kalman Gain
    def kalman_gain(self, Ht_T):
        self.KG = torch.matmul(self.m2x_prior, Ht_T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))

    def estimate_state(self, y, ut, Qt, Rt, Ft, Ht, Ft_T, H_t_T):
        self.predict(ut, Qt, Rt, Ft, Ht, Ft_T, H_t_T)
        self.kalman_gain(H_t_T)
        self.innovation(y)
        self.correct()
        return self.m1x_posterior,self.m2x_posterior;

    
    def generate_sequence(self, y, u, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T]).to(dev)
        self.sigma = torch.empty(size=[self.m, self.m, T]).to(dev)

        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

        for t in range(0, T):
            yt = y[:, t]
            ut = u[:, t]
            Qt = self.Q[:, :, t]
            Rt = self.R[:, :, t]

            Ft = self.F[:, :, t]
            Ht = self.H[:, :, t]
            Ft_T = self.F_T[:, :, t]
            Ht_T = self.H_T[:, :, t]

            xt, sigmat = self.estimate_state(yt, ut, Qt, Rt, Ft, Ht, Ft_T, Ht_T)

            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)

        # perform the save task as GenerateSequence,
    # save additionally m2x_prior, KGs and m2y at each iteration
    def generate_sequence_plus(self, y, u, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.empty(size=[self.m, T]).to(dev)
        self.sigma = torch.empty(size=[self.m, self.m, T]).to(dev)

        # save KGs at every iteration
        self.KG_all = torch.empty(size=[self.m, self.n, T]).to(dev)
        # save m2y at every iteration
        self.m2y_all = torch.empty(size=[self.n, self.n, T]).to(dev)
        # save m2x_prior at every iteration
        self.m2x_prior_all = torch.empty(size=[self.m, self.m, T]).to(dev)
        # save all innovations at every iteration
        self.dy_all = torch.empty(size=[self.n, self.T]).to(dev)
        
        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0
        
        for t in range(0, T):
            yt = y[:, t]
            ut = u[:, t]
            Qt = self.Q[:, :, t]
            Rt = self.R[:, :, t]  

            Ft = self.F[:, :, t]
            Ht = self.H[:, :, t]
            Ft_T = self.F_T[:, :, t]
            H_t_T = self.H_T[:, :, t]   
                  
            xt, sigmat = self.estimate_state(yt, ut, Qt, Rt, Ft, Ht, Ft_T, H_t_T)

            self.x[:, t] = torch.squeeze(xt)
            self.sigma[:, :, t] = torch.squeeze(sigmat)

            # save m2x_prior, KG and m2y in addition
            self.KG_all[:, :, t] = self.KG
            self.m2y_all[:, :, t] = self.m2y
            self.m2x_prior_all[:, :, t] = self.m2x_prior
            self.dy_all[:, t] = self.dy