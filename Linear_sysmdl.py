
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

class SystemModel:
    # q: std dev of model error
    # r: std dev of observation error
    def __init__(self, F, q, H, r, T, T_test, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.F = F
        self.m = self.F.size()[0]
        self.q = q
        self.Q = q * q * torch.eye(self.m)


        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.n = self.H.size()[0]
        self.r = r
        self.R = r * r * torch.eye(self.n)

        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S

    def f(self, x):
        return torch.matmul(self.F, x)
    
    def h(self, x):
        return torch.matmul(self.H, x)
        
    #####################
    ### Init Sequence ###
    #####################
    # m1x_0: 1st moment of x_0
    # m2x_0: 2nd moment of x_0
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R


    #########################
    ### Generate Sequence ###
    #########################
    # Q_gen[:, :] is static
    # R_gen[:, :] is static
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for states
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, T])
        # Pre allocate an array for states
        self.u = torch.empty(size=[self.m, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################
            if self.q==0:
                xt = self.F.matmul(self.x_prev) 
                self.u[:, t] = torch.zeros(self.m)           
            else:
                xt = self.F.matmul(self.x_prev)
                mean = torch.zeros([self.m])
                # Noise generator              
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                # Additive Process Noise
                xt = torch.add(xt, eq)
                # save input noise
                self.u[:, t] = eq

            ################
            ### Emission ###
            ################
            # Observation Noise
            if self.r==0:
                yt = self.H.matmul(xt)           
            else:
                yt = self.H.matmul(xt)
                mean = torch.zeros([self.n])
                # noise generator            
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                # observation noise
                er = distrib.rsample()              
                # Additive Observation Noise
                yt = torch.add(yt, er)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, randomInit=False, seqInit=False, T_test=0):

        # Allocate Empty Array for Input
        self.Observed = torch.empty(size, self.n, T)

        # Allocate Empty Array for Target
        self.State_Target = torch.empty(size, self.m, T)
        self.Input_Target = torch.empty(size, self.m, T)

        ### Generate Examples
        initConditions = self.m1x_0

        for i in range(0, size):
            # Randomize initial conditions to get a rich dataset
            if(randomInit):
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
            if(seqInit):
                initConditions = self.x_prev
                if((i*T % T_test)==0):
                    initConditions = torch.zeros_like(self.m1x_0)
            
            # Generate Sequence for batch i
            self.InitSequence(initConditions, self.m2x_0)
            self.GenerateSequence(self.Q, self.R, T)

            # Training sequence input
            self.Observed[i, :, :] = self.y

            # Training sequence output
            self.State_Target[i, :, :] = self.x
            self.Input_Target[i, :, :] = self.u


    def sampling(self, q, r, gain):

        if (gain != 0):
            gain_q = 0.1
            #aq = gain * q * np.random.randn(self.m, self.m)
            aq = gain_q * q * torch.eye(self.m)
            #aq = gain_q * q * torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        else:
            aq = 0

        Aq = q * torch.eye(self.m) + aq
        Q_gen = torch.transpose(Aq, 0, 1) * Aq

        if (gain != 0):
            gain_r = 0.5
            #ar = gain * r * np.random.randn(self.n, self.n)
            ar = gain_r * r * torch.eye(self.n)
            #ar = gain_r * r * torch.tensor([[1.0, 1.0], [1.0, 1.0]])

        else:
            ar = 0

        Ar = r * torch.eye(self.n) + ar
        R_gen = torch.transpose(Ar, 0, 1) * Ar

        return [Q_gen, R_gen]

# System model with noise evolution (NE)
# Q_evo[:, :, t] is now time-variant
# R_evo[:, :, t] is now time-variant
class System_Model_NE():
    def __init__(self, F, H, T, Q_evo, R_evo):
        # state transition
        self.F = F
        # output matrix
        self.H = H

        # dim of state
        self.m = self.F.size()[0]
        # dim of output
        self.n = self.H.size()[0]

        # noise evolutions
        self.Q_evo = Q_evo
        self.R_evo = R_evo

        # length of traj
        self.T = T
    
    def reset_cov(self, Q_evo, R_evo):
        self.Q_evo = Q_evo
        self.R_evo = R_evo

    #####################
    ### Init Sequence ###
    #####################
    # m1x_0: 1st moment of x_0
    # m2x_0: 2nd moment of x_0
    def init_sequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0
    
    #############################
    ### Generate NUV Sequence ###
    #############################
    # Q_evo[:, :, t] is time-variant
    # R_evo[:, :, t] is time-variant
    def generate_nuv_sequence(self, T):
        # Pre allocate an array for states
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for observation
        self.y = torch.empty(size=[self.n, T])
        # Pre allocate an array for states
        self.u = torch.empty(size=[self.m, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):     
            ########################
            #### State Evolution ###
            ########################
            Qt = self.Q_evo[:, :, t]
            zero_Q = torch.zeros(size=[self.m, self.m])             
            if torch.equal(Qt, zero_Q):
                xt = self.F.matmul(self.x_prev) 
                self.u[:, t] = torch.zeros(self.m)           
            else:
                xt = self.F.matmul(self.x_prev)
                mean = torch.zeros([self.m])
                # Noise generator
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Qt)
                eq = distrib.rsample()
                # Additive Process Noise
                xt = torch.add(xt, eq)
                # save input noise
                self.u[:, t] = eq

            ################
            ### Emission ###
            ################
            # Observation Noise
            Rt = self.R_evo[:, :, t]              
            zero_R = torch.zeros(size=[self.n, self.n])
            if torch.equal(Rt, zero_R):
                yt = self.H.matmul(xt)           
            else:
                yt = self.H.matmul(xt)
                mean = torch.zeros([self.n])
                # noise generator  
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Rt)
                # observation noise
                er = distrib.rsample()              
                # Additive Observation Noise
                yt = torch.add(yt, er)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt

    ##########################
    ### Generate NUV Batch ###
    ##########################
    def generate_nuv_batch(self, size, T):

        # Allocate Empty Array for output traj
        self.Y = torch.empty(size, self.n, T)
        # Allocate Empty Array for state traj
        self.X = torch.empty(size, self.m, T)
        # Allocate Empty Array for input traj
        self.U = torch.empty(size, self.m, T)

        ### Generate Examples
        initConditions = self.m1x_0

        for i in range(0, size):           
            # Generate Sequence for batch i
            self.init_sequence(initConditions, self.m2x_0)
            self.generate_nuv_sequence(T)

            # Training sequence input
            self.Y[i, :, :] = self.y

            # Training sequence output
            self.X[i, :, :] = self.x
            self.U[i, :, :] = self.u

# System model with noise evolution (NE)
# F[:, :, t] is now time-variant
# H[:, :, t] is now time-variant
# Q_evo[:, :, t] is now time-variant
# R_evo[:, :, t] is now time-variant
class System_Model_nti(System_Model_NE):
    #############################
    ### Generate NUV Sequence ###
    #############################
    # Q_evo[:, :, t] is time-variant
    # R_evo[:, :, t] is time-variant
    def generate_nuv_sequence(self, T):
        # Pre allocate an array for states
        self.x = torch.empty(size=[self.m, T])
        # Pre allocate an array for observation
        self.y = torch.empty(size=[self.n, T])
        # Pre allocate an array for states
        self.u = torch.empty(size=[self.m, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):     
            ########################
            #### State Evolution ###
            ########################
            Qt = self.Q_evo[:, :, t]
            zero_Q = torch.zeros(size=[self.m, self.m])             
            if torch.equal(Qt, zero_Q):
                xt = self.F[:, :, t].matmul(self.x_prev) 
                self.u[:, t] = torch.zeros(self.m)           
            else:
                xt = self.F[:, :, t].matmul(self.x_prev)
                mean = torch.zeros([self.m])
                # Noise generator
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Qt)
                eq = distrib.rsample()
                # Additive Process Noise
                xt = torch.add(xt, eq)
                # save input noise
                self.u[:, t] = eq

            ################
            ### Emission ###
            ################
            # Observation Noise
            Rt = self.R_evo[:, :, t]              
            zero_R = torch.zeros(size=[self.n, self.n])
            if torch.equal(Rt, zero_R):
                yt = self.H[:, :, t].matmul(xt)           
            else:
                yt = self.H[:, :, t].matmul(xt)
                mean = torch.zeros([self.n])
                # noise generator  
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Rt)
                # observation noise
                er = distrib.rsample()              
                # Additive Observation Noise
                yt = torch.add(yt, er)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt
