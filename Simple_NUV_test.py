import torch
import time
import math
from abc import ABC, abstractmethod
from RTS_NUV import RTS_Simple_NUV
import torch.nn as nn


class Simple_NUV_Tracker(ABC):
    def __init__(self, lin_model):
        self.sys_model = lin_model
        self.rts_uv = RTS_Simple_NUV(lin_model)
    
    # abstraction of maximizaion step for estimating Q or R
    @abstractmethod
    def M_step(self):
        pass
    
    # abstraction of evaluting MSE of input estm or state estm
    @abstractmethod
    def loss(self):
        pass
    
    # abstraction of initializing EM
    @abstractmethod
    def init_em(self):
        pass


    # track the MSE of test_target for NB trajectories
    def track(self, NB, test_input, test_target, itr, unknown_init = 10.0):
        # MSEs[j, k] stores the MSE for j-th batch at iteration k
        MSEs = torch.empty(NB, itr)
        for j in range(0, NB):
            self.init_em(unknown_init)
            for k in range(itr):
                # E-step
                self.rts_uv.fwd_bwd_filter(test_input[j,:,:])
                # M-step
                self.M_step(test_input[j,:,:])
                # evaluate MSE
                MSEs[j, k]= self.loss(test_target[j,:,:]) 
        return MSEs


class Unknown_R_Tracker(Simple_NUV_Tracker):
    def M_step(self, test_input):
        self.rts_uv.update_R(test_input)
    
    def init_em(self, r_init):
        R_init = r_init*r_init*torch.eye(self.rts_uv.n)
        self.rts_uv.set_R(R_init)   
    
class Unknown_R_State_Tracker(Unknown_R_Tracker):
    def loss(self,test_target):
        return self.rts_uv.mse_state(test_target)

class Unknown_R_Input_Tracker(Unknown_R_Tracker):   
    def loss(self,test_target):
        return self.rts_uv.mse_input(test_target)
    
# Class for computing the MSE of input/state estimator for unknown static Q/R
class Simple_NUV_Tester(ABC):
    def __init__(self, AssumedModel):
        self.rts_uv = RTS_Simple_NUV(AssumedModel)
        self.rts_uv.KF.InitSequence(AssumedModel.m1x_0, AssumedModel.m2x_0)
    
    # EM for estimating R or Q
    @abstractmethod
    def estimate_unknown(self):
        pass

    # Show info
    @abstractmethod
    def info(self):
        pass
    
    # Computes the averaged MSE of test_target for NB trajectories
    def test(self, NB, test_input, X_target, U_target, itr=20, init_unknown=1.0):
        loss_fun = nn.MSELoss(reduction='mean')
        # save estimated state and its variance
        self.x_rts = torch.empty_like(X_target)    
        self.sigma_rts = torch.empty(NB, self.rts_uv.m, self.rts_uv.m, self.rts_uv.T)
        # save estimated input and its variance
        self.u_rts = torch.empty_like(U_target)      
        self.post_Q_rts = torch.empty_like(self.sigma_rts)
        
        start = time.time()
        MSEs_state = torch.empty(NB)
        MSEs_input = torch.empty(NB)
        for j in range(0, NB):
            # EM for unknown variance
            self.estimate_unknown(test_input[j, :, :], itr, init_unknown)
            # save smoothing results
            self.x_rts[j, :, :], self.sigma_rts[j, :, :, :] = self.rts_uv.RTS.s_x, self.rts_uv.RTS.s_sigma
            self.u_rts[j, :, :], self.post_Q_rts[j, :, :, :] = self.rts_uv.RTS.s_u, self.rts_uv.RTS.s_Q            
            # loss for j-th traj
            MSEs_state[j], MSEs_input[j] = loss_fun(self.x_rts[j, :, :], X_target[j, :, :]), loss_fun(self.u_rts[j, :, 1:], U_target[j, :, 1:])
        
        # avg loss for all batches
        MSE_state_avg, MSE_input_avg = torch.mean(MSEs_state), torch.mean(MSEs_input)
        MSE_state_avg_dB, MSE_input_avg_dB = 10*torch.log10(MSE_state_avg), 10*torch.log10(MSE_input_avg)

        # std of loss
        std_MSE_state = torch.std(MSEs_state)
        std_MSE_state_dB = 10*torch.log10(MSE_state_avg+std_MSE_state) - MSE_state_avg_dB
        std_MSE_input = torch.std(MSEs_input)
        std_MSE_input_dB = 10*torch.log10(MSE_input_avg+std_MSE_input) - MSE_input_avg_dB

        self.info(MSE_state_avg_dB, MSE_input_avg_dB, time.time()-start)
        return [MSE_state_avg_dB, std_MSE_state_dB, MSE_input_avg_dB, std_MSE_input_dB]


class Unknown_R_Tester(Simple_NUV_Tester):
    def estimate_unknown(self, y_obs, itr, r0):
        self.rts_uv.smooth_unknownR(y_obs, itr, r0)
    
    def info(self, MSE_state_avg_dB, MSE_input_avg_dB, t):
        print("Unknown R - mean MSE state: ", MSE_state_avg_dB, "[dB]")
        print("Unknown R - mean MSE input: ", MSE_input_avg_dB, "[dB]")
        print("Inference time: ", t, "s")

class Unknown_Q_Tester(Simple_NUV_Tester):
    def estimate_unknown(self, y_obs, itr, q0):
        self.rts_uv.smooth_unknownQ(y_obs, itr, q0)
    
    def info(self, MSE_state_avg_dB, MSE_input_avg_dB, t):
        print("Unknown Q - mean MSE state: ", MSE_state_avg_dB, "[dB]")
        print("Unknown Q - mean MSE input: ", MSE_input_avg_dB, "[dB]")
        print("Inference time: ", t, "s")
