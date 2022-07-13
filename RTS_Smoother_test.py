import torch
import torch.nn as nn
import time
from Linear_KF import KalmanFilter, KalmanFilter_NE
from RTS_Smoother import RTS_Smoother_NE, rts_smoother, rts_smoother_in
from abc import ABC, abstractmethod
from param_lin import T 
from multiprocessing import Pool
import os

## worker process for multhiprocessing
# return state/input estimation as well as MSE for *ONE* traj
def test_traj(KF, RTS, y_obs, x_gt, u_gt, u_prior, Q_prior, T):
    loss_fun = nn.MSELoss(reduction='mean')
    KF.GenerateSequence(y_obs, T)
    RTS.GenerateSequence_in(KF.x, KF.sigma, u_prior, Q_prior, T)
    # save smoothing results
    x_rts, sigma_rts = RTS.s_x, RTS.s_sigma
    u_rts, post_Q_rts = RTS.s_u, RTS.s_Q
    # loss for j-th traj
    MSEs_state, MSEs_input = loss_fun(x_rts, x_gt), loss_fun(u_rts[:, 1:], u_gt[:, 1:])
    return [x_rts, sigma_rts, u_rts, post_Q_rts, MSEs_state, MSEs_input]


class KF_Tester:
    def __init__(self, sys_model):
        self.KF = KalmanFilter(sys_model)
        self.KF.InitSequence(sys_model.m1x_0, sys_model.m2x_0)
    
    def info(self, MSE_avg_dB, t):
        print("KF - State Estimation - MSE LOSS:", MSE_avg_dB, "[dB]")
        print("Inference Time:", t)    

    # Perform fwd recursion only for each trajectory
    def test(self, NB, test_input, test_target):  
        loss_fun = nn.MSELoss(reduction='mean')
        # save estimated state and its variance
        self.x_kf = torch.empty(NB, self.KF.m, self.KF.T)    
        self.sigma_kf = torch.empty(NB, self.KF.m, self.KF.m, self.KF.T)    

        start = time.time()
        MSEs = torch.empty(NB)
        for j in range(0, NB):
            # fwd recursion for j-th batch
            self.KF.GenerateSequence(test_input[j, :, :], self.KF.T_test)
            self.x_kf[j, :, :], self.sigma_kf[j, :, :, :] = self.KF.x, self.KF.sigma
            # loss for j-th batch 
            MSEs[j] = loss_fun(self.x_kf[j, :, :], test_target[j, :, :])
        end = time.time()
        t = end - start
        
        # avg loss for all batches
        MSE_avg = torch.mean(MSEs)
        MSE_avg_dB = 10 * torch.log10(MSE_avg)

        # std deviation
        std_MSE = torch.std(MSEs)
        std_MSE_dB = 10*torch.log10(MSE_avg+std_MSE) - MSE_avg_dB

        self.info(MSE_avg_dB, t)
        return [MSE_avg_dB, std_MSE_dB]

class RTS_Tester():
    def __init__(self, SysModel):
        # prepare the filter and smoother
        self.KF = KalmanFilter(SysModel)
        self.KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
        self.RTS = rts_smoother_in(SysModel)
    
    def info(self, MSE_state_dB, MSE_input_dB, t):
        print("RTS Smoother - State Estimation - MSE LOSS:", MSE_state_dB, "[dB]")
        print("RTS Smoother - Input Estimation - MSE LOSS:", MSE_input_dB, "[dB]")
        print("Inference time: ", t, "s")

    def test_mp(self, NB, test_input, X_target, U_target, filter_u, filter_Q): 
        # PC
        core_count = os.cpu_count()
        try:
            # Euler HPC
            core_count = int(os.environ['LSB_DJOB_NUMPROC'])
        except:
            pass   
        start = time.time()
        pool_of_traj = Pool(core_count)
        batch_param_list = [
            (
                self.KF, self.RTS,
                test_input[j,:,:], X_target[j,:,:], U_target[j,:,:],
                filter_u, filter_Q, self.KF.T
            )for j in range(0, NB)
        ]
        # parallel computing for all traj
        batch_result = pool_of_traj.starmap(test_traj, batch_param_list)
        pool_of_traj.close()
        pool_of_traj.join()    

        # process reulsts
        self.x_rts = torch.stack([batch_result[j][0] for j in range(0, NB)])
        self.sigma_rts = torch.stack([batch_result[j][1] for j in range(0, NB)])
        self.u_rts = torch.stack([batch_result[j][2] for j in range(0, NB)])
        self.post_Q_rts = torch.stack([batch_result[j][3] for j in range(0, NB)])
        MSEs_state = torch.stack([batch_result[j][4] for j in range(0, NB)])
        MSEs_input = torch.stack([batch_result[j][5] for j in range(0, NB)])
        
        # avg loss for all batches
        MSE_state_avg, MSE_input_avg = torch.mean(MSEs_state), torch.mean(MSEs_input)
        MSE_state_avg_dB, MSE_input_avg_dB = 10*torch.log10(MSE_state_avg), 10*torch.log10(MSE_input_avg)

        # std
        std_MSE_state = torch.std(MSEs_state)
        std_MSE_state_dB = 10*torch.log10(MSE_state_avg+std_MSE_state) - MSE_state_avg_dB
        std_MSE_input = torch.std(MSEs_input)
        std_MSE_input_dB = 10*torch.log10(MSE_input_avg+std_MSE_input) - MSE_input_avg_dB

        self.info(MSE_state_avg_dB, MSE_input_avg_dB, time.time()-start)
        return [MSE_state_avg_dB, std_MSE_state_dB, MSE_input_avg_dB, std_MSE_input_dB]


    # test_input: contains NB observation trajectories, 
    # test_input[j,:,:] is the j-th traj
    # test_target: contains NB ground truth trajectories (input or state), 
    # X_target[j,:,:] is the j-th traj
    # U_target[j,:,:] is the j-th traj
    # fitler_u, filter_Q: guess of noise mean and covariance
    # return: average MSE of input estimates over NB batches
    def test(self, NB, test_input, X_target, U_target, filter_u, filter_Q): 
        loss_fun = nn.MSELoss(reduction='mean')
        # save estimated state and its variance
        self.x_rts = torch.empty(NB, self.KF.m, self.KF.T)    
        self.sigma_rts = torch.empty(NB, self.KF.m, self.KF.m, self.KF.T)
        # save estimated input and its variance
        self.u_rts = torch.empty_like(self.x_rts)      
        self.post_Q_rts = torch.empty_like(self.sigma_rts)
        
        start = time.time()
        MSEs_state = torch.empty(NB)
        MSEs_input = torch.empty(NB)

        # Perform fwd and bwd recursion for each trajectory
        for j in range(0, NB):
            # fwd recursion for j-th batch
            self.KF.GenerateSequence(test_input[j, :, :], self.KF.T_test)
            # bwd recursion for j-th batch
            self.RTS.GenerateSequence_in(self.KF.x, self.KF.sigma, filter_u, filter_Q, self.RTS.T_test)
            # save smoothing results
            self.x_rts[j, :, :], self.sigma_rts[j, :, :, :] = self.RTS.s_x, self.RTS.s_sigma
            self.u_rts[j, :, :], self.post_Q_rts[j, :, :, :] = self.RTS.s_u, self.RTS.s_Q
            # loss for j-th traj
            MSEs_state[j], MSEs_input[j] = loss_fun(self.x_rts[j, :, :], X_target[j, :, :]), loss_fun(self.u_rts[j, :, 1:], U_target[j, :, 1:])
        
        # avg loss for all batches
        MSE_state_avg, MSE_input_avg = torch.mean(MSEs_state), torch.mean(MSEs_input)
        MSE_state_avg_dB, MSE_input_avg_dB = 10*torch.log10(MSE_state_avg), 10*torch.log10(MSE_input_avg)

        # std
        std_MSE_state = torch.std(MSEs_state)
        std_MSE_state_dB = 10*torch.log10(MSE_state_avg+std_MSE_state) - MSE_state_avg_dB
        std_MSE_input = torch.std(MSEs_input)
        std_MSE_input_dB = 10*torch.log10(MSE_input_avg+std_MSE_input) - MSE_input_avg_dB

        self.info(MSE_state_avg_dB, MSE_input_avg_dB, time.time()-start)
        return [MSE_state_avg_dB, std_MSE_state_dB, MSE_input_avg_dB, std_MSE_input_dB]


## abstract class for testing RTS smoother (non-static Q & R)
class RTS_Tester_NE:
    def __init__(self, SysModel):
        # prepare the filter and smoother
        self.KF = KalmanFilter_NE(SysModel)
        self.KF.init_sequence(SysModel.m1x_0, SysModel.m2x_0)
        self.RTS = RTS_Smoother_NE(SysModel)


    # test_input: contains NB observation trajectories, 
    # test_input[j,:,:] is the j-th traj
    # test_target: contains NB ground truth trajectories (input or state), 
    # X_target[j,:,:] is the j-th traj
    # U_target[j,:,:] is the j-th traj
    # fitler_u, filter_Q: guess of noise mean and covariance
    # return: average MSE of input estimates over NB batches
    def test(self, NB, test_input, X_target, U_target, filter_u, filter_Q): 
        loss_fun = nn.MSELoss(reduction='mean')
        # save estimated state and its variance
        self.x_rts = torch.empty(NB, self.KF.m, self.KF.T)    
        self.sigma_rts = torch.empty(NB, self.KF.m, self.KF.m, self.KF.T)
        # save estimated input and its variance
        self.u_rts = torch.empty_like(self.x_rts)      
        self.post_Q_rts = torch.empty_like(self.sigma_rts)
        
        start = time.time()
        MSEs_state = torch.empty(NB)
        MSEs_input = torch.empty(NB)
        
        # Perform fwd and bwd recursion for each trajectory
        for j in range(0, NB):
            # fwd recursion for j-th batch
            self.KF.generate_sequence(test_input[j, :, :], self.KF.T)
            # bwd recursion for j-th batch
            self.RTS.generate_sequence_in(self.KF.x, self.KF.sigma, filter_u, filter_Q, self.RTS.T)
            # save smoothing results
            self.x_rts[j, :, :], self.sigma_rts[j, :, :, :] = self.RTS.s_x, self.RTS.s_sigma
            self.u_rts[j, :, :], self.post_Q_rts[j, :, :, :] = self.RTS.s_u, self.RTS.s_Q
            # loss for j-th traj
            MSEs_state[j], MSEs_input[j] = loss_fun(self.x_rts[j, :, :], X_target[j, :, :]), loss_fun(self.u_rts[j, :, 1:], U_target[j, :, 1:])
        
        # avg loss for all batches
        MSE_state_avg, MSE_input_avg = torch.mean(MSEs_state), torch.mean(MSEs_input)
        MSE_state_avg_dB, MSE_input_avg_dB = 10*torch.log10(MSE_state_avg), 10*torch.log10(MSE_input_avg)

        # std
        std_MSE_state = torch.std(MSEs_state)
        std_MSE_state_dB = 10*torch.log10(MSE_state_avg+std_MSE_state) - MSE_state_avg_dB
        std_MSE_input = torch.std(MSEs_input)
        std_MSE_input_dB = 10*torch.log10(MSE_input_avg+std_MSE_input) - MSE_input_avg_dB

        self.info(MSE_state_avg_dB, MSE_input_avg_dB, time.time()-start)
        return [MSE_state_avg_dB, std_MSE_state_dB, MSE_input_avg_dB, std_MSE_input_dB]

    def info(self, MSE_state_dB, MSE_input_dB, t):
        print("RTS Smoother - State Estimation - MSE LOSS:", MSE_state_dB, "[dB]")
        print("RTS Smoother - Input Estimation - MSE LOSS:", MSE_input_dB, "[dB]")
        print("Inference time: ", t, "s")