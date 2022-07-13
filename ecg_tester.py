import torch
from math import log10
import time
import torch.nn as nn
from LFKS import smooth_ecg_LFKS, smooth_ecg_KS


pm_conti_y = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])



################################################
############   Tester for LF + KS   ############
################################################
def test_LFKS(NB, test_input, test_target, gamma, itr_r=60,  itr_q=80, r_init=1.0, q_init=1.0, win_q=5, forget_q=0.0):
    T_ecg = test_target.size()[0]
    # effective window length
    L_eff_left = round(-1/log10(gamma))
    L_eff_right = round(-1/log10(gamma))
    t_range = range(L_eff_left, T_ecg - L_eff_right) 

    # allocate memory for fitted signal
    Y_LFKS = torch.empty_like(test_input)

    loss_func = nn.MSELoss(reduction='mean')
    MSEs = torch.empty(NB)

    start = time.time()
    for j in range(0, NB):
        y_j = test_input[j, :]
        Y_LFKS[j, :] = smooth_ecg_LFKS( y_j, gamma, itr_r, itr_q, r_init, q_init, win_q, forget_q)
        MSEs[j] = loss_func(Y_LFKS[j, t_range], test_target[t_range])

    # avg loss
    MSE_avg = torch.mean(MSEs)
    MSE_avg_dB = 10*torch.log10(MSE_avg)

    # std of loss
    std_MSE = torch.std(MSEs)
    std_MSE_dB = 10*torch.log10(MSE_avg+std_MSE) - MSE_avg_dB

    print("LFKS on ECG - MSE: ", MSE_avg_dB, " [dB]")
    print("Interence Time:", time.time()-start, " s")
    return [MSE_avg_dB, std_MSE_dB, Y_LFKS] 


################################################
############## Tester for pure KS ##############
################################################
def test_KS(NB, test_input, test_target, itr_r=60,  itr_q=80, r_init=1.0, q_init=1.0, win_q=5, forget_q=0.0):
    T_ecg = test_target.size()[0]
    t_range = range(0, T_ecg) 

    # allocate memory for fitted signal
    Y_KS = torch.empty_like(test_input)

    loss_func = nn.MSELoss(reduction='mean')
    MSEs = torch.empty(NB)

    start = time.time()
    for j in range(0, NB):
        y_j = test_input[j, :]
        Y_KS[j, :] = smooth_ecg_KS( y_j, itr_r, itr_q, r_init, q_init, win_q, forget_q)
        MSEs[j] = loss_func( Y_KS[j, t_range], test_target[t_range])

    # avg loss
    MSE_avg = torch.mean(MSEs)
    MSE_avg_dB = 10*torch.log10(MSE_avg)

    # std of loss
    std_MSE = torch.std(MSEs)
    std_MSE_dB = 10*torch.log10(MSE_avg+std_MSE) - MSE_avg_dB

    print("KS on ECG - MSE: ", MSE_avg_dB, " [dB]")
    print("Interence Time:", time.time()-start, " s")
    return [MSE_avg_dB, std_MSE_dB, Y_KS] 



################################################
############  Tester for pure LF   ############
################################################
# test_input: Y[idx_NB, idx_time]
# test target: y[idx_time]
def test_local(NB, test_input, test_target, fitter, win_param_fwd, win_param_bwd, post_mult=pm_conti_y):
    T_ecg = test_target.size()[0]
    # effective window length
    L_eff_left = round(-1/log10(win_param_fwd))
    L_eff_right = round(-1/log10(win_param_bwd))
    t_range = range(L_eff_left, T_ecg - L_eff_right)

    # allocate memory for fitted signal
    Y_LF = torch.empty_like(test_input)

    loss_func = nn.MSELoss(reduction='mean')
    MSEs = torch.empty(NB)

    start = time.time()
    for j in range(0, NB):
        y_j = test_input[j, :]
        fitter.fit(y_j.unsqueeze(0), win_param_fwd, win_param_bwd, post_mult)
        Y_LF[j, :] = fitter.generate_signal().squeeze()
        MSEs[j] = loss_func(Y_LF[j, t_range], test_target[t_range])
    
    # avg loss
    MSE_avg = torch.mean(MSEs)
    MSE_avg_dB = 10*torch.log10(MSE_avg)

    # std of loss
    std_MSE = torch.std(MSEs)
    std_MSE_dB = 10*torch.log10(MSE_avg+std_MSE) - MSE_avg_dB

    print("LF on ECG - MSE: ", MSE_avg_dB, " [dB]")
    print("Interence Time:", time.time()-start, " s")
    return [MSE_avg_dB, std_MSE_dB, Y_LF] 