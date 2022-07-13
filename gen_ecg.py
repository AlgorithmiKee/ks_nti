# %%
import torch

# %% [markdown]
# # "ECG" Signal
# A ECG signal consists of 7 segments inside a cylce:
# 1. P wave (round)
# 2. PR segement
# 3. QRS complex:
#     - negative Q wave
#     - positive R wave
#     - negative S wave
# 4. ST segement
# 5. T wave (round)

# %%
from torch.distributions.normal import Normal

# Noise levels
range_prec_r_dB = [-10.0, 0.0, 10.0, 20.0, 30.0]

# helper for generate gaussian impluse
def gen_gaussian_pulse(width, amp=1):
    ndist = Normal(torch.tensor(0), torch.tensor(width/3))
    t = torch.arange(-width/2, 1+width/2)
    gpulse = ndist.log_prob(t).exp()
    gpulse = gpulse - gpulse.min()
    gpulse *= amp
    return gpulse




# generate signal
def generate_ecg(NB, opath):
    amp = 100.0

    p_pulse = gen_gaussian_pulse(width=40, amp=1)
    pr_segm = torch.zeros(30)
    q_pluse = gen_gaussian_pulse(width=8, amp=-0.5)
    r_pluse = gen_gaussian_pulse(width=20, amp=5)
    s_pluse = gen_gaussian_pulse(width=10, amp=-1)
    st_segm = torch.zeros(30)
    t_pulse = gen_gaussian_pulse(width=40, amp=1) 

    # Ground Truth
    y_ecg = amp*torch.cat([ p_pulse,
                        pr_segm,
                        q_pluse,
                        r_pluse,
                        s_pluse,
                        st_segm,
                        t_pulse], dim=0)
    T_ecg = y_ecg.size()[-1]
    
    # Noisy obs
    Y_ECG_noisy = y_ecg.repeat([len(range_prec_r_dB), NB, 1])
    for idx_r, prec_r_dB in enumerate(range_prec_r_dB):
        r = pow(10.0, -prec_r_dB/20)
        err = torch.normal(mean=torch.zeros([NB, T_ecg]), std=r*torch.ones([NB, T_ecg]))
        Y_ECG_noisy[idx_r, :, :] += err

    torch.save(y_ecg, opath+'ECG_gt.pt')
    torch.save(Y_ECG_noisy, opath+'ECG_noisy.pt')

# %%
if __name__ == '__main__':
    NB = 1000
    generate_ecg(NB, opath='Sim_ecg/traj/')