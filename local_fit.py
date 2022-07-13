from pyexpat import model
import torch
from abc import ABC, abstractmethod

def moving_avg(x, win):
    x_ma = torch.empty_like(x)
    T = x.size()[-1]
    for t in range(0, win+1):
        x_ma[:, t] = torch.mean(x[:, 0:t+win+1])
    for t in range(win+1, T-win-1):
        x_ma[:, t] = x_ma[:, t-1] + (x[:, t+win]-x[:, t-1-win])/(2*win+1)
    for t in range(T-win-1, T):
        x_ma[:, t] = torch.mean(x[:, t-win:])
    return x_ma


## local cost ratio
# model1 corresponds to the hypothesis to be tested
# model0 corresponds to null hypothesis
def LCR(model1, model0):
    model1.get_cost()
    model0.get_cost()
    # lcr is larger where model 1 outperforms model 0
    lcr = -0.5*torch.log10(model1.cost/model0.cost)
    return lcr


def merge(model1, model2, thd):
    lcr = LCR(model1, model2)
    T = lcr.size()[0]
    y_merge = torch.clone(model2.y_smooth)
    for t in range(0, T):
        if lcr[t] > thd:
            y_merge[:,t] = model1.y_smooth[:,t]
    return y_merge


class LocalFitter:
    def __init__(self, A_fwd, C_fwd, win_type, A_bwd=None, C_bwd=None):
        self.dim_x = A_fwd.size()[0]
        self.A_fwd = A_fwd
        self.C_fwd = C_fwd        
        self.win_type = win_type

        self.A_bwd = A_bwd
        self.C_bwd = C_bwd

        self.x_optimal = None
        self.kappa = None
        self.xi = None
        self.W = None

        self.y_smooth = None


    def fit(self, y, win_param_fwd, win_param_bwd, post_mult):
        [
            self.x_optimal, # size: [2*dim_x, T]
            self.kappa,
            self.xi,
            self.W
        ] = self.win_type.fit(y, 
                              self.A_fwd, self.C_fwd, win_param_fwd,
                              self.A_bwd, self.C_bwd, win_param_bwd,
                              post_mult)
        self.get_cost()

   
    def fit_fast_2Dline(self, y, win_param_fwd, win_param_bwd, post_mult):
        [
            self.x_optimal, # size: [2*dim_x, T]
            self.kappa,
            self.xi,
            self.W
        ] = self.win_type.fit_fast_2Dline(y, 
                              self.A_fwd, self.C_fwd, win_param_fwd,
                              self.A_bwd, self.C_bwd, win_param_bwd,
                              post_mult)
        self.get_cost()


    def generate_signal(self):
        I = torch.eye(self.dim_x)
        O = torch.zeros([self.dim_x, self.dim_x])
        get_left = torch.cat((I, O), dim=1)
        x_left = torch.matmul(get_left, self.x_optimal)
        self.y_smooth = torch.matmul(self.C_fwd, x_left)
        return self.y_smooth

    def get_cost(self):
        T = self.x_optimal.size()[-1]
        self.cost = torch.empty(T)
        for t in range(0, T):
            xiWxi_t = torch.matmul(self.xi[:,:,t], torch.linalg.pinv(self.W[:,:,t]))
            xiWxi_t = torch.matmul(xiWxi_t, torch.transpose(self.xi[:,:,t], 0, 1)).squeeze()
            self.cost[t] = self.kappa[t] - xiWxi_t


class LocalWin(ABC):   
    
    @abstractmethod
    def fit(self):
        pass

# left sided Exp window
class ExpWin_LS(LocalWin):
    ## fit the model locally
    # y[:, t]: traj of y
    # gamma: win param
    def fit(self, y, A, C, gamma):
        A_T = torch.transpose(A, 0, 1)
        C_T = torch.transpose(C, 0, 1)
        dim_x = A.size()[0]
        dim_y = C.size()[0]         
        T = y.size()[1]

        # store the estimation
        x = torch.zeros(size=[dim_x, T])
        # variables for recursions
        xi = torch.empty(size=[1, dim_x, T])
        W = torch.empty(size=[dim_x, dim_x, T])

        # init
        xi[:,:,0] = torch.zeros_like(xi[:,:,0])
        W[:,:,0] = torch.zeros_like(W[:,:,0])
        for t in range(1, T):
            yt = y[:, t].unsqueeze(1)
            xi_t = gamma*torch.matmul(xi[:,:,t-1], A)
            xi_t += torch.matmul(torch.transpose(yt, 0, 1), C)
            W_t = gamma*torch.matmul(A_T, W[:,:,t-1])
            W_t = torch.matmul(W_t, A)
            W_t += torch.matmul(C_T, C)
            W_inv_t = torch.linalg.pinv(W_t)
            x_t = torch.matmul(W_inv_t, torch.transpose(xi_t, 0, 1))
            
            xi[:,:,t] = xi_t
            W[:,:,t] = W_t
            x[:,t] = x_t.squeeze()
        return x

# double sided Exp window
class ExpWin_DS(LocalWin):
    def update_kappa_fwd(self, kappa_prev, yt, gamma_fwd):
        kappa_t = gamma_fwd*kappa_prev
        kappa_t += torch.matmul(torch.transpose(yt, 0, 1), yt).squeeze()
        return kappa_t
    
    def update_xi_fwd(self, xi_prev, yt, A_fwd, C_fwd, gamma_fwd):
        xi_t = gamma_fwd*torch.matmul(xi_prev, A_fwd)
        xi_t += torch.matmul(torch.transpose(yt, 0, 1), C_fwd)
        return xi_t
    
    def update_W_fwd(self, W_prev, A_fwd, C_fwd, gamma_fwd):
        A_fwd_T = torch.transpose(A_fwd, 0, 1)
        C_fwd_T = torch.transpose(C_fwd, 0, 1)
        W_t = gamma_fwd*torch.matmul(A_fwd_T, W_prev)
        W_t = torch.matmul(W_t, A_fwd)
        W_t += torch.matmul(C_fwd_T, C_fwd)
        return W_t
    
    def compute_Wss_fwd(self, A_fwd, C_fwd, gamma_fwd):
        dim_x = A_fwd.size()[0]
        C_fwd_T = torch.transpose(C_fwd, 0, 1)
        A_fwd_T = torch.transpose(A_fwd, 0, 1)
        CTC = torch.matmul(C_fwd_T, C_fwd)
        A_power_k = torch.eye(dim_x)
        Wss_fwd = torch.zeros(size=[dim_x, dim_x])
        for k in range(0, 100):
            W_increm = torch.matmul(CTC, A_power_k)
            W_increm = torch.matmul(torch.transpose(A_power_k, 0, 1), W_increm)
            Wss_fwd += W_increm
            A_power_k = torch.matmul(A_power_k, A_fwd)
        return A_power_k

    def compute_Wss_fwd_2Dline(self, gamma_fwd):
        Wss_fwd = torch.zeros(size=[2, 2])
        Wss_fwd[0, 0] = 1/(1-gamma_fwd)
        Wss_fwd[0, 1] = gamma_fwd * Wss_fwd[0, 0] / (1-gamma_fwd)
        Wss_fwd[1, 0] = Wss_fwd[0, 1]
        Wss_fwd[1, 1] = (1+gamma_fwd) * Wss_fwd[0, 1] / (1-gamma_fwd)
        return Wss_fwd

    def compute_Wss_bwd_2Dline(self, gamma_bwd):
        Wss_bwd = torch.zeros(size=[2, 2])
        Wss_bwd[0, 0] = gamma_bwd/(1-gamma_bwd)
        Wss_bwd[0, 1] = gamma_bwd * Wss_bwd[0, 0] / (1-gamma_bwd)
        Wss_bwd[1, 0] = Wss_bwd[0, 1]
        Wss_bwd[1, 1] = (1+gamma_bwd) * Wss_bwd[0, 1] / (1-gamma_bwd)
        return Wss_bwd

    def update_kappa_bwd(self, kappa_next, y_next, gamma_bwd):
        kappa_t = kappa_next + torch.matmul(torch.transpose(y_next, 0, 1), y_next).squeeze()
        kappa_t *= gamma_bwd
        return kappa_t
    
    def update_xi_bwd(self, xi_next, y_next, A_bwd, C_bwd, gamma_bwd):
        xi_t = xi_next + torch.matmul(torch.transpose(y_next, 0, 1), C_bwd)
        xi_t = gamma_bwd*torch.matmul(xi_t, A_bwd)
        return xi_t
    
    def update_W_bwd(self, W_prev, A_bwd, C_bwd, gamma_bwd):
        A_bwd_T = torch.transpose(A_bwd, 0, 1)
        C_bwd_T = torch.transpose(C_bwd, 0, 1)        
        W_t = W_prev + torch.matmul(C_bwd_T, C_bwd)
        W_t = torch.matmul(A_bwd_T, W_t)
        W_t = gamma_bwd*torch.matmul(W_t, A_bwd)
        return W_t
    
    def fit(self, y, A_fwd, C_fwd, win_param_fwd, A_bwd, C_bwd, win_param_bwd, post_mult):
        dim_x = A_fwd.size()[0]
        dim_y = C_fwd.size()[0]         
        T = y.size()[1]
        
        # left sided parameter
        kappa_fwd = torch.empty(size=[T])
        xi_fwd = torch.empty(size=[1, dim_x, T])
        W_fwd = torch.empty(size=[dim_x, dim_x, T])
        # right sided parameter
        kappa_bwd = torch.empty(size=[T])
        xi_bwd = torch.empty(size=[1, dim_x, T])
        W_bwd= torch.empty(size=[dim_x, dim_x, T])
        # init
        kappa_fwd[0] = torch.zeros_like(kappa_fwd[0])
        xi_fwd[:,:,0] = torch.zeros_like(xi_fwd[:,:,0])
        W_fwd[:,:,0] = torch.zeros_like(W_fwd[:,:,0])
        kappa_bwd[-1] = torch.zeros_like(kappa_bwd[-1])
        xi_bwd[:,:,-1] = torch.zeros_like(xi_bwd[:,:,-1])
        W_bwd[:,:,-1] = torch.zeros_like(W_bwd[:,:,-1])
        
        # fwd pass
        for t in range(1, T):
            yt = y[:, t].unsqueeze(1)
            kappa_fwd[t] = self.update_kappa_fwd(kappa_fwd[t-1], yt, win_param_fwd)
            xi_fwd[:,:,t] = self.update_xi_fwd(xi_fwd[:,:,t-1], yt, A_fwd, C_fwd, win_param_fwd)
            W_fwd[:,:,t] = self.update_W_fwd(W_fwd[:,:,t-1], A_fwd, C_fwd, win_param_fwd)
        # bwd pass
        for t in range(T-2, -1, -1):
            y_next = y[:,t+1].unsqueeze(1)
            kappa_bwd[t] = self.update_kappa_bwd(kappa_bwd[t+1], y_next, win_param_bwd)
            xi_bwd[:,:,t] = self.update_xi_bwd(xi_bwd[:,:,t+1], y_next, A_bwd, C_bwd, win_param_bwd)
            W_bwd[:,:,t] = self.update_W_bwd(W_bwd[:,:,t+1], A_bwd, C_bwd, win_param_bwd)
        # combine fwd and bwd pass
        kappa = kappa_fwd + kappa_bwd
        xi = torch.cat((xi_fwd, xi_bwd), dim=1)
        W_list = [torch.block_diag(W_fwd[:,:,t], W_bwd[:,:,t]) for t in range(0, T)]
        W = torch.stack(W_list, dim=2)
        
        # post multiplier
        dim_subspace = post_mult.size()[1]
        post_mult_top = torch.transpose(post_mult, 0, 1)
        xi_post = torch.empty(size=[1, dim_subspace, T])
        W_post = torch.empty(size=[dim_subspace, dim_subspace, T])
        for t in range(0, T):
            xi_post[:,:,t] = torch.matmul(xi[:,:,t], post_mult)
            W_post_t = torch.matmul(post_mult_top, W[:,:,t])
            W_post[:,:,t] = torch.matmul(W_post_t, post_mult)

        # optimal solution
        x = torch.empty(size=[2*dim_x, T])
        for t in range(0, T):
            # optimal solution in subspace
            optimizer_t = torch.matmul(torch.linalg.pinv(W_post[:,:,t]), torch.transpose(xi_post[:,:,t], 0, 1))
            x[:,t] = torch.matmul(post_mult, optimizer_t).squeeze()
        return [x, kappa, xi_post, W_post]

    def fit_fast_2Dline(self, y, A_fwd, C_fwd, win_param_fwd, A_bwd, C_bwd, win_param_bwd, post_mult):
        dim_x = A_fwd.size()[0]
        dim_y = C_fwd.size()[0]         
        T = y.size()[1]

        # off line computation of W
        W_ss_fwd = self.compute_Wss_fwd_2Dline(win_param_fwd)
        W_ss_bwd = self.compute_Wss_bwd_2Dline(win_param_bwd)

        # left sided parameter
        kappa_fwd = torch.empty(size=[T])
        xi_fwd = torch.empty(size=[1, dim_x, T])
        W_fwd = W_ss_fwd.unsqueeze(2).repeat(1, 1, T)
        # right sided parameter
        kappa_bwd = torch.empty(size=[T])
        xi_bwd = torch.empty(size=[1, dim_x, T])
        W_bwd= W_ss_bwd.unsqueeze(2).repeat(1, 1, T)

        # init
        kappa_fwd[0] = torch.zeros_like(kappa_fwd[0])
        xi_fwd[:,:,0] = torch.zeros_like(xi_fwd[:,:,0])
        kappa_bwd[-1] = torch.zeros_like(kappa_bwd[-1])
        xi_bwd[:,:,-1] = torch.zeros_like(xi_bwd[:,:,-1])
        
        # fwd pass
        for t in range(1, T):
            yt = y[:, t].unsqueeze(1)
            kappa_fwd[t] = self.update_kappa_fwd(kappa_fwd[t-1], yt, win_param_fwd)
            xi_fwd[:,:,t] = self.update_xi_fwd(xi_fwd[:,:,t-1], yt, A_fwd, C_fwd, win_param_fwd)
        # bwd pass
        for t in range(T-2, -1, -1):
            y_next = y[:,t+1].unsqueeze(1)
            kappa_bwd[t] = self.update_kappa_bwd(kappa_bwd[t+1], y_next, win_param_bwd)
            xi_bwd[:,:,t] = self.update_xi_bwd(xi_bwd[:,:,t+1], y_next, A_bwd, C_bwd, win_param_bwd)
        # combine fwd and bwd pass
        kappa = kappa_fwd + kappa_bwd
        xi = torch.cat((xi_fwd, xi_bwd), dim=1)
        W_list = [torch.block_diag(W_fwd[:,:,t], W_bwd[:,:,t]) for t in range(0, T)]
        W = torch.stack(W_list, dim=2)
        
        # post multiplier
        dim_subspace = post_mult.size()[1]
        post_mult_top = torch.transpose(post_mult, 0, 1)
        xi_post = torch.empty(size=[1, dim_subspace, T])
        W_post = torch.empty(size=[dim_subspace, dim_subspace, T])
        for t in range(0, T):
            xi_post[:,:,t] = torch.matmul(xi[:,:,t], post_mult)
            W_post_t = torch.matmul(post_mult_top, W[:,:,t])
            W_post[:,:,t] = torch.matmul(W_post_t, post_mult)

        # optimal solution
        x = torch.empty(size=[2*dim_x, T])
        for t in range(0, T):
            # optimal solution in subspace
            optimizer_t = torch.matmul(torch.linalg.pinv(W_post[:,:,t]), torch.transpose(xi_post[:,:,t], 0, 1))
            x[:,t] = torch.matmul(post_mult, optimizer_t).squeeze()
        return [x, kappa, xi_post, W_post]

# left sided Rect window
class RectWin_LS(LocalWin):
    # win_param[0]: 0.99999 for numerical stability
    # win_param[1]: win length
    def fit(self, y, A, C, win_param):
        A_T = torch.transpose(A, 0, 1)
        C_T = torch.transpose(C, 0, 1)
        dim_x = A.size()[0]
        dim_y = C.size()[0]         
        T = y.size()[1]        
        
        gamma = win_param[0]
        L = win_param[1]
        T = y.size()[1]
        # store the estimation
        x = torch.zeros(size=[dim_x, T])
        # variables for recursions
        xi = torch.empty(size=[1, dim_x, T])
        W = torch.empty(size=[dim_x, dim_x, T])

        # init
        xi[:,:,0] = torch.zeros_like(xi[:,:,0])
        W[:,:,0] = torch.zeros_like(W[:,:,0])
        A_L = torch.linalg.matrix_power(A, L)
        gamma_L = pow(gamma, L)
        for t in range(1, L):
            yt = y[:, t].unsqueeze(1)
            yt_T = torch.transpose(yt, 0, 1)

            xi_t = gamma*torch.matmul(xi[:,:,t-1], A)
            xi_t += torch.matmul(yt_T, C)

            W_t = gamma*torch.matmul(A_T, W[:,:,t-1])
            W_t = torch.matmul(W_t, A)
            W_t += torch.matmul(C_T, C)

            W_inv_t = torch.linalg.pinv(W_t)
            x_t = torch.matmul(W_inv_t, torch.transpose(xi_t, 0, 1))
            
            xi[:,:,t] = xi_t
            W[:,:,t] = W_t
            x[:,t] = x_t.squeeze()        
        for t in range(L, T):
            yt = y[:, t].unsqueeze(1)
            yt_T = torch.transpose(yt, 0, 1)

            xi_t = gamma*torch.matmul(xi[:,:,t-1], A)
            xi_t += torch.matmul(yt_T, C)

            W_t = gamma*torch.matmul(A_T, W[:,:,t-1])
            W_t = torch.matmul(W_t, A)
            W_t += torch.matmul(C_T, C)            
            
            # correction terms
            y0 = y[:, t-L].unsqueeze(1)
            y0_T = torch.transpose(y0, 0, 1)
            correct_xi = gamma_L*torch.matmul(y0_T, C)
            correct_xi = torch.matmul(correct_xi, A_L)
            xi_t -= correct_xi
            correct_W = torch.matmul(C, A_L)
            correct_W = gamma_L*torch.matmul(torch.transpose(correct_W, 0, 1), correct_W)
            W_t -= correct_W

            W_inv_t = torch.linalg.pinv(W_t)
            x_t = torch.matmul(W_inv_t, torch.transpose(xi_t, 0, 1))
            
            xi[:,:,t] = xi_t
            W[:,:,t] = W_t
            x[:,t] = x_t.squeeze()
        return x
