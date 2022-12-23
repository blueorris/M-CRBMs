import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

class CRBM(nn.Module):

    def __init__(self, n_vis, n_d1, n_d2, n_d3, n_hid, k=1, use_cuda=True):
        super(CRBM, self).__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.k = k

        self.W = nn.Parameter(torch.randn(n_vis, n_hid) * 1e-2) # weight mtx between vis and hid
        self.D1 = nn.Parameter(torch.randn(n_d1, n_hid) * 1e-2) # weight mtx between r1 and hid
        self.D2 = nn.Parameter(torch.randn(n_d2, n_hid) * 1e-2) # weight mtx between r2 and hid
        self.D3 = nn.Parameter(torch.randn(n_d3, n_hid) * 1e-2) # weight mtx between r3 and hid
        self.v_bias = nn.Parameter(torch.zeros(n_vis)) # bias vector of vis
        self.h_bias = nn.Parameter(torch.zeros(n_hid)) # bias vector of hid
        self.r1_bias = nn.Parameter(torch.zeros(n_d1))
        self.r2_bias = nn.Parameter(torch.zeros(n_d2))
        self.r3_bias = nn.Parameter(torch.zeros(n_d3))

        self.params = [self.W, self.D1, self.D2, self.D3, self.v_bias, self.h_bias, self.r1_bias, self.r2_bias, self.r3_bias]


    def sample_h_given_v_r(self, v, r1, r2, r3):
        p_h = torch.sigmoid(torch.matmul(v, self.W) + torch.matmul(r1, self.D1) + torch.matmul(r2, self.D2) + torch.matmul(r3, self.D3) + self.h_bias)
        h_act = torch.bernoulli(p_h) # based on the existance probability of hidden node, do random sampling
        return p_h, h_act


    def sample_v_given_h(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        v_act = torch.bernoulli(p_v)
        return p_v, v_act


    def cd_k(self, p_h, r1, r2, r3):
        for _ in range(self.k):
            p_v, v_act = self.sample_v_given_h(p_h)
            p_h, h_act = self.sample_h_given_v_r(p_v, r1, r2, r3)
        return p_v, p_h


    def forward(self, v, r1, r2, r3):
        p_h, h_act = self.sample_h_given_v_r(v, r1, r2, r3) # initial sampling
        p_v, v_act = self.sample_v_given_h(p_h)
        return p_v, v_act


    def free_energy(self, v, r1, r2, r3):
        vbias_term = torch.matmul(v, self.v_bias) + torch.matmul(r1, self.r1_bias) + torch.matmul(r2, self.r2_bias) + torch.matmul(r3, self.r3_bias)
        wx_b = self.h_bias + torch.matmul(v, self.W) + torch.matmul(r1, self.D1) + torch.matmul(r2, self.D2) + torch.matmul(r3, self.D3)
        hidden_term = torch.sum((1 + wx_b.exp()).log(), axis=1)
        return (-vbias_term-hidden_term).mean()


    def get_model_loss(self, input_data, r1, r2, r3):
        pos_p_h, pos_h_act = self.sample_h_given_v_r(input_data, r1, r2, r3)
        # cd-k
        neg_v, neg_h = self.cd_k(pos_h_act, r1, r2, r3)
        # Compute free energy loss
        loss = self.free_energy(input_data, r1, r2, r3) - self.free_energy(neg_v, r1, r2, r3)
        # Compute reconstruction error
        error = torch.sum((input_data - neg_v)**2)
        return error, loss
