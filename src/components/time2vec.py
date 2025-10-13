# src/components/time2vec.py

import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    """
    Time2Vec module that learns a vector representation of time.
    As described in: https://arxiv.org/abs/1907.05321
    """
    def __init__(self, in_features, out_features):
        super(Time2Vec, self).__init__()
        self.out_features = out_features
        # Learnable parameters for the linear term
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.phi0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        # Learnable parameters for the periodic terms
        self.wi = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.phii = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))

    def forward(self, t):
        # t shape: [batch_size, seq_len, 1]
        
        # Linear term
        v0 = torch.matmul(t, self.w0) + self.phi0
        
        # Periodic terms
        vi = torch.sin(torch.matmul(t, self.wi) + self.phii)
        
        # Concatenate linear and periodic features
        return torch.cat([v0, vi], -1)