from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma


class GeneralizedExtremeValue(torch.autograd.Function):
    def __init__(self, 
                xi: float | torch.tensor,
                mu: float | torch.tensor, 
                sigma: float | torch.tensor):
        self.xi = xi
        self.mu = mu
        self.sigma = sigma
    
    def log_prob(self, x: torch.tensor) -> torch.tensor:
        x = self.xi / self.sigma * (x - self.mu)
        v = 1 / self.sigma * (1+x) ** (-(1+1/self.xi)) * torch.exp(- (1 + x) ** (-1/self.xi))
        return torch.log(v)


class Wu(nn.Module):
    def __init__(self,  
            xi: float = 0.774, 
            mu: float = 0.078, 
            sigma: float = 0.083,
            alpha: float = 1.006, 
            beta: float = 1.404,
            ) -> None:
        super(Wu, self).__init__()
        self.xi = nn.Parameter(torch.tensor(xi))
        self.mu = nn.Parameter(torch.tensor(mu))
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.paired_dist = GeneralizedExtremeValue(self.xi, self.mu, self.sigma)
        self.unpaired_dist = Gamma(self.alpha, self.beta)


    def forward(self, seq: list[str], paired: list[torch.tensor], targets: list[torch.Tensor]):
        lls = []
        for i in range(len(seq)):
            tgt = targets[i]
            valid = tgt > -1 # to ignore missing values (-999)
            ll = torch.mean(self.paired_dist.log_prob(tgt[valid].clip(min=1e-2)) * paired[i][valid] 
                            + self.unpaired_dist.log_prob(tgt[valid].clip(min=1e-2)) * (1-paired[i][valid]))
            lls.append(ll)
        return torch.stack(lls)


class Foo(nn.Module):
    def __init__(self,  
            p_alpha: float = 0.540,
            p_beta: float = 1.390,
            u_alpha: float = 1.006, 
            u_beta: float = 1.404,
            ) -> None:
        super(Foo, self).__init__()
        self.p_alpha = nn.Parameter(torch.tensor(p_alpha))
        self.p_beta = nn.Parameter(torch.tensor(p_beta))
        self.u_alpha = nn.Parameter(torch.tensor(u_alpha))
        self.u_beta = nn.Parameter(torch.tensor(u_beta))
        self.paired_dist = Gamma(self.p_alpha, self.p_beta)
        self.unpaired_dist = Gamma(self.u_alpha, self.u_beta)


    def forward(self, seq: list[str], paired: list[torch.tensor], targets: list[torch.Tensor]):
        lls = []
        for i in range(len(seq)):
            tgt = targets[i]
            valid = tgt > -1 # to ignore missing values (-999)
            ll = torch.mean(self.paired_dist.log_prob(tgt[valid].clip(min=1e-2)) * paired[i][valid] 
                            + self.unpaired_dist.log_prob(tgt[valid].clip(min=1e-2)) * (1-paired[i][valid]))
            lls.append(ll)
        return torch.stack(lls)
