#%%
import sys
import torch
from torch.distributions import log_normal, uniform

mu_paired = -2.2424660388377395
sig_paired = 1.3329744191550479
mu_unpaired = -1.000297503438639
sig_unpaired = 1.2267076066807234

dist_paired = log_normal.LogNormal(loc=mu_paired, scale=sig_paired)
dist_unpaired = log_normal.LogNormal(loc=mu_unpaired, scale=sig_unpaired)

#%%
def read_bpseq(file):
    seq = []
    pair = []
    with open(file) as fh:
        for l in fh:
            idx, c, p = l.rstrip().split()
            seq.append(c)
            pair.append(int(p))

    return seq, pair


def fake_reactivity(seq, pair, dist_unpaired, dist_paired):
    pair = torch.tensor(pair)
    is_paired = pair>0
    reactivity = torch.zeros(pair.shape)
    reactivity[is_paired] = dist_paired.sample(reactivity[is_paired].shape)
    reactivity[torch.logical_not(is_paired)] = dist_unpaired.sample(reactivity[torch.logical_not(is_paired)].shape)
    return reactivity


def calculate_scores(reactivity, dist_unpaired, dist_paired):
    nll_unpaired = -dist_unpaired.log_prob(reactivity)
    nll_paired = -dist_paired.log_prob(reactivity)
    nll = torch.stack([nll_unpaired, nll_paired])
    nll = -torch.logsumexp(-nll, dim=0)
    nll_unpaired = nll_unpaired - nll
    nll_paired = nll_paired - nll
    return nll_unpaired, nll_paired

def output(nll_unpaired, nll_paired, noise_rate=0.0):
    d = uniform.Uniform(0, 1)
    for i, (c, nll_u, nll_p) in enumerate(zip(seq, nll_unpaired, nll_paired)):
        if d.sample() > noise_rate:
            print(i+1, c, float(nll_u), float(nll_p))
        else:
            print(i+1, c, '-', '-')

#%%
seq, pair = read_bpseq('data/TrainSetA/0.bpseq')
reactivity = fake_reactivity(seq, pair, dist_unpaired, dist_paired)
nll_unpaired, nll_paired = calculate_scores(reactivity, dist_unpaired, dist_paired)
output(nll_paired, nll_unpaired, 0.05)

# def fake_data_simple(file):
#     with open(file) as fh:
#         for l in fh:
#             idx, c, p = l.rstrip().split()
#             if int(p) > 0:
#                 print(' '.join([idx, c, '1', '0']))
#             else:
#                 print(' '.join([idx, c, '0', '1']))

# fake_data_simple(sys.argv[1])


# %%
