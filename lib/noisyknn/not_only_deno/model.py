
# -- python deps --
from tqdm import tqdm
import copy,math
import torch
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

import torchvision.utils as tvu

# -- neural networks --
import torch.nn as nn
import torch.nn.functional as nnF

# -- local --
from .misc import get_3d_inds

#
# -- Model for Learning --
#

def get_model(shape,device):

    t,h,w,c,f = shape
    num = t * h * w
    model = PmModel(num,t,h,w,f).to(device)
    model = model.train()
    return model

class PmModel(nn.Module):
    def __init__(self,num,t,h,w,n_ftrs):
        super().__init__()
        self.num = num
        self.t,self.h,self.w = t,h,w
        self.lin_list = []
        for n in range(self.num):
            lin_n = nn.Linear(n_ftrs,n_ftrs,bias=True)
            self.setup_lin(lin_n)
            self.lin_list.append(lin_n)
        self.lin_list = nn.ModuleList(self.lin_list)

    def fwd(self,nl_inds,patches):
        # -- unpack params --
        outs = []
        B = len(nl_inds)
        for b in range(B):
            outs_b = self.fwd_row(nl_inds[b],patches)
            outs.append(outs_b)
        outs = th.stack(outs)
        return outs

    def fwd_row(self,nl_inds,X):
        # -- prepare --
        K = len(nl_inds)

        # -- foward pass --
        outs = []
        for k in range(K):
            ind_k = nl_inds[k]
            lin_id = self.get_raster_id(ind_k)
            X_k = X[ind_k[0],ind_k[1],ind_k[2]]
            W = self.lin_list[lin_id].weight
            b = self.lin_list[lin_id].bias
            out_k = nnF.linear(X_k, W, b)
            outs.append(out_k)
        outs = th.stack(outs)
        return outs

    def get_raster_id(self,ind):
        hw,h = self.h*self.w,self.h
        return ind[0] * hw + ind[1] * h + ind[2]

    def setup_lin(self,layer):
        for name,weights in layer.named_parameters():
            weights.data[...] = 0
            if name == "weight":
                weights.data.fill_diagonal_(1.)
            # weights.data.requires_grad_(True)

    def forward(self,x):

        # -- forward --
        h,w = self.h,self.w
        npix = self.t*self.h*self.w
        inds = th.arange(0,npix,device=x.device,dtype=th.int32)[:,None]
        inds = get_3d_inds(inds,h,w)
        out = self.fwd_row(inds,x)

        # -- reshape --
        return inds,out


