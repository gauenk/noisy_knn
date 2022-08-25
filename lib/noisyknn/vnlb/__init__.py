"""

An example script for a video non-local bayes.

"""

# -- imports --
import tqdm
import dnls
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- package --
from noisyknn.utils import color
from noisyknn.utils.misc import optional

# -- local --
from .step import run_step

def run(noisy,sigma,flows,params):

    # -- unpack --
    device = noisy.device
    vshape = noisy.shape

    # -- convert --
    color.rgb2yuv(noisy)

    # -- Run Steps --
    step = 1
    search,unfold,fold = get_loop_fxns(vshape,device,flows,step,params)
    basic = run_step(noisy,noisy,sigma,search,unfold,fold,step,params)

    step = 2
    search,unfold,fold = get_loop_fxns(vshape,device,flows,step,params)
    deno = run_step(noisy,basic,sigma,search,unfold,fold,step,params)

    # -- color convert --
    color.yuv2rgb(noisy)
    color.yuv2rgb(basic)
    color.yuv2rgb(deno)

    return deno,basic


def get_loop_fxns(vshape,device,flows,step,params):

    # -- unpack [standard params] --
    k1 = optional(params,'k1',100)
    k2 = optional(params,'k2',60)
    k = k1 if step == 1 else k2
    ps_s = optional(params,'ps_s',7)
    ps_d = optional(params,'ps_d',7)
    ws = optional(params,'ws',27)
    wt = optional(params,'wt',3)
    chnls_s_def = 1 if step == 1 else -1
    chnls_s = optional(params,'chnls_s',chnls_s_def)
    dil = optional(params,'dilation',1)
    stride0 = optional(params,'stride0',3)
    stride1 = optional(params,'stride1',1)
    pt = optional(params,'pt',1)
    rbounds = True

    # -- unpack [misc params] --
    nreps = optional(params,'nreps',1)
    use_rand = optional(params,'use_rand',False)
    exact_foldk = optional(params,'exact_foldk',True)

    # -- init iunfold and ifold --
    search = dnls.search.init("l2_with_index",flows.fflow,flows.bflow,
                              k,ps_s,pt,ws,wt,chnls=chnls_s,
                              stride0=stride0,stride1=stride1,dilation=dil,
                              reflect_bounds=rbounds)
    unfold = dnls.UnfoldK(ps_d,pt,dilation=dil,device=device,reflect_bounds=rbounds)
    if exact_foldk:
        fold = dnls.FoldK(vshape,use_rand=use_rand,nreps=nreps,
                          device=device,exact=True)
        # exit(0)
    else:
        fold = dnls.FoldK(vshape,use_rand=use_rand,nreps=nreps,device=device)

    return search,unfold,fold

