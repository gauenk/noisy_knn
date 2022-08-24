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

# -- local --
from .deno import run_deno
from .step import run_step

def run(noisy,sigma,params):

    # -- load video --
    # device = "cuda:0"
    # th.cuda.set_device(device)
    # clean = th.from_numpy(clean).to(device)
    # noisy = clean + sigma * th.randn_like(clean)

    # -- params --
    vshape = noisy.shape
    t,c,h,w = noisy.shape
    pt = 1 # patch size across time
    stride0 = 5 # spacing between patch centers to search
    stride1 = 1 # spacing between patch centers when searching
    dilation = 1 # spacing between kernels
    batch_size = 32*1024 # num of patches per batch

    # -- race condition params --
    use_rand,nreps_1,nreps_2 = False,1,1

    # -- search params --
    flow = None # no flow
    k = 100 # number of neighbors
    ws = 27 # spatial-search space in each 2-dim direction
    wt = 3 # time-search space across in each fwd-bwd direction
    chnls = 1 # number of channels to use for search
    verbose = True
    ps_s,ps_d = 11,11

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #       Primary Logic Starts Here
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- convert --
    color.rgb2yuv(noisy)


    # -- setup for 2nd step --
    k = optional(params,'k',60)
    fold = dnls.FoldK(clean.shape,use_rand=use_rand,nreps=nreps_2,device=device)
    search.k = k

    # -- Run Steps --
    step = 1
    search,unfold,fold = get_loop_fxns(vshape,device,step,params)
    basic = step.run(noisy,noisy,search,unfold,fold,step,params)

    step = 2
    search,unfold,fold = get_loop_fxns(vshape,device,step,params)
    deno = step.run(noisy,basic,search,unfold,fold,step,params)


    return deno


def get_loop_fxns(vshape,device,step,params):

    # -- unpack [standard params] --
    k1 = optional(params,'k1',100)
    k2 = optional(params,'k1',100)
    k = k1 if step == 1 else k2
    ps_s = optional(params,'ps_s',11)
    ps_d = optional(params,'ps_d',11)
    ws = optional(params,'ws',27)
    wt = optional(params,'wt',3)
    chnls_s = optional(params,'chnls_s',1)
    dil = optional(params,'dilation',1)
    stride0 = optional(params,'stride0',1)
    stride1 = optional(params,'stride1',1)
    pt = optional(params,'pt',1)

    # -- unpack [misc params] --
    nreps = optional(params,'nreps',1)
    use_rand = optional(params,'use_rand',False)
    exact_foldk = optional(params,'exact_foldk',True)

    # -- init iunfold and ifold --
    search = dnls.search.init("l2_with_index",None,None,k,ps_s,pt,ws,wt,chnls=chnls_s,
                              stride0=stride0,stride1=stride1,dilation=dil)
    unfold = dnls.UnfoldK(ps_d,pt,dilation=dil,device=device)
    if exact_foldk:
        exit(0)
    else:
        fold = dnls.FoldK(vshape,use_rand=use_rand,nreps=nreps_1,device=device)

    return search,unfold,fold

