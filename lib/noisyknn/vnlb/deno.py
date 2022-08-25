
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange

# -- opts --
from noisyknn.utils.misc import optional

# -- vnlb denoiser --
def run_denoiser(npatches,bpatches,dists,flat,step,sigma,params):
    """
    n = noisy, b = basic
    npatches.shape = (batch_size,k,pt,c,ps,ps)
    with k == 1 and pt == 1
    k = number of neighbors
    pt = temporal patch size
    """

    # -- params --
    sigma2 = (sigma/255.)**2
    rank = optional(params,'rank',39)
    thresh1 = optional(params,'thresh1',2.7)
    thresh2 = optional(params,'thresh2',0.7)
    beta = optional(params,'beta',10e-3)
    thresh = thresh1 if step == 1 else thresh2
    sigmab2 = sigma2 if step == 1 else beta*sigma2

    # -- reshape --
    b,k,pt,c,ph,pw = npatches.shape
    shape_str = 'b k pt c ph pw -> b c k (pt ph pw)'
    npatches = rearrange(npatches,shape_str)
    bpatches = rearrange(bpatches,shape_str)

    # -- normalize --
    bpatches /= 255.
    npatches /= 255.

    # -- filter --
    # run_flat_deno(npatches[flat_inds],bpatches[flat_inds],params)
    # run_bayes_deno(npatches[nonflat_inds],bpatches[nonflat_inds],params)
    if step == 1:
        run_bayes_deno(npatches,bpatches,step,sigma,params)
    else:

        # -- split inds --
        flat_inds,nonflat_inds = get_flat_inds(flat)

        # -- run flat --
        _npatches = npatches[flat_inds]
        run_flat_deno(_npatches,bpatches[flat_inds],params)
        npatches[flat_inds] = _npatches

        # -- run non-flat --
        _npatches = npatches[nonflat_inds]
        run_bayes_deno(_npatches,bpatches[nonflat_inds],step,sigma,params)
        npatches[nonflat_inds] = _npatches
        # run_bayes_deno(npatches,bpatches,step,sigma,params)

    # -- reshape --
    shape_str = 'b c k (pt ph pw) -> b k pt c ph pw'
    npatches = rearrange(npatches,shape_str,c=c,ph=ph,pw=pw)
    npatches *= 255.
    npatches = npatches.contiguous()

    return npatches

def get_flat_inds(flat):
    flat_inds = th.where(flat == 1)[0]
    nonflat_inds = th.where(flat == 0)[0]
    # print(len(nonflat_inds)/(len(flat)*1.)) # % non-flat
    return flat_inds,nonflat_inds

def run_flat_deno(npatches,bpatches,params):
    npatches[:,:,:] = bpatches[:,:,:].mean((2,3),keepdim=True)

def run_bayes_deno(npatches,bpatches,step,sigma,params):

    # -- params --
    sigma2 = (sigma/255.)**2
    rank = optional(params,'rank',39)
    thresh1 = optional(params,'thresh1',2.7)
    thresh2 = optional(params,'thresh2',0.2)
    beta = optional(params,'beta',1.)
    thresh = thresh1 if step == 1 else thresh2
    sigma2 = beta * sigma2
    sigmab2 = sigma2 if step == 1 else 0.

    # -- normalizing --
    b_centers = bpatches.mean(dim=2,keepdim=True)
    centers = npatches.mean(dim=2,keepdim=True)
    c_bpatches = bpatches - b_centers
    c_npatches = npatches - centers

    # -- group batches --
    shape_str = 'b c k p -> (b c) k p'
    B,C,K,P = c_bpatches.shape
    c_bpatches = rearrange(c_bpatches,shape_str)
    c_npatches = rearrange(c_npatches,shape_str)
    centers = rearrange(centers,shape_str)
    bsize,num,pdim = c_npatches.shape

    # -- flat batch & color --
    C = th.matmul(c_bpatches.transpose(2,1),c_bpatches)/num
    # eigVals,eigVecs = run_scipy_eigh(C,rank)
    eigVals,eigVecs = th.linalg.eigh(C)
    eigVals = th.flip(eigVals,dims=(1,))[...,:rank]
    eigVecs = th.flip(eigVecs,dims=(2,))[...,:rank]

    # -- denoise eigen values --
    if sigmab2 > 0:
        th_sigmab2 = th.FloatTensor([sigmab2]).reshape(1,1)
        th_sigmab2 = th_sigmab2.to(eigVals.device)
        emin = th.min(eigVals,th_sigmab2)
        eigVals -= emin

    # -- filter coeffs --
    geq = th.where(eigVals > (thresh*sigma2))
    leq = th.where(eigVals <= (thresh*sigma2))
    eigVals[geq] = 1. / (1. + sigma2 / eigVals[geq])
    eigVals[leq] = 0.

    # -- denoise patches --
    bsize = c_npatches.shape[0]
    Z = th.matmul(c_npatches,eigVecs)
    R = eigVecs * eigVals[:,None]
    tmp = th.matmul(Z,R.transpose(2,1))
    c_npatches[...] = tmp

    # -- add patches --
    c_npatches[...] += centers

    # -- reshape --
    c_npatches = rearrange(c_npatches,'(b c) k p -> b c k p',c=3)

    # -- fill --
    npatches[...] = c_npatches[...]

def run_scipy_eigh(C,R):
    import scipy
    B,D,_ = C.shape
    device = C.device
    eigVals = th.zeros((B,R),device=device)
    eigVecs = th.zeros((B,D,R),device=device)
    for i in range(B):
        rtns = scipy.linalg.lapack.ssyevx(C[i].cpu().numpy(),compute_v=1,
                                          range='I',lower=0,
                                          overwrite_a=0,
                                          vl=0.0,vu=1.0,
                                          il=D-R+1,iu=D,abstol=0.0,
                                          lwork=8*D)
        eigVals_i = rtns[0]
        eigVecs_i = rtns[1]
        eigVals[i] = th.from_numpy(eigVals_i).to(device)[:R]
        eigVecs[i] = th.from_numpy(eigVecs_i).to(device)
        info = rtns[4]
    return eigVals,eigVecs
