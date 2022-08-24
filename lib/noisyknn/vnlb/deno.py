

from noisyknn.misc import optional

# -- vnlb denoiser --
def run_denoiser(npatches,bpatches,dists,step,sigma,params):
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
    sigmab2 = sigma2 if step == 1 else beta

    # -- reshape --
    b,k,pt,c,ph,pw = npatches.shape
    shape_str = 'b k pt c ph pw -> b c k (pt ph pw)'
    npatches = rearrange(npatches,shape_str)
    bpatches = rearrange(bpatches,shape_str)

    # -- normalize --
    bpatches /= 255.
    npatches /= 255.
    b_centers = bpatches.mean(dim=2,keepdim=True)
    centers = npatches.mean(dim=2,keepdim=True)
    c_bpatches = bpatches - b_centers
    c_npatches = npatches - centers

    # -- group batches --
    shape_str = 'b c k p -> (b c) k p'
    c_bpatches = rearrange(c_bpatches,shape_str)
    c_npatches = rearrange(c_npatches,shape_str)
    centers = rearrange(centers,shape_str)
    bsize,num,pdim = c_npatches.shape

    # -- flat batch & color --
    C = th.matmul(c_bpatches.transpose(2,1),c_bpatches)/num
    eigVals,eigVecs = th.linalg.eigh(C)
    eigVals = th.flip(eigVals,dims=(1,))[...,:rank]
    eigVecs = th.flip(eigVecs,dims=(2,))[...,:rank]

    # -- denoise eigen values --
    eigVals = rearrange(eigVals,'(b c) r -> b c r',b=b)
    th_sigmab2 = th.FloatTensor([sigmab2]).reshape(1,1,1)
    th_sigmab2 = th_sigmab2.to(eigVals.device)
    emin = th.min(eigVals,th_sigmab2)
    eigVals -= emin
    eigVals = rearrange(eigVals,'b c r -> (b c) r')

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
    shape_str = '(b c) k (pt ph pw) -> b k pt c ph pw'
    patches = rearrange(c_npatches,shape_str,b=b,c=c,ph=ph,pw=pw)
    patches *= 255.
    patches = patches.contiguous()
    return patches

