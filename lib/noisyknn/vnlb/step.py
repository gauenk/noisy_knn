
# -- imports --
import dnls
import tqdm

# -- linalg --
import torch as th
from einops import rearrange

# -- package --
from noisyknn.utils.misc import optional

# -- local --
from .deno import run_denoiser
MAX_BATCH_SIZE = 15*1024

def run_step(noisy,basic,sigma,search,unfold,fold,step,params):

    # -- unpack --
    t,c,h,w = noisy.shape
    vshape = noisy.shape
    device = noisy.device

    # -- batching info --
    nh,nw = dnls.utils.get_nums_hw(vshape,search.stride0,search.ps,search.dilation)
    ntotal = t * nh * nw
    batch_size = optional(params,'batch_size',ntotal//t)
    batch_size = min(min(batch_size,MAX_BATCH_SIZE),ntotal)
    nbatches = (ntotal-1) // batch_size + 1

    # -- alloc flat inds --
    flat = th.zeros((batch_size),dtype=th.int8,device=device)

    # -- run batches --
    for batch in tqdm.tqdm(range(nbatches)):

        # -- get batch --
        index = min(batch_size * batch,ntotal)
        nbatch_i = min(batch_size,ntotal-index)

        # -- search patches --
        dists,inds = search(basic/255.,index,nbatch_i)

        # -- get patches --
        noisy_patches = unfold(noisy,inds)
        basic_patches = unfold(basic,inds)

        # -- subset patches [keep non-flat] --
        if step == 2:
            fill_flat(flat,basic_patches,sigma,params)

        # -- process --
        patches_mod = run_denoiser(noisy_patches,basic_patches,dists,
                                   flat,step,sigma,params)

        # -- regroup --
        ones = th.ones_like(dists)
        fold(patches_mod,ones,inds)

    # -- normalize --
    deno,weights = fold.vid,fold.wvid
    deno /= weights
    zargs = th.where(weights == 0)
    print(len(zargs[0]))
    deno[zargs] = basic[zargs]

    return deno

def fill_flat(flat,patches,sigma,params):

    # -- unpack --
    gamma = optional(params,'gamma',0.1) # smaller = fewer flat patches
    sigma2 = (sigma/255.)**2

    # -- shapes --
    bsize,num,ps_t,c,ps,ps = patches.shape
    pflat = rearrange(patches,'b n pt c ph pw -> b c (n pt ph pw)')/255.

    # -- compute var --
    var = th.std(pflat,2).pow(2).mean(1)

    # -- compute thresh --
    thresh = gamma*sigma2
    n = len(var)
    flat[...] = 0
    # flat[:n] = var < thresh
    # flat[n:] = -1

