

import dnls
from .deno import run_denoiser


def run_step(noisy,basic,search,unfold,fold,step,params):

    # -- unpack --
    vshape = noisy.shape
    device = noisy.device

    # -- batching info --
    nh,nw = dnls.utils.get_nums_hw(vshape,search.stride0,search.ps,search.dilation)
    ntotal = t * nh * nw
    batch_size = optional(parmas,'batch_size',batch_size)
    nbatches = (ntotal-1) // batch_size + 1

    # -- run batches --
    for batch in tqdm.tqdm(range(nbatches)):

        # -- get batch --
        index = min(batch_size * batch,ntotal)
        nbatch_i = min(batch_size,ntotal-index)

        # -- search patches --
        dists,inds = search(noisy,index,nbatch_i)

        # -- get patches --
        noisy_patches = unfold(noisy,inds)
        basic_patches = unfold(basic,inds)

        # -- subset patches [remove flat] --


        # -- process --
        patches_mod = run_denoiser(noisy_patches,basic_patches,dists,
                                   step,sigma,parmas)

        # -- fill denoed patches [only non-flat] --


        # -- regroup --
        ones = th.ones_like(dists)
        fold(patches_mod,ones,inds)

    # -- normalize --
    deno,weights = fold.vid,fold.wvid
    deno /= weights
    zargs = th.where(weights == 0)
    deno[zargs] = basic[zargs]

    return deno
