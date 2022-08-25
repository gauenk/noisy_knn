# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- data mngmnt --
import pandas as pd
from easydict import EasyDict as edict

# -- io --
from noisyknn.utils import io

def get_at_index(patches,nl_inds):
    K = len(nl_inds)
    patches_k = []
    for k in range(K):
        _i = nl_inds[k]
        patch_k = patches[_i[0],_i[1],_i[2]]
        patches_k.append(patch_k)
    patches_k = th.stack(patches_k)
    return patches_k

def get_raster_id(ind,h,w):
    hw = h*w
    return ind[0] * hw + ind[1] * h + ind[2]

def get_3d_inds(inds,h,w):

    # -- unpack --
    hw = h*w
    bsize,num = inds.shape
    device = inds.device

    # -- shortcuts --
    tdiv = th.div
    tmod = th.remainder

    # -- init --
    aug_inds = th.zeros((3,bsize,num),dtype=th.int64)
    aug_inds = aug_inds.to(inds.device)

    # -- fill --
    aug_inds[0,...] = tdiv(inds,hw,rounding_mode='floor') # inds // hw
    aug_inds[1,...] = tdiv(tmod(inds,hw),w,rounding_mode='floor') # (inds % hw) // w
    aug_inds[2,...] = tmod(inds,w)
    aug_inds = rearrange(aug_inds,'three b n -> (b n) three')

    return aug_inds

# OLD CODE!
def compute_pme(patches,m_inds):
    # -- pm error [clean values from new inds] --
    m_inds = rearrange(m_inds,'(t h w) k tr -> t h w k tr',t=t,h=h,w=w)
    pm_error = th.zeros(ntotal)
    for ti in range(t):
        for hi in range(h):
            for wi in range(w):

                inds_i = m_inds[ti,hi,wi]
                cpatches_b = get_at_index(patches,inds_i)

                ref = cpatches_b[[0]]
                gpatches = cpatches_b[1:]

                # error_b = compute_patch_psnrs(gpatches,ref,1.).mean().item()
                error_b = th.mean((gpatches - ref)**2,(1,2))
                if ti == 0 and hi == 0 and wi == 0:
                    print(error_b)
                error_b = (-10*th.log10(error_b)).mean()

                raster_id = get_raster_id([ti,hi,wi],h,w)
                pm_error[raster_id] = error_b
    pme = pm_error.mean().item()
    return pme

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         Manging Results
#
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def aggregate_psnrs(records,exps):
    """
    I used to have "psnrs" for each frame. I want to collapse this.
    """
    agg_records = []
    ekey = list(exps.keys())
    fields = list(records.columns)
    for e,df in records.groupby(ekey):
        agg_e = {}
        for field in fields:
            if field in ['psnrs','pme','pme_noisy','pme_clean']:
                agg_e[field] = df[field].mean().item()
            elif field == "image":
                agg_e[field] = np.stack(df[field])
            else:
                agg_e[field] = df[field][0].item()
        agg_records.append(agg_e)
    agg_records = pd.DataFrame(agg_records)
    return agg_records

def save_example(records,name):

    # -- psnrs --
    psnrs = records['psnrs'].to_numpy()
    pme = records['pme'].to_numpy()
    k = records['k'].to_numpy()
    images = np.stack(records['image'].to_numpy())
    marg = np.argmin(psnrs)
    image = images[marg]

    # -- get other info --
    print("Image Info.")
    print(k[marg])
    print(pme[marg])
    print(psnrs[marg])

    # -- save --
    ename = "example_%s" % name
    io.save_burst(image,"./output/not_only_deno/",ename)



