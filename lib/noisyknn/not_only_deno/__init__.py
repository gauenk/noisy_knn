
# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- data mngmnt --
from easydict import EasyDict as edict

# -- non-local patches --
import dnls

# -- package --
from noisyknn.utils import io
from noisyknn.utils.misc import optional
from noisyknn.utils.metrics import compute_psnrs

# -- local --
from .model import get_model
from .misc import get_at_index,get_raster_id,get_3d_inds


def run(noisy,clean,flows,cfg):

    #
    # -- SETUP --
    #

    # -- dont change input? --
    clean = clean.clone()


    # -- unpack [search & patches] --
    vshape = clean.shape
    t,c,h,w = vshape
    device = clean.device
    stride0 = optional(cfg,'stride0',1)
    stride1 = optional(cfg,'stride1',1)
    dil = optional(cfg,'dilation',1)
    chnls_s = optional(cfg,'chnls_s',1)
    k = optional(cfg,'k',7)
    ps = optional(cfg,'ps',7)
    pt = optional(cfg,'pt',1)
    ws = optional(cfg,'ws',15)
    wt = optional(cfg,'wt',2)
    rbounds = optional(cfg,'reflect_bounds',True)

    # -- unpack [training] --
    nepochs = optional(cfg,'nepochs',50)
    nreps = optional(cfg,'nreps',5)
    lr = optional(cfg,'lr',1e-0)
    tgt_sigma = optional(cfg,'tgt_sigma',0.)

    # -- init search logic --
    search = dnls.search.init("l2_with_index",flows.fflow,flows.bflow,
                              k,ps,pt,ws,wt,chnls=chnls_s,
                              stride0=stride0,stride1=stride1,dilation=dil,
                              reflect_bounds=rbounds,anchor_self=True)
    unfold = dnls.UnfoldK(ps,pt,dilation=dil,device=device,reflect_bounds=rbounds)
    fold = dnls.FoldK(vshape,use_rand=False,nreps=1,device=device,exact=True)


    # -- batching info --
    nh,nw = dnls.utils.get_nums_hw(vshape,stride0,ps,dil)
    ntotal = t * nh * nw
    batch_size = optional(cfg,'batch_size',ntotal//t)
    batch_size = ntotal

    #
    # -- Core Logic --
    #

    # -- search & unfold all  --
    dists,inds = search(clean,0,ntotal)
    print(dists.shape)
    print(dists[:3,:4])
    print(inds[:3,0])
    patches = unfold(clean,inds[:,[0]])
    patches = rearrange(patches,'n k pt c ph pw -> n k c (pt ph pw)')
    patches = rearrange(patches,'(t h w) 1 c f -> t h w c f',t=t,h=nh,w=nw)
    inds = rearrange(inds,'(t h w) k three -> t h w k three',t=t,h=nh,w=nw)
    print(inds[0,:3,:3,0])
    print(inds[1,:3,:3,0])
    print(inds[2,:3,:3,0])


    # -- init model --
    model = get_model(patches.shape,device)
    optim = th.optim.SGD(model.parameters(),lr=lr)

    # -- compute loss = \sum_{i,j}A_{i,j}|D_i - D_j|^2 --
    for nr in range(nepochs):
        th.cuda.empty_cache()
        optim.zero_grad()
        loss = 0
        b_inds = th.randperm(ntotal,device=device,dtype=th.int32)[:nepochs]
        b_inds = get_3d_inds(b_inds[:,None],h,w)
        if (nr > 0) and ((nr+1) % 10) == 0: print(f"[{nr+1}/{nepochs}]")
        for r in range(nreps):
            _ind_r = b_inds[r]
            inds_r = inds[_ind_r[0],_ind_r[1],_ind_r[2]]
            mpatches = model.fwd_row(inds_r,patches)
            ref = mpatches[0][None,:]

            # -- noisy reference --
            ref_n = ref + tgt_sigma * th.randn_like(ref)
            gpatches = mpatches[1:]
            loss += th.mean((gpatches - ref_n)**2)

	    # -- clean loss --
            # cpatches_b = get_at_index(cpatches,inds_r)
            # loss += -lam_c * th.mean((mpatches - cpatches_b)**2)

        # -- backward! --
        loss.backward()
        optim.step()

    # -- output again --
    with th.no_grad():
        minds,mpatches = model(patches)
    mpatches = mpatches.detach()

    # -- reshape for agg --
    mpatches = rearrange(mpatches,'num c (ph pw) -> num 1 1 c ph pw',ph=ps,pw=ps)

    # -- aggregate patches --
    num = mpatches.shape[0]
    g_inds = rearrange(inds,'t h w k tr -> (t h w) k tr')
    g_inds = g_inds.contiguous()
    ones = th.ones_like(g_inds[:,[0],0]).float()
    print(g_inds[:3,0])
    fold(mpatches,ones,g_inds[:,[0]])
    vid,wvid = fold.vid,fold.wvid
    io.save_burst(wvid,"./output/not_only_deno","wvid")
    print(th.all(wvid>0))
    vid /= wvid
    mimage = vid
    # save_images(mimage.cpu().numpy(),"deno.png",1.)
    # save_images(images.clean.cpu().numpy(),"clean.png",1.)

    # -- compute adjacency matrix with new image --
    srch_img = mimage
    m_dists,m_inds = search(srch_img,0,ntotal)
    m_inds = rearrange(m_inds,'(t h w) k tr -> t h w k tr',t=t,h=h,w=w)

    # -- pm error [clean values from new inds] --
    pm_error = th.zeros(ntotal)
    for ti in range(t):
        for hi in range(h):
            for wi in range(w):
                raster_id = get_raster_id([ti,hi,wi],h,w)
                inds_i = m_inds[ti,hi,wi]
                cpatches_b = get_at_index(patches,inds_i)
                ref = cpatches_b[[0]]
                gpatches = cpatches_b[1:]
                error_b = th.mean((gpatches - ref)**2).item()
                pm_error[raster_id] = error_b
    pme = th.mean(pm_error).item()

    # -- noisy --
    n_nlDists,n_inds = search(noisy,0,ntotal)
    n_inds = rearrange(n_inds,'(t h w) k tr -> t h w k tr',t=t,h=h,w=w)
    noisy_pm_error = th.zeros(ntotal)
    for ti in range(t):
        for hi in range(h):
            for wi in range(w):
                raster_id = get_raster_id([ti,hi,wi],h,w)
                inds_i = n_inds[ti,hi,wi]
                cpatches_b = get_at_index(patches,inds_i)
                ref = cpatches_b[[0]]
                gpatches = cpatches_b[1:]
                error_b = th.mean((gpatches - ref)**2).item()
                noisy_pm_error[raster_id] = error_b
    pme_noisy = th.mean(noisy_pm_error).item()

    # -- [true] pm error --
    oracle_pm_error = th.zeros(ntotal)
    for ti in range(t):
        for hi in range(h):
            for wi in range(w):
                raster_id = get_raster_id([ti,hi,wi],h,w)
                inds_i = inds[ti,hi,wi]
                cpatches_b = get_at_index(patches,inds_i)
                ref = cpatches_b[[0]]
                gpatches = cpatches_b[1:]
                error_b = th.mean((gpatches - ref)**2).item()
                oracle_pm_error[raster_id] = error_b
    pme_clean = th.mean(oracle_pm_error).item()

    # -- psnrs --
    print(mimage.min(),mimage.max())
    psnrs = compute_psnrs(clean,mimage,1.).mean().item()
    print(psnrs)
    print(pme)
    print(pme_noisy)
    print(pme_clean)

    return mimage,psnrs,pme,pme_noisy,pme_clean

# ------------------------------
#
# -->        Helpers         <--
#
# ------------------------------


def normalize(images,f1,f2):
    img_f1 = images[f1]
    img_f2 = repeat(images[f2][:,None],'t 1 h w -> t c h w',c=3)
    nz = th.where(img_f2>0)
    img_f1[nz] = img_f1[nz] / img_f2[nz]
    images[f1] = img_f1

def compute_error(clean,batch,start,c,pt):

    # -- compute end --
    stop = clean.shape[0] + start

    # -- compute error (yuv) --
    clean = rearrange(clean,'b n pt c ph pw -> b n c (pt ph pw)')
    clean = clean/255.
    delta = (clean[:,[0],:] - clean[...,:,:])**2

    # -- fill --
    errors = delta.mean((1,2,3))
    return errors

def reweight_vals(images):
    nmask_before = images.weights.sum().item()
    index = th.nonzero(images.weights,as_tuple=True)
    images.vals[index] /= images.weights[index]
    irav = images.vals.ravel().cpu().numpy()
    print(np.quantile(irav,[0.1,0.2,0.5,0.8,0.9]))
    # thresh = 0.00014
    thresh = 1e-3
    nz = th.sum(images.vals < thresh).item()
    noupdate = th.nonzero(images.vals > thresh,as_tuple=True)
    images.weights[noupdate] = 0
    th.cuda.synchronize()
    nmask_after = images.weights.sum().item()
    delta_nmask = nmask_before - nmask_after
    print("tozero: [%d/%d]" % (nmask_after,nmask_before))


def fill_valid_patches(vpatches,patches,bufs):
    valid = th.nonzero(th.all(bufs.inds!=-1,1),as_tuple=True)
    for key in patches:
        if (key in patches.tensors) and not(patches[key] is None):
            patches[key][valid] = vpatches[key]

def get_valid_vals(bufs):
    valid = th.nonzero(th.all(bufs.inds!=-1,1),as_tuple=True)
    nv = len(valid[0])
    vals = bufs.vals[valid]
    return vals

def get_valid_patches(patches,bufs):
    valid = th.nonzero(th.all(bufs.inds!=-1,1),as_tuple=True)
    nv = len(valid[0])
    vpatches = edict()
    for key in patches:
        if (key in patches.tensors) and not(patches[key] is None):
            vpatches[key] = patches[key][valid]
        else:
            vpatches[key] = patches[key]
    vpatches.shape[0] = nv
    return vpatches
