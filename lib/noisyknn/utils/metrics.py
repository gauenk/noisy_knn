import torch as th
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as compute_ssim_ski

def compute_ssims(clean,deno,div=255.):
    nframes = clean.shape[0]
    ssims = []
    for t in range(nframes):
        clean_t = clean[t].cpu().numpy().transpose((1,2,0))/div
        deno_t = deno[t].cpu().numpy().transpose((1,2,0))/div
        ssim_t = compute_ssim_ski(clean_t,deno_t,channel_axis=-1,
                                  data_range=1.)
        ssims.append(ssim_t)
    ssims = np.array(ssims)
    return ssims

def compute_psnrs(clean,deno,div=255.):
    t = clean.shape[0]
    clean = clean.detach().cpu().numpy()
    deno = deno.detach().cpu().numpy()
    psnrs = []
    t = clean.shape[0]
    for ti in range(t):
        psnr_ti = comp_psnr(clean[ti,:,:,:], deno[ti,:,:,:], data_range=div)
        psnrs.append(psnr_ti)
    return np.array(psnrs)

def compute_patch_psnrs(patches,ref,div=255.):
    k = patches.shape[0]
    patches = patches.detach().cpu().numpy()
    ref = ref.detach().cpu().numpy()
    psnrs = []
    for ki in range(k):
        psnr_ki = comp_psnr(patches[ki], ref[0], data_range=div)
        psnrs.append(psnr_ki)
    return np.array(psnrs)


