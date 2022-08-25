
# -- python imports --
import sys,os,time,copy
import argparse

# -- io --
from pathlib import Path

# -- linalg --
import torch as th
import numpy as np
from einops import repeat,rearrange

# -- data mgnmnt --
import pandas as pd
from easydict import EasyDict as edict

# -- caching --
import cache_io

# -- data hub --
import data_hub

# -- misc for data --
from noisyknn.utils.misc import set_seed,optional,rslice

# -- optical flow --
from noisyknn.flow import run as run_flow

# -- main function --
from noisyknn.not_only_deno import run as run_nod
from noisyknn.not_only_deno import save_example
from noisyknn.not_only_deno import compare_psnr_vs_pme
from noisyknn.utils import io

# -- vision --
import torch.nn.functional as tnnf

def run_exp(cfg):

    # -- set seed --
    device = "cuda:0"
    set_seed(cfg.seed)

    #
    # -- load data --
    #

    # -- load video --
    clean = load_vid(cfg)/255.

    # -- rescale --
    clean = apply_scale(clean,cfg.scale)

    # -- apply motion --
    clean = apply_motion(clean,cfg.motion,cfg.exp_nframes)

    # -- get noisy --
    noisy = clean + cfg.sigma/255. * th.randn_like(clean)

    # -- compute flows --
    flows = run_flow(clean,cfg.sigma)

    #
    # -- run exp --
    #

    # -- exec --
    print(noisy.shape)
    output = run_nod(noisy,clean,flows,cfg)
    image,psnrs,pme,pme_noisy,pme_clean = output

    #
    # -- wrap it up --
    #

    # -- formating --
    print(type(psnrs))
    print(psnrs)
    results = edict()
    results.image = image.cpu().numpy()
    results.psnrs = psnrs
    results.pme = pme
    results.pme_noisy = pme_noisy
    results.pme_clean = pme_clean

    return results

def load_vid(cfg):
    # -- load set --
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    imax = 255.

    # -- unpack sample --
    sample = data.te[indices[0]]
    region = sample['region']
    clean = sample['clean']

    # -- optional crop --
    clean = rslice(clean,region)

    # -- gpu --
    clean = clean.to(cfg.device)

    return clean

def apply_scale(clean,scale):
    kwargs = {"scale_factor":scale,"mode":"bicubic","align_corners":False}
    clean = tnnf.interpolate(clean,**kwargs)
    return clean

def apply_motion(clean,motion,nframes):
    # post_nframes
    if motion == "none":
        clean = repeat(clean[[0]],'1 c h w -> t c h w',t=nframes)
    else:
        skip = int(motion)
        clean = clean[::skip][:nframes]
    return clean

def default_config():
    cfg = edict()
    cfg.device = "cuda:0"
    cfg.save_dir = "./output/not_only_deno/"
    cfg.nframes = 30 # max
    cfg.isize = "128_128"
    return cfg

def main():

    # -- init --
    pid = os.getpid()
    print("PID: ",pid)
    verbose = True

    # -- (1) Create Cache --
    cache_dir = ".cache_io"
    cache_name = "not_only_deno"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- Data Select --
    dname = ["set8"]
    vid_name = ["motorbike"]

    # -- (2) Create Mesh of Experiments --
    nreps = 3
    seed = list(np.arange(nreps))
    # k = [2,10,50]
    k = [10]
    lams = [0.]
    scale = [0.5]
    sigma = [30.]
    tgt_sigma = [15.]#,1.5]
    # motion = ["none","1","5"]
    # motion = ["none"]
    motion = ["1"]

    # -- long-range motion --
    exp_grid = {"sigma":sigma,"dname":dname,"vid_name":vid_name,
                "seed":seed,"k":k,"lams":lams,
                "tgt_sigma":tgt_sigma,"scale":scale,
                "motion":motion,"exp_nframes":[3]}
    exps = cache_io.mesh_pydicts(exp_grid) # create mesh

    # -- create config --
    cfg = default_config() # parse inputs
    cache_io.append_configs(exps,cfg) # merge the two

    # -- (3) Execute Experiments --
    nexps = len(exps)
    for exp_num,config in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            print(config)

        # -- logic --
        uuid = cache.get_uuid(config) # assing ID to each Dict in Meshgrid
        # cache.clear_exp(uuid) # RESET
        results = cache.load_exp(config) # possibly load result
        if results is None: # check if no result
            results = run_exp(config)
            cache.save_exp(uuid,config,results) # save to cache

    # -- (4) Gather Results --
    records = cache.load_flat_records(exps)
    # records = aggregate_psnrs(records,exp_grid)
    # records = records[records['tgt_sigma'] > 2]

    # -- (5) Report PSNR v.s PME --
    for sname,sdf in records.groupby("motion"):
        # if sname == "v1": sname = "sim"
        # elif sname == "v2": sname = "real"
        # else: raise ValueError(f"Uknown sname [{sname}]")
        print(sname,sdf.columns)
        # io.save_burst(sdf,cfg.save_dir,"deno")
        save_example(sdf,sname)
        compare_psnr_vs_pme(sdf,sname)

if __name__ == "__main__":
    main()
