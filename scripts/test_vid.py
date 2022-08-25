
# -- misc --
import os,math,tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- svnlb --
# import svnlb # old optical flow

# -- caching results --
import cache_io

# -- network --
import noisyknn
import noisyknn.vnlb as vnlb
import noisyknn.utils.gpu_mem as gpu_mem
from noisyknn.flow import run as run_flow
from noisyknn.utils.metrics import compute_psnrs
from noisyknn.utils.metrics import compute_ssims
from noisyknn.utils.misc import set_seed,optional,rslice

def run_exp(cfg):

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- set seed --
    set_seed(cfg.seed)

    # -- init results --
    results = edict()
    results.noisy_psnrs = []
    results.psnrs = []
    results.ssims = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.mem_res = []
    results.mem_alloc = []
    results.timer_flow = []
    results.timer_deno = []

    # -- data --
    # data,loaders = data_hub.sets.load(cfg)
    # groups = data.te.groups
    # indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    # imax = 255.

    # -- optional filter --
    # frame_start = optional(cfg,"frame_start",0)
    # frame_end = optional(cfg,"frame_end",0)
    # if frame_start >= 0 and frame_end > 0:
    #     def fbnds(fnums,lb,ub): return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
    #     indices = [i for i in indices if fbnds(data.te.paths['fnums'][groups[i]],
    #                                            cfg.frame_start,cfg.frame_end)]

    imax = 255.
    path = "../svnlb/data/davis_baseball_64x64/"
    fnums = 3
    vid = data_hub.videos.read_rgb_video(path,fnums,"%05d","jpg")
    indices = [0]
    set_seed(cfg.seed)
    noisy = np.random.normal(vid.copy(),scale=cfg.sigma).astype(np.float32)
    vid = th.from_numpy(vid).contiguous()
    noisy = th.from_numpy(noisy).contiguous()
    # noisy = vid + cfg.sigma * th.randn_like(vid)
    # noisy = noisy.contiguous()
    sample = {"region":None,"noisy":noisy,"clean":vid,"fnums":fnums}
    data = edict()
    data.te = [sample]
    print(vid.shape)


    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()

        # -- unpack --
        sample = data.te[index]
        region = sample['region']
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames = sample['fnums']
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- optional crop --
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- create timer --
        timer = noisyknn.utils.timer.ExpTimer()

        # -- size --
        nframes = noisy.shape[0]
        ngroups = int(25 * 37./nframes)
        batch_size = 32*1024

        # -- optical flow --
        with timer("flow"):
            if cfg.flow == "true":
                # noisy_np = noisy.cpu().numpy()
                # flows = svnlb.compute_flow(noisy_np,cfg.sigma)
                # flows = edict({k:th.from_numpy(v).to(cfg.device)
                #                for k,v in flows.items()})
                flows = run_flow(noisy,cfg.sigma)
            else:
                t,c,h,w = noisy.shape
                zflow = th.zeros((t,2,h,w),device=cfg.device,dtype=th.float32)
                flows = edict()
                flows.fflow = zflow
                flows.bflow = zflow

        # -- pack params --
        batch_size = 10**8#256#85*1024#390*100
        params = edict()
        params.batch_size = batch_size

        # -- denoise --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with timer("deno"):
            with th.no_grad():
                deno,basic = vnlb.run(noisy,cfg.sigma,flows,params)
        mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- save example --
        out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        deno_fns = noisyknn.utils.io.save_burst(deno,out_dir,"deno")

        # -- psnr --
        noisy_psnrs = compute_psnrs(noisy,clean,div=imax)
        psnrs = compute_psnrs(deno,clean,div=imax)
        ssims = compute_ssims(deno,clean,div=imax)
        print("Noisy: ",noisy_psnrs)
        print("Deno: ",psnrs)
        basic_psnrs = compute_psnrs(basic,clean,div=imax)
        print("Basic: ",basic_psnrs)


        # -- append results --
        results.psnrs.append(psnrs)
        results.ssims.append(ssims)
        results.noisy_psnrs.append(noisy_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        results.mem_res.append([mem_res])
        results.mem_alloc.append([mem_alloc])
        for name,time in timer.items():
            results[name].append(time)

    return results


def compute_ssim(clean,deno,div=255.):
    nframes = clean.shape[0]
    ssims = []
    for t in range(nframes):
        clean_t = clean[t].cpu().numpy().squeeze().transpose((1,2,0))/div
        deno_t = deno[t].cpu().numpy().squeeze().transpose((1,2,0))/div
        ssim_t = compute_ssim_ski(clean_t,deno_t,channel_axis=-1)
        ssims.append(ssim_t)
    ssims = np.array(ssims)
    return ssims

def compute_psnr(clean,deno,div=255.):
    t = clean.shape[0]
    deno = deno.detach()
    clean_rs = clean.reshape((t,-1))/div
    deno_rs = deno.reshape((t,-1))/div
    mse = th.mean((clean_rs - deno_rs)**2,1)
    psnrs = -10. * th.log10(mse).detach()
    psnrs = psnrs.cpu().numpy()
    return psnrs

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.saved_dir = "./output/saved_results/"
    # cfg.isize = "none"
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    # cfg.isize = "512_512"
    # cfg.isize = "256_256"
    # cfg.isize = "128_128"
    return cfg

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "test_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- get mesh --
    # dnames = ["toy"]
    # vid_names = ["text_tourbus"]
    # mtypes = ["rand"]
    mtypes = ["rand"]#,"sobel"]
    dnames = ["set8"]
    vid_names = ["sunflower","snowboard","tractor","motorbike",
                 "hypersmooth","park_joy","rafting","touchdown"]
    vid_names = ["sunflower"]
    # dnames = ["davis"]
    # vid_names = ["baseball"]
    # sigmas = [50,30,10]#,30,10]
    sigmas = [50]
    # ws,wt = [29],[0]
    ws,wt = [29],[3]
    flow = ["true"]
    isizes = ["none"]#,"512_512","256_256"]
    # isizes = ["156_156"]#"256_256"]
    model_type = ["batched"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "flow":flow,"ws":ws,"wt":wt,
                 "isize":isizes,"model_type":model_type}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh

    # -- alt search --
    exp_lists['ws'] = [29]
    exp_lists['wt'] = [3]
    exp_lists['flow'] = ["false"]

    # exp_lists['model_type'] = ["refactored"]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    exps = exps_a# + exps_b

    # -- defaults --
    # exp_lists['ws'] = [29]
    # exp_lists['wt'] = [0]
    # exp_lists['flow'] = ["false"]
    # exp_lists['model_type'] = ["refactored"]
    # exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    # exps = exps_b# + exps_a

    # -- group with default --
    cfg = default_cfg()
    cfg.seed = 123
    cfg.nframes = 3
    cfg.isize = "128_128"
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cfg.frame_end = 0 if cfg.frame_end < 0 else cfg.frame_end
    cache_io.append_configs(exps,cfg) # merge the two
    # pp.pprint(exps[0])
    # exit(0)


    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        results = cache.load_exp(exp) # possibly load result
        # if not(results is None) and len(results['psnrs']) == 0:
        #     results = None
        #     cache.clear_exp(uuid)
        # if exp.flow == "true" and results:
        #     cache.clear_exp(uuid)
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(records)
    print(records.filter(like="timer"))

    # -- print by dname,sigma --
    for dname,ddf in records.groupby("dname"):
        for cflow,fdf in ddf.groupby("flow"):
            for ws,wsdf in fdf.groupby("ws"):
                for wt,wtdf in wsdf.groupby("wt"):
                    for sigma,sdf in wtdf.groupby("sigma"):
                        ave_psnr,ave_ssim,ave_time,num_vids = 0,0,0,0
                        for vname,vdf in sdf.groupby("vid_name"):
                            psnr_v = vdf.psnrs[0].mean()
                            ssim_v = vdf.ssims[0].mean()
                            ave_psnr += psnr_v
                            ave_ssim += ssim_v
                            ave_time += vdf['timer_deno'].iloc[0]/len(vdf)
                            num_vids += 1
                            uuid = vdf['uuid'].iloc[0]
                            msg = "\t[%s]: %2.2f %s" % (vname,psnr_v,uuid)
                            print(msg)

                        ave_psnr /= num_vids
                        ave_ssim /= num_vids
                        ave_time /= num_vids
                        total_frames = len(sdf)
                        mem = np.stack(vdf['mem_res'].to_numpy()).mean()
                        fields = (sigma,ave_psnr,ave_ssim,ave_time,total_frames,mem)
                        print("[%d]: %2.3f,%2.3f @ ave %2.2f sec for %d seq at %2.2f" % fields)


if __name__ == "__main__":
    main()
