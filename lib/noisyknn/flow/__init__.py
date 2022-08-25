"""
Wrap the opencv optical flow

"""

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- misc --
from easydict import EasyDict as edict

# -- opencv --
import cv2 as cv

# -- local --
from noisyknn.utils import color

def run(vid_in,sigma):

    # -- init --
    device = vid_in.device
    vid = vid_in.clone() # copy data for no-rounding-error from RGB <-> YUV
    t,c,h,w = vid.shape

    # -- color2gray --
    vid = th.clamp(vid,0,255.).type(th.uint8)
    color.rgb2yuv(vid)
    vid = vid[:,[0],:,:]
    vid = rearrange(vid,'t c h w -> t h w c')

    # -- alloc --
    fflow = th.zeros((t,2,h,w),device=device)
    bflow = th.zeros((t,2,h,w),device=device)

    # -- computing --
    for ti in range(t-1):
        fflow[ti] = pair2flow(vid[ti],vid[ti+1],device)
    for ti in reversed(range(t-1)):
        bflow[ti] = pair2flow(vid[ti+1],vid[ti],device)

    # -- final shaping --
    # fflow = rearrange(fflow,'t h w c -> t c h w')
    # bflow = rearrange(bflow,'t h w c -> t c h w')

    # -- packing --
    flows = edict()
    flows.fflow = fflow
    flows.bflow = bflow

    # -- gray2color --
    color.yuv2rgb(vid)

    return flows


def pair2flow(frame_a,frame_b,device):

    # -- create opencv-gpu frames --
    gpu_frame_a = cv.cuda_GpuMat()
    gpu_frame_b = cv.cuda_GpuMat()
    gpu_frame_a.upload(frame_a.cpu().numpy())
    gpu_frame_b.upload(frame_b.cpu().numpy())

    # -- create flow object --
    gpu_flow = cv.cuda_FarnebackOpticalFlow.create(5, 0.5, False,
                                                   15, 3, 5, 1.2, 0)

    # -- exec flow --
    flow = cv.cuda_FarnebackOpticalFlow.calc(gpu_flow, gpu_frame_a,
                                             gpu_frame_b, None)
    flow = flow.download()
    flow = flow.transpose(2,0,1)
    flow = th.from_numpy(flow).to(device)

    return flow
