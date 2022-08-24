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

def run(vid_in):

    # -- init --
    vid = vid_in.clone() # copy data for no-rounding-error from RGB <-> YUV
    t,c,h,w = vid.shape
    vid = rearrange(vid,'t c h w -> t h w c')

    # -- color2gray --
    rgb2yuv(vid)

    # -- alloc --
    fflow = np.zeros((t,h,w,2))
    bflow = np.zeros((t,h,w,2))

    # -- computing --
    for ti in range(t-1):
        fflow[ti] = pair2flow(vid[ti],vid[ti+1])
    for ti in reversed(range(t-1)):
        bflow[ti] = pair2flow(vid[ti+1],vid[ti])

    # -- final shaping --
    fflow = rearrange(fflow,'t h w c -> t c h w')
    bflow = rearrange(bflow,'t h w c -> t c h w')

    # -- packing --
    flows = edict()
    flows.fflow = fflow
    flows.bflow = bflow

    # -- gray2color --
    yuv2rgb(vid)

    return flows


def pair2flow(vid_c,vid_n):
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
