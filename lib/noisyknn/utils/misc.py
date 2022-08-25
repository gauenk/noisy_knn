import random
import numpy as np
import torch as th
from easydict import EasyDict as edict

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)

def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def rslice(vid,coords):
    if coords is None: return vid
    if len(coords) == 0: return vid
    if th.is_tensor(coords):
        coords = coords.type(th.int)
        coords = list(coords.cpu().numpy())
    fs,fe,t,l,b,r = coords
    return vid[fs:fe,:,t:b,l:r].contiguous()

def slice_flows(flows,t_start,t_end):
    if flows is None: return flows
    flows_t = edict()
    flows_t.fflow = flows.fflow[t_start:t_end].contiguous()
    flows_t.bflow = flows.bflow[t_start:t_end].contiguous()
    return flows_t


