import os
import torch
from glob import glob
from torch.autograd import Function
from torch.utils.cpp_extension import load

# from .ext import raytrace as raytrace_impl

EXT  = os.path.join(os.path.dirname(__file__), "ext")
NAME = "raytrace"

src_list = []
src_list += glob(os.path.join(EXT, NAME, "*.c"))
src_list += glob(os.path.join(EXT, NAME, "*.cpp"))
src_list += glob(os.path.join(EXT, NAME, "*.cu"))

raytrace_impl = load(name=NAME, sources=src_list, verbose=True)

class PrimaryRayFunction(Function):
    @staticmethod
    def forward(ctx, faces, query_ray, culling=False, negdist=False):

        query = query_ray
        i_map, d_map, p_map = raytrace_impl.primary_ray_forward(
            faces, query, culling, negdist)
        return i_map, d_map, p_map

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError