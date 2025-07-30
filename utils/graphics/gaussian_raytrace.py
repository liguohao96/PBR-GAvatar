import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from torch.autograd import Function
from torch.utils.cpp_extension import load

EXT  = os.path.join(os.path.dirname(__file__), "ext")
NAME = "diff_gaussian_raytrace"

include_paths = [os.path.join(EXT, NAME, "include")]

# COPY from https://github.com/NVlabs/nvdiffrecmc/blob/main/render/optixutils/ops.py
# Compiler options.
opts = [
    "-DNVDR_TORCH"]
    # "-DCUDA_NVRTC_OPTIONS='-std=c++11,-arch compute_70'"]

# // #define CUDA_NVRTC_OPTIONS  \
# //   "-std=c++11", \
# //   "-arch", \
# //   "compute_70", \
# //   "-use_fast_math", \
# //   "-lineinfo", \
# //   "-default-device", \
# //   "-rdc", \
# //   "true", \
# //   "-D__x86_64", \
# //   "-D__OPTIX__"

# Linker options.
if os.name == 'posix':
    ldflags = ['-lcuda', '-lnvrtc']
elif os.name == 'nt':
    ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

src_list = []
src_list += glob(os.path.join(EXT, NAME, "*.c"))
src_list += glob(os.path.join(EXT, NAME, "*.cpp"))
src_list += glob(os.path.join(EXT, NAME, "*.cu"))

diff_grt_impl = load(name=NAME, sources=src_list, 
                     extra_cflags=opts, extra_cuda_cflags=opts, 
                     extra_include_paths=include_paths,
                     extra_ldflags=ldflags,
                     with_cuda=True,
                     verbose=True)


# def diff_gaussian_raytrace(rgs_xyz, rgs_rot, rgs_sca, rgs_opa, 
#                            rgs_rgb, rgs_nrm, 
#                            bg_raw, 
#                            ray_dir):
    
#     BS, N = ray_dir.shape[:2]
    
#     geometry = build_proxy_geometry(rgs_xyz, rgs_rot, rgs_sca, rgs_opa)  # B, N

#     GaussianRayTraceFunction.apply(geometry, rgs_xyz, rgs_rot, rgs_sca, rgs_opa, rgs_rgb, rgs_nrm, ray_dir)

#     ambient    = torch.ones(BS, N, 3, device=device)
#     diffuse    = torch.ones(BS, N, 3, device=device)
#     specular   = torch.ones(BS, N, 3, device=device)
#     ao         = torch.ones(BS, N, 3, device=device)
#     background = torch.ones(BS, N, 3, device=device)

#     return (ambient), (diffuse), (specular), ao, (background)

# MODIFY from https://github.com/NVlabs/nvdiffrecmc/blob/main/denoiser/denoiser.py#L17
class BilateralDenoiser(nn.Module):
    def __init__(self, influence=1.0):
        super().__init__()
        self.set_influence(influence)

    def set_influence(self, factor):
        import math
        self.sigma = max(factor * 2, 0.0001)
        self.variance = self.sigma**2.
        self.N = 2 * math.ceil(self.sigma * 2.5) + 1

    def forward(self, col, nrm, zdz):
        # col    = input[..., 0:3]
        # nrm    = F.normalize(input[..., 3:6], dim=-1) # Bent normals can produce normals of length < 1 here
        # zdz    = input[..., 6:8]
    	# return ou.bilateral_denoiser(col, nrm, zdz, self.sigma)

        # https://github.com/NVlabs/nvdiffrecmc/blob/286bb47dd766a6e0504e83c7afe657c43702a078/render/optixutils/ops.py#L139

        col_w = BilateralDenoiserFunction.apply(col, nrm, zdz, self.sigma)
        return col_w[..., 0:3] / col_w[..., 3:4]

class GaussianRayTracer(nn.Module):
    def __init__(self):
        super().__init__()

        self.optix_ctx_list = []

        self.shd_optix_ctx = create_optix_context()

        self.n_samples_x = 4
        self.rnd_seed    = 0

        self.shading = False

        self.train_steps = 0

        self.position_source = "depth"
        self.normal_source   = "render"

        # self.denoiser = None
        self.denoiser = BilateralDenoiser()

        self.denoiser_demodulate = True

    def depth_to_position(self, depth, ray_org, ray_dir):
        pos = ray_org.expand_as(ray_dir) + ray_dir * depth
        return pos

    def depth_to_normal(self, depth, ray_org, ray_dir):
        pos = ray_org.expand_as(ray_dir) + ray_dir * depth

        u2d = pos[:, 1:-1, 2:] - pos[:, 1:-1, :-2]
        v2d = pos[:, 2:, 1:-1] - pos[:, :-2, 1:-1]
        dep_nrm = F.normalize(torch.cross(v2d, u2d, dim=-1), dim=-1)
        dep_nrm = F.pad(dep_nrm, (0, 0, 1, 1, 1, 1), "constant", 0)

        return dep_nrm
    
    def forward(self, 
        rgs_xyz, rgs_rot, rgs_sca, rgs_opa, 
        rgs_rgb, rgs_nrm, 
        bg_raw, 
        ray_org, ray_dir, T_min=0.001, alpha_min=0.01, iclight_mask=None):

        BS,H,W = ray_dir.shape[:3]            
        device = ray_dir.device

        if self.training:
            self.train_steps += 1

        if len(self.optix_ctx_list) < BS:
            self.optix_ctx_list += [ create_optix_context() for _ in range(BS-len(self.optix_ctx_list)) ]
        
        optix_ctx_list = self.optix_ctx_list

        bg_raw = bg_raw.expand(BS, -1, -1, -1)

        aux = {}
        image_list, alpha_list, depth_list, normal_list = [], [], [], []
        depth2_list, depthd_list = [], []
        diffuse_list, specular_list = [], []
        for bi in range(BS):
            xyz, rot, sca, opa = rgs_xyz[bi], rgs_rot[bi], rgs_sca[bi], rgs_opa[bi]

            rgb, gs_nrm = rgs_rgb[bi], rgs_nrm[bi]
            ray_org_ = ray_org[bi:bi+1]
            ray_dir_ = ray_dir[bi:bi+1]

            proxy_ver, proxy_tri = build_proxy_geometry(optix_ctx_list[bi], xyz, rot, sca, opa) # Np,Nv,3 Np,Nf,3

            # https://forums.developer.nvidia.com/t/opacity-accumulation-along-the-ray/303268
            # https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src
            # https://github.com/nvpro-samples/optix_advanced_samples/tree/master/src/optixIntroduction

            img, buf, dbg = GaussianRayTraceFunction.apply(
                optix_ctx_list[bi], 
                xyz, rot, sca, opa, 
                rgb, gs_nrm, 
                ray_org_, ray_dir_,
                proxy_ver, proxy_tri,
                T_min, alpha_min)
                # img.mean().backward()
            
            alp, nrm, dep, pos, T, dep2, ldd = torch.split(buf, (1, 3, 1, 3, 1, 1, 1), dim=-1)
            
            nrm = F.normalize(nrm, dim=-1)

            if self.shading:

                if self.position_source == "render":
                    gb_pos = pos
                elif self.position_source == "depth":
                    gb_pos = self.depth_to_position(dep, ray_org_.expand_as(ray_dir_), ray_dir_)

                if self.normal_source == "render":
                    gb_nrm = nrm
                elif self.normal_source == "depth":
                    gb_nrm = self.depth_to_normal(dep, ray_org_.expand_as(ray_dir_), ray_dir_)

                shd_ro = gb_pos + gb_nrm*0.005  # 5 mm
                cam_ro = ray_org[bi:bi+1]

                albedo, roughness, metallic = torch.split(img, [3, 1, 1], dim=-1)
                # mat_kd = img[None,..., :3]
                # mat_ks = F.pad(img[None,..., 3:5], (1, 0), "constant", 1.0)   # [1, roughness, metallic]
                mat_kd = albedo
                # mat_ks = torch.cat([torch.ones_like(roughness), roughness, metallic], dim=-1)
                mat_ks = torch.cat([torch.zeros_like(roughness), roughness, metallic], dim=-1)
                # mat_ks = torch.cat([roughness*roughness, roughness, metallic], dim=-1)  # alpha, roughness, metallic

                light = bg_raw[bi]
                # pdf   = light[..., 0]
                # cols = torch.cumsum(pdf, dim=1)
                # rows = torch.cumsum(cols[:, -1:].repeat([1, cols.shape[1]]), dim=0)
                with torch.no_grad():
                    # Compute PDF
                    # Y = util.pixel_grid(self.base.shape[1], self.base.shape[0])[..., 1]
                    height, width = light.shape[:2]
                    center_x = center_y = 0.5
                    y, x = torch.meshgrid(
                            (torch.arange(0, height, dtype=torch.float32, device=light.device) + center_y) / height, 
                            (torch.arange(0, width, dtype=torch.float32, device=light.device) + center_x) / width)
                    Y = y

                    pdf = torch.max(light, dim=-1)[0] * torch.sin(Y * np.pi) # Scale by sin(theta) for lat-long, https://cs184.eecs.berkeley.edu/sp18/article/25
                    pdf = pdf / torch.sum(pdf)

                    # Compute cumulative sums over the columns and rows
                    cols = torch.cumsum(pdf, dim=1)
                    rows = torch.cumsum(cols[:, -1:].repeat([1, cols.shape[1]]), dim=0)

                    # Normalize
                    cols = cols / torch.where(cols[:, -1:] > 0, cols[:, -1:], torch.ones_like(cols))
                    rows = rows / torch.where(rows[-1:, :] > 0, rows[-1:, :], torch.ones_like(rows))

                bsdf = 'pbr'
                # bsdf = 'white'

                iBSDF         = ['pbr', 'diffuse', 'white'].index(bsdf)
                n_samples_x   = self.n_samples_x
                rnd_seed      = self.rnd_seed
                # shadow_scale  = 0.5 # shadow_ramp = min(iteration / 1750, 1.0)
                shadow_scale  = min(self.train_steps / 1000, 1.0)
                # self.rnd_seed += 1

                # print(alp.shape, shd_ro.shape, gb_pos.shape, gb_nrm.shape, cam_ro.shape, mat_kd.shape, mat_ks.shape)

                # def forward(ctx, optix_ctx, mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, light, pdf, rows, cols, BSDF, n_samples_x, rnd_seed, shadow_scale):
                diffuse, specular = EnvShadingFunction.apply(optix_ctx_list[bi], 
                    xyz, rot, sca, opa,
                    proxy_ver, proxy_tri,
                    T_min, alpha_min,

                    alp[...,0], shd_ro.detach(), gb_pos.detach(), gb_nrm, cam_ro, mat_kd, mat_ks, 
                    light, pdf, rows, cols, 
                    iBSDF, n_samples_x, rnd_seed, shadow_scale)
                self.rnd_seed += 1
                
                # https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf
                # https://github.com/jacquespillet/SVGF
                # TODO: how to obtain dz?
                # use this to get d_dep https://github.com/jacquespillet/SVGF/blob/c0506894c6fe67ddf33db1bcdb6bec6c1786b913/resources/shaders/GBuffer.frag#L71

                with torch.no_grad():
                    dfdx  = F.pad(torch.diff(dep, dim=2), (0, 0, 1, 0),       "constant", 0.0)
                    dfdy  = F.pad(torch.diff(dep, dim=1), (0, 0, 0, 0, 1, 0), "constant", 0.0)
                    # print("dfdx dfdy dep", dfdx.shape, dfdy.shape, dep.shape)
                    d_dep = torch.maximum( dfdx.abs(), dfdy.abs() )
                
                    zdz   = torch.cat((dep, d_dep), dim=-1)

                if self.denoiser is not None and self.denoiser_demodulate is True:
                    diffuse  = self.denoiser.forward(diffuse,  nrm, zdz)
                    specular = self.denoiser.forward(specular, nrm, zdz)
                
                if bsdf == 'white' or bsdf == 'diffuse':
                    kd  = 1
                    img = diffuse * mat_kd
                else:
                    kd = mat_kd * (1.0 - mat_ks[..., 2:3]) # kd * (1.0 - metalness)
                    img = diffuse * kd + specular

                if self.denoiser is not None and self.denoiser_demodulate is False:
                    # img = self.denoiser.forward(torch.cat((img, nrm, dep, d_dep), dim=-1))
                    img = self.denoiser.forward(img, nrm, zdz)

                # print("diff/spec/img", diffuse.max(), specular.max(), img.mean(), img.shape)

                dep = dep/alp
                dep = torch.nan_to_num(dep, 0.0, neginf=0.0, posinf=0.0)

                image_list.append( img)
                alpha_list.append( alp)
                normal_list.append(nrm)
                depth_list.append( torch.cat([dep, pos], dim=-1))

                depth2_list.append(dep2)
                depthd_list.append(ldd)

                diffuse_list.append( (kd*diffuse))
                specular_list.append(specular)
            else:
                dep = dep/alp
                dep = torch.nan_to_num(dep, 0.0, neginf=0.0, posinf=0.0)

                image_list.append( img)
                alpha_list.append( alp)
                normal_list.append(nrm)
                depth_list.append( torch.cat([dep, pos], dim=-1))

                depth2_list.append(dep2)
                depthd_list.append(ldd)

        image   = torch.cat(image_list,  dim=0)
        alpha   = torch.cat(alpha_list,  dim=0)
        depth   = torch.cat(depth_list,  dim=0)
        normal  = torch.cat(normal_list, dim=0)

        depth2  = torch.cat(depth2_list, dim=0)
        depthd  = torch.cat(depthd_list, dim=0)

        if self.shading:
            aux["diffuse"]  = torch.cat(diffuse_list, dim=0)
            aux["specular"] = torch.cat(specular_list, dim=0)
        
        aux["depth_square"]     = depth2
        aux["depth_distortion"] = depthd

        return image, alpha, depth, normal, aux

def quater2rotation(r):

    q = F.normalize(r, dim=-1)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    R = torch.stack([
                 1 - 2 * (y*y + z*z),
                 2 * (x*y - w*z),
                 2 * (x*z + w*y),
                 2 * (x*y + w*z),
                 1 - 2 * (x*x + z*z),
                 2 * (y*z - w*x),
                 2 * (x*z - w*y),
                 2 * (y*z + w*x),
                 1 - 2 * (x*x + y*y),
                ], dim=-1).unflatten(-1, (3, 3))
    return R

def create_optix_context():
    return diff_grt_impl.OptiXStateWrapper(os.path.join(os.path.dirname(__file__), "ext", "diff_gaussian_raytrace"), torch.utils.cpp_extension.CUDA_HOME)

# v -1  0  0
# v  0  1  0
# v  1  0  0
# v  0 -1  0
# v  0  0  1
# v  0  0 -1
# f  1  5  2
# f  2  5  3
# f  1  2  6
# f  6  2  3
# f  1  4  5
# f  3  5  4
# f  1  6  4
# f  3  4  6
PROXY_MESH_VER = np.array([
            [-1, 0, 0],
            [ 0, 1, 0],
            [ 1, 0, 0],
            [ 0,-1, 0],
            [ 0, 0, 1],
            [ 0, 0,-1],
], dtype=np.float32)
PROXY_MESH_TRI = np.array([
    [1, 5, 2],
    [2, 5, 3],
    [1, 2, 6],
    [6, 2, 3],
    [1, 4, 5],
    [3, 5, 4],
    [1, 6, 4],
    [3, 4, 6],
], dtype=np.int32) - 1

# FROM https://emmaprats.github.io/Icosahedra/index.html
# X = 0.1921241333535732
# Z = 0.4069254445019854
# PROXY_MESH_VER = np.array([
#     [-X,  Z,  0],
#     [ X,  Z,  0],
#     [-X, -Z,  0],
#     [ X, -Z,  0],
#     [ 0, -X,  Z],
#     [ 0,  X,  Z],
#     [ 0, -X, -Z],
#     [ 0,  X, -Z],
#     [ Z,  0, -X],
#     [ Z,  0,  X],
#     [-Z,  0, -X],
#     [-Z,  0,  X],
# ], dtype=np.float32)
# PROXY_MESH_TRI = np.array([
#     [ 1, 12,  6],
#     [ 1,  6,  2],
#     [ 1,  2,  8],
#     [ 1,  8, 11],
#     [ 1, 11, 12],
#     [ 2,  6, 10],
#     [ 6, 12,  5],
#     [12, 11,  3],
#     [11,  8,  7],
#     [ 8,  2,  9],
#     [ 4, 10,  5],
#     [ 4,  5,  3],
#     [ 4,  3,  7],
#     [ 4,  7,  9],
#     [ 4,  9, 10],
#     [ 5, 10,  6],
#     [ 3,  5, 12],
#     [ 7,  3, 11],
#     [ 9,  7,  8],
#     [10,  9,  2],
# ], dtype=np.int32) - 1
# FROM https://schneide.blog/2016/07/15/generating-an-icosphere-in-c/
X = 0.525731112119133606
Z = 0.850650808352039932
N = 0
PROXY_MESH_VER = np.array([
    [-X, N, Z], [ X, N, Z], [-X, N,-Z], [ X, N,-Z],
    [ N, Z, X], [ N, Z,-X], [ N,-Z, X], [ N,-Z,-X],
    [ Z, X, N], [-Z, X, N], [ Z,-X, N], [-Z,-X, N]
], dtype=np.float32)
PROXY_MESH_TRI = np.array([
    [ 0, 4, 1], [ 0, 9, 4], [ 9, 5, 4], [ 4, 5, 8], [ 4, 8, 1],
    [ 8,10, 1], [ 8, 3,10], [ 5, 3, 8], [ 5, 2, 3], [ 2, 7, 3],
    [ 7,10, 3], [ 7, 6,10], [ 7,11, 6], [11, 0, 6], [ 0, 1, 6],
    [ 6, 1,10], [ 9, 0,11], [ 9,11, 2], [ 9, 2, 5], [ 7, 2,11]
], dtype=np.int32)

def build_proxy_geometry(optix_ctx, rgs_xyz, rgs_rot, rgs_sca, rgs_opa):
    eps = 1e-6

    assert rgs_xyz.ndim == 2, f"{rgs_xyz.shape}"

    device = rgs_xyz.device
    N_GS   = rgs_xyz.size(0)

    # vers = torch.as_tensor([
    #         [-1, 0, 0],
    #         [ 0, 1, 0],
    #         [ 1, 0, 0],
    #         [ 0,-1, 0],
    #         [ 0, 0, 1],
    #         [ 0, 0,-1],
    #     ], dtype=torch.float32, device=device)  # Nv, 3

    # tris = torch.as_tensor([
    #             [0, 4, 1],
    #             [1, 4, 2],
    #             [0, 1, 5],
    #             [5, 1, 2],
    #             [0, 3, 4],
    #             [2, 4, 3],
    #             [0, 5, 3],
    #             [2, 3, 5],
    #     ], dtype=torch.int32, device=device)  # Nf, 3
    
    vers = torch.as_tensor(PROXY_MESH_VER, device=device)
    tris = torch.as_tensor(PROXY_MESH_TRI, device=device)
    # tris = tris[:, [0, 2, 1]]

    N_VER, N_TRI = vers.size(0), tris.size(0)

    rot_mat = rgs_rot if rgs_rot.ndim == 3 else quater2rotation(rgs_rot)

    # vers = vers.unsqueeze(0) * rgs_sca.unsqueeze(1) + rgs_xyz.unsqueeze(1)

    # scale = rgs_sca * (2*torch.log(rgs_opa.clamp_min(0.01+eps)/0.01)).sqrt() # alpha_min = 0.01
    scale = rgs_sca * (2*torch.log(rgs_opa/0.01).clamp_min(eps)).sqrt() # alpha_min = 0.01

    # vers = torch.einsum("vc,pcj->pvj", vers, rot_mat*rgs_sca.unsqueeze(-2)) + rgs_xyz.unsqueeze(1)
    vers = torch.einsum("pvc,pjc->pvj", vers.unsqueeze(0)*scale.unsqueeze(1), rot_mat) + rgs_xyz.unsqueeze(1)

    # def check_tensor(t):
    #     assert torch.isfinite(t).all(), f"Inf/NaN:{torch.isinf(t).sum().item()}/{torch.isnan(t).sum().item()}"
    # check_tensor(rgs_sca)
    # check_tensor(rgs_opa)
    # check_tensor(torch.log(rgs_opa.clamp_min(0.01)/0.01))
    # check_tensor(vers)

    tris = tris.unsqueeze(0).expand(rgs_sca.size(0), -1, -1) + torch.arange(N_GS, dtype=torch.int32, device=device).reshape(-1, 1, 1)*N_VER

    diff_grt_impl.optix_build_bvh(optix_ctx, vers.flatten(0, 1).contiguous(), tris.flatten(0, 1).contiguous(), 1)

    return vers, tris

class GaussianRayTraceFunction(Function):
    @staticmethod
    def forward(ctx, optix_ctx, g_xyz, g_rot, g_sca, g_opa, g_rgb, g_nrm, ray_org, ray_dir, proxy_ver, proxy_tri, T_min, alpha_min):

        img, buf, dbg = diff_grt_impl.gaussian_raytrace_fwd(optix_ctx, 
            g_xyz, g_rot, g_sca, g_opa, g_rgb, g_nrm, 
            ray_org, ray_dir, proxy_ver, proxy_tri, 
            T_min, alpha_min)
        
        ctx.save_for_backward(
            g_xyz, g_rot, g_sca, g_opa, g_rgb, g_nrm,
            ray_org, ray_dir, proxy_ver, proxy_tri,
            img, buf, dbg)

        ctx.optix_ctx = optix_ctx
        ctx.T_min     = T_min
        ctx.alpha_min = alpha_min
        
        return img, buf, dbg

    @staticmethod
    def backward(ctx, d_img, d_buf, d_dbg):
        g_xyz, g_rot, g_sca, g_opa, g_rgb, g_nrm, ray_org, ray_dir, proxy_ver, proxy_tri, img, buf, dbg = ctx.saved_tensors

        # print("backward", d_img.shape, d_buf.shape)

        optix_ctx = ctx.optix_ctx
        T_min     = ctx.T_min
        alpha_min = ctx.alpha_min

        d_xyz, d_rot, d_sca, d_opa, d_rgb, d_nrm, d_ro, d_rg = diff_grt_impl.gaussian_raytrace_bwd(optix_ctx, 
            g_xyz, g_rot, g_sca, g_opa, g_rgb, g_nrm, 
            ray_org, ray_dir, proxy_ver, proxy_tri,
            T_min, alpha_min,
            img, buf, dbg,
            d_img, d_buf
            )
        
        return None, d_xyz, d_rot, d_sca, d_opa, d_rgb, d_nrm, d_ro, d_rg, None, None, None, None

# MODIFIED from https://github.com/NVlabs/nvdiffrecmc/blob/286bb47dd766a6e0504e83c7afe657c43702a078/render/optixutils/ops.py#L78
class EnvShadingFunction(Function):
    _random_perm = {}

    @staticmethod
    def forward(ctx, optix_ctx, 
        g_xyz, g_rot, g_sca, g_opa, 
        proxy_ver, proxy_tri,
        T_min, alpha_min,
        mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, light, pdf, rows, cols, 
        BSDF, n_samples_x, rnd_seed, shadow_scale):
        _rnd_seed = np.random.randint(2**31) if rnd_seed is None else rnd_seed
        if n_samples_x not in EnvShadingFunction._random_perm:
            # Generate (32k) tables with random permutations to decorrelate the BSDF and light stratified samples
            EnvShadingFunction._random_perm[n_samples_x] = torch.argsort(torch.rand(32768, n_samples_x * n_samples_x, device="cuda"), dim=-1).int()

        diff, spec = diff_grt_impl.env_shade_fwd(optix_ctx, 
                    g_xyz, g_rot, g_sca, g_opa, 
                    proxy_ver, proxy_tri,
                    T_min, alpha_min,
                    mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, 
                    light, pdf, rows, cols, EnvShadingFunction._random_perm[n_samples_x], 
                    BSDF, n_samples_x, _rnd_seed, shadow_scale)
        ctx.save_for_backward(
                    g_xyz, g_rot, g_sca, g_opa, 
                    proxy_ver, proxy_tri,
                    mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, 
                    light, pdf, rows, cols)
        ctx.optix_ctx = optix_ctx

        ctx.T_min     = T_min
        ctx.alpha_min = alpha_min

        ctx.BSDF         = BSDF
        ctx.n_samples_x  = n_samples_x
        ctx.rnd_seed     = rnd_seed
        ctx.shadow_scale = shadow_scale
        return diff, spec
    
    @staticmethod
    def backward(ctx, diff_grad, spec_grad):
        optix_ctx = ctx.optix_ctx

        _rnd_seed = np.random.randint(2**31) if ctx.rnd_seed is None else ctx.rnd_seed

        g_xyz, g_rot, g_sca, g_opa, \
        proxy_ver, proxy_tri, \
        mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, \
        light, pdf, rows, cols = ctx.saved_tensors

        gb_pos_grad, gb_normal_grad, gb_kd_grad, gb_ks_grad, light_grad = diff_grt_impl.env_shade_bwd(optix_ctx, 
                    g_xyz, g_rot, g_sca, g_opa, 
                    proxy_ver, proxy_tri,
                    ctx.T_min, ctx.alpha_min,
                    mask, ro, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, 
                    light, pdf, rows, cols, EnvShadingFunction._random_perm[ctx.n_samples_x], 
                    ctx.BSDF, ctx.n_samples_x, _rnd_seed, ctx.shadow_scale, 
                    diff_grad, spec_grad)

        return None, \
               None, None, None, None, \
               None, None, \
               None, None, \
               None, None, gb_pos_grad, gb_normal_grad, None, gb_kd_grad, gb_ks_grad, light_grad, None, None, None, \
               None, None, None, None

class BilateralDenoiserFunction(Function):
    @staticmethod
    def forward(ctx, col, nrm, zdz, sigma):
        ctx.save_for_backward(col, nrm, zdz)
        ctx.sigma = sigma
        out = diff_grt_impl.bilateral_denoiser_fwd(col, nrm, zdz, sigma)
        return out
    
    @staticmethod
    def backward(ctx, out_grad):
        col, nrm, zdz = ctx.saved_tensors
        col_grad = diff_grt_impl.bilateral_denoiser_bwd(col, nrm, zdz, ctx.sigma, out_grad)
        return col_grad, None, None, None