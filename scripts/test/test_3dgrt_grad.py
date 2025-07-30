'''
test 3D GS blendshape with Global Illumination
'''
import os
import sys
import platform
import subprocess

host = platform.node()
plat = f"{platform.python_version()}"
p    = subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__)"], 
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
tver = p.stdout.decode().rstrip()

assert 4 < len(tver) < 20, f"got torch.__version__ = '{tver}', by {sys.executable}, {p.stderr.decode()}"

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(os.environ["HOME"], f"TORCH_EXT_pbropen_{host}_{plat}_{tver}")
import torch
import torch.nn.functional as F

import numpy as np

ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

sys.path.insert(0, ROOT)
from utils.graphics           import gaussian_raytrace
from utils.graphics.raytrace  import PrimaryRayFunction
sys.path.pop(0)

def quater2rotation(r):

    q = F.normalize(r, dim=-1)
    q = r
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

# float4 gaussian_particle_fwd(const float3& mu, const float3& scale, const Rot3DMat& rot_3d, const float opacity, const float3& ray_org, const float3& ray_dir){
def gaussian_fwd(mu, scale, q_vec, opacity, ro, rd, n=4):

    mu = mu.clone()
    mu.retain_grad()

    scale = scale.clone()
    scale.retain_grad()

    q_vec = q_vec.clone()
    q_vec.retain_grad()

    S_inv    = torch.diag(1/(scale+1e-6))
    R        = quater2rotation(q_vec.unsqueeze(0)).squeeze(0)  # 3,3
    S_inv_Rt = (S_inv @ R.transpose(-2, -1)).type(mu.dtype)
    S_inv.retain_grad()
    R.retain_grad()
    S_inv_Rt.retain_grad()

    o_g = (S_inv_Rt@(ro - mu))
    d_g = (S_inv_Rt@(rd))
    o_g.retain_grad()
    d_g.retain_grad()

    d_g_0 = d_g.clone()
    d_g_1 = d_g.clone()
    d_g_0.retain_grad()
    d_g_1.retain_grad()

    t = - (o_g*d_g_0).sum(dim=-1, keepdim=True)/(d_g_1*d_g_1).sum(dim=-1, keepdim=True)
    t = t.clone()
    t.retain_grad()

    x_g = o_g + t*d_g
    x_g.retain_grad()

    xtx = (x_g*x_g).sum(dim=-1)
    # rho = torch.exp(-0.5*xtx.pow(n))
    rho = torch.exp(-(3**(2-n))*xtx.pow(n/2))   # equation 12 in 3DGUT

    alpha = opacity*rho

    value_dict = {
        "mu":       mu,
        "scale":    scale,
        "q_vec":    q_vec,
        "S_inv":    S_inv,
        "R":        R,
        "S_inv_Rt": S_inv_Rt,
        "o_g":      o_g,
        "d_g":      d_g,
        "d_g_0":    d_g_0,
        "d_g_1":    d_g_1,
        "t":        t,
        "x_g":      x_g,
        "xtx":      xtx,
        "rho":      rho,
        "alpha":    alpha,
    }

    return t, alpha, x_g, rho, value_dict

@torch.no_grad()
def gaussian_bwd(mu, scale, q_vec, opacity, ro, rd, out, out_grad, n=4, value_dict=None):
    t, alpha = out[:2]
    x_g, rho = out[2:4]

    named_interm = out[4]

    grad_t, grad_alpha = out_grad

    inv_s = 1/scale

    S_inv = torch.diag(1/scale)
    R     = quater2rotation(q_vec.unsqueeze(0)).squeeze(0)  # 3,3

    col0 = R[:, 0]
    col1 = R[:, 1]
    col2 = R[:, 2]

    row0 = R[0, :]
    row1 = R[1, :]
    row2 = R[2, :]

    # s_col0 = R[:, 0] / scale[0]
    # s_col1 = R[:, 1] / scale[1]
    # s_col2 = R[:, 2] / scale[2]

    S_inv_Rt = S_inv @ R.transpose(-2, -1)

    # o_g = (S_inv_Rt@(ro - mu))
    # d_g = (S_inv_Rt@(rd))

    o_g = torch.stack([col0@(ro-mu), col1@(ro-mu), col2@(ro-mu)]) / scale
    d_g = torch.stack([col0@(rd), col1@(rd), col2@(rd)]) / scale

    t = - (o_g*d_g).sum(dim=-1, keepdim=True)/(d_g*d_g).sum(dim=-1, keepdim=True)

    x_g = o_g + t*d_g
    xtx = (x_g*x_g).sum(dim=-1)
    # NOT USED rho = torch.exp(-0.5*xtx)
    # rho = torch.exp(-(3**(2-n))*xtx.pow(n/2))   # equation 12 in 3DGUT

    # x_g_grad = (-grad_out*rho).unsqueeze(-1)*x_g
    # mu_grad_0 = (R@(-x_g_grad/s).unsqueeze(-1)).squeeze(-1)
    # mu_grad_1 = (R@(d_g/s).unsqueeze(-1) * (t_grad/(d_g*d_g).sum(dim=-1))[..., None, None] ).squeeze(-1)

    grad_rho     = grad_alpha*opacity
    grad_opacity = grad_alpha*rho

    # NOT USED grad_x_g   = -grad_rho*rho*x_g
    grad_x_g   = grad_rho*(-(3**(2-n))*(n/2)*xtx.pow(n/2-1))*rho*2*x_g
    grad_o_g   = grad_x_g.clone()
    grad_t     = grad_t + (grad_x_g * d_g).sum(dim=-1)  # scalar
    grad_d_g   = grad_x_g * t

    grad_t_o_g = -grad_t*d_g/(d_g*d_g).sum(dim=-1, keepdim=True)
    grad_t_d_g_0 = -grad_t*o_g/(d_g*d_g).sum(dim=-1, keepdim=True)
    grad_t_d_g_1 = -grad_t*2*d_g*t/(d_g*d_g).sum(dim=-1, keepdim=True)

    grad_o_g = grad_o_g + grad_t_o_g
    grad_d_g = grad_d_g + grad_t_d_g_0 + grad_t_d_g_1

    # grad_S_inv_Rt = grad_o_g.unsqueeze(-1) @ (ro - mu).unsqueeze(0) + grad_d_g.unsqueeze(-1) @ rd.unsqueeze(0)

    grad_S_inv_Rt = torch.stack([
        grad_o_g[0]*(ro-mu)[0] + grad_d_g[0]*rd[0],
        grad_o_g[0]*(ro-mu)[1] + grad_d_g[0]*rd[1],
        grad_o_g[0]*(ro-mu)[2] + grad_d_g[0]*rd[2],

        grad_o_g[1]*(ro-mu)[0] + grad_d_g[1]*rd[0],
        grad_o_g[1]*(ro-mu)[1] + grad_d_g[1]*rd[1],
        grad_o_g[1]*(ro-mu)[2] + grad_d_g[1]*rd[2],

        grad_o_g[2]*(ro-mu)[0] + grad_d_g[2]*rd[0],
        grad_o_g[2]*(ro-mu)[1] + grad_d_g[2]*rd[1],
        grad_o_g[2]*(ro-mu)[2] + grad_d_g[2]*rd[2],
    ]).reshape(3, 3)


    # grad_S_inv = grad_S_inv_Rt @ R
    # grad_R     = grad_S_inv_Rt.T @ S_inv
    # grad_S_inv = torch.stack([
    #     grad_S_inv_Rt[0] @ col0, grad_S_inv_Rt[0] @ col1, grad_S_inv_Rt[0] @ col2,
    #     grad_S_inv_Rt[1] @ col0, grad_S_inv_Rt[1] @ col1, grad_S_inv_Rt[1] @ col2,
    #     grad_S_inv_Rt[2] @ col0, grad_S_inv_Rt[2] @ col1, grad_S_inv_Rt[2] @ col2,
    # ]).reshape(3, 3)
    grad_S_inv_diag = torch.stack([
        grad_S_inv_Rt[0] @ col0, 
        grad_S_inv_Rt[1] @ col1,
        grad_S_inv_Rt[2] @ col2,
    ])
    grad_R = torch.stack([
        grad_S_inv_Rt[0][0]/scale[0], grad_S_inv_Rt[1][0]/scale[1], grad_S_inv_Rt[2][0]/scale[2],
        grad_S_inv_Rt[0][1]/scale[0], grad_S_inv_Rt[1][1]/scale[1], grad_S_inv_Rt[2][1]/scale[2],
        grad_S_inv_Rt[0][2]/scale[0], grad_S_inv_Rt[1][2]/scale[1], grad_S_inv_Rt[2][2]/scale[2],
    ]).reshape(3, 3)

    # RtR = torch.stack([
    #     col0 @ col0, col0 @ col1, col0 @ col2,
    #     col1 @ col0, col1 @ col1, col1 @ col2,
    #     col2 @ col0, col2 @ col1, col2 @ col2,
    # ]).reshape(3, 3)
    # assert torch.allclose(RtR, R.transpose(-2, -1)@R), f"{RtR} {R.transpose(-2, -1)@R}"

    grad_R_flat= grad_R.flatten()

    # grad_mu    = -S_inv_Rt.T@(grad_o_g)
    # grad_mu    = -torch.stack([row0@(grad_o_g/scale), row1@(grad_o_g/scale), row2@(grad_o_g/scale)])
    # grad_scale = -torch.diag(grad_S_inv)/scale.square()
    # grad_scale = -grad_S_inv_diag/scale.square()
    grad_mu    = torch.zeros_like(mu)
    grad_mu[0] = -(col0[0]*grad_o_g[0]*inv_s[0] + col1[0]*grad_o_g[1]*inv_s[1] + col2[0]*grad_o_g[2]*inv_s[2])
    grad_mu[1] = -(col0[1]*grad_o_g[0]*inv_s[0] + col1[1]*grad_o_g[1]*inv_s[1] + col2[1]*grad_o_g[2]*inv_s[2])
    grad_mu[2] = -(col0[2]*grad_o_g[0]*inv_s[0] + col1[2]*grad_o_g[1]*inv_s[1] + col2[2]*grad_o_g[2]*inv_s[2])
    grad_scale = -grad_S_inv_diag*inv_s*inv_s

    w, x, y, z = q_vec

    grad_q_w = 2*(-z)*grad_R_flat[1] + 2*(y)*grad_R_flat[2] + 2*(z)*grad_R_flat[3] + \
               2*(-x)*grad_R_flat[5] + 2*(-y)*grad_R_flat[6] + 2*(x)*grad_R_flat[7]

    grad_q_x = 2*(y)*grad_R_flat[1] + 2*(z)*grad_R_flat[2] + 2*(y)*grad_R_flat[3] + 4*(-x)*grad_R_flat[4] + \
               2*(-w)*grad_R_flat[5] + 2*(z)*grad_R_flat[6] + 2*(w)*grad_R_flat[7] + 4*(-x)*grad_R_flat[8]

    grad_q_y = 4*(-y)*grad_R_flat[0] + 2*(x)*grad_R_flat[1] + 2*(w)*grad_R_flat[2] + 2*(x)*grad_R_flat[3] + \
               2*(z)*grad_R_flat[5] + 2*(-w)*grad_R_flat[6] + 2*(z)*grad_R_flat[7] + 4*(-y)*grad_R_flat[8] 

    grad_q_z = 4*(-z)*grad_R_flat[0] + 2*(-w)*grad_R_flat[1] + 2*(x)*grad_R_flat[2] + 2*(w)*grad_R_flat[3] + \
               4*(-z)*grad_R_flat[4] + 2*(y)*grad_R_flat[5] + 2*(x)*grad_R_flat[6] + 2*(y)*grad_R_flat[7] 


    # R = torch.stack([
    #              1 - 2 * (y*y + z*z),           0
    #              2 * (x*y - w*z),               1
    #              2 * (x*z + w*y),               2
    #              2 * (x*y + w*z),               3
    #              1 - 2 * (x*x + z*z),           4
    #              2 * (y*z - w*x),               5
    #              2 * (x*z - w*y),               6
    #              2 * (y*z + w*x),               7
    #              1 - 2 * (x*x + y*y),           8
    #             ], dim=-1).unflatten(-1, (3, 3))

    # assume already normalized
    # q_vec_nrm  = (q_vec*q_vec).sum(dim=-1)
    grad_q_vec = torch.zeros_like(q_vec)
    grad_q_vec[0] = grad_q_w
    grad_q_vec[1] = grad_q_x
    grad_q_vec[2] = grad_q_y
    grad_q_vec[3] = grad_q_z

    grad_ro = S_inv_Rt.T@grad_o_g
    grad_rd = S_inv_Rt.T@grad_d_g

    if value_dict is not None:
        assert torch.allclose(grad_x_g, value_dict["x_g"].grad), f"{(grad_x_g - value_dict['x_g'].grad).abs().mean()} {grad_x_g} {value_dict['x_g'].grad}"
        assert torch.allclose(d_g,      value_dict["d_g"]),      f"{(d_g - value_dict['d_g']).abs().mean()} {d_g} {value_dict['d_g']}"

        if torch.allclose(grad_t,   value_dict["t"].grad) is False:
            print("debug grad_t: isclose: grad_x_g", torch.allclose(grad_x_g, value_dict["x_g"].grad) )
            print("debug grad_t: isclose: d_g",      torch.allclose(d_g,      value_dict["d_g"]) )
            print("grad_t:", grad_t, (grad_x_g*d_g).sum(dim=-1), value_dict["t"].grad)
            # print("ref grad_t:", value_dict["t"].grad)

        assert torch.allclose(grad_t,   value_dict["t"].grad),   f"{(grad_t - value_dict['t'].grad).abs().mean()} {grad_t} {value_dict['t'].grad}"

        assert torch.allclose(grad_t_d_g_0,value_dict["d_g_0"].grad), f"{(grad_t_d_g_0 - value_dict['d_g_0'].grad).abs().mean()} {grad_t_d_g_0} {value_dict['d_g_0'].grad}"
        assert torch.allclose(grad_t_d_g_1,value_dict["d_g_1"].grad), f"{(grad_t_d_g_1 - value_dict['d_g_1'].grad).abs().mean()} {grad_t_d_g_1} {value_dict['d_g_1'].grad}"

        assert torch.allclose(grad_d_g, value_dict["d_g"].grad), f"{(grad_d_g - value_dict['d_g'].grad).abs().mean()} {grad_d_g} {value_dict['d_g'].grad}"
        assert torch.allclose(grad_S_inv_Rt, value_dict["S_inv_Rt"].grad), f"{(grad_S_inv_Rt - value_dict['S_inv_Rt'].grad).abs().mean()} {grad_S_inv_Rt} {value_dict['S_inv_Rt'].grad}"

        # assert torch.allclose(grad_S_inv, value_dict["S_inv"].grad), f"{(grad_S_inv - value_dict['S_inv'].grad).abs().mean()} {grad_S_inv} {value_dict['S_inv'].grad}"
        assert torch.allclose(grad_R,     value_dict["R"].grad),     f"{(grad_R - value_dict['R'].grad).abs().mean()} {grad_R} {value_dict['R'].grad}"

        assert torch.allclose(grad_scale, value_dict["scale"].grad), f"{(grad_scale - value_dict['scale'].grad).abs().mean()} {grad_scale} {value_dict['scale'].grad}"
        assert torch.allclose(grad_q_vec, value_dict["q_vec"].grad), f"{(grad_q_vec - value_dict['q_vec'].grad).abs().mean()} {grad_q_vec} {value_dict['q_vec'].grad}"
        assert torch.allclose(grad_mu, value_dict["mu"].grad), f"{(grad_mu - value_dict['mu'].grad).abs().mean()} {grad_mu} {value_dict['mu'].grad}"

    return grad_mu, grad_scale, grad_q_vec, grad_opacity, grad_ro, grad_rd

# s = torch.as_tensor([0.1, 0.2, 0.3]).requires_grad_(True)

# m = torch.diag(1/s)
# g = torch.arange(m.numel(), dtype=m.dtype).reshape(m.shape)
# m.backward(g)

# s_grad = -torch.diag(g)/s/s

# print("diag backward behavior")
# print("out_grad:", g)
# print("input grad", s.grad)
# print("compute grad:", s_grad)

v = torch.as_tensor([0.91, -10.2, 0.3]).requires_grad_(True)
b = torch.as_tensor([0.1, 0.2, 0.3]).requires_grad_(True)
s = torch.as_tensor([0.5]).requires_grad_(True)

m = b + s*v
g = torch.arange(m.numel(), dtype=m.dtype).reshape(m.shape)
m.backward(g)

s_grad = (v*g).sum(dim=-1)

print("o+t*d backward behavior")
print("out_grad:", g)
print("reference grad", s.grad)
print("compute grad:", s_grad)

def test_pytorch_impl():

    # order = torch.randperm(xyz.size(0))

    N = 4

    dtype = torch.float64

    mu      = torch.randn(N, 3, dtype=dtype).requires_grad_(True)
    scale   = (0.001+torch.rand( N, 3, dtype=dtype)).requires_grad_(True)
    q_vec   = F.normalize(torch.randn(N, 4, dtype=dtype), dim=-1).requires_grad_(True)
    opacity = torch.rand(N, 1, dtype=dtype).requires_grad_(True)
    rgb     = torch.rand(N, 3, dtype=dtype).requires_grad_(True)

    ro = torch.rand(3, dtype=dtype).requires_grad_(True)
    rd = F.normalize(torch.randn(3, dtype=dtype), dim=-1).requires_grad_(True)

    order = np.arange(len(mu))
    np.random.shuffle(order)

    color = torch.zeros(rgb.size(-1)).requires_grad_(True)
    I_alp = torch.zeros(1).requires_grad_(True)
    I_dep = torch.zeros(1).requires_grad_(True)
    I_dep2= torch.zeros(1).requires_grad_(True)
    I_ldd = torch.zeros(1).requires_grad_(True)

    intermedia_alpha, intermedia_value, intermedia_weight = [], [], []
    # reference fwd
    T = 1
    for i, idx in enumerate(order):
        t, alpha, _, _, value = gaussian_fwd(mu[idx], scale[idx], q_vec[idx], opacity[idx], ro, rd)

        alpha.retain_grad()
        weight = T*alpha

        intermedia_alpha.append(alpha)
        intermedia_value.append(value)

        weight.retain_grad()
        intermedia_weight.append(weight)

        I_ldd  = I_ldd + weight * (t.pow(2)*I_alp - 2*t*I_dep + I_dep2)

        color  = color + rgb[idx]*weight
        I_alp  = I_alp + weight
        I_dep  = I_dep + t*weight
        I_dep2 = I_dep2 + t.pow(2)*weight

        T = T*(1-alpha)

    # auto-bwd
    grad_color = torch.randn_like(color)
    grad_I_dep = torch.randn_like(I_dep)
    grad_I_alp = torch.randn_like(I_alp)
    grad_I_ldd = torch.randn_like(I_ldd)
    grad_I_dep2= torch.randn_like(I_dep2)
    grad_T     = torch.randn_like(T)

    color.backward(grad_color, retain_graph=True)
    I_dep.backward(grad_I_dep, retain_graph=True)
    I_alp.backward(grad_I_alp, retain_graph=True)
    I_ldd.backward(grad_I_ldd, retain_graph=True)
    I_dep2.backward(grad_I_dep2, retain_graph=True)
    T.backward(grad_T, retain_graph=True)

    # our-bwd
    mu_grad      = torch.zeros_like(mu)
    scale_grad   = torch.zeros_like(scale)
    q_vec_grad   = torch.zeros_like(q_vec)
    opacity_grad = torch.zeros_like(opacity)
    rgb_grad     = torch.zeros_like(rgb)
    ro_grad      = torch.zeros_like(ro)
    rd_grad      = torch.zeros_like(rd)

    _I_alp = I_alp.clone().detach()
    _I_dep = I_dep.clone().detach()
    _I_dep2= I_dep2.clone().detach()

    b_alp = torch.zeros(1).requires_grad_(True)
    b_dep = torch.zeros(1).requires_grad_(True)
    b_dep2= torch.zeros(1).requires_grad_(True)
    b_ldd = torch.zeros(1).requires_grad_(True)
    T = 1
    acc_g_w = 0
    for i, idx in enumerate(order):
        t, alpha = gaussian_fwd(mu[idx], scale[idx], q_vec[idx], opacity[idx], ro, rd)[:2]
        weight = T*alpha

        # ComputeGrad
        rgb_grad[idx] += grad_color*weight

        # accumulated grad*weight
        grad_w = (grad_color*rgb[idx]).sum(dim=-1)
        grad_w = grad_w + grad_I_dep*t
        grad_w = grad_w + grad_I_alp

        # gradient of loss depth distrotion
        # ldd_e  = (t.pow(2)*b_alp - 2*t*b_dep + b_dep2)
        # b_ldd  = b_ldd + weight * ldd_e

        # b_alp  = b_alp  + weight
        # b_dep  = b_dep  + t*weight
        # b_dep2 = b_dep2 + t.pow(2)*weight

        grad_w = grad_w + grad_I_dep2*t.pow(2)
        grad_w = grad_w + grad_I_ldd*( t.pow(2)*(_I_alp - weight) - 2*t*(_I_dep - t*weight) + (_I_dep2 - t.pow(2)*weight) )

        # accumulate from front-to-back
        acc_g_w = acc_g_w + grad_w*weight
        T       = T*(1-alpha)
    
    print("loss depth distrotion:", I_ldd, b_ldd)
    
    acc_T = T

    b_alp = torch.zeros(1).requires_grad_(True)
    b_dep = torch.zeros(1).requires_grad_(True)
    b_dep2= torch.zeros(1).requires_grad_(True)
    b_ldd = torch.zeros(1).requires_grad_(True)
    T = 1
    for i, idx in enumerate(order):
        fwd_ret = gaussian_fwd(mu[idx], scale[idx], q_vec[idx], opacity[idx], ro, rd)
        t, alpha= fwd_ret[:2]
        weight = T*alpha

        b_alp  = b_alp + weight
        b_dep  = b_dep + t*weight

        # ComputeGrad
        grad_w = (grad_color*rgb[idx]).sum(dim=-1)
        grad_w = grad_w + grad_I_dep*t
        grad_w = grad_w + grad_I_alp
        grad_w = grad_w + grad_I_dep2*t.pow(2)
        grad_w = grad_w + grad_I_ldd*( t.pow(2)*(_I_alp - weight) - 2*t*(_I_dep - t*weight) + (_I_dep2 - t.pow(2)*weight) )  # gradient of loss depth distrotion

        grad_alpha = grad_w*T + (acc_g_w - grad_w*weight)/(alpha-1) + grad_T*acc_T/(alpha-1)

        assert torch.allclose(grad_w,     intermedia_weight[i].grad, rtol=1e-4, atol=1e-5), f"[{i}] {grad_w} {intermedia_weight[i].grad}"
        assert torch.allclose(grad_alpha, intermedia_alpha[i].grad,  rtol=1e-4, atol=1e-5), f"[{i}] {grad_alpha} {intermedia_alpha[i].grad}"

        grad_t = grad_I_dep*weight
        grad_t = grad_t + 2*grad_I_dep2*weight*t
        # grad_t = grad_t + 2*grad_I_ldd *weight*(t*(_I_alp - b_alp) - (_I_dep - b_dep))
        # grad_t = grad_t + 2*grad_I_ldd *weight*(t*b_alp - b_dep)
        grad_t = grad_t + 2*grad_I_ldd *weight*(t*(_I_alp) - (_I_dep))
        grad_output = (grad_t, grad_alpha)

        g_mu, g_scale, g_q_vec, g_opacity, g_ro, g_rd = \
            gaussian_bwd(mu[idx], scale[idx], q_vec[idx], opacity[idx], ro, rd, fwd_ret, grad_output, 4, intermedia_value[i])
        mu_grad[idx]      += g_mu
        scale_grad[idx]   += g_scale
        q_vec_grad[idx]   += g_q_vec
        opacity_grad[idx] += g_opacity

        ro_grad += g_ro
        rd_grad += g_rd

        acc_g_w -= grad_w * weight

        T      = T*(1-alpha)
    
    def compare(name, ref_val, our_val):
        shape_match = ref_val.shape == our_val.shape
        value_match = torch.allclose(ref_val, our_val)
        value_dist  = (ref_val - our_val).abs().mean()
        if shape_match and value_match:
            print(f"{name} succ !")
        else:
            print(f"{name} fail ! same_shape:{shape_match} allclose:{value_match} err:{value_dist}")
    
    compare("mu",      mu.grad,      mu_grad)
    compare("scale",   scale.grad,   scale_grad)
    compare("q_vec",   q_vec.grad,   q_vec_grad)
    compare("opacity", opacity.grad, opacity_grad)
    compare("RGB",     rgb.grad,     rgb_grad)

    compare("ro",      ro.grad,      ro_grad)
    compare("rd",      rd.grad,      rd_grad)

def test_cuda_impl():

    # order = torch.randperm(xyz.size(0))

    N = 16

    dtype = torch.float64

    order  = np.arange(N)
    np.random.shuffle(order)

    ref_mu      = (F.pad(torch.linspace(1, N/2, N)[:,None], (2,0), "constant", 0.0) + 0.1*torch.randn(N, 3, dtype=dtype))
    ref_mu      = ref_mu[order].requires_grad_(True)
    ref_scale   = (0.01+0.2*torch.rand( N, 3, dtype=dtype)).requires_grad_(True)
    ref_q_vec   = F.normalize(torch.randn(N, 4, dtype=dtype), dim=-1).requires_grad_(True)
    ref_opacity = (0.5+0.2*torch.rand(N, 1, dtype=dtype)).requires_grad_(True)
    ref_rgb     = torch.rand(N, 3, dtype=dtype).requires_grad_(True)
    ref_nrm     = torch.randn(N, 3, dtype=dtype).requires_grad_(True)

    ref_ro = (0.1*torch.randn(3, dtype=dtype))
    ref_ro[2] -= 0.5
    ref_ro = ref_ro.clone().detach().requires_grad_(True)
    ref_rd = F.normalize(torch.as_tensor([0.01, 0.01, 1], dtype=dtype), dim=-1).requires_grad_(True)

    # print(ref_mu)

    # print(order, (ref_mu - ref_ro).norm(dim=-1).sort().indices)
    # order  = (ref_mu - ref_ro).norm(dim=-1).sort().indices.cpu().numpy().tolist()
    optix_ctx = gaussian_raytrace.create_optix_context()
    proxy_ver, proxy_tri = gaussian_raytrace.build_proxy_geometry(optix_ctx, ref_mu.float(), ref_q_vec.float(), ref_scale.float(), ref_opacity.float()) # Np,Nv,3 Np,Nf,3

    # proxy_ver, proxy_tri = proxy_ver.cpu(), proxy_tri.cpu()

    order = []
    faces = proxy_ver.flatten(0, 1)[proxy_tri.flatten().long()].reshape(1, -1, 3, 3).float()
    faces = faces.cuda()
    hit_t = 0
    while True:
        query_ray = torch.cat([ref_ro + hit_t*ref_rd, ref_rd], dim=-1).reshape(1, -1, 6).float()
        i, d, _ = PrimaryRayFunction.apply(faces, query_ray.cuda(), True)
        i = i.item()
        if i < 0:
            break
        hit_i = i//proxy_tri.size(1)
        hit_t += d.item() + 1e-3
        # print("faces.shape", faces.shape, query_ray.shape, hit_i, hit_t)
        order.append(hit_i)
    
    print("order", order)

    color = torch.zeros(ref_rgb.size(-1)).requires_grad_(True)
    I_nrm = torch.zeros(3).requires_grad_(True)
    I_alp = torch.zeros(1).requires_grad_(True)
    I_dep = torch.zeros(1).requires_grad_(True)
    I_pos = torch.zeros(3).requires_grad_(True)
    I_dep2= torch.zeros(1).requires_grad_(True)
    I_ldd = torch.zeros(1).requires_grad_(True)

    intermedia_t_max, intermedia_alpha, intermedia_weight, intermedia_value = [], [], [], []
    # reference fwd
    T = torch.ones(1, dtype=dtype)
    acc_hit = 0

    primid = []
    report_i = list(range(0, len(order), 8))[1:] + [len(order)-1]
    report_i = set(report_i)
    for i, idx in enumerate(order):
        t, alpha, _, _, value = gaussian_fwd(ref_mu[idx], ref_scale[idx], ref_q_vec[idx], ref_opacity[idx], ref_ro, ref_rd)

        t.retain_grad()
        alpha.retain_grad()
        intermedia_t_max.append(t)
        intermedia_alpha.append(alpha)
        intermedia_value.append(value)

        if alpha.item() > 0.01:
            primid.append(idx)

            weight = T*alpha

            weight.retain_grad()
            intermedia_weight.append(weight)

            I_ldd  = I_ldd + weight * (t.pow(2)*I_alp - 2*t*I_dep + I_dep2)

            color  = color + ref_rgb[idx]*weight
            I_nrm  = I_nrm + ref_nrm[idx]*weight
            I_pos  = I_pos + ref_mu[idx]*weight
            I_alp  = I_alp + weight
            I_dep  = I_dep + t*weight
            I_dep2 = I_dep2+ t.pow(2)*weight

            T = T*(1-alpha)

        if i in report_i:
            for j in range(i+1):
                print(f"forward t_max ({intermedia_t_max[j].item()}, {order[j]}), ", end="")
            print(f"T={T.item()}")

        acc_hit += 1

        if i % 8 == 7 and T.item() < 0.005: # mimic chunk behavior
            break
    
    # auto-bwd
    grad_color = torch.randn_like(color)         .zero_()
    grad_I_dep = torch.randn_like(I_dep)         # .zero_()
    grad_I_nrm = torch.randn_like(I_nrm)         # .zero_()
    grad_I_pos = torch.randn_like(I_pos)         # .zero_()
    grad_I_alp = torch.randn_like(I_alp)         # .zero_()
    grad_I_ldd = torch.randn_like(I_ldd)         # .zero_()
    grad_I_dep2= torch.randn_like(I_dep2)        # .zero_()
    grad_T     = torch.randn_like(T)             # .zero_()

    color.backward(grad_color, retain_graph=True)
    I_nrm.backward(grad_I_nrm, retain_graph=True)
    I_pos.backward(grad_I_pos, retain_graph=True)
    I_dep.backward(grad_I_dep, retain_graph=True)
    I_alp.backward(grad_I_alp, retain_graph=True)
    I_ldd.backward(grad_I_ldd, retain_graph=True)
    I_dep2.backward(grad_I_dep2, retain_graph=True)
    T.backward(grad_T, retain_graph=True)

    print("primid",  primid)
    print("grad_w", [t.grad.item() for t in intermedia_weight])
    # print("grad_t", [t.grad.item() for t in intermedia_t_max])

    # our-bwd
    device  = torch.device("cuda")
    mu      = ref_mu     .float().clone().to(device).detach().requires_grad_(True)
    scale   = ref_scale  .float().clone().to(device).detach().requires_grad_(True)
    q_vec   = ref_q_vec  .float().clone().to(device).detach().requires_grad_(True)
    opacity = ref_opacity.float().clone().to(device).detach().requires_grad_(True)
    rgb     = ref_rgb    .float().clone().to(device).detach().requires_grad_(True)
    nrm     = ref_nrm    .float().clone().to(device).detach().requires_grad_(True)

    ro      = ref_ro.float().clone().to(device).detach().requires_grad_(True)
    rd      = ref_rd.float().clone().to(device).detach().requires_grad_(True)

    optix_ctx = gaussian_raytrace.create_optix_context()
    proxy_ver, proxy_tri = gaussian_raytrace.build_proxy_geometry(optix_ctx, mu, q_vec, scale, opacity) # Np,Nv,3 Np,Nf,3

    img, buf, dbg = gaussian_raytrace.GaussianRayTraceFunction.apply(
        optix_ctx, 
        mu, q_vec, scale, opacity, 
        rgb, nrm, 
        ro[None, None, None], rd[None, None, None],
        proxy_ver, proxy_tri,
        0.005, 0.01)
    
    alp = buf[..., 0:1]
    nrm = buf[..., 1:4]
    dep = buf[..., 4:5]
    pos = buf[..., 5:8]
    T   = buf[..., 8:9]
    dep2= buf[..., 9:10]
    ldd = buf[..., 10:11]

    print("ref T/acc_hit", T.item(), acc_hit)
    print("our T/acc_hit", buf[0,0,0,8].item())

    print("ref dep2/ldd", I_dep2.item(), I_ldd.item())
    print("our dep2/ldd", dep2[0,0,0].item(), ldd[0,0,0].item())

    # print(img.shape, grad_color.shape)
    # print(dep.shape, grad_I_dep.shape)
    # print(alp.shape, grad_I_alp.shape)
    
    # auto-bwd
    img.backward(grad_color[None, None, None].to(device), retain_graph=True)
    # nrm.backward(grad_I_nrm[None, None, None].to(device), retain_graph=True)
    # pos.backward(grad_I_pos[None, None, None].to(device), retain_graph=True)
    # dep.backward(grad_I_dep[None, None, None].to(device), retain_graph=True)
    # alp.backward(grad_I_alp[None, None, None].to(device), retain_graph=True)
    # ldd.backward(grad_I_ldd[None, None, None].to(device), retain_graph=True)
    # dep2.backward(grad_I_dep2[None, None, None].to(device), retain_graph=True)
    # T.backward(grad_T[None, None, None].to(device), retain_graph=True)

    buf.backward(torch.cat([grad_I_alp, grad_I_nrm, grad_I_dep, grad_I_pos, grad_T, grad_I_dep2, grad_I_ldd], dim=-1)[None, None, None].to(device), retain_graph=True)

    def compare(name, ref_val, our_val):
        if our_val is None:
            print(f"{name} fail ! targe_shape:{ref_val.shape} our:{our_val}")
            return

        our_val = our_val.cpu()
        ref_val = ref_val.cpu().type(our_val.dtype)
        shape_match = ref_val.shape == our_val.shape
        value_match = torch.allclose(ref_val, our_val, atol=1e-4, rtol=1e-6)
        value_dist  = (ref_val - our_val).abs().mean()
        if shape_match and value_match:
            print(f"{name} succ !")
        else:
            print(f"{name} fail ! same_shape:{shape_match} allclose:{value_match} err:{value_dist}")
    
    compare("RGB",     ref_rgb.grad,     rgb.grad)
    compare("mu",      ref_mu.grad,      mu.grad)
    compare("scale",   ref_scale.grad,   scale.grad)
    compare("q_vec",   ref_q_vec.grad,   q_vec.grad)
    compare("opacity", ref_opacity.grad, opacity.grad)

    compare("ro",      ref_ro.grad,      ro.grad)
    compare("rd",      ref_rd.grad,      rd.grad)

print("============ PyTorch Implement ==============")
# test_pytorch_impl()

print("============ CUDA/OptiX Implement ==============")
test_cuda_impl()