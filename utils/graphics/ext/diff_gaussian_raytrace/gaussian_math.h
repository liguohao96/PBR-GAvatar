#pragma once

#ifdef __CUDACC__

#define INV_SCALE_EPS   1e-6f
#ifndef GAUSSIAN_KERNEL_DEGREE
    #define GAUSSIAN_KERNEL_DEGREE 4
#endif
#ifndef GAUSSIAN_KERNEL_R
    #define GAUSSIAN_KERNEL_R 3
#endif

struct Rot3DMat{
    float3 col0;
    float3 col1;
    float3 col2;
};

struct GaussianGrad{
    float4 q_vec;
    float3 mu;
    float  opacity;
    float3 scale;
    float3 ro;
    float3 rd;
};

Rot3DMat quaterion2rotation(const float4 q_vec){
    Rot3DMat ret;
    const float x = q_vec.x;
    const float y = q_vec.y;
    const float z = q_vec.z;
    const float w = q_vec.w;

    const float r00 = 1 - 2 * (y*y + z*z);
    const float r01 = 2 * (x*y - w*z);
    const float r02 = 2 * (x*z + w*y);
    const float r10 = 2 * (x*y + w*z);
    const float r11 = 1 - 2 * (x*x + z*z);
    const float r12 = 2 * (y*z - w*x);
    const float r20 = 2 * (x*z - w*y);
    const float r21 = 2 * (y*z + w*x);
    const float r22 = 1 - 2 * (x*x + y*y);

    ret.col0 = make_float3(r00, r10, r20);
    ret.col1 = make_float3(r01, r11, r21);
    ret.col2 = make_float3(r02, r12, r22);
    return ret;
}

float4 gaussian_particle_fwd(const float3& mu, const float3& scale, const Rot3DMat& rot_3d, const float opacity, const float3& ray_org, const float3& ray_dir){

    const float3& col0 = rot_3d.col0;
    const float3& col1 = rot_3d.col1;
    const float3& col2 = rot_3d.col2;

    const float3 inv_s = 1/(scale + INV_SCALE_EPS);

    const float3 off = (ray_org - mu);
    const float3 dir = (ray_dir);
    const float3 o_g = {dot(col0, off)*inv_s.x, dot(col1, off)*inv_s.y, dot(col2, off)*inv_s.z};
    const float3 d_g = {dot(col0, dir)*inv_s.x, dot(col1, dir)*inv_s.y, dot(col2, dir)*inv_s.z};

    // max response distance
    const float    t = -dot(o_g, d_g)/dot(d_g, d_g);
    const float3 x_g = o_g + t*d_g;

    // float4 ret = make_float4(x_g.x, x_g.y, x_g.z, opacity*exp(-0.5*dot(x_g, x_g)));
    #if GAUSSIAN_KERNEL_DEGREE==2
    float4 ret = make_float4(t, 0.0f, 0.0f, opacity*exp(-dot(x_g, x_g)));  // match with the 3DGRT and 3DGUT paper
    #else
    const float n = (float)(GAUSSIAN_KERNEL_DEGREE);
    const float r = (float)(GAUSSIAN_KERNEL_R);
    float4 ret = make_float4(t, 0.0f, 0.0f, opacity*exp(-pow(r, 2-n)*pow(dot(x_g, x_g), n/2)));
    #endif
    return ret;
}

float4 gaussian_particle_fwd(const float3& mu, const float3& scale, const float4& q_vec, const float opacity, const float3& ray_org, const float3& ray_dir){
    const Rot3DMat rot_3d = quaterion2rotation(q_vec);

    return gaussian_particle_fwd(mu, scale, rot_3d, opacity, ray_org, ray_dir);
}

GaussianGrad gaussian_particle_bwd(const float3& mu, const float3& scale, const float4 q_vec, const Rot3DMat& rot_3d, const float opacity, const float3& ray_org, const float3& ray_dir, const float4& output, const float4& grad_output){
    GaussianGrad ret;

    const float d_L_alpha = grad_output.w;

    const float3& col0 = rot_3d.col0;
    const float3& col1 = rot_3d.col1;
    const float3& col2 = rot_3d.col2;

    const float3 inv_s = 1/(scale + INV_SCALE_EPS);

    const float3 off = (ray_org - mu);
    const float3 dir = (ray_dir);
    const float3 o_g = {dot(col0, off)*inv_s.x, dot(col1, off)*inv_s.y, dot(col2, off)*inv_s.z};
    const float3 d_g = {dot(col0, dir)*inv_s.x, dot(col1, dir)*inv_s.y, dot(col2, dir)*inv_s.z};

    const float inv_square_d_g = 1/dot(d_g, d_g);
    // max response distance
    // const float    t = -dot(o_g, d_g)*inv_square_d_g;
    // const float3 x_g = make_float3(output.x, output.y, output.z);
    const float    t = output.x;
    const float3 x_g = o_g + t*d_g;
    #if GAUSSIAN_KERNEL_DEGREE==2
    const float rho  = exp(-dot(x_g, x_g));
    #else
    // const float n    = GAUSSIAN_KERNEL_DEGREE;
    // const float r    = GAUSSIAN_KERNEL_R;
    const float n    = (float)(GAUSSIAN_KERNEL_DEGREE);
    const float r    = (float)(GAUSSIAN_KERNEL_R);
    const float xtx  = dot(x_g, x_g);
    const float rho  = exp(-pow(r, 2-n)*pow(xtx, n/2));
    #endif

    // grad_rho     = out_grad*opacity
    // grad_opacity = out_grad*rho
    const float grad_rho = d_L_alpha*opacity;
    ret.opacity = d_L_alpha*rho;

    // grad_x_g   = -grad_rho*rho*x_g
    // grad_o_g   = grad_x_g
    // grad_t     = (grad_x_g * d_g).sum(dim=-1)  # scalar
    // grad_d_g   = grad_x_g * t
    #if GAUSSIAN_KERNEL_DEGREE==2
    const float3 grad_x_g   = -2*grad_rho*rho*x_g;
    #else
    const float3 grad_x_g   = grad_rho*(-pow(3.0f, 2-n))*((n/2)*pow(xtx, n/2-1))*rho*2*x_g;
    #endif
    const float  grad_t     = dot(grad_x_g, d_g) + grad_output.x;
    float3 grad_o_g   = grad_x_g;
    float3 grad_d_g   = grad_x_g * t;

    // grad_t_o_g = -grad_t*d_g/(d_g*d_g).sum(dim=-1, keepdim=True)
    // grad_t_d_g_0 = -grad_t*o_g/(d_g*d_g).sum(dim=-1, keepdim=True)
    // grad_t_d_g_1 = -grad_t*2*d_g*t/(d_g*d_g).sum(dim=-1, keepdim=True)
    // grad_o_g = grad_o_g + grad_t_o_g
    // grad_d_g = grad_d_g + grad_t_d_g_0 + grad_t_d_g_1
    const float3 grad_t_o_g   = -grad_t*d_g*inv_square_d_g;
    const float3 grad_t_d_g_0 = -grad_t*o_g*inv_square_d_g;
    const float3 grad_t_d_g_1 = -grad_t*2*d_g*t*inv_square_d_g;
    grad_o_g = grad_o_g + grad_t_o_g;
    grad_d_g = grad_d_g + grad_t_d_g_0 + grad_t_d_g_1;

    // grad_S_inv_Rt = grad_o_g.unsqueeze(-1) @ (ro - mu).unsqueeze(0) + grad_d_g.unsqueeze(-1) @ rd.unsqueeze(0)
    // grad_S_inv = grad_S_inv_Rt @ R
    // grad_R     = grad_S_inv_Rt.T @ S_inv
    float  grad_S_inv_Rt[3][3];
    float3 grad_S_inv_diag;
    float  grad_R[9];

    grad_S_inv_Rt[0][0] = grad_o_g.x*off.x + grad_d_g.x*dir.x;
    grad_S_inv_Rt[0][1] = grad_o_g.x*off.y + grad_d_g.x*dir.y;
    grad_S_inv_Rt[0][2] = grad_o_g.x*off.z + grad_d_g.x*dir.z;

    grad_S_inv_Rt[1][0] = grad_o_g.y*off.x + grad_d_g.y*dir.x;
    grad_S_inv_Rt[1][1] = grad_o_g.y*off.y + grad_d_g.y*dir.y;
    grad_S_inv_Rt[1][2] = grad_o_g.y*off.z + grad_d_g.y*dir.z;

    grad_S_inv_Rt[2][0] = grad_o_g.z*off.x + grad_d_g.z*dir.x;
    grad_S_inv_Rt[2][1] = grad_o_g.z*off.y + grad_d_g.z*dir.y;
    grad_S_inv_Rt[2][2] = grad_o_g.z*off.z + grad_d_g.z*dir.z;

    grad_S_inv_diag.x = grad_S_inv_Rt[0][0]*col0.x + grad_S_inv_Rt[0][1]*col0.y + grad_S_inv_Rt[0][2]*col0.z;
    grad_S_inv_diag.y = grad_S_inv_Rt[1][0]*col1.x + grad_S_inv_Rt[1][1]*col1.y + grad_S_inv_Rt[1][2]*col1.z;
    grad_S_inv_diag.z = grad_S_inv_Rt[2][0]*col2.x + grad_S_inv_Rt[2][1]*col2.y + grad_S_inv_Rt[2][2]*col2.z;

    grad_R[0] = grad_S_inv_Rt[0][0]*inv_s.x;
    grad_R[1] = grad_S_inv_Rt[1][0]*inv_s.y;
    grad_R[2] = grad_S_inv_Rt[2][0]*inv_s.z;
    grad_R[3] = grad_S_inv_Rt[0][1]*inv_s.x;
    grad_R[4] = grad_S_inv_Rt[1][1]*inv_s.y;
    grad_R[5] = grad_S_inv_Rt[2][1]*inv_s.z;
    grad_R[6] = grad_S_inv_Rt[0][2]*inv_s.x;
    grad_R[7] = grad_S_inv_Rt[1][2]*inv_s.y;
    grad_R[8] = grad_S_inv_Rt[2][2]*inv_s.z;

    // # grad_mu    = -S_inv_Rt.T@(grad_o_g)
    // # grad_scale = -grad_S_inv_diag/scale.square()
    // grad_mu    = torch.zeros_like(mu)
    // grad_mu[0] = -(col0[0]*grad_o_g[0]*inv_s[0] + col1[0]*grad_o_g[1]*inv_s[1] + col2[0]*grad_o_g[2]*inv_s[2])
    // grad_mu[1] = -(col0[1]*grad_o_g[0]*inv_s[0] + col1[1]*grad_o_g[1]*inv_s[1] + col2[1]*grad_o_g[2]*inv_s[2])
    // grad_mu[2] = -(col0[2]*grad_o_g[0]*inv_s[0] + col1[2]*grad_o_g[1]*inv_s[1] + col2[2]*grad_o_g[2]*inv_s[2])
    // grad_scale = -grad_S_inv_diag/scale.square()
    const float3 temp_0 = grad_o_g*inv_s;
    ret.mu.x = -(col0.x*temp_0.x + col1.x*temp_0.y + col2.x*temp_0.z);
    ret.mu.y = -(col0.y*temp_0.x + col1.y*temp_0.y + col2.y*temp_0.z);
    ret.mu.z = -(col0.z*temp_0.x + col1.z*temp_0.y + col2.z*temp_0.z);
    ret.scale = -grad_S_inv_diag*inv_s*inv_s;

    // if (dot(ret.scale, ret.scale) > 100.0f)
    //     printf("scale grad_S_inv_diag:(%0.8f,%0.8f,%0.8f) inv_s:(%0.8f,%0.8f,%0.8f) grad_scale:(%0.8f,%0.8f,%0.8f)\n", 
    //         grad_S_inv_diag.x, grad_S_inv_diag.y, grad_S_inv_diag.z, 
    //         inv_s.x, inv_s.y, inv_s.z,
    //         ret.scale.x, ret.scale.y, ret.scale.z);

    // w, x, y, z = q_vec
    // grad_q_w = 2*(-z)*grad_R_flat[1] + 2*(y)*grad_R_flat[2] + 2*(z)*grad_R_flat[3] + \
    //            2*(-x)*grad_R_flat[5] + 2*(-y)*grad_R_flat[6] + 2*(x)*grad_R_flat[7]
    // grad_q_x = 2*(y)*grad_R_flat[1] + 2*(z)*grad_R_flat[2] + 2*(y)*grad_R_flat[3] + 4*(-x)*grad_R_flat[4] + \
    //            2*(-w)*grad_R_flat[5] + 2*(z)*grad_R_flat[6] + 2*(w)*grad_R_flat[7] + 4*(-x)*grad_R_flat[8]
    // grad_q_y = 4*(-y)*grad_R_flat[0] + 2*(x)*grad_R_flat[1] + 2*(w)*grad_R_flat[2] + 2*(x)*grad_R_flat[3] + \
    //            2*(z)*grad_R_flat[5] + 2*(-w)*grad_R_flat[6] + 2*(z)*grad_R_flat[7] + 4*(-y)*grad_R_flat[8] 
    // grad_q_z = 4*(-z)*grad_R_flat[0] + 2*(-w)*grad_R_flat[1] + 2*(x)*grad_R_flat[2] + 2*(w)*grad_R_flat[3] + \
    //            4*(-z)*grad_R_flat[4] + 2*(y)*grad_R_flat[5] + 2*(x)*grad_R_flat[6] + 2*(y)*grad_R_flat[7] 
    const float x = q_vec.x;
    const float y = q_vec.y;
    const float z = q_vec.z;
    const float w = q_vec.w;
    ret.q_vec.w = 2*(-z)*grad_R[1] + 2*(y)* grad_R[2] + 2*(z)*grad_R[3] + \
                  2*(-x)*grad_R[5] + 2*(-y)*grad_R[6] + 2*(x)*grad_R[7];
    ret.q_vec.x = 2*(y)* grad_R[1] + 2*(z)* grad_R[2] + 2*(y)*grad_R[3] + 4*(-x)*grad_R[4] + \
                  2*(-w)*grad_R[5] + 2*(z)* grad_R[6] + 2*(w)*grad_R[7] + 4*(-x)*grad_R[8];
    ret.q_vec.y = 4*(-y)*grad_R[0] + 2*(x)* grad_R[1] + 2*(w)*grad_R[2] + 2*(x)* grad_R[3] + \
                  2*(z)* grad_R[5] + 2*(-w)*grad_R[6] + 2*(z)*grad_R[7] + 4*(-y)*grad_R[8];
    ret.q_vec.z = 4*(-z)*grad_R[0] + 2*(-w)*grad_R[1] + 2*(x)*grad_R[2] + 2*(w)* grad_R[3] + \
                  4*(-z)*grad_R[4] + 2*(y)* grad_R[5] + 2*(x)*grad_R[6] + 2*(y)* grad_R[7];

    // grad_ro = S_inv_Rt.T@grad_o_g
    // grad_rd = S_inv_Rt.T@grad_d_g
    const float3 temp_1 = grad_o_g*inv_s;
    ret.ro.x = (col0.x*temp_1.x + col1.x*temp_1.y + col2.x*temp_1.z);
    ret.ro.y = (col0.y*temp_1.x + col1.y*temp_1.y + col2.y*temp_1.z);
    ret.ro.z = (col0.z*temp_1.x + col1.z*temp_1.y + col2.z*temp_1.z);
    const float3 temp_2 = grad_d_g*inv_s;
    ret.rd.x = (col0.x*temp_2.x + col1.x*temp_2.y + col2.x*temp_2.z);
    ret.rd.y = (col0.y*temp_2.x + col1.y*temp_2.y + col2.y*temp_2.z);
    ret.rd.z = (col0.z*temp_2.x + col1.z*temp_2.y + col2.z*temp_2.z);

    return ret;
}

GaussianGrad gaussian_particle_bwd(const float3& mu, const float3& scale, const float4& q_vec, const float opacity, const float3& ray_org, const float3& ray_dir, const float4& output, const float4& grad_output){
    const Rot3DMat rot_3d = quaterion2rotation(q_vec);

    return gaussian_particle_bwd(mu, scale, q_vec, rot_3d, opacity, ray_org, ray_dir, output, grad_output);
}
#endif
