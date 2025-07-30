// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#define OPTIXU_MATH_DEFINE_IN_NAMESPACE

#include <optix.h>
#include <math_constants.h>

#include "params.h"
#include "../common.h"
#include "../gaussian_math.h"

#define MIN_ROUGHNESS 0.08f

// #define HIT_BUFF_SIZE 8

template<typename T>
inline void swap(T &a, T &b){
    T temp = b;
    b = a;
    a = temp;
};

// inline float dot(const float3 a, const float3 b) {
//     float ret = static_cast<float>(0);
//     ret += a.x * b.x;
//     ret += a.y * b.y;
//     ret += a.z * b.z;
//     return ret;
// };

struct PerRayData 
{
    float        dist [HIT_BUFF_SIZE];
    // float        alpha[8];
    unsigned int index[HIT_BUFF_SIZE];

    // float        T;
    // unsigned int cdim;
    unsigned int length;

    unsigned int  isfull() {return length==HIT_BUFF_SIZE?1:0;};
    unsigned int isempty() {return length==0?1:0;};
    // bool   isdone() {return T<0.005?1:0;};
    void    clear() {length=0;};

};

// COPY from https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/rtigo3/shaders/per_ray_data.h#L113
typedef union
{
  PerRayData* ptr;
  uint2       dat;
} Payload;

__forceinline__ __device__ uint2 splitPointer(PerRayData* ptr)
{
  Payload payload;

  payload.ptr = ptr;

  return payload.dat;
}

__forceinline__ __device__ PerRayData* mergePointer(unsigned int p0, unsigned int p1)
{
  Payload payload;

  payload.dat.x = p0;
  payload.dat.y = p1;

  return payload.ptr;
}

extern "C" {
__constant__ GaussianTracingParams params;
}

//==============================================================================
// Optix kernels
//==============================================================================

extern "C" __global__ void __closesthit__ch()
{
    #if HIT_BUFF_SIZE == 1
    PerRayData&           pld = *(PerRayData*)mergePointer(optixGetPayload_0(), optixGetPayload_1());

    const unsigned int i_prim = optixGetPrimitiveIndex() / params.proxy_geometry_nfaces;
    const float         t_hit = optixGetRayTmax();  // In Any-Hit, this returns the current hitT

    float        h_t = t_hit;
    unsigned int h_i = i_prim;

    pld.dist[ 0] = h_t;
    pld.index[0] = h_i;
    pld.length   = 1;

    optixSetPayload_2(optixGetPayload_2()+1);
    #endif
}

extern "C" __global__ void __anyhit__ah()
{
    // if ( optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE )
    if ( optixIsBackFaceHit() ){
        optixIgnoreIntersection();
        return;
    }

    #if HIT_BUFF_SIZE > 1

    PerRayData&           pld = *(PerRayData*)mergePointer(optixGetPayload_0(), optixGetPayload_1());

    const unsigned int i_prim = optixGetPrimitiveIndex() / params.proxy_geometry_nfaces;
    const float         t_hit = optixGetRayTmax();  // In Any-Hit, this returns the current hitT

    float        h_t = t_hit;
    unsigned int h_i = i_prim;

    for (unsigned int i=0; i<pld.length; ++i){
        if (h_t < pld.dist[i]){
            // swap
            swap(h_t, pld.dist [i]);
            swap(h_i, pld.index[i]);
        }
    }

    if (!pld.isfull()){
        pld.dist[ pld.length] = h_t;
        pld.index[pld.length] = h_i;
        pld.length += 1;

        // ignore hits when the buffer is not full
        optixIgnoreIntersection();
    } else {
        // ignore k-closest hits to prevent the traversal from stopping
        if (t_hit < pld.dist[HIT_BUFF_SIZE-1])
            optixIgnoreIntersection();
    }

    #endif
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // skip empty pixel
    const unsigned int fwd_hit = __float_as_uint(params.out_dbg[idx.z][idx.y][idx.x][3]);
    if (fwd_hit == 0)
        return;

    // https://raytracing-docs.nvidia.com/optix8/guide/index.html#device_side_functions#13203

    // Read per-pixel constant input tensors, ray_origin, g-buffer entries etc.
    const float3 ray_org = fetch3(params.ro, idx.z, idx.y, idx.x);
    const float3 ray_dir = fetch3(params.rd, idx.z, idx.y, idx.x);

    float3 grad_ro = make_float3(0.0f, 0.0f, 0.0f);
    float3 grad_rd = make_float3(0.0f, 0.0f, 0.0f);

    PerRayData pld;

    uint2 payload = splitPointer(&pld);
    unsigned int hit, acc_hit = 0;

    // const float _t_min     = 0.001f;   // min ray distance
    // const float _t_max     = 1e4f;     // max ray distance
    // const float _T_min     = 0.005f;   // min Transmittance
    // const float _alpha_min = 0.01f;

    const float _t_min     = 0.001f;   // min ray distance
    const float _t_max     = 1e4f;     // max ray distance
    const float _T_min     = params.T_min;      // min Transmittance
    const float _alpha_min = params.alpha_min;

    float acc_g_w = 0.0f;

    // float d_L_o = 0.0f;
    // // ComputeGrad
    // for (unsigned int ci=0; ci<params.rgb.size(2); ++ci){
    //     d_L_o += params.out_rgb_grad[idx.y][idx.x][ci];
    // }

    const float acc_alp = params.out_buf[idx.z][idx.y][idx.x][0];
    const float acc_dep = params.out_buf[idx.z][idx.y][idx.x][4];
    const float acc_dep2= params.out_buf[idx.z][idx.y][idx.x][9];
    const float acc_ldd = params.out_buf[idx.z][idx.y][idx.x][10];

    const float grad_dep2= params.out_buf_grad[idx.z][idx.y][idx.x][9];
    const float grad_ldd = params.out_buf_grad[idx.z][idx.y][idx.x][10];

    // first pass: compute appearance grad: d_L/d_a[i] = d_L/d_i[y,x] * w[i]
    //             accumulate weight grad: G_acc = sum( d_L/d_w[i]*w[i] for i in range(N) )
    {
        float T      = 1.0f;
        float t_cur  = _t_min;
        unsigned int i_cur = 0;

        // float  acc_alp = 0.0f;
        // float3 acc_nrm = make_float3(0.0f);
        // float4 acc_pos = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        while (t_cur < _t_max && T > _T_min ){

            pld.clear();

            optixTrace(
                    params.handle,
                    ray_org,
                    ray_dir,
                    t_cur,               // Min intersection distance
                    1e16f,               // Max intersection distance
                    0.0f,                // rayTime -- used for motion blur
                    OptixVisibilityMask( 255 ), // Specify always visible
                    OPTIX_RAY_FLAG_NONE | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                    0,                   // SBT offset   -- See SBT discussion
                    1,                   // SBT stride   -- See SBT discussion
                    0,                   // missSBTIndex -- See SBT discussion
                    payload.x, payload.y, hit );

            for (unsigned int i=0; i<pld.length; ++i){
                // GetParticleIndex
                unsigned int primID = pld.index[i];

                // ComputeResponse
                float3 mu    = {params.xyz[primID][0], params.xyz[primID][1], params.xyz[primID][2]};
                float3 scale = {params.sca[primID][0], params.sca[primID][1], params.sca[primID][2]};
                float3 normal= {params.nrm[primID][0], params.nrm[primID][1], params.nrm[primID][2]};
                // float4 q  = {params.rot[primID][1], params.rot[primID][2], params.rot[primID][3],  params.rot[primID][0]};
                float opacity= params.opa[primID][0];            // TODO: compute alpha

                float4 q_vec = {params.rot[primID][1], params.rot[primID][2], params.rot[primID][3],  params.rot[primID][0]};

                float4 gau_fwd_ret = gaussian_particle_fwd(mu, scale, q_vec, opacity, ray_org, ray_dir);
                float        t_max = gau_fwd_ret.x;
                float        alpha = gau_fwd_ret.w;
                // float          pho = gau_fwd_ret.w;
                // float        alpha = opacity * pho;

                if (alpha > _alpha_min){
                    const float weight    = T*alpha;

                    // ComputeGrad
                    for (unsigned int ci=0; ci<params.rgb.size(2); ++ci){
                        atomicAdd(&params.rgb_grad[primID][ci], params.out_rgb_grad[idx.z][idx.y][idx.x][ci] * weight);
                    }
                    atomicAdd(&params.nrm_grad[primID][0], params.out_buf_grad[idx.z][idx.y][idx.x][1] * weight);
                    atomicAdd(&params.nrm_grad[primID][1], params.out_buf_grad[idx.z][idx.y][idx.x][2] * weight);
                    atomicAdd(&params.nrm_grad[primID][2], params.out_buf_grad[idx.z][idx.y][idx.x][3] * weight);
                    atomicAdd(&params.xyz_grad[primID][0], params.out_buf_grad[idx.z][idx.y][idx.x][5] * weight);
                    atomicAdd(&params.xyz_grad[primID][1], params.out_buf_grad[idx.z][idx.y][idx.x][6] * weight);
                    atomicAdd(&params.xyz_grad[primID][2], params.out_buf_grad[idx.z][idx.y][idx.x][7] * weight);

                    // accumulated grad*weight
                    float grad_w = 0.0f;  // d_L/d_w[i] = sum( out[c]*inp[c] for c in range(C) )
                    for (unsigned int ci=0; ci<params.rgb.size(2); ++ci){
                        grad_w += (params.out_rgb_grad[idx.z][idx.y][idx.x][ci] * params.rgb[primID][ci]);
                    }
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][0]);            // alpha
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][1] * normal.x); // normal
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][2] * normal.y);
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][3] * normal.z);
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][4] * t_max);    // depth
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][5] * mu.x);     // position
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][6] * mu.y);
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][7] * mu.z);

                    // T, channel index 8
                    // depth*depth, channel index 9
                    const float t_2 = pow(t_max, 2.0f);
                    grad_w += (grad_dep2 * t_2); // t*t
                    // depth distortion, channel index 10
                    grad_w += (grad_ldd  * ( t_2*(acc_alp - weight) - 2*t_max*(acc_dep - t_max*weight) + (acc_dep2 - t_2*weight) ) ); // depth distortion

                    acc_g_w += grad_w * weight;
                    // acc_alp += weight;
                    // acc_nrm += weight*normal;
                    // acc_pos += weight*make_float4(mu.x, mu.y, mu.z, t_max);
                    // printf("loop1 [%d] weight=%f, grad_w=%f\n", primID, weight, grad_w);

                    T = T * (1-alpha);
                }

                t_cur = pld.dist[i];
                i_cur = pld.index[i];
            }

            acc_hit += pld.length;
            if (pld.length == 0)
                break;
        }

        // if (acc_hit != fwd_hit){
        //     printf("[%d,%d] acc_hit=%d, len=%d, T=%f\n", idx.y, idx.x, (fwd_hit - acc_hit), pld.length, T);
        // }
    }

    const float grad_T = params.out_buf_grad[idx.z][idx.y][idx.x][8];
    const float out_T  = params.out_buf[idx.z][idx.y][idx.x][8];

    acc_hit = 0;
    // second pass: compute d_L/d_alpha[i] = sum( d_L/d_w[j] * d_w[j]/d_alpha[i] for j in range(N) )
    //                                     = (d_L/d_w[i])*(d_w[i]/alpha[i]) + 1/(alpha[i]-1)*sum( d_L/d_w[j]*w_[j] for j in range(i+1, N) )
    //                                     = (d_L/d_w[i])*(d_w[i]/alpha[i]) + 1/(alpha[i]-1)*(G_acc - d_L/d_w[i]*w[i] ) );      G_acc -= d_L/d_w[i]*w[i]
    {
        float T      = 1.0f;
        float t_cur  = _t_min;
        unsigned int i_cur = 0;

        // float bwd_acc_alp = 0.0f;
        // float bwd_acc_dep = 0.0f;

        while (t_cur < _t_max && T > _T_min ){

            pld.clear();
            hit = 0;

            optixTrace(
                    params.handle,
                    ray_org,
                    ray_dir,
                    t_cur,               // Min intersection distance
                    1e16f,               // Max intersection distance
                    0.0f,                // rayTime -- used for motion blur
                    OptixVisibilityMask( 255 ), // Specify always visible
                    OPTIX_RAY_FLAG_NONE | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                    0,                   // SBT offset   -- See SBT discussion
                    1,                   // SBT stride   -- See SBT discussion
                    0,                   // missSBTIndex -- See SBT discussion
                    payload.x, payload.y, hit );

            for (unsigned int i=0; i<pld.length; ++i){
                // GetParticleIndex
                unsigned int primID = pld.index[i];

                // ComputeResponse
                float3 mu    = {params.xyz[primID][0], params.xyz[primID][1], params.xyz[primID][2]};
                float3 scale = {params.sca[primID][0], params.sca[primID][1], params.sca[primID][2]};
                float3 normal= {params.nrm[primID][0], params.nrm[primID][1], params.nrm[primID][2]};
                // float4 q  = {params.rot[primID][1], params.rot[primID][2], params.rot[primID][3],  params.rot[primID][0]};
                float opacity= params.opa[primID][0];            // TODO: compute alpha

                float4 q_vec = {params.rot[primID][1], params.rot[primID][2], params.rot[primID][3],  params.rot[primID][0]};  // 3DGS order wxyz -> xyzw

                float4 gau_fwd_ret = gaussian_particle_fwd(mu, scale, q_vec, opacity, ray_org, ray_dir);
                float        t_max = gau_fwd_ret.x;
                float        alpha = gau_fwd_ret.w;
                // float          pho = gau_fwd_ret.w;
                // float        alpha = opacity * pho;

                if (alpha > _alpha_min){
                    const float weight    = T*alpha;

                    // bwd_acc_alp += weight;
                    // bwd_acc_dep += t_max*weight;

                    // ComputeGrad
                    float grad_w = 0.0f;  // d_L/d_w[i] = sum( out[c]*inp[c] for c in range(C) )

                    for (unsigned int ci=0; ci<params.rgb.size(2); ++ci){
                        grad_w += (params.out_rgb_grad[idx.z][idx.y][idx.x][ci] * params.rgb[primID][ci]);
                    }
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][0]);            // alpha
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][1] * normal.x); // normal
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][2] * normal.y);
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][3] * normal.z);
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][4] * t_max);    // depth
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][5] * mu.x);     // position
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][6] * mu.y);
                    grad_w += (params.out_buf_grad[idx.z][idx.y][idx.x][7] * mu.z);

                    // T, channel index 8
                    // depth*depth, channel index 9
                    const float t_2      = pow(t_max, 2.0f);
                    grad_w += (grad_dep2 * t_2); // t*t
                    // depth distortion, channel index 10
                    grad_w += (grad_ldd  * ( t_2*(acc_alp - weight) - 2*t_max*(acc_dep - t_max*weight) + (acc_dep2 - t_2*weight) ) ); // depth distortion

                    float grad_alpha = grad_w*T + (acc_g_w - grad_w*weight)/(alpha-1);       // gradient from weights
                    grad_alpha += grad_T*out_T/(alpha-1);                                    // gradient from T

                    float4 gau_out_grad;
                    gau_out_grad.x = params.out_buf_grad[idx.z][idx.y][idx.x][4]*weight;  // grad_t from accumulate depth
                    gau_out_grad.x+= 2*grad_dep2*weight*t_max;                            // grad_t from accumulate depth*depth
                    gau_out_grad.x+= 2*grad_ldd *weight*(t_max*acc_alp - acc_dep); // grad_t from depth distortion
                    gau_out_grad.w = grad_alpha;


                    GaussianGrad grad = gaussian_particle_bwd(mu, scale, q_vec, opacity, ray_org, ray_dir, gau_fwd_ret, gau_out_grad);

                    // atomicadd
                    atomicAdd(&params.xyz_grad[primID][0], grad.mu.x);
                    atomicAdd(&params.xyz_grad[primID][1], grad.mu.y);
                    atomicAdd(&params.xyz_grad[primID][2], grad.mu.z);

                    atomicAdd(&params.sca_grad[primID][0], grad.scale.x);
                    atomicAdd(&params.sca_grad[primID][1], grad.scale.y);
                    atomicAdd(&params.sca_grad[primID][2], grad.scale.z);

                    atomicAdd(&params.rot_grad[primID][0], grad.q_vec.w);
                    atomicAdd(&params.rot_grad[primID][1], grad.q_vec.x);
                    atomicAdd(&params.rot_grad[primID][2], grad.q_vec.y);
                    atomicAdd(&params.rot_grad[primID][3], grad.q_vec.z);

                    atomicAdd(&params.opa_grad[primID][0], grad.opacity);

                    // normal accumulate
                    grad_ro += grad.ro;
                    grad_rd += grad.rd;

                    // printf("loop2 [%d] weight=%f, grad_t=%f, grad_w=%f\n", primID, weight, gau_out_grad.x, grad_w);

                    acc_g_w -= grad_w * weight;
                    T = T * (1-alpha);
                }

                t_cur = pld.dist[i];
                i_cur = pld.index[i];
            }

            acc_hit += pld.length;
            if (pld.length == 0)
                break;
        }

        // if (acc_hit != fwd_hit){
        //     printf("[%d,%d] acc_hit=%d, len=%d, T=%f\n", idx.y, idx.x, (fwd_hit - acc_hit), pld.length, T);
        // }
    }

    // printf("accumulated grad_weight %f\n", acc_g_w);

    params.ro_grad[idx.z][idx.y][idx.x][0] = grad_ro.x;
    params.ro_grad[idx.z][idx.y][idx.x][1] = grad_ro.y;
    params.ro_grad[idx.z][idx.y][idx.x][2] = grad_ro.z;

    params.rd_grad[idx.z][idx.y][idx.x][0] = grad_rd.x;
    params.rd_grad[idx.z][idx.y][idx.x][1] = grad_rd.y;
    params.rd_grad[idx.z][idx.y][idx.x][2] = grad_rd.z;

    // // alpha:1
    // params.out_buf[idx.y][idx.x][0] = acc_alp;
    // // normal:3
    // params.out_buf[idx.y][idx.x][1] = acc_nrm.x;
    // params.out_buf[idx.y][idx.x][2] = acc_nrm.y;
    // params.out_buf[idx.y][idx.x][3] = acc_nrm.z;
    // // depth:1, position:3
    // params.out_buf[idx.y][idx.x][4] = acc_pos.w;
    // params.out_buf[idx.y][idx.x][5] = acc_pos.x;
    // params.out_buf[idx.y][idx.x][6] = acc_pos.y;
    // params.out_buf[idx.y][idx.x][7] = acc_pos.z;

    // params.out_dbg[idx.y][idx.x][0] = T;
    // params.out_dbg[idx.y][idx.x][1] = t_cur;
    // params.out_dbg[idx.y][idx.x][2] = __uint_as_float(i_cur);
    // params.out_dbg[idx.y][idx.x][3] = __uint_as_float(acc_hit);
}

extern "C" __global__ void __miss__ms()
{
    // optixSetPayload_2(0);
}