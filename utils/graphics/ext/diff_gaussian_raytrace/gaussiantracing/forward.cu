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

    // void    insert(float d, unsigned int i){

    //     // float tmp_d=d, tmp_a=a;
    //     // unsigned int tmp_i = i;

    //     for (unsigned int i=0; i<length; ++i){
    //         if (d < dist[i]){
    //             // swap

    //             swap(d, dist [i]);
    //             swap(i, index[i]);
    //             // swap(a, alpha[i]);
    //         }
    //     }

    //     dist[length]  = d;
    //     index[length] = i;
    //     // alpha[length] = a;
    //     length += 1;
    // };
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

    // the following is not correct; it will just stop when the buffer length first reach K
    // if (!pld.isfull()){
    //     pld.dist[ pld.length] = h_t;
    //     pld.index[pld.length] = h_i;
    //     pld.length += 1;
    // }

    // // ignore 𝑘-closest hits to prevent the traversal from stopping
    // if ((!pld.isfull()) || t_hit < pld.dist[pld.length-1])
    //     optixIgnoreIntersection();

    #endif
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // if (idx.x == 0 && idx.y == 0){
    //   // we could of course also have used optixGetLaunchDims to query
    //   // the launch size, but accessing the optixLaunchParams here
    //   // makes sure they're not getting optimized away (because
    //   // otherwise they'd not get used)
    //   printf("############################################\n");
    //   printf("Hello world from OptiX 7 raygen program!\n");
    //   printf("############################################\n");
    // }

    // https://raytracing-docs.nvidia.com/optix8/guide/index.html#device_side_functions#13203

    // Read per-pixel constant input tensors, ray_origin, g-buffer entries etc.
    const float3 ray_org = fetch3(params.ro, idx.z, idx.y, idx.x);
    const float3 ray_dir = fetch3(params.rd, idx.z, idx.y, idx.x);

    PerRayData pld;

    // pld.T      = 1;
    // pld.length = 0;
    // pld.cdim   = params.rgb.size(1);

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

    float T      = 1.0f;
    float t_cur  = _t_min;
    unsigned int i_cur = 0;

    float  acc_alp = 0.0f;
    float3 acc_nrm = make_float3(0.0f);
    float4 acc_pos = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float  acc_ldd = 0.0f;
    float  acc_t_2 = 0.0f;

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
        
        // if (hit != 0)
        //     printf("%d [%f %f %f %f %f %f %f %f]\n", pld.length, pld.dist[0], pld.dist[1], pld.dist[2], pld.dist[3], pld.dist[4], pld.dist[5], pld.dist[6], pld.dist[7]);

        for (unsigned int i=0; i<pld.length; ++i){
            // GetParticleIndex
            unsigned int primID = pld.index[i];

            // ComputeResponse
            float3 mu    = {params.xyz[primID][0], params.xyz[primID][1], params.xyz[primID][2]};
            float3 scale = {params.sca[primID][0], params.sca[primID][1], params.sca[primID][2]};
            float3 normal= {params.nrm[primID][0], params.nrm[primID][1], params.nrm[primID][2]};
            float opacity= params.opa[primID][0];            // TODO: compute alpha

            float4 q_vec = {params.rot[primID][1], params.rot[primID][2], params.rot[primID][3],  params.rot[primID][0]};

            float4 gau_fwd_ret = gaussian_particle_fwd(mu, scale, q_vec, opacity, ray_org, ray_dir);
            float        t_max = gau_fwd_ret.x;
            float        alpha = gau_fwd_ret.w;
            // float        alpha = opacity * pho;

            if (alpha > _alpha_min){
                const float weight    = T*alpha;

                // ComputeRadiance
                for (unsigned int ci=0; ci<params.rgb.size(2); ++ci){
                    params.out_rgb[idx.z][idx.y][idx.x][ci] += weight*params.rgb[primID][ci];
                }

                // T
                acc_ldd += weight*(pow(t_max, 2.0f)*acc_alp - 2*t_max*acc_pos.w + acc_t_2); // compute depth distortion first!!!
                acc_alp += weight; // weight*1
                acc_nrm += weight*normal;
                acc_pos += weight*make_float4(mu.x, mu.y, mu.z, t_max);
                acc_t_2 += weight*pow(t_max, 2.0f);

                T = T * (1-alpha);
            }

            t_cur = pld.dist[i];
            i_cur = pld.index[i];
            // t_cur = t_max;
        }
        
        // if (idx.y == 256 && idx.x == 256){
        // if (idx.y == 289 && idx.x == 259){
        // if (idx.y == 328 && idx.x == 226){
        // if (idx.y == 324 && idx.x == 240 ){
        // if ( 1 ){
        // #if HIT_BUFF_SIZE == 1
        //     printf("%f %d %d\n", pld.dist[0], pld.index[0], hit);
        // #else
        //     for (unsigned int i=0; i<pld.length; ++i)
        //         printf("forward (%f, %d), ", pld.dist[i], pld.index[i]);
        //     // printf("\n");
        //     printf("T=%f t_last=%f i_last=%d\n", T, t_cur, i_cur);
        // #endif
        // }

        acc_hit += pld.length;
        // acc_hit += hit;
        // if (hit == 0 || !pld.isfull())
        // if (hit == 0)
        if (pld.length == 0)
            break;
    }

    // if (acc_hit != 0){
    //     acc_nrm = acc_nrm / dot(acc_nrm, acc_nrm);
    // }

    // alpha:1
    params.out_buf[idx.z][idx.y][idx.x][0] = acc_alp;
    // normal:3
    params.out_buf[idx.z][idx.y][idx.x][1] = acc_nrm.x;
    params.out_buf[idx.z][idx.y][idx.x][2] = acc_nrm.y;
    params.out_buf[idx.z][idx.y][idx.x][3] = acc_nrm.z;
    // depth:1, position:3, T:1, depth*depth:1, 
    params.out_buf[idx.z][idx.y][idx.x][4] = acc_pos.w;
    params.out_buf[idx.z][idx.y][idx.x][5] = acc_pos.x;
    params.out_buf[idx.z][idx.y][idx.x][6] = acc_pos.y;
    params.out_buf[idx.z][idx.y][idx.x][7] = acc_pos.z;
    params.out_buf[idx.z][idx.y][idx.x][8] = T;
    params.out_buf[idx.z][idx.y][idx.x][9] = acc_t_2;
    params.out_buf[idx.z][idx.y][idx.x][10] = acc_ldd;

    params.out_dbg[idx.z][idx.y][idx.x][0] = T;
    params.out_dbg[idx.z][idx.y][idx.x][1] = t_cur;
    params.out_dbg[idx.z][idx.y][idx.x][2] = __uint_as_float(i_cur);
    params.out_dbg[idx.z][idx.y][idx.x][3] = __uint_as_float(acc_hit);
}

extern "C" __global__ void __miss__ms()
{
    // optixSetPayload_2(0);
}