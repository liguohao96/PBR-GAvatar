#include <torch/types.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include "utils.h"

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
namespace raytrace{
// start of namespace

// #define max(a,b) \
//    ({ __typeof__ (a) _a = (a); \
//        __typeof__ (b) _b = (b); \
//      _a > _b ? _a : _b; })

// #define min(a,b) \
//    ({ __typeof__ (a) _a = (a); \
//        __typeof__ (b) _b = (b); \
//      _a < _b ? _a : _b; })

#define min3(a,b,c) (min(min(a,b), c))
#define max3(a,b,c) (max(max(a,b), c))

#define USE_SHARED

template <typename scalar_t, typename index_t, const int THREAD_W>
__global__ void primary_ray_forward_cuda_impl(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> faces,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> query,
          torch::PackedTensorAccessor32<index_t,  2, torch::RestrictPtrTraits> i_buf,
          torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_buf,
          torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> p_buf,
    const bool culling,
    const bool negdist
){
    const int BATCH_SIZE = faces.size(0);
    const int NUM_FACES  = faces.size(1);
    const int NUM_QUERY  = query.size(1);

    const int bx  = blockIdx.x,  by = blockIdx.y, bz = blockIdx.z;
    const int tx  = threadIdx.x, ty = threadIdx.y;

    const int bi = bz;
    const int qi = bx*THREAD_W + tx;

    __shared__ vec3<scalar_t> local_v33[THREAD_W][3];

    const bool qi_in_range = qi < NUM_QUERY;

    index_t        tid;
    scalar_t       dis;
    vec3<scalar_t> dir, org, xyz, uvw;

    if(qi_in_range){
        org.x = query[bi][qi][0];
        org.y = query[bi][qi][1];
        org.z = query[bi][qi][2];
        dir.x = query[bi][qi][3];
        dir.y = query[bi][qi][4];
        dir.z = query[bi][qi][5];
    }

    const int fi_step   = THREAD_W;
    const int fi_offset = tx;

    tid   = -1;
    xyz.x = static_cast<scalar_t>(INFINITY);
    xyz.y = static_cast<scalar_t>(INFINITY);
    xyz.z = static_cast<scalar_t>(INFINITY);
    dis   = static_cast<scalar_t>(INFINITY);

    for(int fi=0; fi<NUM_FACES; fi+=fi_step){
        bool in_range = (fi + fi_offset) < NUM_FACES;

        if(in_range){
            for(int i=0; i<3; ++i){
                // local_v33[fi_offset][i].x = faces[b_i*NUM_FACES*9 + (fi+fi_offset)*9 + i*3 + 0];
                // local_v33[fi_offset][i].y = faces[b_i*NUM_FACES*9 + (fi+fi_offset)*9 + i*3 + 1];
                // local_v33[fi_offset][i].z = faces[b_i*NUM_FACES*9 + (fi+fi_offset)*9 + i*3 + 2];
                local_v33[fi_offset][i].x = faces[bi][fi+fi_offset][i][0];
                local_v33[fi_offset][i].y = faces[bi][fi+fi_offset][i][1];
                local_v33[fi_offset][i].z = faces[bi][fi+fi_offset][i][2];
            }
        }
            
        __syncthreads();

        const int lt_len = min(fi_step, NUM_FACES-fi);

        for(int lti=0; lti<lt_len; ++lti){
            bool skip = false;

            vec3<scalar_t> v0 = local_v33[lti][0];
            vec3<scalar_t> v1 = local_v33[lti][1];
            vec3<scalar_t> v2 = local_v33[lti][2];
                
            vec3<scalar_t> v0v1 = v1 - v0;
            vec3<scalar_t> v0v2 = v2 - v0;

            vec3<scalar_t> pvec = dir.cross(v0v2);
            scalar_t det        = v0v1.dot(pvec);
            scalar_t invDet     = 1 / det;

            skip |= (culling && det < 0);  // back-face culling
            skip |= (abs(det) <= 1e-8);

            vec3<scalar_t> tvec = org - v0;

            scalar_t u, v, d;
            u = tvec.dot(pvec) * invDet;
            vec3<scalar_t> qvec = tvec.cross(v0v1);
            v = dir.dot(qvec) * invDet;

            skip |= (u < 0 || u > 1);
            skip |= (v < 0 || v > 1);
            skip |= (u + v > 1);

            d = v0v2.dot(qvec) * invDet;
            skip |= (~negdist && d < 0);  // skip_negdist = negdist ? (False) : (d<0)

            if(!skip && d < dis){
                dis = d;
                tid = fi + lti;

                uvw.x = u;
                uvw.y = v;
            }

        }
    }

    if(qi_in_range){
        xyz = dir * dis + org;

        i_buf[bi][qi]    = tid;
        d_buf[bi][qi]    = dis;
        // p_buf[bi][0][qi] = xyz.x;
        // p_buf[bi][1][qi] = xyz.y;
        p_buf[bi][0][qi] = (1 - uvw.x - uvw.y);
        p_buf[bi][1][qi] = uvw.x;
    }
}

// Moller-Trumbore algorithm
// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html

std::vector<torch::Tensor> primary_ray_forward_cuda(
    torch::Tensor faces,
    torch::Tensor prays,
    bool culling,
    bool negdist
){
    cudaError_t err;
    const unsigned int  batch_size = faces.size(0);
    const unsigned int  num_tri    = faces.size(1);
    const unsigned int  NQ         = prays.size(1);

    auto tri_ind_map_b   = torch::full({batch_size, NQ},
                                     static_cast<int64_t>(-1),
                                     faces.options().dtype(torch::kInt64));
    // float min_value      = __half2float(std::numeric_limits<half>::lowest());
    float max_value      = 1;
    auto depth_map       = torch::full({batch_size, NQ}, max_value, faces.options());
    auto point_map       = torch::empty({batch_size, 2, NQ}, faces.options());

    const unsigned int THREAD_W = 32;
    const unsigned int THREAD_H = 1;
    // const unsigned int THREAD_Z = 16;

    const unsigned int BLOCK_W  = (NQ%THREAD_W == 0) ? (NQ/THREAD_W) : (NQ/THREAD_W+1);

    const dim3 block_conf{THREAD_W, THREAD_H};

    const dim3 grid_conf {BLOCK_W, 1, batch_size};


    err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Error before forward_ray_cuda_kernel: %s\n", cudaGetErrorString(err));
        return {tri_ind_map_b, depth_map, point_map};
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(faces.scalar_type(), "forward_zbuffer_cuda_kernel", [&]{

        primary_ray_forward_cuda_impl<scalar_t, int64_t, THREAD_W>
        <<<grid_conf, block_conf>>>(
            faces.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            prays.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),

            tri_ind_map_b.packed_accessor32<int64_t,  2, torch::RestrictPtrTraits>(),
            depth_map.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            point_map.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            culling, negdist
        );

    });

    err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Error after forward_ray_cuda_kernel: %s\n", cudaGetErrorString(err));
        return {tri_ind_map_b, depth_map, point_map};
    }

    return {tri_ind_map_b, depth_map, point_map};
}

// end of namespace
}