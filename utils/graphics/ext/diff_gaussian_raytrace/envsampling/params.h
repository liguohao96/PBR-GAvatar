// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "../accessor.h"

#define HIT_BUFF_SIZE 8

struct EnvSamplingParams
{
    // Ray data
    PackedTensorAccessor32<float, 4>    ro;             // ray origin

    // Gaussian Geometry
    PackedTensorAccessor32<float, 2>    xyz;
    PackedTensorAccessor32<float, 2>    rot;
    PackedTensorAccessor32<float, 2>    sca;
    PackedTensorAccessor32<float, 2>    opa;
    PackedTensorAccessor32<float, 2>    xyz_grad;
    PackedTensorAccessor32<float, 2>    rot_grad;
    PackedTensorAccessor32<float, 2>    sca_grad;
    PackedTensorAccessor32<float, 2>    opa_grad;
    
    // GBuffer
    PackedTensorAccessor32<float, 3>    mask;
    PackedTensorAccessor32<float, 4>    gb_pos;
    PackedTensorAccessor32<float, 4>    gb_pos_grad;
    PackedTensorAccessor32<float, 4>    gb_normal;
    PackedTensorAccessor32<float, 4>    gb_normal_grad;
    PackedTensorAccessor32<float, 4>    gb_view_pos;
    PackedTensorAccessor32<float, 4>    gb_kd;
    PackedTensorAccessor32<float, 4>    gb_kd_grad;
    PackedTensorAccessor32<float, 4>    gb_ks;
    PackedTensorAccessor32<float, 4>    gb_ks_grad;
    
    // Light
    PackedTensorAccessor32<float, 3>    light;
    PackedTensorAccessor32<float, 3>    light_grad;
    PackedTensorAccessor32<float, 2>    pdf;        // light pdf
    PackedTensorAccessor32<float, 1>    rows;       // light sampling cdf
    PackedTensorAccessor32<float, 2>    cols;       // light sampling cdf

    // Output
    PackedTensorAccessor32<float, 4>    diff;
    PackedTensorAccessor32<float, 4>    diff_grad;
    PackedTensorAccessor32<float, 4>    spec;
    PackedTensorAccessor32<float, 4>    spec_grad;
    // PackedTensorAccessor32<float, 5>    dbg;

    // Table with random permutations for stratified sampling
    PackedTensorAccessor32<int, 2>      perms;

    OptixTraversableHandle              handle;
    unsigned int                        BSDF;
    unsigned int                        n_samples_x;
    unsigned int                        rnd_seed;
    unsigned int                        backward;
    float                               shadow_scale;
    float                               T_min;
    float                               alpha_min;
    unsigned int                        proxy_geometry_nfaces;
};

