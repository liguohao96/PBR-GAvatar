// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "../accessor.h"

#define HIT_BUFF_SIZE 8

struct GaussianTracingParams
{
    // Ray data
    PackedTensorAccessor32<float, 4>    ro;             // ray origin,     B,H,W,3
    PackedTensorAccessor32<float, 4>    rd;             // ray direction
    PackedTensorAccessor32<float, 4>    ro_grad;
    PackedTensorAccessor32<float, 4>    rd_grad;
    
    // Gaussian Geometry
    PackedTensorAccessor32<float, 2>    xyz;
    PackedTensorAccessor32<float, 2>    rot;
    PackedTensorAccessor32<float, 2>    sca;
    PackedTensorAccessor32<float, 2>    opa;
    PackedTensorAccessor32<float, 2>    xyz_grad;
    PackedTensorAccessor32<float, 2>    rot_grad;
    PackedTensorAccessor32<float, 2>    sca_grad;
    PackedTensorAccessor32<float, 2>    opa_grad;

    // Gaussian Appearance
    PackedTensorAccessor32<float, 2>    rgb;
    PackedTensorAccessor32<float, 2>    nrm;
    PackedTensorAccessor32<float, 2>    rgb_grad;
    PackedTensorAccessor32<float, 2>    nrm_grad;


    // Proxy Geometry
    PackedTensorAccessor32<float, 3>    proxy_ver;
    PackedTensorAccessor32<int32_t, 3>  proxy_tri;
    
    // // Light
    // PackedTensorAccessor32<float, 3>    light;
    // PackedTensorAccessor32<float, 3>    light_grad;
    // PackedTensorAccessor32<float, 2>    pdf;        // light pdf
    // PackedTensorAccessor32<float, 1>    rows;       // light sampling cdf
    // PackedTensorAccessor32<float, 2>    cols;       // light sampling cdf

    // Output
    PackedTensorAccessor32<float, 4>    out_rgb;
    PackedTensorAccessor32<float, 4>    out_buf;
    PackedTensorAccessor32<float, 4>    out_dbg;

    PackedTensorAccessor32<float, 4>    out_rgb_grad;
    PackedTensorAccessor32<float, 4>    out_buf_grad;

    unsigned int                        backward;

    float                               T_min;
    float                               alpha_min;
    unsigned int                        proxy_geometry_nfaces;

    OptixTraversableHandle              handle;
};

