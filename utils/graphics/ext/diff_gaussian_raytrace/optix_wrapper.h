// COPY from https://github.com/NVlabs/nvdiffrecmc/blob/main/render/optixutils/c_src/optix_wrapper.h
// Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include <optix.h>
#include <string>

//------------------------------------------------------------------------
// Python OptiX state wrapper.

struct OptiXState
{
    OptixDeviceContext     context;
    OptixTraversableHandle gas_handle;
    CUdeviceptr            d_gas_output_buffer;

    // OptixPipeline pipelineEnvSampling;
    // OptixShaderBindingTable sbtEnvSampling;
    // OptixModule moduleEnvSampling;

    // Differentiable gaussian ray-tracing
    OptixPipeline           fwd_ppl;
    OptixShaderBindingTable fwd_sbt;
    OptixModule             fwd_mod;

    OptixPipeline           bwd_ppl;
    OptixShaderBindingTable bwd_sbt;
    OptixModule             bwd_mod;

    // Differentiable env sampling
    OptixPipeline           shd_ppl;
    OptixShaderBindingTable shd_sbt;
    OptixModule             shd_mod;
};


class OptiXStateWrapper
{
public:
    OptiXStateWrapper     (const std::string &path, const std::string &cuda_path);
    ~OptiXStateWrapper    (void);
    
    OptiXState*           pState;
};
