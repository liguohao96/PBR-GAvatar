#include <vector>
#include <stdio.h>
#include <algorithm>
#include <string>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

#include <optix_stubs.h>

#include "common.h"
#include "optix_wrapper.h"
#include "envsampling/params.h"
#include "gaussiantracing/params.h"
#include "denoising.h"
// #include "accessor.h"

//------------------------------------------------------------------------
// CUDA kernels

void bilateral_denoiser_fwd_kernel(BilateralDenoiserParams params);
void bilateral_denoiser_bwd_kernel(BilateralDenoiserParams params);

template<class T, int N, template <typename U> class PtrTraits = DefaultPtrTraits> PackedTensorAccessor32<T, N> packed_accessor32(torch::Tensor tensor)
{
    return PackedTensorAccessor32<T,N,PtrTraits>(static_cast<typename PtrTraits<T>::PtrType>(tensor.data_ptr<T>()), tensor.sizes().data(), tensor.strides().data());
}

//------------------------------------------------------------------------
// OptiX tracer

namespace raytrace{

void optix_build_bvh(OptiXStateWrapper& stateWrapper,torch::Tensor grid_verts, torch::Tensor grid_tris, unsigned int rebuild)
{
        //
    // accel handling
    //

    // Clear BVH GPU memory
    {
        // Use default options for simplicity.  In a real use case we would want to
        // enable compaction, etc
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;

        if (rebuild > 0)
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( stateWrapper.pState->d_gas_output_buffer ) ) ); 
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
        }
        else 
        {
            accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
        }
        CUdeviceptr d_vertices = (CUdeviceptr)grid_verts.data_ptr<float>();
        CUdeviceptr d_indices = (CUdeviceptr)grid_tris.data_ptr<int>();

        // Our build input is a simple list of non-indexed triangle vertices
        // const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE};
        const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
        // const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE | OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
        OptixBuildInput triangle_input = {};
        triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices   = (uint32_t)grid_verts.size(0);
        triangle_input.triangleArray.vertexBuffers = &d_vertices;
        triangle_input.triangleArray.indexFormat   = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.numIndexTriplets = (uint32_t)grid_tris.size(0);
        triangle_input.triangleArray.indexBuffer   = d_indices;
        triangle_input.triangleArray.flags         = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage(
                    stateWrapper.pState->context,
                    &accel_options,
                    &triangle_input,
                    1, // Number of build inputs
                    &gas_buffer_sizes
                    ) );
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &d_temp_buffer_gas ),
                    gas_buffer_sizes.tempSizeInBytes
                    ) );

        if (rebuild > 0)
        {
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &stateWrapper.pState->d_gas_output_buffer ),
                        gas_buffer_sizes.outputSizeInBytes
                        ) );
        }

        OPTIX_CHECK( optixAccelBuild(
                    stateWrapper.pState->context,
                    0,                  // CUDA stream
                    &accel_options,
                    &triangle_input,
                    1,                  // num build inputs
                    d_temp_buffer_gas,
                    gas_buffer_sizes.tempSizeInBytes,
                    stateWrapper.pState->d_gas_output_buffer,
                    gas_buffer_sizes.outputSizeInBytes,
                    &stateWrapper.pState->gas_handle,
                    nullptr,            // emitted property list
                    0                   // num emitted properties
                    ) );

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
    }
}

std::vector<torch::Tensor> gaussian_raytrace_fwd(
    OptiXStateWrapper& stateWrapper, 
    torch::Tensor xyz, 
    torch::Tensor rot, 
    torch::Tensor sca, 
    torch::Tensor opa, 

    torch::Tensor rgb, 
    torch::Tensor nrm, 

    torch::Tensor org,   // N,3
    torch::Tensor dir,   // N,3

    torch::Tensor proxy_ver, 
    torch::Tensor proxy_tri,

    float T_min,
    float alpha_min
    )
{
    //
    // launch OptiX kernel
    //
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out_rgb = torch::zeros({ dir.size(0), dir.size(1), dir.size(2), rgb.size(1) }, opts) ;
    torch::Tensor out_buf = torch::zeros({ dir.size(0), dir.size(1), dir.size(2), 11 }, opts) ;
    torch::Tensor out_dbg = torch::zeros({ dir.size(0), dir.size(1), dir.size(2), 4 }, opts) ;

    GaussianTracingParams params;
    params.handle       = stateWrapper.pState->gas_handle;

    params.xyz = packed_accessor32<float, 2>(xyz);
    params.rot = packed_accessor32<float, 2>(rot);
    params.sca = packed_accessor32<float, 2>(sca);
    params.opa = packed_accessor32<float, 2>(opa);

    params.rgb = packed_accessor32<float, 2>(rgb);
    params.nrm = packed_accessor32<float, 2>(nrm);

    params.proxy_ver = packed_accessor32<float,   3>(proxy_ver);
    params.proxy_tri = packed_accessor32<int32_t, 3>(proxy_tri);

    params.ro  = packed_accessor32<float, 4>(org);
    params.rd  = packed_accessor32<float, 4>(dir);

    params.out_rgb  = packed_accessor32<float, 4>(out_rgb);
    params.out_buf  = packed_accessor32<float, 4>(out_buf);
    params.out_dbg  = packed_accessor32<float, 4>(out_dbg);

    params.backward  = 0;
    params.T_min     = T_min;
    params.alpha_min = alpha_min;
    params.proxy_geometry_nfaces = proxy_tri.size(1);

    CUdeviceptr d_param;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( GaussianTracingParams ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_param ),
                &params, sizeof( params ),
                cudaMemcpyHostToDevice
                ) );

    OPTIX_CHECK( optixLaunch( stateWrapper.pState->fwd_ppl, stream, d_param, sizeof( GaussianTracingParams ), 
                              &stateWrapper.pState->fwd_sbt, dir.size(2), dir.size(1), dir.size(0) ) );
                              // width, height, depth

    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    return {out_rgb, out_buf, out_dbg};
}

std::vector<torch::Tensor> gaussian_raytrace_bwd(
    OptiXStateWrapper& stateWrapper, 
    torch::Tensor xyz, 
    torch::Tensor rot, 
    torch::Tensor sca, 
    torch::Tensor opa, 

    torch::Tensor rgb, 
    torch::Tensor nrm, 

    torch::Tensor org,   // N,3
    torch::Tensor dir,   // N,3

    torch::Tensor proxy_ver, 
    torch::Tensor proxy_tri,

    float T_min,
    float alpha_min,

    torch::Tensor out_rgb, 
    torch::Tensor out_buf,
    torch::Tensor out_dbg,

    torch::Tensor out_rgb_grad, 
    torch::Tensor out_buf_grad
    )
{
    //
    // launch OptiX kernel
    //
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor xyz_grad = torch::zeros({ xyz.size(0), xyz.size(1) }, xyz.options()) ;
    torch::Tensor rot_grad = torch::zeros({ rot.size(0), rot.size(1) }, rot.options()) ;
    torch::Tensor sca_grad = torch::zeros({ sca.size(0), sca.size(1) }, sca.options()) ;
    torch::Tensor opa_grad = torch::zeros({ opa.size(0), opa.size(1) }, opa.options()) ;
    torch::Tensor rgb_grad = torch::zeros({ rgb.size(0), rgb.size(1) }, rgb.options()) ;
    torch::Tensor nrm_grad = torch::zeros({ nrm.size(0), nrm.size(1) }, nrm.options()) ;
    torch::Tensor org_grad = torch::zeros({ org.size(0), org.size(1), org.size(2), 3 }, org.options()) ;
    torch::Tensor dir_grad = torch::zeros({ dir.size(0), dir.size(1), org.size(2), 3 }, dir.options()) ;

    GaussianTracingParams params;
    params.handle       = stateWrapper.pState->gas_handle;

    params.xyz = packed_accessor32<float, 2>(xyz);
    params.rot = packed_accessor32<float, 2>(rot);
    params.sca = packed_accessor32<float, 2>(sca);
    params.opa = packed_accessor32<float, 2>(opa);
    params.xyz_grad = packed_accessor32<float, 2>(xyz_grad);
    params.rot_grad = packed_accessor32<float, 2>(rot_grad);
    params.sca_grad = packed_accessor32<float, 2>(sca_grad);
    params.opa_grad = packed_accessor32<float, 2>(opa_grad);

    params.rgb = packed_accessor32<float, 2>(rgb);
    params.nrm = packed_accessor32<float, 2>(nrm);
    params.rgb_grad = packed_accessor32<float, 2>(rgb_grad);
    params.nrm_grad = packed_accessor32<float, 2>(nrm_grad);

    params.proxy_ver = packed_accessor32<float,   3>(proxy_ver);
    params.proxy_tri = packed_accessor32<int32_t, 3>(proxy_tri);

    params.ro  = packed_accessor32<float, 4>(org);
    params.rd  = packed_accessor32<float, 4>(dir);
    params.ro_grad  = packed_accessor32<float, 4>(org_grad);
    params.rd_grad  = packed_accessor32<float, 4>(dir_grad);

    params.out_rgb  = packed_accessor32<float, 4>(out_rgb);
    params.out_buf  = packed_accessor32<float, 4>(out_buf);
    params.out_dbg  = packed_accessor32<float, 4>(out_dbg);

    params.out_rgb_grad = packed_accessor32<float, 4>(out_rgb_grad);
    params.out_buf_grad = packed_accessor32<float, 4>(out_buf_grad);

    params.backward  = 1;
    params.T_min     = T_min;
    params.alpha_min = alpha_min;
    params.proxy_geometry_nfaces = proxy_tri.size(1);

    CUdeviceptr d_param;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( GaussianTracingParams ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_param ),
                &params, sizeof( params ),
                cudaMemcpyHostToDevice
                ) );

    OPTIX_CHECK( optixLaunch( stateWrapper.pState->bwd_ppl, stream, d_param, sizeof( GaussianTracingParams ), 
                              &stateWrapper.pState->bwd_sbt, dir.size(2), dir.size(1), dir.size(0) ) );
                              // width, height, depth

    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    return {xyz_grad, rot_grad, sca_grad, opa_grad, rgb_grad, nrm_grad, org_grad, dir_grad};
}
}

namespace envshade{
std::vector<torch::Tensor> env_shade_fwd(
    OptiXStateWrapper& stateWrapper, 

    torch::Tensor xyz, 
    torch::Tensor rot, 
    torch::Tensor sca, 
    torch::Tensor opa, 
    torch::Tensor proxy_ver, 
    torch::Tensor proxy_tri,
    float T_min,
    float alpha_min,

    torch::Tensor mask, 
    torch::Tensor ro, 
    torch::Tensor gb_pos, 
    torch::Tensor gb_normal, 
    torch::Tensor gb_view_pos, 
    torch::Tensor gb_kd, 
    torch::Tensor gb_ks, 
    torch::Tensor light, 
    torch::Tensor pdf, 
    torch::Tensor rows, 
    torch::Tensor cols,
    torch::Tensor perms,

    unsigned int BSDF,
    unsigned int n_samples_x,
    unsigned int rnd_seed,
    float shadow_scale
    )
{
    //
    // launch OptiX kernel
    //
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor diff = torch::zeros({ ro.size(0), ro.size(1), ro.size(2), 3 }, opts) ;
    torch::Tensor spec = torch::zeros({ ro.size(0), ro.size(1), ro.size(2), 3 }, opts) ;

    EnvSamplingParams params;
    params.handle       = stateWrapper.pState->gas_handle;

    params.xyz          = packed_accessor32<float, 2>(xyz);
    params.rot          = packed_accessor32<float, 2>(rot);
    params.sca          = packed_accessor32<float, 2>(sca);
    params.opa          = packed_accessor32<float, 2>(opa);

    params.mask         = packed_accessor32<float, 3>(mask);
    params.ro           = packed_accessor32<float, 4>(ro);
    params.gb_pos       = packed_accessor32<float, 4>(gb_pos);
    params.gb_normal    = packed_accessor32<float, 4>(gb_normal);
    params.gb_view_pos  = packed_accessor32<float, 4>(gb_view_pos);
    params.gb_kd        = packed_accessor32<float, 4>(gb_kd);
    params.gb_ks        = packed_accessor32<float, 4>(gb_ks);
    params.light        = packed_accessor32<float, 3>(light);
    params.pdf          = packed_accessor32<float, 2>(pdf);
    params.rows         = packed_accessor32<float, 1>(rows);
    params.cols         = packed_accessor32<float, 2>(cols);
    params.diff         = packed_accessor32<float, 4>(diff);
    params.spec         = packed_accessor32<float, 4>(spec);
    params.perms        = packed_accessor32<int, 2>(perms);
    params.BSDF         = BSDF;
    params.n_samples_x  = n_samples_x;
    params.rnd_seed     = rnd_seed;
    params.backward     = 0;
    params.shadow_scale = shadow_scale;

    params.T_min     = T_min;
    params.alpha_min = alpha_min;
    params.proxy_geometry_nfaces = proxy_tri.size(1);

    // torch::Tensor dbg  = torch::zeros({ ro.size(0), ro.size(1), ro.size(2), n_samples_x*n_samples_x, 16 }, opts) ;
    // params.dbg          = packed_accessor32<float, 5>(dbg);

    CUdeviceptr d_param;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( EnvSamplingParams ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_param ),
                &params, sizeof( params ),
                cudaMemcpyHostToDevice
                ) );

    OPTIX_CHECK( optixLaunch( stateWrapper.pState->shd_ppl, stream, d_param, sizeof( EnvSamplingParams ), 
                              &stateWrapper.pState->shd_sbt, ro.size(2), ro.size(1), ro.size(0) ) );

    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    return {diff, spec};
}

std::vector<torch::Tensor> env_shade_bwd(
    OptiXStateWrapper& stateWrapper, 

    torch::Tensor xyz, 
    torch::Tensor rot, 
    torch::Tensor sca, 
    torch::Tensor opa, 
    torch::Tensor proxy_ver, 
    torch::Tensor proxy_tri,
    float T_min,
    float alpha_min,

    torch::Tensor mask, 
    torch::Tensor ro, 
    torch::Tensor gb_pos, 
    torch::Tensor gb_normal, 
    torch::Tensor gb_view_pos, 
    torch::Tensor gb_kd, 
    torch::Tensor gb_ks, 
    torch::Tensor light,
    torch::Tensor pdf, 
    torch::Tensor rows, 
    torch::Tensor cols, 
    torch::Tensor perms,
    unsigned int BSDF,
    unsigned int n_samples_x,
    unsigned int rnd_seed,
    float shadow_scale,
    torch::Tensor diff_grad,
    torch::Tensor spec_grad)
{
    //
    // launch OptiX kernel
    //
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    EnvSamplingParams params;
    params.handle       = stateWrapper.pState->gas_handle;

    params.xyz          = packed_accessor32<float, 2>(xyz);
    params.rot          = packed_accessor32<float, 2>(rot);
    params.sca          = packed_accessor32<float, 2>(sca);
    params.opa          = packed_accessor32<float, 2>(opa);

    params.mask         = packed_accessor32<float, 3>(mask);
    params.ro           = packed_accessor32<float, 4>(ro);
    params.gb_pos       = packed_accessor32<float, 4>(gb_pos);
    params.gb_normal    = packed_accessor32<float, 4>(gb_normal);
    params.gb_view_pos  = packed_accessor32<float, 4>(gb_view_pos);
    params.gb_kd        = packed_accessor32<float, 4>(gb_kd);
    params.gb_ks        = packed_accessor32<float, 4>(gb_ks);
    params.light        = packed_accessor32<float, 3>(light);
    params.pdf          = packed_accessor32<float, 2>(pdf);
    params.rows         = packed_accessor32<float, 1>(rows);
    params.cols         = packed_accessor32<float, 2>(cols);
    params.diff_grad    = packed_accessor32<float, 4>(diff_grad);
    params.spec_grad    = packed_accessor32<float, 4>(spec_grad);
    params.perms        = packed_accessor32<int, 2>(perms);
    params.BSDF         = BSDF;
    params.n_samples_x  = n_samples_x;
    params.rnd_seed     = rnd_seed;
    params.backward     = 1;
    params.shadow_scale = shadow_scale;

    params.T_min     = T_min;
    params.alpha_min = alpha_min;
    params.proxy_geometry_nfaces = proxy_tri.size(1);

    // Create gradient tensor for pos
    torch::Tensor gb_pos_grad = torch::zeros({ ro.size(0), ro.size(1), ro.size(2), gb_pos.size(3) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    params.gb_pos_grad = packed_accessor32<float, 4>(gb_pos_grad);

    torch::Tensor gb_normal_grad = torch::zeros({ ro.size(0), ro.size(1), ro.size(2), gb_normal.size(3) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    params.gb_normal_grad = packed_accessor32<float, 4>(gb_normal_grad);

    torch::Tensor gb_kd_grad = torch::zeros({ ro.size(0), ro.size(1), ro.size(2), gb_kd.size(3) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    params.gb_kd_grad = packed_accessor32<float, 4>(gb_kd_grad);

    torch::Tensor gb_ks_grad = torch::zeros({ ro.size(0), ro.size(1), ro.size(2), gb_ks.size(3) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    params.gb_ks_grad = packed_accessor32<float, 4>(gb_ks_grad);

    // Create gradient tensor for light
    torch::Tensor light_grad = torch::zeros({ light.size(0), light.size(1), light.size(2) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    params.light_grad = packed_accessor32<float, 3>(light_grad);

    CUdeviceptr d_param;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( EnvSamplingParams ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_param ),
                &params, sizeof( params ),
                cudaMemcpyHostToDevice
                ) );

    OPTIX_CHECK( optixLaunch( stateWrapper.pState->shd_ppl, stream, d_param, sizeof( EnvSamplingParams ), 
                              &stateWrapper.pState->shd_sbt, ro.size(2), ro.size(1), ro.size(0) ) );

    CUDA_CHECK( cudaStreamSynchronize( stream ) );

    return {gb_pos_grad, gb_normal_grad, gb_kd_grad, gb_ks_grad, light_grad};
} 

}

namespace denoise{
torch::Tensor bilateral_denoiser_fwd(torch::Tensor col, torch::Tensor nrm, torch::Tensor zdz, float sigma)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::zeros({ col.size(0), col.size(1), col.size(2), 4 }, opts);

    dim3 blockSize(8, 8, 1);
    dim3 gridSize((col.size(2) - 1) / blockSize.x + 1, (col.size(1) - 1) / blockSize.y + 1, (col.size(0) - 1) / blockSize.z + 1);

    BilateralDenoiserParams params;
    params.col = packed_accessor32<float, 4>(col);
    params.nrm = packed_accessor32<float, 4>(nrm);
    params.zdz = packed_accessor32<float, 4>(zdz);
    params.out = packed_accessor32<float, 4>(out);
    params.sigma = sigma;

    void *args[] = {&params};
    CUDA_CHECK(cudaLaunchKernel((const void *)bilateral_denoiser_fwd_kernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor bilateral_denoiser_bwd(torch::Tensor col, torch::Tensor nrm, torch::Tensor zdz, float sigma, torch::Tensor out_grad)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor col_grad = torch::zeros({ col.size(0), col.size(1), col.size(2), col.size(3) }, opts);

    dim3 blockSize(8, 8, 1);
    dim3 gridSize((col.size(2) - 1) / blockSize.x + 1, (col.size(1) - 1) / blockSize.y + 1, (col.size(0) - 1) / blockSize.z + 1);

    BilateralDenoiserParams params;
    params.col = packed_accessor32<float, 4>(col);
    params.nrm = packed_accessor32<float, 4>(nrm);
    params.zdz = packed_accessor32<float, 4>(zdz);
    params.out_grad = packed_accessor32<float, 4>(out_grad);
    params.col_grad = packed_accessor32<float, 4>(col_grad);
    params.sigma = sigma;

    void *args[] = {&params};
    CUDA_CHECK(cudaLaunchKernel((const void *)bilateral_denoiser_bwd_kernel, gridSize, blockSize, args, 0, stream));

    return col_grad;
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<OptiXStateWrapper>(m, "OptiXStateWrapper").def(pybind11::init<const std::string &, const std::string &>());

    m.def("optix_build_bvh",       &raytrace::optix_build_bvh,         "optix_build_bvh");
    m.def("gaussian_raytrace_fwd", &raytrace::gaussian_raytrace_fwd,   "gaussian_raytrace_fwd");
    m.def("gaussian_raytrace_bwd", &raytrace::gaussian_raytrace_bwd,   "gaussian_raytrace_bwd");

    m.def("env_shade_fwd",         &envshade::env_shade_fwd,   "env_shade_fwd");
    m.def("env_shade_bwd",         &envshade::env_shade_bwd,   "env_shade_bwd");

    m.def("bilateral_denoiser_fwd", &denoise::bilateral_denoiser_fwd, "bilateral_denoiser_fwd");
    m.def("bilateral_denoiser_bwd", &denoise::bilateral_denoiser_bwd, "bilateral_denoiser_bwd");    
}