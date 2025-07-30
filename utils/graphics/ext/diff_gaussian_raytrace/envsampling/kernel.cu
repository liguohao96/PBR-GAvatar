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
// #define MIN_ROUGHNESS 0.001f

extern "C" {
__constant__ EnvSamplingParams params;
}

//==============================================================================
// Math / utility functions
//==============================================================================

#include "../bsdf.h"

// from https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
__device__ unsigned int rand_pcg(unsigned int &rng_state)
{
    unsigned int word = ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
    rng_state = rng_state * 747796405u + 2891336453u;
    return (word >> 22u) ^ word;
}

__device__ unsigned int hash_pcg(unsigned int global_seed, unsigned int sample_seed)
{
    return rand_pcg(global_seed) ^ rand_pcg(sample_seed);
}

__device__ float uniform_pcg(unsigned int &rng_state)
{
    return (float)(rand_pcg(rng_state) & 0xFFFFFF) / (float)0x1000000;
}

__device__ float3 tolocal(const float3& a, const float3& u, const float3& v, const float3& w) 
{
    return make_float3(dot(a, u), dot(a, v), dot(a, w));
}

__device__ float3 toworld(const float3& a, const float3& u, const float3& v, const float3& w) 
{
    return u * a.x + v * a.y + w * a.z;
}

__device__ float3 cosine_sample(float3 N, float u, float v, float& pdf)
{   
    // construct local frame
    N = safe_normalize(N);
    float3 dx, dy;
    branchlessONB(N, dx, dy); 

    // cosine sampling in local frame
    float phi = 2.0 * CUDART_PI * u;
    float costheta = sqrt(v);
    float sintheta = sqrt(1.0 - v);

    // Cartesian vector in local space
    float x = cos(phi)*sintheta;
    float y = sin(phi)*sintheta;
    float z = costheta;

    pdf = max(0.000001f, costheta / CUDART_PI);

    // Local to world
    float3 vec = dx*x + dy*y + N*z;
    return safe_normalize(vec);
}

__device__ float albedo(const float3& baseColor, const float eta, const float3& wo, const float3& N)
{
    // Construct tangent frame
    float3 W = safe_normalize(N);
    float3 U,V;
    branchlessONB(W, U, V);
    float3 wo_l = safe_normalize(tolocal(wo, U, V, W));

    const float cosNO = wo_l.z;
    if (!(cosNO > 0))
        return 0.0f;

    return luminance(fwdFresnelSchlick(baseColor, make_float3(1.f, 1.f, 1.f), cosNO));
}

//==============================================================================
// Shadow ray test. Note: This code ignores the shadow gradient boundary term.
// We saw no benefit to boundary term gradients in our experiments. 
//==============================================================================

__device__ float shadow_test(uint3 idx, float3 ray_origin, float3 ray_dir, float vis_grad);

//==============================================================================
// Light probe functions
//==============================================================================

// MODIFIED
__device__ float2 _dir_to_tc(float3 dir)
{
    // float u = atan2f(dir.x, -dir.z) / (2.0f * CUDART_PI) + 0.5f;
    // float v = acosf(clamp(dir.y, -1.0f, 1.0f)) / CUDART_PI;
    float u = atan2f(dir.z, dir.x) / (2.0f * CUDART_PI) + 0.5f;
    float v = acosf(clamp(dir.y, -1.0f, 1.0f)) / CUDART_PI;
    return make_float2(u, v);
}

// MODIFIED
__device__ float3 _tc_to_dir(float2 uv)
{
    float sinphi, cosphi;
    sincos((uv.x * 2.0f - 1.0f) * CUDART_PI, &sinphi, &cosphi);
    float sintheta, costheta;
    sincos(uv.y * CUDART_PI, &sintheta, &costheta);
    // return make_float3(sintheta*sinphi, costheta, -sintheta*cosphi);
    return make_float3(sintheta*cosphi, costheta, sintheta*sinphi);
}

template<class T> __device__ float sample_cdf(const T &cdf, float x, unsigned int &idx, float &pdf)
{
    x = min(x, 0.99999994f);

    // Binary search to find next index above
    unsigned int _min = 0;
    unsigned int _max = cdf.size(0) - 1;
    unsigned int m = int(ceil(log2((float)_max))) + 1;
    for (int i=0; i<m; ++i)
    {
        unsigned int mid = (_min + _max) / 2;
        _min = x >= cdf[mid] ? mid :_min;
        _max = x < cdf[mid] ? mid : _max;
    }
    idx = _max;

    float sample;
    if (idx == 0) {
        pdf = cdf[0];
        sample = x;
    }
    else {
        float data0 = cdf[idx];
        float data1 = cdf[idx-1];
        pdf = data0 - data1;
        sample = (x - data1);
    }
    // keep result in [0,1)
    return min(sample / pdf, 0.99999994f);
}

__device__ float lightPDF(const float3& dir)
{
    // Sample light
    float2 coord = _dir_to_tc(dir);

    // retrieve nearest neighbor
    int x = clamp((int)(coord.x * params.pdf.size(1)), 0, params.pdf.size(1) - 1);
    int y = clamp((int)(coord.y * params.pdf.size(0)), 0, params.pdf.size(0) - 1);

    float pdf_weight = params.cols.size(0) * params.cols.size(1) / (2.0f * CUDART_PI * CUDART_PI * max(sinf(coord.y * CUDART_PI), 0.0001f));
    return params.pdf[y][x] * pdf_weight;
}

__device__ float3 lightSample(float u, float v, float& pdf)
{
    float row_pdf, col_pdf;
    unsigned int x, y;
    float ry = sample_cdf(params.rows, v, y, row_pdf);
    float rx = sample_cdf(params.cols[y], u, x, col_pdf);
    float3 rnd_dir = _tc_to_dir(make_float2((x+rx)/params.cols.size(1), (y+ry)/params.cols.size(0)));
    pdf = lightPDF(rnd_dir);
    return rnd_dir;
}

__device__ float3 eval_light_fwd(float2 coord)
{
    coord = coord * make_float2(params.light.size(1), params.light.size(0)); 
    int x = clamp((int)coord.x, 0, params.light.size(1) - 1);
    int y = clamp((int)coord.y, 0, params.light.size(0) - 1);
    return fetch3(params.light, y, x);
}

__device__ void eval_light_bwd(float2 coord, float3 light_grad)
{
    coord = coord * make_float2(params.light.size(1), params.light.size(0)); 
    int x = clamp((int)coord.x, 0, params.light.size(1) - 1);
    int y = clamp((int)coord.y, 0, params.light.size(0) - 1);
    atomicAdd(&params.light_grad[y][x][0], light_grad.x);
    atomicAdd(&params.light_grad[y][x][1], light_grad.y);
    atomicAdd(&params.light_grad[y][x][2], light_grad.z);
}

//==============================================================================
// BSDF evaluation & importance sampling
//==============================================================================

__device__ float evalNdfGGX(float alpha, float cosTheta)
{
    float a2 = alpha * alpha;
    float d = ((cosTheta * a2 - cosTheta) * cosTheta + 1);
    return a2 / (d * d * CUDART_PI);
}

__device__ float evalG1GGX(float alphaSqr, float cosTheta)
{
    if (cosTheta <= 0) return 0;
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1.0f - cosThetaSqr, 0.0f) / cosThetaSqr;
    return 2 / (1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

__device__ float evalPdfGGX_VNDF(float alpha, float3 wo, float3 h)
{
    float G1 = evalG1GGX(alpha * alpha, wo.z);
    float D = evalNdfGGX(alpha, h.z);
    return G1 * D * max(0.f, dot(wo, h)) / wo.z;
}

// Samples the GGX (Trowbridge-Reitz) using the distribution of visible normals (VNDF).
// See http://jcgt.org/published/0007/04/01/paper.pdf
__device__ float3 sampleGGX_VNDF(float alpha, float3 wo, float ux, float uy, float& pdf)
{
    // Transform the view vector to the hemisphere configuration.
    float3 Vh = safe_normalize(make_float3(alpha * wo.x, alpha * wo.y, wo.z));

    // Construct orthonormal basis (Vh,T1,T2).
    float3 T1 = (Vh.z < 0.9999f) ? safe_normalize(cross(make_float3(0.f, 0.f, 1.f), Vh)) : make_float3(1.f, 0.f, 0.f);
    float3 T2 = cross(Vh, T1);

    // Parameterization of the projected area of the hemisphere.
    float r = sqrtf(ux);
    float phi = (2.f * M_PI) * uy;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.f + Vh.z);
    t2 = (1.f - s) * sqrtf(1.f - t1 * t1) + s * t2;

    // Reproject onto hemisphere.
    float3 Nh = T1 * t1 + T2* t2 + Vh * sqrtf(max(0.f, 1.f - t1 * t1 - t2 * t2));

    // Transform the normal back to the ellipsoid configuration. This is our half vector.
    float3 h = safe_normalize(make_float3(alpha * Nh.x, alpha * Nh.y, max(0.f, Nh.z)));

    pdf = evalPdfGGX_VNDF(alpha, wo, h);
    return h;
}

__device__ float3 ggx_sample(float3 N, float3 wo, float u, float v, float alpha, float& pdf)
{
    // Construct tangent frame
    float3 W = safe_normalize(N);
    float3 U,V;
    branchlessONB(W, U, V);

    float3 wo_l = safe_normalize(tolocal(wo, U, V, W));
    const float cosNO = wo_l.z;
    if (!(cosNO > 0)) {
        pdf = 0.f;
        return make_float3(0.f, 0.f, 0.f);
    }

    float3 h = sampleGGX_VNDF(alpha, wo_l, u, v, pdf);    // pdf = G1(wo) * D(h) * max(0,dot(wo,h)) / wo.z

    // Reflect the outgoing direction to find the incident direction.
    float woDotH = dot(wo_l, h);
    float3 wi_l = h * woDotH * 2.0f - wo_l;
    pdf /= (4.0f * woDotH); // Jacobian of the reflection operator.

    float3 wi_o = toworld(wi_l, U, V, W);
    return safe_normalize(wi_o);
}

__device__ float evalLambdaGGX(float alphaSqr, float cosTheta)
{
    if (cosTheta <= 0) return 0;
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1 - cosThetaSqr, 0.0f) / cosThetaSqr;
    return 0.5 * (-1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

__device__ float ggx_pdf(float3 N, const float3 wo, const float3 wi, float alpha)
{
    // Construct tangent frame
    float3 W = safe_normalize(N);
    float3 U,V;
    branchlessONB(W, U, V);

    // wo_l : V
    // wi_l : L
    float3 wo_l = tolocal(wo, U, V, W);
    float3 wi_l = tolocal(wi, U, V, W);

    float pdf = 0.0f;
    if (wo_l.z > 0 && wi_l.z > 0) {
        float3 m = safe_normalize(wi_l + wo_l);
        const float woDotH = dot(m, wo_l);
        const float D = evalNdfGGX(alpha, m.z);
        float G1 = evalG1GGX(alpha * alpha, wo_l.z);
        pdf = G1 * D * max(0.f, dot(wo_l, m)) / wo_l.z;
        pdf /= (4 * woDotH);
    }
    return pdf; 
}

__device__ void update_pdf(float* pdf, float opdf, float b)
{
    if (b > 0.000001f)
    {
        opdf *= b;
        *pdf += opdf;
    }
}

__device__ float3 bsdf_sample(float pDiffuse, float pSpecular, float3 N, float3 wo, float3 s, float alpha, float& pdf)
{
    float3 rnd = s;
    pdf = 0.0f;
    float3 wi_o; 

    if (rnd.z < pDiffuse) // Sample diffuse lobe
    {
        if (pDiffuse < 0.0001f)
        { 
            pdf = 1.0f;
            return N;
        }

        wi_o = cosine_sample(N, rnd.x, rnd.y, pdf);
        pdf *= pDiffuse;

        // we sampled the diffuse lobe, now figure out how much the other bsdf contribute to the chosen direction
        if (pSpecular > 0)
        {
            float bsdf_pdf = ggx_pdf(N, wo, wi_o, alpha);
            update_pdf(&pdf, bsdf_pdf, 1.0f - pDiffuse);
        }
    }
    else // Sample specular lobe
    {
        wi_o = ggx_sample(N, wo, rnd.x, rnd.y, alpha, pdf);
        pdf *= 1.f - pDiffuse;

        // we sampled PDF 1, now figure out how much the other bsdf contribute to the chosen direction
        if (pDiffuse > 0)
        {
            float bsdf_pdf = max(dot(N, wi_o), 0.0) / CUDART_PI; // cosine sampling pdf
            update_pdf(&pdf, bsdf_pdf, pDiffuse);
        }
    }

    return wi_o;
}

__device__ float bsdf_pdf(float pDiffuse, float pSpecular, float3 N, const float3 wo, const float3 wi, float alpha)
{
    // Check that L and V are in the positive hemisphere.
    // The G term on the correlated form is not robust for NdotL = NdotV = 0.0.
    float NdotL = dot(N, wi);
    float NdotV = dot(N, wo);
    static const float kMinCosTheta = 1e-6f;
    float pdf = 0.0f;
    if (min(NdotV, NdotL) < kMinCosTheta)
        return 1.0f;

    if (pDiffuse > 0)
    {
        float bsdf_pdf = max(dot(N, wi), 0.0) / CUDART_PI; // cosine sampling pdf
        update_pdf(&pdf, bsdf_pdf, pDiffuse);
    }

    if (pSpecular > 0)
    {
        float bsdf_pdf = ggx_pdf(N, wo, wi, alpha); // ggx sampling pdf
        update_pdf(&pdf, bsdf_pdf, 1.0f - pDiffuse);
    }
    return pdf;
}

//==============================================================================
// Optix kernels
//==============================================================================

__device__ void process_sample(uint3 idx, float3 ray_origin, float3 ray_dir, float3 gb_pos, float3 gb_normal, float3 gb_view_pos, 
    float3 gb_kd, float3 gb_ks, float pdfSum, float weight, float3 &diff, float3 &spec, float3 diff_grad, float3 spec_grad)
{
    float2 coord = _dir_to_tc(ray_dir);
    float3 light_col = eval_light_fwd(coord);

    float mis_weight = 1.0 / max(pdfSum, 0.0001f); // MIS balance heuristic
    float alpha = gb_ks.y * gb_ks.y;

    float3 _diff = make_float3(0), _spec = make_float3(0);
    if (params.BSDF == 1 || params.BSDF == 2)
        _diff = make_float3(fwdLambert(gb_normal, ray_dir));
    else
        fwdPbrBSDF(gb_kd, gb_ks, gb_pos, gb_normal, gb_view_pos, ray_dir, 0.08f, _diff, _spec);

    // Trace shadow ray for current sample
    float V_grad = sum((diff_grad * _diff + spec_grad * _spec) * light_col * mis_weight * weight) * params.shadow_scale;
    float V = shadow_test(idx, ray_origin, ray_dir, V_grad) * params.shadow_scale + (1 - params.shadow_scale);
    // float V = 1;

    if (params.backward) 
    {
        float3 light_grad = (diff_grad * _diff + spec_grad * _spec) * V * mis_weight * weight;
        eval_light_bwd(coord, light_grad);

        float3 _diff_grad = diff_grad * light_col * V * mis_weight * weight;
        float3 _spec_grad = spec_grad * light_col * V * mis_weight * weight;
        float3 gb_kd_grad = make_float3(0), gb_ks_grad = make_float3(0), gb_pos_grad = make_float3(0), gb_normal_grad = make_float3(0), gb_view_pos_grad = make_float3(0), ray_dir_grad = make_float3(0);
        if (params.BSDF == 1 || params.BSDF == 2) // params.BSDF : 0 : 'pbr', 1 : 'diffuse', 2 : 'white'
        {
            float3 wi_grad = make_float3(0);
            float lambert = fwdLambert(gb_normal, ray_dir);
            float lambert_grad = sum(_diff_grad);
            bwdLambert(gb_normal, ray_dir, gb_normal_grad, wi_grad, lambert_grad);
        }
        else
        {
            bwdPbrBSDF( gb_kd, gb_ks, gb_pos, gb_normal, gb_view_pos, ray_dir, 0.08f,  
                        gb_kd_grad, gb_ks_grad, gb_pos_grad, gb_normal_grad, gb_view_pos_grad, ray_dir_grad, _diff_grad, _spec_grad);
        }
        params.gb_pos_grad[idx.z][idx.y][idx.x][0] += gb_pos_grad.x;
        params.gb_pos_grad[idx.z][idx.y][idx.x][1] += gb_pos_grad.y;
        params.gb_pos_grad[idx.z][idx.y][idx.x][2] += gb_pos_grad.z;

        params.gb_normal_grad[idx.z][idx.y][idx.x][0] += gb_normal_grad.x;
        params.gb_normal_grad[idx.z][idx.y][idx.x][1] += gb_normal_grad.y;
        params.gb_normal_grad[idx.z][idx.y][idx.x][2] += gb_normal_grad.z;

        params.gb_kd_grad[idx.z][idx.y][idx.x][0] += gb_kd_grad.x;
        params.gb_kd_grad[idx.z][idx.y][idx.x][1] += gb_kd_grad.y;
        params.gb_kd_grad[idx.z][idx.y][idx.x][2] += gb_kd_grad.z;

        params.gb_ks_grad[idx.z][idx.y][idx.x][0] += gb_ks_grad.x;
        params.gb_ks_grad[idx.z][idx.y][idx.x][1] += gb_ks_grad.y;
        params.gb_ks_grad[idx.z][idx.y][idx.x][2] += gb_ks_grad.z;
    }

    diff = _diff * light_col * V * mis_weight * weight;
    spec = _spec * light_col * V * mis_weight * weight;
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Read per-pixel constant input tensors, ray_origin, g-buffer entries etc.
    float  mask        = params.mask[idx.z][idx.y][idx.x];
    float3 ray_origin  = fetch3(params.ro, idx.z, idx.y, idx.x);
    float3 gb_pos      = fetch3(params.gb_pos, idx.z, idx.y, idx.x);
    float3 gb_normal   = fetch3(params.gb_normal, idx.z, idx.y, idx.x);
    float3 gb_view_pos = fetch3(params.gb_view_pos, idx.z, idx.y, idx.x);
    float3 gb_kd       = fetch3(params.gb_kd, idx.z, idx.y, idx.x);
    float3 gb_ks       = fetch3(params.gb_ks, idx.z, idx.y, idx.x);

    if (mask <= 0) return; // Early exit masked pixels

    float3 diff_grad, spec_grad;
    if (params.backward)
    {
        diff_grad = fetch3(params.diff_grad, idx.z, idx.y, idx.x);
        spec_grad = fetch3(params.spec_grad, idx.z, idx.y, idx.x);
    }

    float3 diffAccum = make_float3(0.0f, 0.0f, 0.0f);
    float3 specAccum = make_float3(0.0f, 0.0f, 0.0f);

    float strata_frac = 1.0f / params.n_samples_x;
    float sample_frac = 1.0f / (params.n_samples_x * params.n_samples_x);
    float alpha = gb_ks.y * gb_ks.y; // roughness squared
    float3 wo = safe_normalize(gb_view_pos - gb_pos); // view direction

    float metallic = gb_ks.z;
    float3 baseColor = gb_kd;
    float3 specColor = make_float3(0.04f, 0.04f, 0.04f) * (1.0f - metallic) + baseColor * metallic;
    float diffuseWeight = (1.f - metallic) * luminance(baseColor);
    float eta = 1.0f;
    float specularWeight = albedo(specColor, eta, wo, gb_normal);
    float pDiffuse = (diffuseWeight + specularWeight) > 0.f ? diffuseWeight / (diffuseWeight + specularWeight) : 1.f;
    float pSpecular = 1.0f - pDiffuse;

    unsigned int rng_state = hash_pcg(params.rnd_seed, (idx.z * dim.y + idx.y) * dim.x + idx.x);
    unsigned int lightIdx = rand_pcg(rng_state) % params.perms.size(0), bsdfIdx = rand_pcg(rng_state) % params.perms.size(0);

    for (int i = 0; i < params.n_samples_x * params.n_samples_x; ++i)
    {
        float3 ray_dir, diff, spec;
        float sx, sy, sz = 0.f, pdf_light, pdf_bsdf;

        // Light importance sampling
        sx = ((float)(params.perms[lightIdx][i] % params.n_samples_x) + uniform_pcg(rng_state)) * strata_frac;
        sy = ((float)(params.perms[lightIdx][i] / params.n_samples_x) + uniform_pcg(rng_state)) * strata_frac;
        ray_dir = lightSample(sx, sy, pdf_light);
        pdf_bsdf = bsdf_pdf(pDiffuse, pSpecular, gb_normal, wo, ray_dir, alpha);
        process_sample(idx, ray_origin, ray_dir, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, pdf_light + pdf_bsdf, sample_frac, diff, spec, diff_grad, spec_grad);
        diffAccum = diffAccum + diff;
        specAccum = specAccum + spec;

        // params.dbg[idx.z][idx.y][idx.x][i][0] = ray_dir.x;
        // params.dbg[idx.z][idx.y][idx.x][i][1] = ray_dir.y;
        // params.dbg[idx.z][idx.y][idx.x][i][2] = ray_dir.z;
        // params.dbg[idx.z][idx.y][idx.x][i][3] = pdf_light;
        // params.dbg[idx.z][idx.y][idx.x][i][4] = pdf_bsdf;
        // params.dbg[idx.z][idx.y][idx.x][i][5] = diff.x;
        // params.dbg[idx.z][idx.y][idx.x][i][6] = diff.y;
        // params.dbg[idx.z][idx.y][idx.x][i][7] = diff.z;

        // BSDF sampling (sample either the diffuse or specular lobe)
        sx = ((float)(params.perms[bsdfIdx][i] % params.n_samples_x) + uniform_pcg(rng_state)) * strata_frac;
        sy = ((float)(params.perms[bsdfIdx][i] / params.n_samples_x) + uniform_pcg(rng_state)) * strata_frac;
        sz = uniform_pcg(rng_state);
        ray_dir = bsdf_sample(pDiffuse, pSpecular, gb_normal, wo, make_float3(sx, sy, sz), alpha, pdf_bsdf);
        pdf_light = lightPDF(ray_dir);
        process_sample(idx, ray_origin, ray_dir, gb_pos, gb_normal, gb_view_pos, gb_kd, gb_ks, pdf_light + pdf_bsdf, sample_frac, diff, spec, diff_grad, spec_grad);
        diffAccum = diffAccum + diff;
        specAccum = specAccum + spec;

        // params.dbg[idx.z][idx.y][idx.x][i][8+0] = ray_dir.x;
        // params.dbg[idx.z][idx.y][idx.x][i][8+1] = ray_dir.y;
        // params.dbg[idx.z][idx.y][idx.x][i][8+2] = ray_dir.z;
        // params.dbg[idx.z][idx.y][idx.x][i][8+3] = pdf_light;
        // params.dbg[idx.z][idx.y][idx.x][i][8+4] = pdf_bsdf;
        // params.dbg[idx.z][idx.y][idx.x][i][8+5] = diff.x;
        // params.dbg[idx.z][idx.y][idx.x][i][8+6] = diff.y;
        // params.dbg[idx.z][idx.y][idx.x][i][8+7] = diff.z;
    }

    // Record results in our output raster
    if (!params.backward)
    {
        params.diff[idx.z][idx.y][idx.x][0] = diffAccum.x;
        params.diff[idx.z][idx.y][idx.x][1] = diffAccum.y;
        params.diff[idx.z][idx.y][idx.x][2] = diffAccum.z;
        params.spec[idx.z][idx.y][idx.x][0] = specAccum.x;
        params.spec[idx.z][idx.y][idx.x][1] = specAccum.y;
        params.spec[idx.z][idx.y][idx.x][2] = specAccum.z;
    }
}

//==============================================================================
// Shadow ray test. Note: This code ignores the shadow gradient boundary term.
// We saw no benefit to boundary term gradients in our experiments. 
//==============================================================================

template<typename T>
inline void swap(T &a, T &b){
    T temp = b;
    b = a;
    a = temp;
};

struct PerRayData 
{
    float        dist [HIT_BUFF_SIZE];
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

extern "C" __global__ void __miss__ms()
{
    // optixSetPayload_0(1);
}

// extern "C" __global__ void __closesthit__ch()
// {
// }

// extern "C" __global__ void __anyhit__ah()
// {
// }

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

__device__ float shadow_test(uint3 idx, float3 ray_org, float3 ray_dir, float vis_grad)
{
    // unsigned int isVisible = 0;
    // optixTrace(
    //     params.handle,
    //     ray_origin,
    //     ray_dir,
    //     0.0f,                                       // Min intersection distance
    //     1e16,                                       // Max intersection distance
    //     0.0f,                                       // rayTime -- used for motion blur
    //     OptixVisibilityMask(0xFF),
    //     OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
    //     0,                                          // SBT offset
    //     0,                                          // SBT stride
    //     0,                                          // missSBTIndex
    //     isVisible);

    PerRayData pld;

    uint2 payload = splitPointer(&pld);
    unsigned int hit, acc_hit = 0;

    const float _t_min     = 0.001f;   // min ray distance
    const float _t_max     = 1e4f;     // max ray distance
    const float _T_min     = params.T_min;      // min Transmittance
    const float _alpha_min = params.alpha_min;

    float T      = 1.0f;
    float t_cur  = _t_min;
    unsigned int i_cur = 0;

    // float  acc_alp = 0.0f;

    // while (t_cur < _t_max && T > _T_min ){
    {
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
            float opacity= params.opa[primID][0];            // TODO: compute alpha

            float4 q_vec = {params.rot[primID][1], params.rot[primID][2], params.rot[primID][3],  params.rot[primID][0]};

            float4 gau_fwd_ret = gaussian_particle_fwd(mu, scale, q_vec, opacity, ray_org, ray_dir);
            // float        t_max = gau_fwd_ret.x;
            float        alpha = gau_fwd_ret.w;
            // float        alpha = opacity * pho;

            if (alpha > _alpha_min){
                // const float weight    = T*alpha;

                // T
                T = T * (1-alpha);
            }

            t_cur = pld.dist[i];
            i_cur = pld.index[i];
        }
        
        acc_hit += pld.length;
        // if (pld.length == 0)
        //     break;
    }

    // return T;
    return T > 0.5 ? 1.0f : 0.0f;
}