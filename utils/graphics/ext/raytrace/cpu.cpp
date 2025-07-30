#include <torch/types.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include "utils.h"

namespace raytrace{
    using std::max;
    using std::min;
    #define min3(a,b,c) (min(min(a,b), c))
    #define max3(a,b,c) (max(max(a,b), c))

template <typename scalar_t>
__inline__ void ray_tri_intersect_MT(
    vec3<scalar_t>& org,
    vec3<scalar_t>& dir,
    vec3<scalar_t>& v0,
    vec3<scalar_t>& v1,
    vec3<scalar_t>& v2,
    bool&           hit,
    vec3<scalar_t>& out
){
    bool skip = false;

    vec3<scalar_t> v0v1 = v1 - v0;
    vec3<scalar_t> v0v2 = v2 - v0;

    vec3<scalar_t> pvec = dir.cross(v0v2);
    scalar_t det        = v0v1.dot(pvec);
    scalar_t invDet     = 1 / det;

    skip |= (abs(det) <= 1e-6);

    vec3<scalar_t> tvec = org - v0;

    scalar_t u, v, d;
    u = tvec.dot(pvec) * invDet;
    vec3<scalar_t> qvec = tvec.cross(v0v1);
    v = dir.dot(qvec) * invDet;

    skip |= (u < 0 || u > 1);
    skip |= (v < 0 || v > 1);
    skip |= (u + v > 1);

    d = v0v2.dot(qvec) * invDet;

    hit   = !skip;
    out.x = d;
    out.y = u;
    out.z = v;
}

template <typename scalar_t, typename index_t>
void primary_ray_forward_cpu_impl(
    const torch::TensorAccessor<scalar_t, 4> faces, // b, nf, 3, 3
    const torch::TensorAccessor<scalar_t, 4> prays, // h, w, K, 6

    torch::TensorAccessor<index_t,   3>      i_buf, // b, h, w
    torch::TensorAccessor<scalar_t,  3>      z_buf, // b, h, w
    torch::TensorAccessor<scalar_t,  4>      p_buf, // b, 2, h, w
    const bool culling
){
    const int HEIGHT     = prays.size(0);
    const int WIDTH      = prays.size(1);
    const int NUM_RAYS   = prays.size(2);
    const int BATCH_SIZE = faces.size(0);
    const int NUM_FACES  = faces.size(1);

    for(int bi=0; bi<BATCH_SIZE; bi+=1){
        for(int hi=0; hi<HEIGHT; hi+=1){
        for(int wi=0; wi<WIDTH;  wi+=1){
        for(int ki=0; ki<NUM_RAYS; ki+=1){

            vec3<scalar_t> dir, org, xyz;

            vec3<scalar_t> out;
            vec3<scalar_t> local_v33[3];
            bool hit;

            org.x = prays[hi][wi][ki][0];
            org.y = prays[hi][wi][ki][1];
            org.z = prays[hi][wi][ki][2];
            dir.x = prays[hi][wi][ki][3];
            dir.y = prays[hi][wi][ki][4];
            dir.z = prays[hi][wi][ki][5];

            scalar_t dis = static_cast<scalar_t>(INFINITY);
            index_t  tid = -1;

            for(int fi=0; fi<NUM_FACES; fi+=1){

                local_v33[0].x = faces[bi][fi][0][0];
                local_v33[0].y = faces[bi][fi][0][1];
                local_v33[0].z = faces[bi][fi][0][2];
                local_v33[1].x = faces[bi][fi][1][0];
                local_v33[1].y = faces[bi][fi][1][1];
                local_v33[1].z = faces[bi][fi][1][2];
                local_v33[2].x = faces[bi][fi][2][0];
                local_v33[2].y = faces[bi][fi][2][1];
                local_v33[2].z = faces[bi][fi][2][2];

                ray_tri_intersect_MT<scalar_t>(org, dir, local_v33[0], local_v33[1],local_v33[2], hit, out);

                if (hit && out.x < dis){
                    dis = out.x;
                    tid = fi;
                }
            }

            xyz = dir*dis + org;
            i_buf[bi][hi][wi]    = tid;
            p_buf[bi][0][hi][wi] = xyz.x;
            p_buf[bi][1][hi][wi] = xyz.y;
            z_buf[bi][hi][wi]    = dis;
        }
        }
        }
    }
}

std::vector<torch::Tensor> primary_ray_forward_cpu(
    torch::Tensor faces,
    torch::Tensor prays,
    bool culling,
    bool negdist
){
    const unsigned int  batch_size = faces.size(0);
    const unsigned int  num_tri    = faces.size(1);
    const unsigned int  H          = prays.size(0);
    const unsigned int  W          = prays.size(1);
    const unsigned int  PR_NUM     = prays.size(2);

    auto tri_ind_map_b   = torch::full({batch_size, H, W},
                                     static_cast<int64_t>(-1),
                                     faces.options().dtype(torch::kInt64));
    float max_value      = 0;
    auto depth_map       = torch::full({batch_size, H, W}, max_value, faces.options());
    auto point_map       = torch::empty({batch_size, 2, H, W}, faces.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(faces.scalar_type(), "forward_ray_cpu_kernel", [&]{

        primary_ray_forward_cpu_impl<scalar_t, int64_t>(
            faces.accessor<scalar_t, 4>(),
            prays.accessor<scalar_t, 4>(),

            tri_ind_map_b.accessor<int64_t,  3>(),
            depth_map.accessor<scalar_t, 3>(),
            point_map.accessor<scalar_t, 4>(),
            culling
        );
    });

    return {tri_ind_map_b, depth_map, point_map};
}

// end of namespace
}