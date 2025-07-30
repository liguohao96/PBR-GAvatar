#include <torch/extension.h>
#include <vector>
#include <stdio.h>

namespace raytrace{
    std::vector<torch::Tensor> primary_ray_forward_cuda(
        torch::Tensor faces,
        torch::Tensor prays,
        bool culling,
        bool negdist
    );
    std::vector<torch::Tensor> primary_ray_forward_cpu(
        torch::Tensor faces,
        torch::Tensor prays,
        bool culling,
        bool negdist
    );

    std::vector<torch::Tensor> primary_ray_forward(
        torch::Tensor faces,
        torch::Tensor prays,
        bool culling,
        bool negdist
    ){
        switch(faces.device().type()){
            case c10::DeviceType::CUDA:
                // printf("[RUN] zbuffer forward on CUDA\n");
                return primary_ray_forward_cuda(faces, prays, culling, negdist);
                break;
            case c10::DeviceType::CPU:
                // printf("[RUN] zbuffer forward on CPU\n");
                return primary_ray_forward_cpu(faces, prays, culling, negdist);
                break;
            default :
                TORCH_CHECK(0, "unsupport dispatch type ", faces.device().type());
        }
    }


    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("primary_ray_forward", &primary_ray_forward, "Primary Ray forward");
    }
}