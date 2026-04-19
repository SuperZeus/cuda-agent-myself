#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>

#include "../binding_registry.h"

extern "C" void sigmoid_mul_launcher(float* out, const float* x, int size, int config, cudaStream_t stream);

static torch::Tensor sigmoid_mul_forward(torch::Tensor x, int config = 0) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    auto out = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sigmoid_mul_launcher(out.data_ptr<float>(), x.data_ptr<float>(), static_cast<int>(x.numel()), config, stream);
    return out;
}

static void register_sigmoid_mul(pybind11::module& module) {
    module.def("sigmoid_mul_forward", &sigmoid_mul_forward, py::arg("x"), py::arg("config") = 0);
}

REGISTER_BINDING(sigmoid_mul, register_sigmoid_mul);
