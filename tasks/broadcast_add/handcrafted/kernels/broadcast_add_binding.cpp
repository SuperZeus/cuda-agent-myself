#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>

#include "../binding_registry.h"

extern "C" void broadcast_add_launcher(
    float* out,
    const float* x,
    const float* bias,
    int width,
    int size,
    int config,
    cudaStream_t stream
);

static torch::Tensor broadcast_add_forward(torch::Tensor x, torch::Tensor bias, int config = 0) {
    TORCH_CHECK(x.is_cuda() && bias.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && bias.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(x.size(-1) == bias.size(0), "last x dimension must equal bias length");

    auto out = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    broadcast_add_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        static_cast<int>(bias.numel()),
        static_cast<int>(x.numel()),
        config,
        stream
    );
    return out;
}

static void register_broadcast_add(pybind11::module& module) {
    module.def("broadcast_add_forward", &broadcast_add_forward, py::arg("x"), py::arg("bias"), py::arg("config") = 0);
}

REGISTER_BINDING(broadcast_add, register_broadcast_add);
