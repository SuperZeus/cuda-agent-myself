#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>

#include <vector>

#include "../binding_registry.h"

extern "C" void sum_lastdim_launcher(float* out, const float* x, int rows, int cols, int config, cudaStream_t stream);

static torch::Tensor sum_lastdim_forward(torch::Tensor x, int config = 0) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() >= 2, "x must have at least two dimensions");

    int64_t cols = x.size(-1);
    int64_t rows = x.numel() / cols;
    std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end() - 1);
    auto out = torch::empty(out_sizes, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sum_lastdim_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols),
        config,
        stream
    );
    return out;
}

static void register_sum_lastdim(pybind11::module& module) {
    module.def("sum_lastdim_forward", &sum_lastdim_forward, py::arg("x"), py::arg("config") = 0);
}

REGISTER_BINDING(sum_lastdim, register_sum_lastdim);
