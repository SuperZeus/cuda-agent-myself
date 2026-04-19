#include <cuda_runtime.h>

template<int THREADS>
__global__ void broadcast_add_kernel(float* out, const float* x, const float* bias, int width, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        out[i] = x[i] + bias[i % width];
    }
}

extern "C" void broadcast_add_launcher(
    float* out,
    const float* x,
    const float* bias,
    int width,
    int size,
    int config,
    cudaStream_t stream
) {
    if (size <= 0) {
        return;
    }
    int threads = config == 1 ? 128 : 256;
    int blocks = (size + threads - 1) / threads;
    if (config == 1) {
        broadcast_add_kernel<128><<<blocks, threads, 0, stream>>>(out, x, bias, width, size);
    } else {
        broadcast_add_kernel<256><<<blocks, threads, 0, stream>>>(out, x, bias, width, size);
    }
}
