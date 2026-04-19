#include <cuda_runtime.h>

template<int THREADS>
__global__ void axpby_kernel(float* out, const float* a, const float* b, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        out[i] = alpha * a[i] + b[i];
    }
}

extern "C" void axpby_launcher(
    float* out,
    const float* a,
    const float* b,
    float alpha,
    int size,
    int config,
    cudaStream_t stream
) {
    if (size <= 0) {
        return;
    }
    int threads = config == 1 ? 128 : (config == 2 ? 512 : 256);
    int blocks = (size + threads - 1) / threads;
    if (config == 1) {
        axpby_kernel<128><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
    } else if (config == 2) {
        axpby_kernel<512><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
    } else {
        axpby_kernel<256><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
    }
}
