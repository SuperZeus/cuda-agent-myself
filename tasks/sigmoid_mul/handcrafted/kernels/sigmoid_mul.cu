#include <cuda_runtime.h>
#include <math.h>

template<int THREADS>
__global__ void sigmoid_mul_kernel(float* out, const float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        float value = x[i];
        out[i] = value / (1.0f + expf(-value));
    }
}

extern "C" void sigmoid_mul_launcher(float* out, const float* x, int size, int config, cudaStream_t stream) {
    if (size <= 0) {
        return;
    }
    int threads = config == 1 ? 128 : 256;
    int blocks = (size + threads - 1) / threads;
    if (config == 1) {
        sigmoid_mul_kernel<128><<<blocks, threads, 0, stream>>>(out, x, size);
    } else {
        sigmoid_mul_kernel<256><<<blocks, threads, 0, stream>>>(out, x, size);
    }
}
