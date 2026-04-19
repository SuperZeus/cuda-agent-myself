#include <cuda_runtime.h>

template<int THREADS>
__global__ void sum_lastdim_kernel(float* out, const float* x, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float smem[THREADS];
    float local = 0.0f;
    int base = row * cols;
    for (int col = threadIdx.x; col < cols; col += THREADS) {
        local += x[base + col];
    }
    smem[threadIdx.x] = local;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[row] = smem[0];
    }
}

extern "C" void sum_lastdim_launcher(float* out, const float* x, int rows, int cols, int config, cudaStream_t stream) {
    (void)config;
    if (rows <= 0 || cols <= 0) {
        return;
    }
    sum_lastdim_kernel<256><<<rows, 256, 0, stream>>>(out, x, rows, cols);
}
