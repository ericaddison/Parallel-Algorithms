#ifndef _CUDA_FFT_KERNELS_
#define _CUDA_FFT_KERNELS_

#include <cuda_runtime.h>
#include "fft_cuda.h"

// CUDA kernels
__global__ void fft_kernel_shared(thCdouble *x, int n, direction dir);
__global__ void fft_kernel_finish(thCdouble *x, int n, direction dir);
__global__ void bit_reverse_kernel(thCdouble *out, thCdouble *in, int n);

#endif
