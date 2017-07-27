#ifndef _CUDA_FFT_
#define _CUDA_FFT_

#include <cuda_runtime.h>
#include <thrust/complex.h>
#include "ft_helpers.h"

#define MIN(x,y) ((x<y)?x:y)
#define MAX(x,y) ((x>y)?x:y)
#define cuPI 3.14159265359
#define MAX_THREADS 1024

typedef thrust::complex<double> thCdouble;

__global__ void fft_kernel_shared(thCdouble *x, int n, direction dir);

#endif
