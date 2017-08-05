#ifndef _FFT_CUDA_
#define _FFT_CUDA_

#include <thrust/complex.h>
#include "ft_helpers.h"

#define MIN(x,y) ((x<y)?x:y)
#define MAX(x,y) ((x>y)?x:y)
#define cuPI 3.14159265359
#define MAX_THREADS 1024

typedef thrust::complex<double> thCdouble;

// Host side functions
void cuda_init();
float fft_cuda_transform(thCdouble* h_A, int n, direction dir);
float fft_cuda(thCdouble* h_A, int n);
float ifft_cuda(thCdouble* h_A, int n);
float fft_cufft(double *d, int n);

#endif
