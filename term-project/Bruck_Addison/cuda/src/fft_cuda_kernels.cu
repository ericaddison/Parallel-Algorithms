#include "fft_cuda.h"
#include "fft_cuda_kernels.h"
#include <stdio.h>


__device__ int d_bitReverse(unsigned int b, int d)
{
    b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
    b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
    b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
    b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
    b = ((b >> 16) | (b << 16));
    return b>>(32-d);
}



__global__ void bit_reverse_kernel(thCdouble *out, thCdouble *in, int n)
{ 
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int myId = tid + offset;
    int d = roundf(log2f(n));
    n = MIN(n, MAX_THREADS);
    out[ myId ] = in[ d_bitReverse(myId, d) ];
}


// iterative FFT with shared memory
__global__ void fft_kernel_shared(thCdouble *x, int n, direction dir)
{
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int myId = tid + offset;
    n = MIN(n, MAX_THREADS);

    // perform bit-reverse swapping (from global memory to shared)
    // i.e. from global x into block shared array of complexes
    extern __shared__ thCdouble sdata[];
    sdata[tid] = x[myId];
    __syncthreads();

    // perform iterative fft loop
    // m = number of elements at current iteration (tree level)
    // so each iteration is computing the m element dft of the elements in its subtree
    thCdouble t;
    thCdouble u;
    for(int m=2; m <= n; m<<=1)
    {
    // set up index variables
        double v = (2*(dir==REVERSE)-1) * 2 * cuPI / m;
        int i = tid/m;    // which node am I in at level m/2 (counting from bottom)?
       int j = tid%m;    // which element am I within that node?
       int k = i*m + j;    // what is my offset into sdata?

        // compute value
        thCdouble wj = thrust::polar(1.0, v*j);
        t = (j<m/2) ? sdata[k] : sdata[k-m/2];
        u = (j<m/2) ? wj*sdata[k+m/2] : wj*sdata[k];
        __syncthreads();
        sdata[k] = t+u;
    }

    // write result
    x[myId] = sdata[tid];
}



// kernel to call after the shared memory kernel
// has performed fft on blocks
// this one will keep going, but working in global memory
// at this point in the computation, each block holds a complete fft
// of the elements it started with. Need to continue fft merging those
// elements until all done
__global__ void fft_kernel_finish(thCdouble *x, int m, direction dir)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;

    // set up index variables
    double v = (2*(dir==REVERSE)-1) * 2 * cuPI / m;
    int i = myId/(m/2);
    int j = myId%(m/2);
    int k = i*m + j;

    // compute value
    thCdouble wj = thrust::polar(1.0, v*j);
    thCdouble t = x[k];
    thCdouble u = wj*x[k+m/2];
    x[k] = t+u;
    x[k+m/2] = t-u;
}

