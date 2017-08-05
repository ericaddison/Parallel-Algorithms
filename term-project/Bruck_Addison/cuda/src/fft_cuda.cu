#include "fft_cuda.h"
#include "fft_cuda_kernels.h"
#include <stdio.h>

void cuda_init()
{
    cudaFree(0);
    return;
}

float fft_cuda_transform(thCdouble* h_A, int n, direction dir)
{
    // define device data
    thCdouble *d_A;
    cudaMalloc((thCdouble**) &d_A, n*sizeof(thCdouble));
    thCdouble *d_B;
    cudaMalloc((thCdouble**) &d_B, n*sizeof(thCdouble));
    cudaMemcpy(d_B, h_A, n*sizeof(thCdouble), cudaMemcpyHostToDevice);

    // cuda timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // invoke kernel
    int threadsPerBlock = MIN(n,MAX_THREADS);
    int nBlocks = (n-1)/threadsPerBlock + 1;

    // block level kernel call
    cudaEventRecord(start);
    bit_reverse_kernel<<<nBlocks,threadsPerBlock>>>(d_A, d_B, n);
    cudaThreadSynchronize();
    fft_kernel_shared<<<nBlocks,threadsPerBlock,threadsPerBlock*sizeof(thCdouble)>>>(d_A, n, dir);
    cudaThreadSynchronize();

    // continue FFT in global memory
    if(nBlocks>1)
        for(int m=2*MAX_THREADS; m<=n; m<<=1)
        {
            fft_kernel_finish<<<nBlocks, threadsPerBlock>>>(d_A, m, dir);
            cudaThreadSynchronize();
        }

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copy back to host
    cudaMemcpy(h_A, d_A, n*sizeof(thCdouble), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    

    return milliseconds;
}



float fft_cuda(thCdouble* h_A, int n)
{
    return fft_cuda_transform(h_A, n, FORWARD);
}



float ifft_cuda(thCdouble* h_A, int n)
{
    float t = fft_cuda_transform(h_A, n, REVERSE);
    for(int i=0; i<n; i++)
        h_A[i] /= n;
    return t;
}




