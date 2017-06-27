//***********************************************
// Problem 1b functions: last-digit O(1)
#include "p1.h"

__global__ void last_digit_kernel(int * d_out, int * d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    d_out[myId] = d_in[myId]%10;
}


lastDigitResult last_digit_cuda(int *a, int n)
{

// allocate device memory
    int *d_A, *d_B;
    cudaMalloc((int**) &d_A, sizeof(int)*n);
    cudaMalloc((int**) &d_B, sizeof(int)*n);

// copy input array to device
    cudaMemcpy(d_A, a, n*sizeof(int), cudaMemcpyHostToDevice);

// call kernel
    int threadsPerBlock = MIN(n,1024);
    int nBlocks = (n-1)/threadsPerBlock + 1;
	nBlocks = MAX(1,nBlocks);

    last_digit_kernel<<<nBlocks,threadsPerBlock>>>(d_B,d_A);
	cudaThreadSynchronize();

// copy result back to host
    int* digits = (int*)malloc(n*sizeof(int));
    cudaMemcpy(digits, d_B, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

	lastDigitResult res = {digits};
    return res;

}



lastDigitResult last_digit_seq(int* a, int n)
{
	int* b = (int*)malloc(n*sizeof(int));
    for(int i=0; i<n; i++)
      b[i] = a[i]%10;
	
	lastDigitResult res = {b};
	return res;
}


