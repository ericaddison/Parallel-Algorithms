//***********************************************
// Problem 1b functions: last-digit O(1)
#include "p1.h"

/**
 * last_digit_kernel
 * The CUDA kernel for mapping the last base-10 digit of 
 * the element of an array to a new array.
 *
 * @param d_out result array. Should be same size as d_in
 * @param d_in input array.
 */
__global__ void last_digit_kernel(int * d_out, int * d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    d_out[myId] = d_in[myId]%10;
}


/**
 * last_digit_cuda
 * Host-side function to set up and launch CUDA last digit kernel.
 *
 * @param a input array
 * @param n size of input array a
 * @return lastDigitResult struct holding minimum of input array
 */
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


/**
 * last_digit_seq
 * Sequential implementation of last digit mapping. Simple O(n) loop.
 *
 * @param a input array
 * @param n size of input array a
 * @return lastDigitResult struct holding minimum of input array
 */
lastDigitResult last_digit_seq(int* a, int n)
{
	int* b = (int*)malloc(n*sizeof(int));
    for(int i=0; i<n; i++)
      b[i] = a[i]%10;
	
	lastDigitResult res = {b};
	return res;
}


