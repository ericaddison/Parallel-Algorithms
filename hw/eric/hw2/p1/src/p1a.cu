/**
 * Problem 1a functions: reduce-min
 */
#include "p1.h"


/**
 * reduce_min_kernel
 * The CUDA kernel for running reduce with a min operator.
 * Simple binary-tree style implementation.
 *
 * @param B result array. Should be same size as A
 * @param A input array
 * @param n size of input array A
 */
__global__ void reduce_min_kernel(int * B, int * A, int n)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;

	// reduce block array
	int n2 = d_next_pow2(blockDim.x);
	for(int s=n2/2; s > 0; s>>=1)
	{
		if( d_checkReduceIndex(myId, s, n) )
			A[myId] = MIN(A[myId],A[myId+s]);
		__syncthreads();
	}

	// thread 0 write result
	if(threadIdx.x==0)
		B[blockIdx.x] = A[myId];
}


/**
 * find_min_cuda
 * Host-side function to set up and launch CUDA min kernel.
 *
 * @param a input array
 * @param n size of input array a
 * @return minResult struct holding minimum of input array
 */
minResult find_min_cuda(int *a, int n)
{

// allocate device memory
    int *d_A;
    cudaMalloc((int**) &d_A, n*sizeof(int));

// copy input array to device
    cudaMemcpy(d_A, a, n*sizeof(int), cudaMemcpyHostToDevice);

// call kernel
    int threadsPerBlock = MIN(n,MAX_THREADS);
    int nBlocks = (n-1)/threadsPerBlock + 1;
	nBlocks = MAX(1,nBlocks);

	int * d_temp;
    cudaMalloc((int**) &d_temp, nBlocks*sizeof(int));

// block level kernel call
	reduce_min_kernel<<<nBlocks,threadsPerBlock>>>(d_temp, d_A, n);
	cudaThreadSynchronize();
    cudaFree(d_A);

// reduce block results
	int *d_min;
	do
	{
		int new_nBlocks = (nBlocks-1)/MAX_THREADS+1;
		threadsPerBlock = MIN(nBlocks,MAX_THREADS);

		cudaMalloc((int**) &d_min, new_nBlocks*sizeof(int));

		reduce_min_kernel<<<new_nBlocks,threadsPerBlock>>>(d_min, d_temp, nBlocks);
		cudaThreadSynchronize();

		cudaFree(d_temp);
		d_temp = d_min;
		nBlocks = new_nBlocks;
	} while( nBlocks > 1 );


// copy result back to host
    int min = 0;
    cudaMemcpy(&min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_min);

	minResult res = {min};
    return res;

}


/**
 * find_min_seq
 * Sequential implementation of find min. Simple O(n) loop.
 *
 * @param a input array
 * @param n size of input array a
 * @return minResult struct holding minimum of input array
 */
minResult find_min_seq(int* a, int n)
{
	int min = INT_MAX;
    for(int i=0; i<n; i++)
      min = MIN(min,a[i]);
	
	minResult res = {min};
	return res;
}


