/**
 * Problem 2a functions: use global memory
 */
#include "p2.h"


/**
 * hs_scan_kernel
 * In-place Hillis-Steele scan CUDA kernel.
 *
 * @param A input array
 * @param n size of input array
 */
__global__ void hs_scan_kernel(int *A, int n)
{
	extern __shared__ int sdata[];
	int myId = threadIdx.x + blockIdx.x * blockDim.x;

	int myVal = A[myId];
	sdata[myId] = A[myId];
	__syncthreads();

	for(int s=1; s<n; s*=2)
	{
		if(myId>=s)
			myVal = sdata[myId] + sdata[myId-s];
		__syncthreads();
		sdata[myId] = myVal;
		__syncthreads();
	}

	A[myId] = myVal;

}
