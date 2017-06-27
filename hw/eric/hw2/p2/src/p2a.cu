//***********************************************
// Problem 2a functions: use global memory

#include "p2.h"

__global__ void range_count_kernel_global(int * count, int * B, int * A, int n)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
	int rangeBin = blockIdx.y;
	int * C = B + n*rangeBin;
 
	// create indicator array
	if(myId<n)
		C[myId] = ((A[myId]/100)==rangeBin);
	__syncthreads();

	// reduce indicator array
	int n2 = d_next_pow2(blockDim.x);
	for(int s=n2/2; s > 0; s>>=1)
	{
		if( d_checkReduceIndex(myId, s, n) )
			C[myId] += C[myId+s];
		__syncthreads();
	}

	// thread 0 write result
	if(threadIdx.x==0)
		count[blockIdx.x + gridDim.x*rangeBin] = C[myId];
}


__global__ void reduce_add_kernel_global(int * B, int * A, int n)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
	int rangeBin = blockIdx.y;
	int * C = A + n*rangeBin;

	// reduce block array
	int n2 = d_next_pow2(blockDim.x);
	for(int s=n2/2; s > 0; s>>=1)
	{
		if( d_checkReduceIndex(myId, s, n) )
			C[myId] += C[myId+s];
		__syncthreads();
	}

	// thread 0 write result
	if(threadIdx.x==0)
		B[blockIdx.x + rangeBin*gridDim.x] = C[myId];
}


