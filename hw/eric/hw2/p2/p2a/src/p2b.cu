//***********************************************
// Problem 2b functions: use shared memory

#include "p2.h"

__global__ void range_count_kernel_shared(int * count, int * A, int n)
{
	extern __shared__ int sdata[];
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
	int rangeBin = blockIdx.y;
 
	// create indicator array
	sdata[tid] = ((A[myId]/100)==rangeBin);
	__syncthreads();

	// reduce indicator array
	int n2 = d_next_pow2(blockDim.x);
	for(int s=n2/2; s > 0; s>>=1)
	{
		if( d_checkReduceIndex(myId, s, n) )
			sdata[tid] += sdata[tid+s];
		__syncthreads();
	}

	// thread 0 write result
	if(tid==0)
		count[blockIdx.x + gridDim.x*rangeBin] = sdata[tid];
}


__global__ void reduce_add_kernel_shared(int * B, int * A, int n)
{
	extern __shared__ int sdata[];
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
	int rangeBin = blockIdx.y;

	// copy value from global to shared memory
	sdata[tid] = A[myId + n*rangeBin];

	// reduce block array
	int n2 = d_next_pow2(blockDim.x);
	for(int s=n2/2; s > 0; s>>=1)
	{
		if( d_checkReduceIndex(myId, s, n) )
			sdata[tid] += sdata[tid+s];
		__syncthreads();
	}

	// thread 0 write result
	if(tid==0)
		B[blockIdx.x + rangeBin*gridDim.x] = sdata[tid];
}
