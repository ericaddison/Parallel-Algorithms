#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
extern "C"
{
	#include "randomInts.h"
}

#define MIN(x,y) ((x<y)?x:y)
#define MAX(x,y) ((x>y)?x:y)
#define MAX_THREADS 1024

typedef struct {
    int* counts;
} result;


//***********************************************
// Device convenience functions

__device__ inline int d_next_pow2(int n)
{
    int nBits = 0;
	while( (n>>nBits) > 0 )
		nBits++;
    return 1<<nBits;
}


__device__ inline int d_checkReduceIndex(int myId, int s, int n)
{
	return (threadIdx.x<s) && (threadIdx.x+s)<blockDim.x && (myId+s)<n;
}


//***********************************************
// Problem 2a functions: use global memory

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




//***********************************************
// Problem 2b functions: use shared memory

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



//***********************************************
// Host calling functions

result range_count_cuda(int *a, int n, const char shared)
{

// allocate device memory
    int *d_A, *d_temp;
    cudaMalloc((int**) &d_A, n*sizeof(int));
    cudaMalloc((int**) &d_temp, 10*n*sizeof(int));
	cudaMemset(d_temp, 0, 10*n*sizeof(int));

// copy input array to device
    cudaMemcpy(d_A, a, n*sizeof(int), cudaMemcpyHostToDevice);

// call kernel
    int threadsPerBlock = MIN(n,MAX_THREADS);
    int nBlocks = (n-1)/threadsPerBlock + 1;
	nBlocks = MAX(1,nBlocks);

	int *d_all_counts;
    cudaMalloc((int**) &d_all_counts, sizeof(int)*10*nBlocks);

// block level kernel call
	dim3 blocks(nBlocks,10);
	if(shared)
		range_count_kernel_shared<<<blocks,threadsPerBlock,threadsPerBlock*sizeof(int)>>>(d_all_counts, d_A, n);
	else
		range_count_kernel_global<<<blocks,threadsPerBlock>>>(d_all_counts,d_temp,d_A, n);
	cudaThreadSynchronize();
    cudaFree(d_A);
	cudaFree(d_temp);

// reduce block results
	int *d_counts;
	do
	{
		int new_nBlocks = (nBlocks-1)/MAX_THREADS+1;
		threadsPerBlock = MIN(nBlocks,MAX_THREADS);

		cudaMalloc((int**) &d_counts,10*new_nBlocks*sizeof(int));

		blocks = dim3(new_nBlocks,10);
		if(shared)
			reduce_add_kernel_shared<<<blocks,threadsPerBlock,threadsPerBlock*sizeof(int)>>>(d_counts, d_all_counts, nBlocks);
		else
			reduce_add_kernel_global<<<blocks,threadsPerBlock>>>(d_counts, d_all_counts, nBlocks);
		cudaThreadSynchronize();

		cudaFree(d_all_counts);
		d_all_counts = d_counts;
		nBlocks = new_nBlocks;
	} while( nBlocks > 1 );


// copy result back to host
    int* counts = (int*)malloc(10*sizeof(int));
    cudaMemcpy(counts, d_counts, 10*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_counts);

	result res = {counts};
    return res;

}


result range_count_seq(int* a, int n)
{
	int* b = (int*)calloc(10,sizeof(int));
    for(int i=0; i<n; i++)
      b[a[i]/100]++;
	
	result res = {b};
	return res;
}


//***********************************************
// Main function

int main(int argc, char** argv)
{

// run through several times for run time stats
	
	const int NRUNS = 10;
    struct timeval t;
    gettimeofday(&t, NULL);
    srand(t.tv_usec);	

	int n = rand()%(1<<28);
	
	for(int irun=0; irun<NRUNS; irun++)
	{
		int* h_A = (int*)malloc(n*(sizeof(int)));

	// make test array
		writeRandomFile(n, "inp.txt");
	   	readIntsFromFile("inp.txt",n,h_A);

	// get CUDA result from global kernel
		result cudaResultGlobal = range_count_cuda(h_A, n, 0);

	// get CUDA result from shared memory kernel
		result cudaResultShared = range_count_cuda(h_A, n, 1);

	// get sequential result
		result seqResult = range_count_seq(h_A, n);

	// print results
		printf("n   SEQ      CUDA_G   CUDA_S\n-----------------\n");
		for(int i=0; i<10; i++)
		{
			printf("%d %8d %8d %8d",i, seqResult.counts[i],
				cudaResultGlobal.counts[i], cudaResultShared.counts[i]);
			if( seqResult.counts[i]!=cudaResultGlobal.counts[i] 
				|| seqResult.counts[i] != cudaResultShared.counts[i])
				printf ("  XXX");
			printf("\n");
		}

	// free array memory
		free(h_A);
		free(seqResult.counts);
		free(cudaResultGlobal.counts);
		free(cudaResultShared.counts);
	}

	printf("n = %d\n",n);
    return 0;
}
