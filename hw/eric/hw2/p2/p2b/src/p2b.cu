#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
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


// loop for reduce-add
// contains protection in case n is not a power of 2
// and from reading off the end of the array
__device__ inline void d_reduce_add_loop(int * B, int myId, int n)
{
	int n2 = d_next_pow2(blockDim.x);
	for(int s=n2/2; s > 0; s>>=1)
	{
		if( d_checkReduceIndex(myId, s, n) )
		{
			B[myId] += B[myId+s];
			
		}
		__syncthreads();
	}
}


__global__ void range_count_kernel(int * count, int * B, int * A, int n)
{
	extern __shared__ int sdata[];
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
	int rangeBin = blockIdx.y;
	int * C = sdata + blockIdx.x*rangeBin;
 
	if(myId<n)
	{
		C[myId] = ((A[myId]/100)==rangeBin);
		__syncthreads();
		d_reduce_add_loop(C, myId, n);		
		if(threadIdx.x==0)
		{
			count[blockIdx.x + gridDim.x*rangeBin] = C[myId];
		}
	}
}


__global__ void reduce_add_kernel(int * B, int * A, int n)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
	int rangeBin = blockIdx.y;

	__syncthreads();
	d_reduce_add_loop(A+n*rangeBin, myId, n);
	if(threadIdx.x==0)
	{
		B[blockIdx.x + rangeBin*gridDim.x] = A[n*rangeBin + myId];
	}
}


result range_count_cuda(int *a, int n)
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
	range_count_kernel<<<blocks,threadsPerBlock,10*threadsPerBlock*sizeof(int)>>>(d_all_counts,d_temp,d_A, n);
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
		reduce_add_kernel<<<blocks,threadsPerBlock>>>(d_counts, d_all_counts, nBlocks);
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


int main(int argc, char** argv)
{

	int exp = 22;
	int n = 1<<exp;
	n -= 234;
	int* h_A = (int*)malloc(n*(sizeof(int)));

// make test array
	writeRandomFile(n, "inp.txt");
   	readIntsFromFile("inp.txt",n,h_A);

// get CUDA result
	result cudaResult = range_count_cuda(h_A, n);

// get sequential result
	result seqResult = range_count_seq(h_A, n);

// print results
	printf("n   SEQ      CUDA\n-----------------\n");
	for(int i=0; i<10; i++)
		printf("%d %8d %8d\n",i, seqResult.counts[i], cudaResult.counts[i]);

// free array memory
	free(h_A);
	free(seqResult.counts);
	free(cudaResult.counts);

    return 0;
}
