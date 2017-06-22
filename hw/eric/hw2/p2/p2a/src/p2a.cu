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


__device__ int d_next_pow2(int n)
{
    int nBits = 0;
	while( (n>>nBits) > 0 )
		nBits++;
    return 1<<nBits;
}

__device__ inline void d_reduce_add_loop(int * B, int myId, int n)
{
	int n2 = d_next_pow2(blockDim.x);
	for(int s=n2/2; s > 0; s>>=1)
	{
		if( (threadIdx.x<s) && (threadIdx.x+s)<blockDim.x && (myId+s)<n)
		{
			B[myId] += B[myId+s];
		}
		__syncthreads();
	}
}


// A and B should be same size
__global__ void range_count_kernel(int * count, int * A, int n)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
	int myA = A[myId];

	for(int rangeBin=0; rangeBin<10; rangeBin++)
	{
		A[myId] = ((myA/100)==rangeBin);
		__syncthreads();
		d_reduce_add_loop(A, myId, n);		
		if(threadIdx.x==0)
		{
			count[blockIdx.x + gridDim.x*rangeBin] = A[myId];
		}
	}

}

__global__ void reduce_add_kernel(int * B, int * A, int rangeBin, int n)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;

	__syncthreads();
	d_reduce_add_loop(A, myId, n);
	if(threadIdx.x==0)
	{
		B[blockIdx.x + rangeBin*gridDim.x] = A[myId];
	}
}


result range_count_cuda(int *a, int n)
{

// allocate device memory
    int *d_A;
    cudaMalloc((int**) &d_A, sizeof(int)*n);

// copy input array to device
    cudaMemcpy(d_A, a, n*sizeof(int), cudaMemcpyHostToDevice);

// call kernel
    int threadsPerBlock = MIN(n,MAX_THREADS);
    int nBlocks = (n-1)/threadsPerBlock + 1;
	nBlocks = MAX(1,nBlocks);

	int *d_all_counts;
    cudaMalloc((int**) &d_all_counts, sizeof(int)*10*nBlocks);

// block level kernel call
	range_count_kernel<<<nBlocks,threadsPerBlock>>>(d_all_counts,d_A, n);
	cudaThreadSynchronize();

// reduce block results
	int *d_counts;
	do
	{
		int new_nBlocks = (nBlocks-1)/MAX_THREADS+1;
		threadsPerBlock = MIN(nBlocks,MAX_THREADS);

		cudaMalloc((int**) &d_counts,10*new_nBlocks*sizeof(int));

		for(int rangeBin=0; rangeBin<10; rangeBin++)
			reduce_add_kernel<<<new_nBlocks,threadsPerBlock>>>(d_counts, d_all_counts+nBlocks*rangeBin, rangeBin, nBlocks);
		cudaThreadSynchronize();

		cudaFree(d_all_counts);
		d_all_counts = d_counts;
		nBlocks = new_nBlocks;
	} while( nBlocks > 1 );


// copy result back to host
    int* counts = (int*)malloc(10*sizeof(int));
    cudaMemcpy(counts, d_counts, 10*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
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

	int exp = 21;
	int n = 1<<exp;
	n -=1;
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
