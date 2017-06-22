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

typedef struct {
    int* counts;
    float time;
    float innerTime;
} result;


__device__ int d_next_pow2(int n)
{
    int nBits = 0;
	while( (n>>nBits) > 0 )
		nBits++;
    return 1<<nBits;
}

__device__ inline void d_reduce_add_loop(int * B, int myId)
{
	int n = d_next_pow2(blockDim.x);
	for(int s=n/2; s > 0; s>>=1)
	{
		if( (threadIdx.x<s) && (threadIdx.x+s)<blockDim.x)
		{
			B[myId] += B[myId+s];
		}
		__syncthreads();
	}
}

// A and B should be same size
__global__ void range_count_kernel(int * count, int * A)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
	int myA = A[myId];

	for(int rangeBin=0; rangeBin<10; rangeBin++)
	{
		A[myId] = ((myA/100)==rangeBin);
		__syncthreads();
		d_reduce_add_loop(A, myId);		
		if(tid==0)
		{
			count[bid + gridDim.x*rangeBin] = A[myId];
		}
	}

}

__global__ void reduce_add_kernel(int * B, int * A, int rangeBin)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;

	__syncthreads();
	d_reduce_add_loop(A, myId);
	if(threadIdx.x==0)
	{
		B[rangeBin + 10*gridDim.x] = A[myId];
		printf("from block=%d, i=%d, wrote %d -- %d\n",blockIdx.x, rangeBin, A[myId],B[rangeBin + 10*gridDim.x]);
	}
}

__global__ void print_vec_kernel(int * v, int n)
{
	for(int i=0; i<n; i++)
		printf("from GPU: v[%d] = %d\n",i,v[i]);
}

result range_count_cuda(int *a, int n)
{

// Cuda timing setup
    cudaEvent_t startInner, stopInner;
    cudaEvent_t startOuter, stopOuter;
    float elapsedTimeInner, elapsedTimeOuter;
    cudaEventCreate(&startInner);
    cudaEventCreate(&startOuter);
    cudaEventCreate(&stopInner);
    cudaEventCreate(&stopOuter);


    cudaEventRecord(startOuter,0);
// allocate device memory
    int *d_A;
    cudaMalloc((int**) &d_A, sizeof(int)*n);

// copy input array to device
    cudaMemcpy(d_A, a, n*sizeof(int), cudaMemcpyHostToDevice);

// call kernel
    cudaEventRecord(startInner,0);
    int threadsPerBlock = MIN(n,1024);
    int nBlocks = (n-1)/threadsPerBlock + 1;
	nBlocks = MAX(1,nBlocks);

	int *d_all_counts;
    cudaMalloc((int**) &d_all_counts, sizeof(int)*10*nBlocks);

	// block level kernel call
	range_count_kernel<<<nBlocks,threadsPerBlock>>>(d_all_counts,d_A);
	cudaThreadSynchronize();

	// at this point, each group of 1024 elements (each block) has computed
	// its counts and stored them in the d_all_counts array
	// next, need to reduce those results, taking care in case the number
	// of blocks is greater than 1024

	// reduce block results
	int *d_counts;
	int n_1024_blocks = (nBlocks-1)/1024+1;
	cudaMalloc((int**) &d_counts,10*n_1024_blocks*sizeof(int));
	threadsPerBlock = MIN(nBlocks,1024);
	for(int rangeBin=0; rangeBin<10; rangeBin++)
		reduce_add_kernel<<<n_1024_blocks,threadsPerBlock>>>(d_counts, d_all_counts+nBlocks*rangeBin, rangeBin);
	cudaThreadSynchronize();


	print_vec_kernel<<<1,1>>>(d_counts,10*n_1024_blocks);

    cudaEventRecord(stopInner,0);
    cudaEventSynchronize(stopInner);
    cudaEventElapsedTime(&elapsedTimeInner, startInner, stopInner);

// copy result back to host
    int* counts = (int*)malloc(10*n_1024_blocks*sizeof(int));
    cudaMemcpy(counts, d_counts, 10*n_1024_blocks*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<10*n_1024_blocks; i++)
		printf("outof: %d, %d, %d\n",i,i/n_1024_blocks, counts[i]);

    cudaFree(d_A);
    cudaFree(d_all_counts);
    cudaFree(d_counts);

    cudaEventRecord(stopOuter,0);
    cudaEventSynchronize(stopOuter);
    cudaEventElapsedTime(&elapsedTimeOuter, startOuter, stopOuter);

	result res = {counts, elapsedTimeOuter, elapsedTimeInner};
    return res;

}



result range_count_seq(int* a, int n)
{
// Cuda timing setup
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);


	int* b = (int*)calloc(10,sizeof(int));
    for(int i=0; i<n; i++)
      b[a[i]/100]++;
	
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

	result res = {b, elapsedTime, 0.0f};
	return res;
}


int main(int argc, char** argv)
{

	const int N_RUNS = 1;
	const int MAX_POW = 20;
	int cnt = 0;

	//for(int exp=20; exp<=MAX_POW; exp++)
	{
		int exp = 20;
		int n = 1<<exp;
		
		for(int iRun=0; iRun<N_RUNS; iRun++)
		{
			int* h_A = (int*)malloc(n*(sizeof(int)));

		// make test array
			writeRandomFile(n, "inp.txt");
    		readIntsFromFile("inp.txt",n,h_A);

		// get CUDA result
    		result cudaResult = range_count_cuda(h_A, n);

		// get sequential result
    		result seqResult = range_count_seq(h_A, n);

			int nErrors = 0;
			for(int i=0; i<10; i++)
				printf("%d: %d -- %d\n",i, seqResult.counts[i], cudaResult.counts[i]);
			
//				nErrors += (cudaResult.lastDigit[i] != seqResult.lastDigit[i]);

//			printf("%d, %d, %d, ",cnt, n, iRun);
//			printf("%d, ",nErrors);
//			printf("%f, %f, %f\n",seqResult.time, cudaResult.time, cudaResult.innerTime);

		// free array memory
			free(h_A);
			free(seqResult.counts);
			free(cudaResult.counts);

		// inrement counter
			cnt++;
		}

	}


    return 0;
}
