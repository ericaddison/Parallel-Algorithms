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
	int n;
    float time;
    float innerTime;
} result;


__device__ int nextPow2(int n)
{
    int nBits = 0;
	while( (n>>nBits) > 0 )
		nBits++;
    return 1<<nBits;
}

__device__ inline void simple_reduce_add(int * B, int myId)
{
	int n = nextPow2(blockDim.x);
	for(int s=n/2; s > 0; s>>=1)
	{
		if( (threadIdx.x<s) && (threadIdx.x+s)<blockDim.x)
		{
			B[myId] += B[myId+s];
		}
		__syncthreads();
	}
}


__global__ void range_count_kernel(int * count, int * reduceArray, int * A)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int bdim = blockDim.x;
    int myId = tid + bid * bdim;

	for(int rangeBin=0; rangeBin<10; rangeBin++)
	{
		reduceArray[myId] = ((A[myId]/100)==rangeBin);
		__syncthreads();
		simple_reduce_add(reduceArray, myId);		
		if(tid==0)
		{
			count[bid + gridDim.x*rangeBin] = reduceArray[myId];
		}
	}

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
    int *d_A, *d_temp, *d_count;
    cudaMalloc((int**) &d_A, sizeof(int)*n);
    cudaMalloc((int**) &d_temp, sizeof(int)*n);

// copy input array to device
    cudaMemcpy(d_A, a, n*sizeof(int), cudaMemcpyHostToDevice);

// call kernel
    cudaEventRecord(startInner,0);
    int threadsPerBlock = MIN(n,1024);
    int nBlocks = n/threadsPerBlock + ((n%threadsPerBlock==0)?0:1);
	nBlocks = MAX(1,nBlocks);
    cudaMalloc((int**) &d_count, sizeof(int)*10*nBlocks);

	// block level kernel call
	range_count_kernel<<<nBlocks,threadsPerBlock>>>(d_count,d_temp,d_A);
	cudaThreadSynchronize();

    cudaEventRecord(stopInner,0);
    cudaEventSynchronize(stopInner);
    cudaEventElapsedTime(&elapsedTimeInner, startInner, stopInner);

// copy result back to host
    int* counts = (int*)malloc(10*nBlocks*sizeof(int));
    cudaMemcpy(counts, d_count, 10*nBlocks*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_temp);
    cudaFree(d_count);

    cudaEventRecord(stopOuter,0);
    cudaEventSynchronize(stopOuter);
    cudaEventElapsedTime(&elapsedTimeOuter, startOuter, stopOuter);

	result res = {counts, nBlocks, elapsedTimeOuter, elapsedTimeInner};
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

	result res = {b, 0, elapsedTime, 0.0f};
	return res;
}


int main(int argc, char** argv)
{

	const int N_RUNS = 1;
	const int MAX_POW = 20;
	int cnt = 0;

	//for(int exp=20; exp<=MAX_POW; exp++)
	{
		int exp = 11;
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
				printf("SEQ: %d: %d\n",i, seqResult.counts[i]);
			for(int i=0; i<10*cudaResult.n; i++)
				printf("%d, %d: %d\n",i,i/cudaResult.n, cudaResult.counts[i]);
			
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
