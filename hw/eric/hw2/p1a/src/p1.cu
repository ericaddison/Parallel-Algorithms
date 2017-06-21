#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C"
{
	#include <randomInts.h>
}

#define MIN(x,y) ((x<y)?x:y)

typedef struct {
    int minVal;
    float time;
    float innerTime;
} minResult;

__device__ int nextPow2(int n)
{
    int nBits = (int)ceilf(log2f(1.0f*n));
    return 1<<nBits;
}



__global__ void reduce_min_kernel(int * d_out, int * d_in, int n)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;

    int n2 = nextPow2(n);
    for (unsigned int s = n2/2; s > 0; s >>= 1)
    {
        if ( (myId < s) && (myId+s) < n )
            d_in[myId] = MIN(d_in[myId],d_in[myId + s]);
        __syncthreads();
    }

    if (myId == 0)
        d_out[0] = d_in[0];
}



minResult min_seq(int* a, int n)
{
// Cuda timing setup
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    int minVal = INT_MAX;
    for(int i=0; i<n; i++)
      minVal = MIN(minVal,a[i]);
	
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

	minResult res = {minVal, elapsedTime, 0.0f};
	return res;
}



minResult min_cuda(int *a, int n)
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
    int *d_A, *d_B;
    cudaMalloc((int**) &d_A, sizeof(int)*n);
    cudaMalloc((int**) &d_B, sizeof(int));

// copy input array to device
    cudaMemcpy(d_A, a, n*sizeof(int), cudaMemcpyHostToDevice);

// call kernel
    cudaEventRecord(startInner,0);
    int threadsPerBlock = 512;
    int nBlocks = n/threadsPerBlock + ((n%threadsPerBlock==0)?0:1);

    reduce_min_kernel<<<nBlocks,threadsPerBlock>>>(d_B,d_A,n);

    cudaEventRecord(stopInner,0);
    cudaEventSynchronize(stopInner);
    cudaEventElapsedTime(&elapsedTimeInner, startInner, stopInner);

// copy result back to host
    int minVal;
    cudaMemcpy(&minVal, d_B, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    cudaEventRecord(stopOuter,0);
    cudaEventSynchronize(stopOuter);
    cudaEventElapsedTime(&elapsedTimeOuter, startOuter, stopOuter);

	minResult res = {minVal, elapsedTimeOuter, elapsedTimeInner};
    return res;

}





int main(int argc, char** argv)
{

	const int N_RUNS = 100;
	const int MAX_POW = 25;
	int cnt = 0;

	for(int exp=10; exp<MAX_POW; exp++)
	{
		int n = 1<<exp;

		for(int iRun=0; iRun<N_RUNS; iRun++)
		{
			int* h_A = (int*)malloc(n*(sizeof(int)));

		// make test array
			writeRandomFile(n, "inp.txt");
    		readIntsFromFile("inp.txt",n,h_A);

		// get CUDA result
    		minResult cudaMin = min_cuda(h_A, n);

		// get sequential result
    		minResult seqMin = min_seq(h_A, n);

			printf("%d, %d, %d, ",cnt, n, iRun);
			printf("%d, %d, ",seqMin.minVal, cudaMin.minVal);
			printf("%f, %f, %f\n",seqMin.time, cudaMin.time, cudaMin.innerTime);

		// free array memory
			free(h_A);


		if(seqMin.minVal!=cudaMin.minVal)
			return 1;

		// inrement counter
			cnt++;
		}

	}


    return 0;
}
