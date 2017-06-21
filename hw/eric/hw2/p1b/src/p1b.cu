#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <cuda_runtime.h>
#include "randomInts.h"

#define MIN(x,y) ((x<y)?x:y)
#define MAX(x,y) ((x>y)?x:y)

typedef struct {
    int* lastDigit;
    float time;
    float innerTime;
} result;


__device__ int d_get_last_digit(int x)
{
	int ndigits = (int)log10f(x);
	for(int i=0; i<ndigits; i++)
	{
		int pow10 = (int)roundf(powf(10,(ndigits-i)));
		int d = (x/pow10);
		x -= d*pow10;
	}

	return x;
}

__global__ void last_digit_kernel(int * d_out, int * d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    d_out[myId] = d_get_last_digit( d_in[myId]);
}


int h_get_last_digit(int x)
{
	int ndigits = (int)log10(x);
	for(int i=0; i<ndigits; i++)
	{
		int pow10 = (int)pow(10,(ndigits-i));
		int d = (x/pow10);
		x -= d*pow10;
	}

	return x;
}


result last_digit_seq(int* a, int n)
{
// Cuda timing setup
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);


	int* b = (int*)malloc(n*sizeof(int));
    for(int i=0; i<n; i++)
      b[i] = h_get_last_digit(a[i]);
	
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

	result res = {b, elapsedTime, 0.0f};
	return res;
}



result last_digit_cuda(int *a, int n)
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
    cudaMalloc((int**) &d_B, sizeof(int)*n);

// copy input array to device
    cudaMemcpy(d_A, a, n*sizeof(int), cudaMemcpyHostToDevice);

// call kernel
    cudaEventRecord(startInner,0);
    int threadsPerBlock = MIN(n,1024);
    int nBlocks = n/threadsPerBlock + ((n%threadsPerBlock==0)?0:1);
	nBlocks = MAX(1,nBlocks);

    last_digit_kernel<<<nBlocks,threadsPerBlock>>>(d_B,d_A);

    cudaEventRecord(stopInner,0);
    cudaEventSynchronize(stopInner);
    cudaEventElapsedTime(&elapsedTimeInner, startInner, stopInner);

// copy result back to host
    int* digits = (int*)malloc(n*sizeof(int));
    cudaMemcpy(digits, d_B, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    cudaEventRecord(stopOuter,0);
    cudaEventSynchronize(stopOuter);
    cudaEventElapsedTime(&elapsedTimeOuter, startOuter, stopOuter);

	result res = {digits, elapsedTimeOuter, elapsedTimeInner};
    return res;

}


int main(int argc, char** argv)
{

	const int N_RUNS = 100;
	const int MAX_POW = 25;
	int cnt = 0;

	//for(int exp=10; exp<MAX_POW; exp++)
	{
		int exp = 3;
		int iRun = 0;
		int n = 1<<exp;
		
		//for(int iRun=0; iRun<N_RUNS; iRun++)
		{
			int* h_A = (int*)malloc(n*(sizeof(int)));

		// make test array
			writeRandomFile(n, "inp.txt");
    		readIntsFromFile("inp.txt",n,h_A);

		// get CUDA result
    		result cudaMin = last_digit_cuda(h_A, n);

		// get sequential result
    		result seqMin = last_digit_seq(h_A, n);

			int nErrors = 0;
			for(int i=0; i<n; i++)
				nErrors += (cudaMin.lastDigit[i] != seqMin.lastDigit[i]);


			printf("%d, %d, %d, ",cnt, n, iRun);
			printf("%d, ",nErrors);
			printf("%f, %f, %f\n",seqMin.time, cudaMin.time, cudaMin.innerTime);

		// free array memory
			free(h_A);
			free(seqMin.lastDigit);
			free(cudaMin.lastDigit);


		// inrement counter
			cnt++;
		}

	}


    return 0;
}
