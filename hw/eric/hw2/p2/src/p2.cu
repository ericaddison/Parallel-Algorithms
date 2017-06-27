#include "p2.h"
extern "C"
{
	#include "randomInts.h"
}

#define MAX_THREADS 1024

typedef struct {
    int* counts;
} result;


//***********************************************
// Host calling functions

result range_count_cuda(int *a, int n, const int shared)
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
		range_count_kernel_shared<<<blocks,threadsPerBlock,threadsPerBlock*sizeof(int)>>>(d_all_counts,d_A, n);
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
    double exp = (26.0*( (double)rand()/(double)RAND_MAX));
    int n = (int)pow(2,exp); 
	int errCnt = 0;
	
	for(int irun=0; irun<NRUNS; irun++)
	{
		int* h_A = (int*)malloc(n*(sizeof(int)));

	// make test array
		writeRandomFile(n, "inp.txt");
	   	readIntsFromFile("inp.txt",n,h_A);

	// get global memory CUDA result
		result cudaGlobalResult = range_count_cuda(h_A, n, 0);

	// get shared memory CUDA result
		result cudaSharedResult = range_count_cuda(h_A, n, 1);

	// get sequential result
		result seqResult = range_count_seq(h_A, n);

	// print results
		printf("\nrun SEQ      CUDA_G  CUDA_S\n-----------------\n");
		for(int i=0; i<10; i++)
		{
			printf("%d %8d %8d %8d",i, seqResult.counts[i], cudaSharedResult.counts[i], cudaSharedResult.counts[i]);
			if( seqResult.counts[i]!=cudaGlobalResult.counts[i] || seqResult.counts[i]!=cudaSharedResult.counts[i])
			{
				printf ("  XXX");
				errCnt++;
			}
			printf("\n");
		}

	// free array memory
		free(h_A);
		free(seqResult.counts);
		free(cudaGlobalResult.counts);
		free(cudaSharedResult.counts);
	}

	printf("n = %d\nerrCnt = %d\n",n,errCnt);
    return 0;
}
