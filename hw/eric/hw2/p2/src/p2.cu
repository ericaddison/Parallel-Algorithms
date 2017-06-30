/**
 * Problem 2: histogram and CDF
 */
#include "p2.h"
extern "C"
{
	#include "randomInts.h"
}

#define MAX_THREADS 1024

/**
 * result struct to hold both histogram and scan results
 * of a call to one part of problem 2.
 */
typedef struct {
    int* histogram;
	int* scan;
} result;



/**
 * range_count_cuda
 * Host-side function to set up and launch CUDA range_count kernel.
 * This function first calls the range_count_kernel for as many 
 * blocks as necessary, and then reduces the resulting histograms
 * repeatedly until all have been aggregated.
 *
 * @param a input array
 * @param n size of input array a
 * @param shared flag whether to call kernel using shared memory, 
 * @return result struct holding histogram and CDF results
 */
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
    int* histogram = (int*)malloc(10*sizeof(int));
    cudaMemcpy(histogram, d_counts, 10*sizeof(int), cudaMemcpyDeviceToHost);

// scan on the resulting histogram for CDF
// known data size n=10
	hs_scan_kernel<<<1,10,10*sizeof(int)>>>(d_counts,10);
	cudaThreadSynchronize();
    int* scan = (int*)malloc(10*sizeof(int));
    cudaMemcpy(scan, d_counts, 10*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_counts);

	result res = {histogram, scan};
    return res;

}



/**
 * range_count_seq
 * Sequential implementation of range_count (i.e. histogram)
 * 
 * @param a input array
 * @param n size of input array a
 * @return result struct holding histogram and CDF results
 */
result range_count_seq(int* a, int n)
{
	int* hist = (int*)calloc(10,sizeof(int));
	int* scan = (int*)calloc(10,sizeof(int));
	
    for(int i=0; i<n; i++)
      hist[a[i]/100]++;
	
	scan[0] = hist[0];
	for(int i=1; i<10; i++)
		scan[i] = scan[i-1] + hist[i];

	result res = {hist, scan};
	return res;
}



/**
 * main
 * Main function for problem 2. Expects paths to input
 * files as command line arguments. Loops through 
 * paths and calls range_count for CUDA global, CUDA shared,
 * and sequential algorithms and compares.
 * Results are printed to STDOUT.
 *
 * @param argc number of command line arguments
 * @param argv command line argument strings
 */
int main(int argc, char** argv)
{

	if(argc<2)
		printf("No input files specified\n\n");

	for(int i=1; i<argc; i++)
	{
		char* nextFile = argv[i];
		printf("\n***********************************\n");
		printf("Running p2 for file %s\n",nextFile);
		printf("***********************************\n\n");

	//  read array file
	   	randArray ra = readIntsFromFile(nextFile);
		int* h_A = ra.A;
		int n = ra.n;
		int errCnt = 0;

	// get global memory CUDA result
		result cudaGlobalResult = range_count_cuda(h_A, n, 0);

	// get shared memory CUDA result
		result cudaSharedResult = range_count_cuda(h_A, n, 1);

	// get sequential result
		result seqResult = range_count_seq(h_A, n);

	// print results
		printf("\nbin SEQ_H    CUDA_G_H CUDA_S_H  |  SEQ_CM  CUDA_G_CM  CUDA_S_CM\n");
		printf("---------------------------------------------------------\n");
		for(int i=0; i<10; i++)
		{
			printf("%d %8d ",i, seqResult.histogram[i]);
			printf("%8d ", cudaSharedResult.histogram[i]);
			printf("%8d", cudaSharedResult.histogram[i]);
			printf("   |  %8d ", seqResult.scan[i]);
			printf("%8d ", cudaSharedResult.scan[i]);
			printf("%8d", cudaSharedResult.scan[i]);
			if( seqResult.histogram[i]!=cudaGlobalResult.histogram[i] 
				|| seqResult.histogram[i]!=cudaSharedResult.histogram[i]
				|| seqResult.scan[i]!=cudaGlobalResult.scan[i]
				|| seqResult.scan[i]!=cudaSharedResult.scan[i])
			{
				printf ("  XXX");
				errCnt++;
			}
			printf("\n");
		}

	// free array memory
		free(h_A);
		free(seqResult.histogram);
		free(cudaGlobalResult.histogram);
		free(cudaSharedResult.histogram);
		free(cudaSharedResult.scan);
		free(cudaGlobalResult.scan);

		printf("n = %d\nerrCnt = %d\n",n,errCnt);
		printf("Done with file %s\n\n",nextFile);
	}

    return 0;
}
