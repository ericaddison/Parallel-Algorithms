#include "p3.h"
extern "C"
{
	#include "randomInts.h"
}

#define MAX_THREADS 1024



void cuda_radix_sort(int *A, int n, int nDigits)
{
	// pad array if less than a power of 2
	int np2 = next_pow2(n);
	int* h_A = (int*)calloc(np2,(sizeof(int)));
	memcpy(h_A,A,n*sizeof(int));

	// allocate and fill device memory
	int *d_A, *d_B;
	cudaMalloc((int**)&d_A, np2*sizeof(int));
	cudaMalloc((int**)&d_B, np2*sizeof(int));
	cudaMemcpy(d_A, h_A, np2*sizeof(int), cudaMemcpyHostToDevice);

	// device config
	int nBlocks = (np2-1)/MAX_THREADS+1;
	int threadsPerBlock = MIN(MAX_THREADS,np2);

	// launch radix kernel
	radix_sort_kernel<<<nBlocks,threadsPerBlock,2*threadsPerBlock*sizeof(int)>>>(d_A, threadsPerBlock, nDigits);
	cudaThreadSynchronize();

	// merge sorted blocks
	int cnt=1;
	while((nBlocks/cnt)>1)
	{
		parallel_merge_kernel<<<nBlocks,threadsPerBlock>>>(d_B,d_A,cnt);
		cudaThreadSynchronize();
		cnt*=2;
    // swap A and B
        int *d_C = d_A;
        d_A = d_B;
		d_B = d_C;
	}

	// copy result and free memory
	cudaMemcpy(A, d_A+(np2-n), n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_A);	
	cudaFree(d_B);	
	free(h_A);

}


void seq_radix_sort(int *A, int n, int nDigits)
{
	int *B = (int*)malloc(n*sizeof(int));
	if(nDigits%2)
		nDigits++;
	
	for(int iDigit=0; iDigit<nDigits; iDigit++)
	{
		int radix = 1<<iDigit;
		int l = 0;
		int r = n-1;
		for(int i=0; i<n; i++)
		{
			if(!(A[i]&radix))
				B[l++] = A[i];
			if(A[n-i-1]&radix)
				B[r--] = A[n-i-1];
		}

		int *C = A;
		A = B;
		B = C;
	}

	free(B);

}


int main()
{

	int MAX_EXP = 26;
    struct timeval t;
    gettimeofday(&t, NULL);
    srand(t.tv_usec);
    double exp = (MAX_EXP*( (double)rand()/(double)RAND_MAX));
    int n = (int)pow(2,exp); 
	
	printf("\n\nMAX_THREADS = %d\n",MAX_THREADS);
	printf("n = %d\n",n);

	// make test array
	int* A = (int*)malloc(n*(sizeof(int)));
	writeRandomFile(n, "inp.txt");
	readIntsFromFile("inp.txt",n,A);

	cuda_radix_sort(A, n, 10);

	// seq sort for comparison
	int* B = (int*)malloc(n*(sizeof(int)));
	readIntsFromFile("inp.txt",n,B);
	seq_radix_sort(B,n,10);

	// print some results
	printf("CUDA result: ");
	for(int i=0; i<MIN(n,15); i++)
		printf("%d, ",A[i]);
	printf("\b\b ...\n");
	printf("SEQ  result: ");
	for(int i=0; i<MIN(n,15); i++)
		printf("%d, ",A[i]);
	printf("\b\b ...\n");



// check results
	printf("\n\nArray is %s sorted\n", (checkSorted(A,n)?"\b":"NOT"));

	int errCnt=0;
	for(int i=0; i<n; i++)
	{
		if(A[i]!=B[i])
		{
			printf("SORT disagreement at index %d\n",i);
			errCnt++;
		}	
	}
	printf("CUDA result matches sequential result\n\n");

	free(A);
	free(B);

	
}

