#include "p3.h"
extern "C"
{
	#include "randomInts.h"
}

#define MAX_THREADS 4

enum repeatCheck { check, noCheck };

__device__ void d_hs_scan(int myId, int *A, int n, int initVal)
{
	if(myId==0)
		A[myId] += initVal;
	int myVal = A[myId];
	__syncthreads();

	for(int s=1; s<n; s*=2)
	{
		if(myId>=s)
			myVal += A[myId-s];
		__syncthreads();
		if(myId<n)
			A[myId] = myVal;
		__syncthreads();
	}
}


// perform a scan for radix sort
__global__ void radix_sort_kernel(int *A, int n, int nDigits)
{
	extern __shared__ int sdata[];
	int *left = sdata;
	int *right = sdata+n;
	int offset = blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	int myId = tid + offset;


	for(int iDigit=0; iDigit<nDigits; iDigit++)
	{
		int myVal = A[myId];
		int radix = 1<<iDigit;
		left[tid] = !(myVal&radix);
		right[tid] = !(left[tid]);
		__syncthreads();
	
	// scan
		d_hs_scan(tid, left, n, 0);
		d_hs_scan(tid, right, n, left[n-1]);

	// scatter
		int index = (myVal&radix)?(right[tid]-1):(left[tid]-1);
		A[index+offset] = myVal;
		__syncthreads();
	}
}



// this will merge two pieces of array A of size treeLevel*blockDim.x into d_out
__global__ void parallel_merge_kernel(int *d_out, int *A, int treeLevel)
{

    int mergeID = blockIdx.x/treeLevel;
    int n = treeLevel*blockDim.x;    

	int offset = (mergeID/(2*treeLevel))*2*n;
	int myInd = threadIdx.x +(blockIdx.x%treeLevel)*blockDim.x;
	//printf("threadID=%d, blockId=%d, mergeID=%d, offset=%d, myInd=%d\n",threadIdx.x, blockIdx.x, mergeID, offset, myInd);
	A = A+offset;
	int *B = A+n;

	// swap A and B if this is an odd block
	if(mergeID%2)
	{
		int *C = A;
		A = B;
		B = C;
	}
		
	int otherInd = d_binary_search(B, A[myInd], n);
	int mergedIndex = myInd + otherInd;
	int nRepeats=0;

	// sensitive to repeated elements if an odd block
	if(mergeID%2)
	{
	 	nRepeats = otherInd - d_binary_search(B,A[myInd]-1,n);		
		mergedIndex -= nRepeats;
	}

	d_out[mergedIndex+offset] = A[myInd];	
}




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

/*
	// print A
	printf("A: ");
		cudaMemcpy(h_A, d_A, np2*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<np2; i++)
	{
		if(i%MAX_THREADS==0 && i>0)
			printf(" | ");
		printf("%d, ",h_A[i]);
	}
	printf("\n");
*/
	// merge sorted blocks
	int cnt=1;
//	int nMerges = nBlocks;
	while((nBlocks/cnt)>1)
	{
//		printf("merging round %d\n",cnt, nBlocks);
		parallel_merge_kernel<<<nBlocks,threadsPerBlock>>>(d_B,d_A,cnt);
		cudaThreadSynchronize();
/*
	// print B
		printf("B: ");
		cudaMemcpy(h_A, d_B, np2*sizeof(int), cudaMemcpyDeviceToHost);
		for(int i=0; i<np2; i++)
		{
			if(i%(MAX_THREADS*cnt*2)==0 && i>0)
				printf(" | ");
			printf("%d, ",h_A[i]);
		}
		printf("\n");
*/
		cnt*=2;
//		nMerges /= 2;
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

}


int main()
{

	int MAX_EXP = 5;
    struct timeval t;
    gettimeofday(&t, NULL);
    srand(t.tv_usec);
    double exp = (MAX_EXP*( (double)rand()/(double)RAND_MAX));
    int n = (int)pow(2,exp); 
	
	printf("MAX_THREADS = %d\n",MAX_THREADS);
	printf("n = %d\n",n);

	// make test array
	int* A = (int*)malloc(n*(sizeof(int)));
	writeRandomFile(n, "inp.txt");
	readIntsFromFile("inp.txt",n,A);

	printf("\n");
	for(int i=0; i<n; i++)
		printf("%d, ",h_A[i]);
	printf("\n");

	cuda_radix_sort(A, n, 10);

	// print A
	printf("A: ");
	for(int i=0; i<n; i++)
		printf("%d, ",h_A[i]);
	printf("\n");


	printf("Array is %s sorted\n", (checkSorted(h_A,n)?"\b":"NOT"));

	free(A);

	
}

