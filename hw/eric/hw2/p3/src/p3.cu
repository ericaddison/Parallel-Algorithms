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



// this will merge two pieces of array A of size n into the first 2n entries of d_out
__global__ void parallel_merge_kernel(int *d_out, int *A, int treeLevel)
{

    int mergeID = blockIdx.x/treeLevel;
    int n = treeLevel*blockDim.x;    

	int offset = (mergeID/2)*n;
	int tid = threadIdx.x + offset + (blockIdx.x%treeLevel)*blockDim.x;
	A = A+offset;
	int *B = A+n;

	// swap A and B if this is blockIdx.y==1
	if(mergeID%2)
	{
		int *C = A;
		A = B;
		B = C;
	}
		
	int otherInd = d_binary_search(B, A[tid], n);
	int mergedIndex = tid + otherInd;
	int nRepeats=0;

	// sensitive to repeated elements if blockIdx.y==1
	if(mergeID%2)
	{
	 	nRepeats = otherInd - d_binary_search(B,A[tid]-1,n);		
		mergedIndex -= nRepeats;
	}

	d_out[mergedIndex+offset] = A[tid];	
}




int main()
{

	int MAX_EXP = 5;
    struct timeval t;
    gettimeofday(&t, NULL);
    srand(t.tv_usec);
    double exp = (MAX_EXP*( (double)rand()/(double)RAND_MAX));
    int n = (int)pow(2,exp); 
	n = 10;
	
	printf("MAX_THREADS = %d\n",MAX_THREADS);
	printf("n = %d\n",n);


	// pad array if less than a power of 2
	int np2 = next_pow2(n);
	int* h_A = (int*)malloc(np2*(sizeof(int)));

	// make test array
	//writeRandomFile(n, "inp.txt");
	readIntsFromFile("inp.txt",n,h_A);
	//int h_A[] = {3, 5, 5, 2, 5, 1, 7, 7};

	int nDigits = 10;
	printf("\n");
	for(int i=0; i<n; i++)
		printf("%d, ",h_A[i]);
	printf("\n");

	
	int *d_A, *d_B;
	cudaMalloc((int**)&d_A, np2*sizeof(int));
	cudaMalloc((int**)&d_B, np2*sizeof(int));
	cudaMemset(d_A, 0, np2*sizeof(int));
	cudaMemcpy(d_A, h_A, n*sizeof(int), cudaMemcpyHostToDevice);

	int nBlocks = (np2-1)/MAX_THREADS+1;
	int threadsPerBlock = MIN(MAX_THREADS,np2);

	radix_sort_kernel<<<nBlocks,threadsPerBlock,2*threadsPerBlock*sizeof(int)>>>(d_A, threadsPerBlock, nDigits);
	cudaThreadSynchronize();


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

	if(nBlocks>1)
	{
		int cnt=1;
		int nMerges = nBlocks;
		while(nMerges>1)
		{
			printf("merging round %d\n",cnt, nBlocks);
			parallel_merge_kernel<<<nBlocks,threadsPerBlock>>>(d_B,d_A,cnt);
			cudaThreadSynchronize();

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
			cnt*=2;
			nMerges /= 2;

            // swap A and B
            int *d_C = d_A;
            d_A = d_B;
            d_B = d_C;

		}
		//cudaMemcpy(h_A, d_B+(np2-n), n*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_A, d_A, np2*sizeof(int), cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaMemcpy(h_A, d_A+(np2-n), n*sizeof(int), cudaMemcpyDeviceToHost);
	}



	// print A
	printf("A: ");
		cudaMemcpy(h_A, d_A, np2*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<np2; i++)
		printf("%d, ",h_A[i]);
	printf("\n");


	printf("Array is %s sorted\n", (checkSorted(h_A,n)?"\b":"NOT"));

	cudaFree(d_A);	
	cudaFree(d_B);	
	free(h_A);
}

