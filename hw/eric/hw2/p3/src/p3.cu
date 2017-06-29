#include "p3.h"
extern "C"
{
	#include "randomInts.h"
}

#define MAX_THREADS 4

enum repeatCheck { check, noCheck };

// H-S scan with initial value
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
	int nth = blockDim.x;
	int myId = tid + offset;

	// compensate for non-full blocks
	if(blockIdx.x == (gridDim.x-1))
		nth = n - nth*(gridDim.x-1);

	// LSB radix sort
	for(int iDigit=0; iDigit<nDigits; iDigit++)
	{
		int myVal = A[myId];
		int radix = 1<<iDigit;
		left[tid] = !(myVal&radix);
		right[tid] = !(left[tid]);
		__syncthreads();
	
	// scan
		d_hs_scan(tid, left, nth, 0);
		d_hs_scan(tid, right, nth, left[nth-1]);

	// scatter
		if(myId<n)
		{
			int index = (myVal&radix)?(right[tid]-1):(left[tid]-1);
			A[index+offset] = myVal;
		}
		__syncthreads();
	}

	if(tid==0)
	{
		printf("%d: %d, %d, %d, %d\n",blockIdx.x,A[0+offset], A[1+offset], A[2+offset], A[3+offset]);
	}
}


__device__ int d_binary_search(int *A, int key, int n)
{
	int l = 0;
	int r = n-1;
	int ind = (l+r)/2;
	while( l<=r )
	{
		if(A[ind]>key)
			r = ind-1;
		else
			l = ind+1;
		ind = (l+r)/2;		
	}
	return (r<0)?0:ind+1;
}



// this will merge two pieces of array A of size n into the first 2n entries of d_out
__global__ void parallel_merge_kernel(int *d_out, int *A, int n)
{
	int tid = threadIdx.x;
	int myInd = tid;
	int *B;
	// swap A and B if this is blockIdx.y==1
	if(blockIdx.y==1)
	{
		int *C = A;
		A = B;
		B = C;
	}
		
	int otherInd = d_binary_search(B, A[tid], n);
	int mergedIndex = myInd + otherInd;
	int nRepeats=0;

	// sensitive to repeated elements if blockIdx.y==1
	if(blockIdx.y==1)
	{
	 	nRepeats = otherInd - d_binary_search(B,A[tid]-1,n);		
		mergedIndex -= nRepeats;
	}

	d_out[mergedIndex] = A[tid];	
}


int main()
{

	int MAX_EXP = 10;
    struct timeval t;
    gettimeofday(&t, NULL);
    srand(t.tv_usec);
    double exp = (MAX_EXP*( (double)rand()/(double)RAND_MAX));
    int n = (int)pow(2,exp); 
	n = 6;//1<<3;


	// pad array to next power of 2
	


	// make test array
	//int* h_A = (int*)malloc(n*(sizeof(int)));
	//writeRandomFile(n, "inp.txt");
	//readIntsFromFile("inp.txt",n,h_A);
	int h_A[] = {3, 5, 5, 2, 5, 1, 7, 7};
	
	int nDigits = MAX_EXP+1;
	printf("\n");
	for(int i=0; i<n; i++)
		printf("%d, ",h_A[i]);
	printf("\n");
	
	int *d_A, *d_B;
	cudaMalloc((int**)&d_A, n*sizeof(int));
	cudaMalloc((int**)&d_B, n*sizeof(int));
	cudaMemcpy(d_A, h_A, n*sizeof(int), cudaMemcpyHostToDevice);

	int nBlocks = (n-1)/MAX_THREADS+1;
	int threadsPerBlock = MIN(MAX_THREADS,n);

	radix_sort_kernel<<<nBlocks,threadsPerBlock,2*threadsPerBlock*sizeof(int)>>>(d_A, n, nDigits);
	cudaThreadSynchronize();

	if(nBlocks>1)
	{
		parallel_merge_kernel<<<dim3(1,2),threadsPerBlock>>>(d_B,d_A,threadsPerBlock);
		cudaThreadSynchronize();
		cudaMemcpy(h_A, d_B, n*sizeof(int), cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaMemcpy(h_A, d_A, n*sizeof(int), cudaMemcpyDeviceToHost);
	}



	for(int i=0; i<n; i++)
		printf("%d, ",h_A[i]);
	printf("\n");

	printf("Array is %s sorted\n", (checkSorted(h_A,n)?"\b":"NOT"));

	cudaFree(d_A);	
	cudaFree(d_B);	
	//free(h_A);
}

