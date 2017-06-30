#include "p3.h"

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

	int offset = (mergeID/2)*2*n;
	int myInd = threadIdx.x +(blockIdx.x%treeLevel)*blockDim.x;
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

