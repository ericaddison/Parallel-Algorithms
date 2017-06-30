/**
 * Problem 3: CUDA functions for radix sort
 */
#include "p3.h"


/**
 * radix_sort_kernel
 * CUDA kernel to perform binary digit radix sort (radix-2).
 * Requires shared memory of size 2n for working space
 * Kernel proceeds by scanning each block of elements
 * twice per bit, and then scattering the elements to
 * the proper location.
 *
 * Array is modified in-place.
 *
 * d_hs_scan() function located in common/src/deviceFunctions.h
 *
 * @param A input array to be sorted
 * @param n size of input array
 * @param nDigits number of binary digits to sort on
 */
__global__ void radix_sort_kernel(int *A, int n, int nDigits)
{
    extern __shared__ int sdata[];
    int *left = sdata;
    int *right = sdata+n;
    int offset = blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int myId = tid + offset;

// loop over all binary digits
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



/**
 * parallel_merge_kernel
 * Parallel merge of a single array A that has been segmented into
 * blocks of sorted elements. It is expected that this kernel will
 * be called in a loop to merge sorted segments of the array of size
 * blockDim.x*iter into half the number of sorted segments of size
 * 2*blockDim.x*iter. The external looping is necessary to 
 * synchronize threads across all blocks, i.e. a global barrier.
 *
 * The merge strategy used is non-optimal rank-based merging. The second
 * of each pair of sorted segments to be merged checks for repeated
 * elements in the first segment for proper placement.
 *
 * d_binary_search() function located in common/src/deviceFunctions.h
 *
 * @param d_out output array
 * @param A input array
 * @param iter merge iteration (level of the binary merge tree)
 */
__global__ void parallel_merge_kernel(int *d_out, int *A, int iter)
{
    int mergeID = blockIdx.x/iter;
    int n = iter*blockDim.x;    

    int offset = (mergeID/2)*2*n;
    int myInd = threadIdx.x +(blockIdx.x%iter)*blockDim.x;
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

