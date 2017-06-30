/**
 * Problem 2b functions: use shared memory
 */
#include "p2.h"


/**
 * range_count_kernel_shared
 * The CUDA kernel for creating a fixed-bin histogram with 
 * bins 0-99, 100-199, ..., 900-999.
 * 
 * This kernel uses shared memory to compute histogram results.
 * 
 * This kernel should be called with a 2D array of blocks, with
 * 10 blocks along the y dimension. Each set of blocks along 
 * the y dimension are respondible for computing the count in
 * one of the histogram bins.
 * 
 * This kernel expects to have shared memory of size n*sizeof(int)
 * avaialble.
 *
 * @param count interim result array. Holds histogram result for each x-block
 * @param A input array
 * @param n size of input array A
 */
__global__ void range_count_kernel_shared(int * count, int * A, int n)
{
    extern __shared__ int sdata[];
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int rangeBin = blockIdx.y;
 
    // create indicator array
    sdata[tid] = ((A[myId]/100)==rangeBin);
    __syncthreads();

    // reduce indicator array
    int n2 = d_next_pow2(blockDim.x);
    for(int s=n2/2; s > 0; s>>=1)
    {
        if( d_checkReduceIndex(myId, s, n) )
            sdata[tid] += sdata[tid+s];
        __syncthreads();
    }

    // thread 0 write result
    if(tid==0)
        count[blockIdx.x + gridDim.x*rangeBin] = sdata[tid];
}


/**
 * reduce_add_kernel_shared
 * CUDA kernel for recude-add in shared memory. This is specific
 * for reducing the results of the interim histograms from 
 * the range_count_kernel_global function. Calling should be 
 * similar, i.e. blockIdx.y determines which bin will be reduced.
 *
 * @param B result array
 * @param A input array
 * @param n size of input array
 */
__global__ void reduce_add_kernel_shared(int * B, int * A, int n)
{
    extern __shared__ int sdata[];
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int rangeBin = blockIdx.y;

    // copy value from global to shared memory
    sdata[tid] = A[myId + n*rangeBin];

    // reduce block array
    int n2 = d_next_pow2(blockDim.x);
    for(int s=n2/2; s > 0; s>>=1)
    {
        if( d_checkReduceIndex(myId, s, n) )
            sdata[tid] += sdata[tid+s];
        __syncthreads();
    }

    // thread 0 write result
    if(tid==0)
        B[blockIdx.x + rangeBin*gridDim.x] = sdata[tid];
}
