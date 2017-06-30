/**
 * Problem 2a functions: use global memory
 */
#include "p2.h"


/**
 * range_count_kernel_global
 * The CUDA kernel for creating a fixed-bin histogram with 
 * bins 0-99, 100-199, ..., 900-999.
 * 
 * This kernel uses global memory to compute histogram results.
 * 
 * This kernel should be called with a 2D array of blocks, with
 * 10 blocks along the y dimension. Each set of blocks along 
 * the y dimension are respondible for computing the count in
 * one of the histogram bins.
 * 
 * A working buffer of size 10n is required because the input array
 * cannot be overwritten, since threads from different blocks depend on
 * reading from it.
 *
 * @param count interim result array. Holds histogram result for each x-block
 * @param B working buffer, size should be 10n
 * @param A input array
 * @param n size of input array A
 */
__global__ void range_count_kernel_global(int * count, int * B, int * A, int n)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
    int rangeBin = blockIdx.y;
    int * C = B + n*rangeBin;
 
    // create indicator array
    if(myId<n)
        C[myId] = ((A[myId]/100)==rangeBin);
    __syncthreads();

    // reduce indicator array
    int n2 = d_next_pow2(blockDim.x);
    for(int s=n2/2; s > 0; s>>=1)
    {
        if( d_checkReduceIndex(myId, s, n) )
            C[myId] += C[myId+s];
        __syncthreads();
    }

    // thread 0 write result
    if(threadIdx.x==0)
        count[blockIdx.x + gridDim.x*rangeBin] = C[myId];
}


/**
 * reduce_add_kernel_global
 * CUDA kernel for recude-add in global memory. This is specific
 * for reducing the results of the interim histograms from 
 * the range_count_kernel_global function. Calling should be 
 * similar, i.e. blockIdx.y determines which bin will be reduced.
 *
 * @param B result array
 * @param A input array
 * @param n size of input array
 */
__global__ void reduce_add_kernel_global(int * B, int * A, int n)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
    int rangeBin = blockIdx.y;
    int * C = A + n*rangeBin;

    // reduce block array
    int n2 = d_next_pow2(blockDim.x);
    for(int s=n2/2; s > 0; s>>=1)
    {
        if( d_checkReduceIndex(myId, s, n) )
            C[myId] += C[myId+s];
        __syncthreads();
    }

    // thread 0 write result
    if(threadIdx.x==0)
        B[blockIdx.x + rangeBin*gridDim.x] = C[myId];
}


