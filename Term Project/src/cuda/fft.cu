#include <cuda_runtime.h>
#include "ft_helpers.h"

// for n elements, should need n/2 threads for n elements
// do iterative because that follows cuda style .. bottom of tree up
__global__ void fft_kernel(carray &x)
{
    int myId = threadIdx.x + blockIdx.x * blockDim.x;
    int n = x.size();
    int d = flog2(n);

    // perform bit-reverse swapping (from global memory to shared)
    // i.e. from global stored carrray x into block shared array of complexes
    extern __shared__ cdouble sdata[];
    sdata[ threadIdx.x ] = x[ bitReverse(myId, d) ];

    // perform iterative fft loop
    // current looping starting at n/2 ... will probably need to adjust for
    // larger than one block worth of elements
    // and do a merger thing
    for(int s=n/2; s > 0; s>>=1)   // same as s loop in seq. fft_iterative
    {

      // inner two loops always loop for s groups of m elements
      // so always a total of s*m/2 elements
      // so maybe could do one loop for l = 0..(s*m/2)
      // then i = l/(m/2)
      // and  j = l%(m/2)
      // does this scheme work for seq?

        if( threadIdx.x < s )
            C[myId] += C[myId+s];
        __syncthreads();
    }

    // thread 0 write result
    if(threadIdx.x==0)
        count[blockIdx.x + gridDim.x*rangeBin] = C[myId];
}
