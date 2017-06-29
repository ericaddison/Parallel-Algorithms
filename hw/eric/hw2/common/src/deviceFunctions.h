//***********************************************
// Device convenience functions

__device__ inline int d_next_pow2(int n)
{
    int nBits = 0;
	while( (n>>nBits) > 0 )
		nBits++;
    return 1<<nBits;
}


__device__ inline int d_checkReduceIndex(int myId, int s, int n)
{
	return (threadIdx.x<s) && (threadIdx.x+s)<blockDim.x && (myId+s)<n;
}


