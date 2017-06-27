//***********************************************
// Device convenience functions

__device__ inline int d_next_pow2(int n)
{
	if(!(n&(n-1)))
		return n;

    int nextPow2 = 1;
	while( n>>=1 )
		nextPow2 <<= 1;
    return nextPow2;
}

__device__ inline int d_checkReduceIndex(int myId, int s, int n)
{
	return (threadIdx.x<s) && (threadIdx.x+s)<blockDim.x && (myId+s)<n;
}
