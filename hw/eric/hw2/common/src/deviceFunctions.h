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
