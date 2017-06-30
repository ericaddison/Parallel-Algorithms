//***********************************************
// Device convenience functions

#ifndef _DEVICE_FUNCS
#define _DEVICE_FUNCS

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


__device__ inline int d_binary_search(int *A, int key, int n)
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


__device__ inline void d_hs_scan(int myId, int *A, int n, int initVal)
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

#endif
