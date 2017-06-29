int next_pow2(int n)
{
	if(n==0)
		return 0;
	
	// check if already a power of 2
	if(!(n&(n-1)))
		return n;

	// find and return next power of 2
    int nextPow2 = 1;
	do
	{	
		nextPow2 <<= 1;
	}while( n>>=1 );

    return nextPow2;
}


int checkSorted(int *A, int n)
{

	for(int i=1; i<n; i++)
		if(A[i]<A[i-1])
			return 0;
	return 1;

}
