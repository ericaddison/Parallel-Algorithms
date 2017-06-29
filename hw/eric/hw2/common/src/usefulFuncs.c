int next_pow2(int n)
{
	if(!(n&(n-1)))
		return n;

    int nextPow2 = 1;
	while( n>>=1 )
		nextPow2 <<= 1;
    return nextPow2;
}


int checkSorted(int *A, int n)
{

	for(int i=1; i<n; i++)
		if(A[i]<A[i-1])
			return 0;
	return 1;

}
