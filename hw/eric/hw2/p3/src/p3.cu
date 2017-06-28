#include "p3.h"
extern "C"
{
	#include "randomInts.h"
}


__device__ void d_hs_scan(int myId, int *A, int n, int initVal)
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
		A[myId] = myVal;
		__syncthreads();
	}
}


// perform a scan for radix sort
__global__ void radix_scan_binary_kernel(int *A, int n, int nDigits)
{
	extern __shared__ int sdata[];
	int *left = sdata;
	int *right = sdata+n;
	int myId = threadIdx.x + blockIdx.x * blockDim.x;


	for(int iDigit=0; iDigit<nDigits; iDigit++)
	{
		int myVal = A[myId];
		int radix = 1<<iDigit;
		left[myId] = !(myVal&radix);
		right[myId] = !(left[myId]);
		__syncthreads();
	
	// scan
		d_hs_scan(myId, left, n, 0);
		d_hs_scan(myId, right, n, left[n-1]);

	// scatter
		int index = (myVal&radix)?(right[myId]-1):(left[myId]-1);
		A[index] = myVal;
		__syncthreads();
	}
}


int checkSorted(int *A, int n)
{

	for(int i=1; i<n; i++)
		if(A[i]<A[i-1])
			return 0;
	return 1;

}


int main()
{
	int MAX_EXP = 10;
    struct timeval t;
    gettimeofday(&t, NULL);
    srand(t.tv_usec);
    double exp = (MAX_EXP*( (double)rand()/(double)RAND_MAX));
    int n = (int)pow(2,exp); 
	n = 1<<10;

	// make test array
	int* h_A = (int*)malloc(n*(sizeof(int)));
	writeRandomFile(n, "inp.txt");
	readIntsFromFile("inp.txt",n,h_A);
	
	int nDigits = MAX_EXP+1;
	printf("\n");
	for(int i=0; i<n; i++)
		printf("%d, ",h_A[i]);
	printf("\n");
	
	int *d_A;
	cudaMalloc((int**)&d_A, n*sizeof(int));
	cudaMemcpy(d_A, h_A, n*sizeof(int), cudaMemcpyHostToDevice);

	radix_scan_binary_kernel<<<1,n,2*n*sizeof(int)>>>(d_A,n, nDigits);
	cudaThreadSynchronize();
	cudaMemcpy(h_A, d_A, n*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i<n; i++)
		printf("%d, ",h_A[i]);
	printf("\n");

	printf("Array is %s sorted\n", (checkSorted(h_A,n)?"\b":"NOT"));

	cudaFree(d_A);	
	free(h_A);


}

