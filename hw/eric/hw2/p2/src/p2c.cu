
//***********************************************
// Problem 2a functions: use global memory

#include "p2.h"

__global__ void hs_scan_kernel(int *A, int n)
{
	extern __shared__ int sdata[];
	int myId = threadIdx.x + blockIdx.x * blockDim.x;

	sdata[myId] = A[myId];
	int toggler=0;
	__syncthreads();

	for(int s=1; s<n; s*=2)
	{
		toggler = !toggler;
		int newInd = myId + n*toggler;
		int oldInd = myId + n*(!toggler);
		sdata[newInd] = sdata[oldInd];
		if(myId>=s)
			sdata[newInd] += sdata[oldInd-s];
		__syncthreads();
		if(myId==0)
		{
			for(int i=0; i<n; i++)
				printf("%d ", sdata[i]);
			printf("  |  ");
			for(int i=0; i<n; i++)
				printf("%d ", sdata[i+n]);
			printf("\n");	
		}
		__syncthreads();
	}

	A[myId] = sdata[myId + n*toggler];

}

int main()
{
	int N = 7;
	int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	
	int *d_A;
	cudaMalloc((int**) &d_A, N*sizeof(int));
	cudaMemcpy(d_A, A, N*sizeof(int), cudaMemcpyHostToDevice);

	hs_scan_kernel<<<1,N,2*N>>>(d_A, N);
	cudaThreadSynchronize();

	int *B = (int*)malloc(N*sizeof(int));
	cudaMemcpy(B,d_A,N*sizeof(int),cudaMemcpyDeviceToHost);


	for(int i=0; i<N; i++)
		printf("%d, ", B[i]);
	printf("\n");

	cudaFree(d_A);
	free(B);
	
}

