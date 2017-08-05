#include "fft_cuda.h"
#include <cufft.h>

void cuda_init()
{
	cudaFree(0);
	return;
}

float fft_cufft(double *h_A, int n)
{

	// define device data
	cufftDoubleComplex *d_A;
	cudaMalloc((cufftDoubleComplex**) &d_A, n*sizeof(cufftDoubleComplex));
	cudaMemcpy(d_A, h_A, n*sizeof(thCdouble), cudaMemcpyHostToDevice);

	// make cufft plan
	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_Z2Z , 1);

	// cuda timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// start timing
	cudaEventRecord(start);
	

	// call cufft
	cufftResult res = cufftExecZ2Z(plan, d_A, d_A, CUFFT_FORWARD);

	// stop timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	if(res != CUFFT_SUCCESS)
		std::cout << "Warning: non-success cufft code found: " << res << std::endl;

	// copy back to host
    cudaMemcpy(h_A, d_A, n*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cufftDestroy(plan);

	return milliseconds;
}

