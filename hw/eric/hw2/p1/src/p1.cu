#include "p1.h"
extern "C"
{
	#include "randomInts.h"
}


void run_p1a(const int n, const int NRUNS)
{
	int errCnt = 0;
	printf("\nrun SEQ  CUDA\n-----------------\n");

	for(int irun=0; irun<NRUNS; irun++)
	{
		int* h_A = (int*)malloc(n*(sizeof(int)));

	// make test array
		writeRandomFile(n, "inp.txt");
	   	readIntsFromFile("inp.txt",n,h_A);

	// get CUDA result
		minResult cudaMinResult = find_min_cuda(h_A, n);

	// get sequential result
		minResult seqMinResult = find_min_seq(h_A, n);

	// print results
		printf(" %d  %3d %3d",irun, seqMinResult.minVal, cudaMinResult.minVal);
		if( seqMinResult.minVal!=cudaMinResult.minVal)
		{
			printf ("  XXX");
			errCnt++;
		}
		printf("\n");

	// free array memory
		free(h_A);
	}

	printf("p1a: n = %d\nerror Count = %d\n\n",n,errCnt);
}


void run_p1b(const int n, const int NRUNS)
{

	int errCnt=0;
	for(int irun=0; irun<NRUNS; irun++)
	{
		int* h_A = (int*)malloc(n*(sizeof(int)));

	// make test array
		writeRandomFile(n, "inp.txt");
	   	readIntsFromFile("inp.txt",n,h_A);

	// get CUDA result
		lastDigitResult cudaDigitResult = last_digit_cuda(h_A, n);

	// get sequential result
		lastDigitResult seqDigitResult = last_digit_seq(h_A, n);

	// print results
		printf("%d:  seq: ",irun);
		for(int i=0; i<10; i++)
			printf("%d, ",seqDigitResult.lastDigit[i]);
		printf("\b\b ...\n");
		printf("%d: CUDA: ",irun);
		for(int i=0; i<10; i++)
			printf("%d, ",cudaDigitResult.lastDigit[i]);
		printf("\b\b ...");

		for(int i=0; i<n; i++)
			if( seqDigitResult.lastDigit[i]!=cudaDigitResult.lastDigit[i])
		{
			printf ("  XXX");
			errCnt++;
		}
		printf("\n\n");

	// free array memory
		free(h_A);
		free(cudaDigitResult.lastDigit);
		free(seqDigitResult.lastDigit);
	}

	printf("p1b: n = %d\nerrCnt = %d\n",n,errCnt);
}

//***********************************************
// Main function

int main(int argc, char** argv)
{

	const int NRUNS = 10;
    struct timeval t;
    gettimeofday(&t, NULL);
    srand(t.tv_usec);
	double exp = (26.0*( (double)rand()/(double)RAND_MAX));
	int n = (int)pow(2,exp);

// p1a: find min
	run_p1a(n, NRUNS);

// p1b: digits
	run_p1b(n, NRUNS);

   return 0;
}
