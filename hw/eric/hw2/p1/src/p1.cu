/**
 * Problem 1
 */
#include "p1.h"
extern "C"
{
	#include "randomInts.h"
}


/**
 * run_p1a
 * Run problem 1a by reading input array from file, calling
 * CUDA find_min implementation, sequential find_min, and 
 * comparing results. Results are printed to STDOUT.
 * All allocated memory is freed.
 *
 * @param filename path to input file
 */
void run_p1a(const char* filename)
{
	int errCnt = 0;

	//  read array file
	   	randArray ra = readIntsFromFile(filename);
		int* h_A = ra.A;
		int n = ra.n;

	// get CUDA result
		minResult cudaMinResult = find_min_cuda(h_A, n);

	// get sequential result
		minResult seqMinResult = find_min_seq(h_A, n);

	// print results
		printf("Seq  min: %3d\nCUDA min: %3d", seqMinResult.minVal, cudaMinResult.minVal);
		if( seqMinResult.minVal!=cudaMinResult.minVal)
		{
			printf ("  XXX");
			errCnt++;
		}
		printf("\n");

	// free array memory
		free(h_A);

	printf("p1a: n = %d\nerror Count = %d\n\n",n,errCnt);
}



/**
 * run_p1b
 * Run problem 1b by reading input array from file, calling
 * CUDA last-digit implementation, sequential last-digit, and 
 * comparing results. Results are printed to STDOUT.
 * All allocated memory is freed.
 *
 * @param filename path to input file
 */
void run_p1b(const char* filename)
{

	int errCnt=0;
	//  read array file
	   	randArray ra = readIntsFromFile(filename);
		int* h_A = ra.A;
		int n = ra.n;

	// get CUDA result
		lastDigitResult cudaDigitResult = last_digit_cuda(h_A, n);

	// get sequential result
		lastDigitResult seqDigitResult = last_digit_seq(h_A, n);

	// print results
		printf("seq: ");
		for(int i=0; i<10; i++)
			printf("%d, ",seqDigitResult.lastDigit[i]);
		printf("\b\b ...\n");
		printf("CUDA: ");
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

	printf("p1b: n = %d\nerrCnt = %d\n",n,errCnt);
}



/**
 * main
 * Main function for problem 1. Expects paths to input
 * files as command line arguments. Loops through 
 * paths and calls p1a and p1b for each file.
 *
 * @param argc number of command line arguments
 * @param argv command line argument strings
 */
int main(int argc, char** argv)
{

	if(argc<2)
		printf("No input files specified\n\n");

	for(int i=1; i<argc; i++)
	{
		char* nextFile = argv[i];
		printf("\n***********************************\n");
		printf("Running p1 for file %s\n",nextFile);
		printf("***********************************\n\n");
	
	// p1a: find min
		run_p1a(nextFile);

	// p1b: digits
		run_p1b(nextFile);
		
		printf("Done with file %s\n\n",nextFile);
	}

   return 0;
}
