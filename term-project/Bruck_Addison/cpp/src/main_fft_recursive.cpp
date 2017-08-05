#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <chrono>
#include "fft_recursive.h"

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::high_resolution_clock;

// sequential recursive fft program

int main(int argc, char ** argv)
{
	if(argc < 2)
	{
		cout << "\nUSAGE: " << argv[0] << " input_file\n\n";
		return 1;
	}
	bool resultOutput = true;
	if(argc >= 3)
		resultOutput = (*argv[2] == 'y');

// step 0: write current algorithm to stdout
  cout << "C++: fft_recursive()" << endl;

// step 1: read input from command line arg
	string filename = argv[1];
	std::vector<double> data = readFile(filename);

	int n = data.size();
	carray x(n);
	for(int i=0; i<n; i++)
		x[i] = data[i];	


// step 2: perform fft with timing in microseconds
	auto t1 = high_resolution_clock::now();
  	fft(x);
	auto t2 = high_resolution_clock::now();
	long time = duration_cast<microseconds>(t2-t1).count();


// step 3: write elapsed time (microseconds) to stdout
	cout << "elapsed time (microseconds): " << time << endl;

// step 4: write result to stdout with format real, imag\n
	if(resultOutput)
	  for(int i=0; i<n; i++)
	    cout << x[i].real() << ", " << x[i].imag() << endl;

  return 0;
}
