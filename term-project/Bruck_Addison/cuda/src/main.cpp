#include "fft_cuda.h"
#include <iostream>
#include <stdlib.h>
#include <chrono>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::high_resolution_clock;



int main(int argc, char** argv)
{
// handle input args
    if(argc < 2)
    {
        cout << "\nUSAGE: " << argv[0] << " input_file [result output y/n]\n\n";
        return 1;
    }
    bool resultOutput = true;
    if(argc >= 3)
        resultOutput = (*argv[2] == 'y');

// step 0: write current algorithm to stdout
    cout << "Cuda fft()" << endl;

// step 1: read input from command line arg
    string filename = argv[1];
    std::vector<double> data = readFile(filename);

    int n = data.size();
    thCdouble *h_A = new thCdouble[n];
    for(int i=0; i<n; i++)
        h_A[i] = data[i];    

// step 2: perform fft with timing in microseconds
    cuda_init();
    auto t1 = high_resolution_clock::now();
    float innerTime = fft_cuda(h_A, n);
    auto t2 = high_resolution_clock::now();
    long time = duration_cast<microseconds>(t2-t1).count();


// step 3: write elapsed time (microseconds) to stdout
    cout << "elapsed time (microseconds): " << time << " / " << innerTime*1000 << endl;

// step 4: write result to stdout with format real, imag\n
    if(resultOutput)
        for(int i=0; i<n; i++)
            cout << h_A[i].real() << ", " << h_A[i].imag() << endl;


    return 0;
}
