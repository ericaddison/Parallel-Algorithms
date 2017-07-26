#include <iostream>
#include <complex>
#include <cmath>
#include "dft.h"
#include "fft.h"

using std::complex;
using namespace std::literals::complex_literals;

int main()
{
  // dft test
  int n = 8;
  carray x(n);
  for(int i=0; i<n; i++)
    x[i] = i+1;

  dft(x);
  for(int i=0; i<n; i++)
    std::cout << i << ": " << x[i] << std::endl;
  std::cout << std::endl;

// fft test
  carray y(n);
  for(int i=0; i<n; i++)
    y[i] = i+1;

  fft(y);
  for(int i=0; i<n; i++)
    std::cout << i << ": " << y[i] << std::endl;
  std::cout << std::endl;

  ifft(y);
  for(int i=0; i<n; i++)
    std::cout << i << ": " << y[i] << std::endl;
  std::cout << std::endl;

  return 0;
}
