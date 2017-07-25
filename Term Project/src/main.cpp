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
  cvec x(n);
  for(int i=0; i<n; i++)
    x[i] = i+1;

  dft mydft(n);

  cvec X(n);
  mydft.forward(X,x);

  cvec xp(n);
  mydft.reverse(xp,X);

  for(int i=0; i<n; i++)
    std::cout << i << ": " << X[i] << std::endl;

  std::cout << std::endl;
  for(int i=0; i<n; i++)
    std::cout << i << ": " << xp[i] << std::endl;
std::cout << std::endl;

// fft test
  cdouble* y = new cdouble[n];
  for(int i=0; i<n; i++)
    y[i] = i+1;

  fft myfft(n);

  cdouble* Y = new cdouble[n];
  myfft.forward(Y,y);

  for(int i=0; i<n; i++)
    std::cout << i << ": " << Y[i] << std::endl;

  std::cout << std::endl;

    delete[] y;
    delete[] Y;

  return 0;
}
