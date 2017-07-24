#include <iostream>
#include <complex>
#include <cmath>
#include "dft.h"

using std::complex;
using namespace std::literals::complex_literals;

int main()
{

  int n = 10;
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

  return 0;
}
