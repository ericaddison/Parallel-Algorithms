#include <iostream>
#include <complex>
#include <cmath>
#include "dft.h"
#include "fft.h"

using std::cout;
using std::endl;

int main()
{
  // dft test
  int n = 8;
  carray x(n);
  for(int i=0; i<n; i++)
    x[i] = i+1;

  dft(x);
  for(int i=0; i<n; i++)
    cout << i << ": " << x[i] << endl;
  cout << endl;

// fft test
  carray y(n);
  for(int i=0; i<n; i++)
    y[i] = i+1;

  fft(y);
  for(int i=0; i<n; i++)
    cout << i << ": " << y[i] << endl;
  cout << endl;

  ifft(y);
  for(int i=0; i<n; i++)
    cout << i << ": " << y[i] << endl;
  cout << endl;

  return 0;
}
