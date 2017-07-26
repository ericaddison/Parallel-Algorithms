#include <iostream>
#include <complex>
#include <cmath>
#include "dft.h"
#include "fft.h"
#include "fft_iterative.h"

using std::cout;
using std::endl;



int main()
{

  // dft test
  int n = 1<<8;
  carray x(n);
  for(int i=0; i<n; i++)
    x[i] = i+1;

  dft(x);
  idft(x);
  /*
  for(int i=0; i<n; i++)
    cout << i << ": " << x[i] << endl;
  cout << endl;
*/

// fft test
  carray y(n);
  for(int i=0; i<n; i++)
    y[i] = i+1;

  fft_iterative(y);
  ifft_iterative(y);
  /*
  for(int i=0; i<n; i++)
    cout << i << ": " << y[i] << endl;
  cout << endl;
*/

cout << "abs error: " << (abs(x-y)).sum().real()/(abs(x).sum().real()) << endl;


  return 0;
}
