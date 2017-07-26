#include "fft.h"



bool isPow2(int n)
{
  if(n==0)
    return true;
  return (!(n&(n-1)));
}



void checkSize(int n)
{
  if(!isPow2(n))
  {
    std::cout << "\n\n*******************************************************\n"
              << "WARNING: this fft implementation will only work\n"
              << "correctly for arrays with power-of-2 number of elements!\n"
              << "*******************************************************\n\n";
  }
}



// slightly modified from https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
void transform(carray& x, direction dir)
{
    const size_t N = x.size();

    // recursion base case
    if (N <= 1)
      return;

    // even side recursive call
    carray E = x[std::slice(0, N/2, 2)];
    transform(E, dir);

    // odd side recursive call
    carray O = x[std::slice(1, N/2, 2)];
    transform(O, dir);

    // combine
    double v = (2*(dir==REVERSE)-1) * 2 * PI / N;
    for (size_t k = 0; k < N/2; ++k)
    {
        cdouble t = std::polar(1.0, v * k) * O[k];
        x[k] = E[k] + t;
        x[k+N/2] = E[k] - t;
    }
}



void fft(carray& x)
{
  checkSize(x.size());
  transform(x, FORWARD);
}



void ifft(carray& x)
{
  checkSize(x.size());
  transform(x, REVERSE);
  for(int i=0; i<x.size(); i++)
    x[i] /= x.size();
}
