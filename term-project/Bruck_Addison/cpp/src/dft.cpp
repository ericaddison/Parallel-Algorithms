#include "dft.h"



void transform_dft(carray& x, direction dir)
{
  const size_t N = x.size();
  const double v = (2*(dir==REVERSE)-1) * 2 * PI / N;

  // make a copy of x and initialize x to zero
  carray x_copy(x);
  x = 0;

  // n^2 dft loop
  for(int k=0; k<N; k++)
    for(int j=0; j<N; j++)
      x[k] += std::polar(1.0, v * j * k) * x_copy[j];
}



void dft(carray& x)
{
  transform_dft(x, FORWARD);
}



void idft(carray& x)
{
  transform_dft(x, REVERSE);
  for(int i=0; i<x.size(); i++)
    x[i] /= x.size();
}
