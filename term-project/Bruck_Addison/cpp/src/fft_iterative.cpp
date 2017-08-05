#include "fft_iterative.h"

void transform_iter(carray& x, direction dir)
{
  bitReverse(x);
  int N = x.size();

  for(int s=N/2; s>=1; s/=2)
  {
    int m = N/s;
    double v = (2*(dir==REVERSE)-1) * 2 * PI / m;
    cdouble w = std::polar(1.0, v);

    // do s groups of m elements
    for(int i=0; i<s; i++)
    {
      cdouble wj = 1;
      for(int j=0; j<m/2; j++)
      {
        int k = i*m + j;
        cdouble t = x[k];
        cdouble u = wj*x[k+m/2];
        x[k] = t+u;
        x[k+m/2] = t-u;
        wj *= w;
      }
    }
  }
}




void fft_iterative(carray& x)
{
  checkSize(x.size());
  transform_iter(x, FORWARD);
}



void ifft_iterative(carray& x)
{
  checkSize(x.size());
  transform_iter(x, REVERSE);
  for(int i=0; i<x.size(); i++)
    x[i] /= x.size();
}
