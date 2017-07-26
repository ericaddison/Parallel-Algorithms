#include "fft_iterative.h"


// adapted from https://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious
int bitReverse(unsigned int b, int d)
{

    b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
    if(d<2)
      return b>>(2-d);

		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
    if(d<4)
      return b>>(4-d);

    b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
    if(d<8)
      return b>>(8-d);

    b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
    if(d<16)
      return b>>(16-d);

    b = ((b >> 16) | (b << 16));
      return b>>(32-d);
}


void bitReverse(carray& x)
{
    int N = x.size();
    int d = log2(N);

    carray y(x);
    for(int i=0; i<N; i++)
      x[bitReverse(i,d)] = y[ i ];
}

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
