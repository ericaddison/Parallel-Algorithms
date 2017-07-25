#include "fft.h"
#include <iostream>

// precompute exponential terms
fft::fft(int n)
{

  // ENSURE n is POWER of 2!

  this->n = n;
  w = std::vector<cvec>(n);
  v = -2*PI/n;

  // precompute exponential terms
  for(int k=0; k<n; k++)
  {
    w[k] = cvec(n);
    for(int j=0; j<n; j++)
      w[k][j] = exp((v*k*j)*1i);
  }
}

fft::~fft()
{}


void fft::forward(cdouble* out, const cdouble* in)
{
  transform(out, in, n, 1, FORWARD);
}

void fft::reverse(cdouble* out, const cdouble* in)
{
  transform(out, in, n, 1, REVERSE);
}

// uses the class variable x as input
// can probably do it without the extra E and O memory, carefully writing over out...
void fft::transform(cdouble* out, const cdouble* in, int N, int stride, direction dir)
{

  // recursion base case
  if(N==1)
  {
    out[0] = in[0];
    return;
  }

  // even side recursive call
  cdouble* E = new cdouble[N/2];
  transform(E, in, N/2, stride*2, dir);

  // odd side recursive call
  cdouble* O = new cdouble[N/2];
  transform(O, &in[stride], N/2, stride*2, dir);

  // combine
  // TODO: replace that exp with cached w values
  cdouble o;
  double v = -2*PI/N;
  for(int k=0; k<N/2; k++)
  {
    o = exp(v*k*1i)*O[k];
    out[k] = E[k] + o;
    out[k+N/2] = E[k] - o;
  }

  delete[] E;
  delete[] O;
}
