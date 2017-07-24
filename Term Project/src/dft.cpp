#include "dft.h"
#include <iostream>

// precompute exponential terms
dft::dft(int n)
{
  this->n = n;
  w = std::vector<cvec>(n);
  double v = -2*PI/n;

  // precompute exponential terms
  for(int k=0; k<n; k++)
  {
    w[k] = cvec(n);
    for(int j=0; j<n; j++)
      w[k][j] = exp((v*k*j)*1i);
  }
}

dft::~dft()
{}

void dft::forward(cvec& out, const cvec& in)
{

  // vector length check
  if(in.size() < n)
    throw new std::invalid_argument("in.size() != n");
  if(out.size() < n)
    throw new std::invalid_argument("out.size() != n");

  for(int k=0; k<n; k++)
  {
    out[k] = 0;
    for(int j=0; j<n; j++)
      out[k] += w[k][j]*in[j];
  }

}

void dft::reverse(cvec& out, const cvec& in)
{
  // vector length check
  if(in.size() < n)
    throw new std::invalid_argument("in.size() != n");
  if(out.size() < n)
    throw new std::invalid_argument("out.size() != n");

  for(int k=0; k<n; k++)
  {
    out[k] = 0;
    for(int j=0; j<n; j++)
      out[k] += std::conj(w[k][j])*in[j];
    out[k] /= n;
  }
}
