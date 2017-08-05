#include "fft.h"
#include <cilk/cilk.h>
#include <iostream>
#include <assert.h>

#define MAX_LINEAR 128

using std::cout;

void transform(carray& x, direction dir);
void cilk_transform(carray& x, direction dir);
void transform_iter(carray& x, direction dir);
void cilk_transform_iter(carray& x, direction dir);
void combine(carray& x, size_t N, size_t start, size_t size, direction dir, carray& E, carray& O);
void cilky_combine(carray& x, size_t N, size_t start, size_t size, direction dir, carray& E, carray& O);
void cilk_combine(carray& x, size_t N, size_t start, size_t size, direction dir, carray& E, carray& O);

void cilk_fft(carray& x)
{
  cilk_transform(x, FORWARD);
}

void cilk_iter_fft(carray& x)
{
  cilk_transform_iter(x, FORWARD);
}

void iter_fft(carray& x)
{
  transform_iter(x, FORWARD);
}

void fft(carray& x)
{
  transform(x, FORWARD);
}

void ifft(carray& x)
{
  transform(x, REVERSE);
  for(int i=0; i<x.size(); i++)
    x[i] /= x.size();
}

void cilk_transform_iter(carray& x, direction dir)
{
  bitReverse(x);
  int N = x.size();

  for(int s=N/2; s>=1; s/=2)
  {
    int m = N/s;
    double v = (2*(dir==REVERSE)-1) * 2 * PI / m;
    cdouble w = std::polar(1.0, v);

    // do s groups of m elements
    cilk_for(int i=0; i<s; i++)
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
    combine(x, N, 0, N/2, dir, E, O);
}

const size_t cilk_max_recombine = MAX_LINEAR;

void combine(carray& x, size_t N, size_t start, size_t size, direction dir, carray& E, carray& O)
{
    // combine
    double v = (2*(dir==REVERSE)-1) * 2 * PI / N;
    for (size_t k = start; k < start+size; ++k)
    {
        cdouble t = std::polar(1.0, v * k) * O[k];
        x[k] = E[k] + t;
        x[k+N/2] = E[k] - t;
    }
}

void cilky_combine(carray& x, size_t N, size_t start, size_t size, direction dir, carray& E, carray& O)
{
    // combine
    double v = (2*(dir==REVERSE)-1) * 2 * PI / N;
    cilk_for (size_t k = start; k < start+size; ++k)
    {
        cdouble t = std::polar(1.0, v * k) * O[k];
        x[k] = E[k] + t;
        x[k+N/2] = E[k] - t;
    }
} 

void cilk_combine(carray& x, size_t N, size_t start, size_t size, direction dir, carray& E, carray& O)
{
    assert (size >= 1);
    if (size <= cilk_max_recombine) {
      // combine
      cilky_combine(x, N, start, size, dir, E, O);
    } else {
          //carray small = x[std::slice(0, N/2, 1)];
          //carray large = x[std::slice(N/2, N/2, 1)];
          
          cilk_spawn cilk_combine(x, N, start       , size/2, dir, E, O);
          cilk_spawn cilk_combine(x, N, start+size/2, size/2, dir, E, O);
          cilk_sync;
    }
} 

void cilk_transform(carray& x, direction dir)
{
    const size_t N = x.size();

    // recursion base case
    if (N <= 1)
      return;

    // even side recursive call
    carray E = x[std::slice(0, N/2, 2)];
    cilk_spawn cilk_transform(E, dir);

    // odd side recursive call
    carray O = x[std::slice(1, N/2, 2)];
    cilk_spawn cilk_transform(O, dir);

    cilk_sync;

    // combine
    cilk_combine(x, N, 0, N/2, dir, E, O);
    //combine(x, N, dir, E, O);
}
