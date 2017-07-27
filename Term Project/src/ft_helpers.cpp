#include "ft_helpers.h"

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

// taken from https://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious
int bitReverse(unsigned int b, int d)
{
    b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
    b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
    b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
    b = ((b >> 16) | (b << 16));
    return b>>(32-d);
}


void bitReverse(carray& x)
{
    int N = x.size();
    int d = log2(N);

    carray y(x);
    for(int i=0; i<N; i++)
      x[i] = y[ bitReverse(i,d) ];
}


