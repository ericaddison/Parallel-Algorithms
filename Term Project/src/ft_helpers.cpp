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
