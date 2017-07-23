#include <iostream>
#include <complex>
#include <cmath>

using std::complex;
using namespace std::literals::complex_literals;

int main()
{

  double PI = acos(-1);

  complex<double> c = 1i;
  complex<double> d = 1i;
  std::cout << "c = " << c << "\n";
  std::cout << "d = " << d << "\n";
  std::cout << "c+d = " << c+d << "\n";
  std::cout << "c-d = " << c-d << "\n";
  std::cout << "c*d = " << c*d << "\n";
  std::cout << "c/d = " << c/d << "\n";
  std::cout << "exp(c) = " << exp(c) << "\n";
  std::cout << "exp(pi*i) = " << exp(PI*1i) << "\n";
  return 0;
}
