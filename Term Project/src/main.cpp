#include <iostream>
#include "complex.h"

int main()
{

  Complex<> c(1,2);
  c.mag2();

  std::cout << "I am main\n";
  return 0;
}
