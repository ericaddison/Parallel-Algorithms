#include <iostream>
#include "matrix.h"
using namespace std;

int main()
{

  Matrix m(3,3);
  m(0,0) = 20;
  cout << "Here is my matrix: " << m(0,0) << ", " << m(0,1) << endl;

  Matrix b = m;
  b(0,1) = 10;
  cout << "Here is my matrix: " << b(0,0) << ", " << b(0,1) << endl;

  Vector v(10);
  v(1) = 10;

  Vector u = v;
  u(0) = 100;

  cout << "Here is my vector: " << v(0) << ", " << v(1) << endl;
  cout << "Here is my vector: " << u(0) << ", " << u(1) << endl;

  return 0;
}
