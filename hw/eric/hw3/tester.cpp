#include <iostream>
#include "matrix.h"
using namespace std;

int main()
{

  Matrix m(3,3);
  m(0,0) = 1;
  m(1,0) = 2;
  m(2,0) = 3;
  m(0,2) = 4;
  m.print();
  cout << endl;

  Matrix b = m;
  b(0,1) = 10;
  b.print();
  cout << endl;

  ColVector v(3);
  v(0) = 1;
  v(1) = 2;
  v(2) = 3;

  ColVector u = v;
  u(0) = 100;

  cout << "trying to print v\n";
  v.print();
  cout << endl;

  u.print();
  cout << endl;

  Matrix r("testMat");
  r.print();
  cout << endl;

  ColVector cv("testVec");
  cv.print();
  cout << endl;

  (m*v).print();

  return 0;
}
