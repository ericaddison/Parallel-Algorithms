#include <iostream>
#include "matrix.h"
#include "quickSort.h"
#include <sys/time.h>
#include <stdlib.h>
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


  cout << "derp\n";

  struct timeval t;
  gettimeofday(&t, NULL);
  srand(t.tv_usec);

  int n = rand()%20+1;
  int p = rand()%10;
  cout << "n = " << n << endl;
  cout << "pivot = " << p << endl;

  int *A = new int[n];

  for(int i=0; i<n; i++)
    A[i] = rand()%10;


  int ind = partition(A, n, p);
  cout << "ind = " << ind << endl;

  delete A;
  return 0;
}
