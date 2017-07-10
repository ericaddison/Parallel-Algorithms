#include <iostream>
using std::cout;
using std::endl;


void swap(int *A, int i, int j)
{
  int temp = A[i];
  A[i] = A[j];
  A[j] = temp;
}

void printArray(int *A, int n, const char* str = "")
{
  cout << str << ": ";
  for(int i=0; i<n; i++)
    cout << A[i] << ", ";
  cout << "\n";
}


void quickSort(int *A, int n)
{

  // base case
  if(n==1)
    return;

  // choose new pivot
  int p = A[0];

  // swap items compared to pivot
  int l = -1;
  int r = n;
  while(1)
  {
    do
    {
      l++;
    } while(A[l]<p);
    do
    {
      r--;
    } while(A[r]>p);
    if(r<l)
      break;
    swap(A,l,r);
  }

  // recursive call
  quickSort(A,l);
  quickSort(A+l,n-l);

}



int main()
{

  // test quickSort
  int n = 7;
  int A[] = {4, 1, 4, 5, 6, 2, 3};

  printArray(A,n,"input");
  quickSort(A,n);

  printArray(A,n,"result");


  return 0;
}
