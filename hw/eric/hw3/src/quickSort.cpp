#include "hw3.h"

void swap(int *A, int i, int j)
{
  int temp = A[i];
  A[i] = A[j];
  A[j] = temp;
}



void quickSort(int *A, int n)
{

  // base case
  if(n<2)
    return;

  // choose new pivot
  int p = A[0];

  // partition array based on pivot value
  int ind = partition(A, n, p);

  // recursive call
  quickSort(A,ind);
  quickSort(A+ind,n-ind);

}



int partition(int *A, int n, int p)
{
  // swap items compared to pivot
  int l = -1;
  int r = n;
  while(1)
  {
    do
    {
      l++;
    } while(A[l]<p && l<n);
    do
    {
      r--;
    } while(A[r]>p && r>=0);
    if(r<l)
      break;
    swap(A,l,r);
  }
    return l;
}
