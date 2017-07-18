#include "quickSort.h"
#include <iostream>
using namespace std;

/**
 * swap
 * Simple swap of two elements in a given array.
 *
 * @param A array with elements to swap
 * @param i first index to swap
 * @param j second index to swap
 */
void swap(int *A, int i, int j)
{
  int temp = A[i];
  A[i] = A[j];
  A[j] = temp;
}



/**
 * quickSort
 * Perform sequential in-place quickSort on the input array.
 *
 * @param A array to sort
 * @param n number of elements in array A
 */
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



/**
 * quickSort
 * Perform sequential in-place quickSort on the input ColVector.
 *
 * @param x reference to ColVector to sort
 */
void quickSort(ColVector &x)
{
  quickSort(x.getValueBuffer(), x.getCount());
}



/**
 * partition
 * Given an array A with length n and a pivot p, swap elements in A such
 * that A[i]<=p for all i in 0..(ind-1), and A[i]>=p for all i in ind..(n-1).
 * Elements equal to p can appear in either side of the array. The index ind
 * is returned.
 *
 * @param A input array
 * @param n number of elements in array
 * @param p pivot value
 */
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
