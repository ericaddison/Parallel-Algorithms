#include "hw3.h"

int lastPow2(int n)
{
  int lp2=1;
  while((n>>=1)>0)
  {
    lp2<<=1;
  }
  return lp2;
}

bool dimensionCheck(int rank, int size)
{
  if(size==-1)
  {
    if(rank==0)
      cerr << "Error! dimension mismatch!\n";
    MPI_Finalize();
    return false;
  }
  return true;
}



int getNrowsForRank(int rank, int nProcs, int totalRows)
{
  int nRows = totalRows/nProcs;
  nRows += (rank < (totalRows - nRows*nProcs));
  return nRows;
}



void sendVectorSegments(int world_size, ColVector &x)
{
  int totalRows = x.m;
  x.m = getNrowsForRank(0, world_size, totalRows);
  int valCnt = x.m;

  for(int iRank=1; iRank<world_size; iRank++)
  {
    int nVals = getNrowsForRank(iRank, world_size, totalRows);
    int* vals = x.getValueBuffer() + valCnt;
    //cout << "sending " << nRows << " rows to rank " << iRank << "\n";
    MPI_Send(&nVals, 1, MPI_INT, iRank, 0, MPI_COMM_WORLD);
    MPI_Send(vals, nVals, MPI_INT, iRank, 0, MPI_COMM_WORLD);
    valCnt += nVals;
  }
}



void receiveVectorSegments(int rank, ColVector &x)
{
  int nVals;
  MPI_Recv(&nVals, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //cout << "Rank " << rank << " expecting " << A.m << " rows \n";
  int* vec = new int[nVals];
  MPI_Recv(vec, nVals, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  x.setValueBuffer(vec, nVals);
}

void swap(int *A, int i, int j)
{
  int temp = A[i];
  A[i] = A[j];
  A[j] = temp;
}

void quickSort(int *A, int n)
{

  // base case
  if(n==1)
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

// merge the sorted arrays in1 and in2 into result
void merge(int *result, int *in1, int n1, int *in2, int n2)
{
  int c = n1+n2-1;
  int c1 = n1-1;
  int c2 = n2-1;
  while(c1>=0 && c2>=0)
  {
    if(in1[c1]>in2[c2])
      result[c--] = in1[c1--];
    else
      result[c--] = in2[c2--];
  }
  while(c1>=0)
    result[c--] = in1[c1--];
  while(c2>=0)
    result[c--] = in2[c2--];
}
