#include "hw3.h"



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
  x.m = nVals;
  //cout << "Rank " << rank << " expecting " << A.m << " rows \n";
  int* vec = new int[nVals];
  MPI_Recv(vec, nVals, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  x.setValueBuffer(vec);
}

oid swap(int *A, int i, int j)
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
