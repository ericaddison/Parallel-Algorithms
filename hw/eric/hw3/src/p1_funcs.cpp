#include "hw3.h"


int readFiles(Matrix &A, ColVector &x, string matrixFile, string vectorFile)
{
    //cout << "Rank 0 reading files " << matrixFile << " and " << vectorFile << endl;
    A.readFromFile(matrixFile);
    x.readFromFile(vectorFile);

    cout << "True answer b = \n";
    (A*x).print();
    cout << endl;

    if(A.getColumnCount()==x.getCount())
      return x.getCount();
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



void sendVector(int rank, int size, ColVector& x)
{
    // rank0 broadcast vector values
    int* vector = (rank==0) ? x.getValueBuffer() : new int[size];
    MPI_Bcast(vector, size, MPI_INT, 0, MPI_COMM_WORLD);

    // other procs create vector
    if(rank>0)
    {
      x.setValueBuffer(vector, size);
    }
}



int getNrowsForRank(int rank, int nProcs, int totalRows)
{
  int nRows = totalRows/nProcs;
  nRows += (rank < (totalRows - nRows*nProcs));
  return nRows;
}



void sendMatrixRows(int world_size, Matrix &A)
{
  int totalRows = A.m;
  A.m = getNrowsForRank(0, world_size, totalRows);
  int rowCnt = A.m;

  for(int iRank=1; iRank<world_size; iRank++)
  {
    int nRows = getNrowsForRank(iRank, world_size, totalRows);
    int* rows = A.getValueBuffer() + rowCnt*A.n;
    //cout << "sending " << nRows << " rows to rank " << iRank << "\n";
    MPI_Send(&nRows, 1, MPI_INT, iRank, 0, MPI_COMM_WORLD);
    MPI_Send(rows, nRows*A.n, MPI_INT, iRank, 0, MPI_COMM_WORLD);
    rowCnt += nRows;
  }
}



void receiveMatrixRows(int rank, Matrix &A)
{
  int nRows;
  MPI_Recv(&nRows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //cout << "Rank " << rank << " expecting " << A.m << " rows \n";
  int* matrix = new int[nRows*A.n];
  MPI_Recv(matrix, nRows*A.n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  A.setValueBuffer(matrix, nRows, A.n);
}



ColVector gatherResults(int world_size, int finalSize, ColVector &result, Matrix &A)
{
    ColVector b(finalSize);
    int cnt = 0;

    // copy in local results
    for(int i=0; i<result.getCount(); i++)
      b(cnt++) = result(i);

    // recieve worker results
    for(int iRank=1; iRank<world_size; iRank++)
    {
      int nRows = getNrowsForRank(iRank, world_size, finalSize);
      int* target = b.getValueBuffer() + cnt;
      MPI_Recv(target, nRows, MPI_INT, iRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      cnt += nRows;
    }

    return b;
}
