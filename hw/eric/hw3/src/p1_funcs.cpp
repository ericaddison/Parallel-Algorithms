#include "hw3.h"


int readFiles(Matrix &A, ColVector &x, string matrixFile, string vectorFile)
{
    //cout << "Rank 0 reading files " << matrixFile << " and " << vectorFile << endl;
    A.readFromFile(matrixFile);
    x.readFromFile(vectorFile, true);

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
      x.m = size;
      x.setValueBuffer(vector);
    }
}



void sendMatrixRows(int world_size, Matrix &A)
{
  int nRowsMax = A.m/world_size+1;
  int rowCnt = nRowsMax;

  for(int iRank=1; iRank<world_size; iRank++)
  {
    int newNrows = nRowsMax - (iRank >= nRowsMax*world_size - A.m);
    int* rows = A.getValueBuffer() + rowCnt*A.n;
    //cout << "sending " << newNrows << " rows => " << newNrows*A.getColumnCount() << " ints\n";
    MPI_Send(&newNrows, 1, MPI_INT, iRank, 0, MPI_COMM_WORLD);
    MPI_Send(rows, newNrows*A.n, MPI_INT, iRank, 0, MPI_COMM_WORLD);
    rowCnt += newNrows;
  }
  A.m = nRowsMax;
}



void receiveMatrixRows(int rank, Matrix &A)
{
  int nRows;
  MPI_Recv(&nRows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  A.m = nRows;
  //cout << "Rank " << rank << " expecting " << A.m << " rows \n";
  int* matrix = new int[nRows*A.n];
  MPI_Recv(matrix, nRows*A.n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  A.setValueBuffer(matrix);
}



ColVector gatherResults(int world_size, int finalSize, ColVector &result, Matrix &A)
{
    ColVector b(finalSize);
    int cnt = 0;

    // copy in local results
    for(int i=0; i<result.getCount(); i++)
      b(cnt++) = result(i);

    // recieve worker results
    int nRowsMax = A.n/world_size+1;
    for(int iRank=1; iRank<world_size; iRank++)
    {
      int newNrows = nRowsMax - (iRank >= nRowsMax*world_size - finalSize);
      int* target = b.getValueBuffer() + cnt;
      MPI_Recv(target, newNrows, MPI_INT, iRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      cnt += newNrows;
    }

    return b;
}
