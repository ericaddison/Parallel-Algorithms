#include <iostream>
#include <mpi.h>
#include "matrix.h"

using std::cerr;
using std::cout;
using std::endl;

#define MPI_VECSIZE_TAG 1001
#define MPI_VECTOR_TAG 1002

int readFiles(int rank, Matrix &A, ColVector &x, string matrixFile, string vectorFile)
{
  if(rank==0)
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
  return -1;
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

void sendMatrixRows(int rank, int world_size, Matrix &A)
{
  int nRowsMax = A.n/world_size+1;
  int nRows = nRowsMax - (rank >= nRowsMax*world_size - A.n);
  A.m = nRows;

  if(rank==0)
  {
    int rowCnt = nRows;
    for(int iRank=1; iRank<world_size; iRank++)
    {
      int newNrows = nRowsMax - (iRank >= nRowsMax*world_size - A.n);
      int* rows = A.getValueBuffer() + rowCnt*A.getColumnCount();
      //cout << "sending " << newNrows << " rows => " << newNrows*A.getColumnCount() << " ints\n";
      MPI_Send(rows, newNrows*A.getColumnCount(), MPI_INT, iRank, 0, MPI_COMM_WORLD);
      rowCnt += newNrows;
    }
  }
  else
  {
    int* matrix = new int[nRows*A.n];
    MPI_Recv(matrix, nRows*A.n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    A.setValueBuffer(matrix);
  }
  //cout << "Rank " << rank << " will process " << nRows << " rows\n";

}


ColVector gatherResults(int rank, int world_size, int finalSize, ColVector &result, Matrix &A)
{
  if(rank==0)
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
        int newNrows = nRowsMax - (iRank >= nRowsMax*world_size - A.n);
        int* target = b.getValueBuffer() + cnt;
        MPI_Recv(target, newNrows, MPI_INT, iRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt += newNrows;
      }

      return b;
  }
  else
  {
    MPI_Send(result.getValueBuffer(), result.getCount(), MPI_INT, 0, 0, MPI_COMM_WORLD);
    return ColVector(0);
  }
}

int main(int argc, char** argv)
{

  // MPI init
  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // check command line args
  if(argc!=3)
  {
    if(world_rank==0)
      cerr << "\nUsage: " << argv[0] << " <matrix file> <vector file>\n\n";
    MPI_Finalize();
    return 0;
  }

  // get command line args
  string matrixFile = string(argv[1]);
  string vectorFile = string(argv[2]);

  // rank0 read files and broadcast vector size
  Matrix A(0,0);
  ColVector x(0);
  int vecSize = readFiles(world_rank, A, x, matrixFile, vectorFile);
  int finalSize = A.m;
  MPI_Bcast(&vecSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  A.n = vecSize;
  //cout << "Rank " << world_rank << " received vecsize " << vecSize << endl;

  // dimension mismatch check
  if(!dimensionCheck(world_rank, vecSize))
    return 0;

  // send the vector
  sendVector(world_rank, vecSize, x);

  // rank0 send out matrix rows
  sendMatrixRows(world_rank, world_size, A);

  // process matrix rows
  ColVector result = A*x;

  // send results back to rank0
  ColVector b = gatherResults(world_rank, world_size, finalSize, result, A);

  // output result to file
  if(world_rank==0)
    b.writeToFile("p1Result");

  // cleanup
  MPI_Finalize();

}
