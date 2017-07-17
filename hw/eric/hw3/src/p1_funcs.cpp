#include "hw3.h"

/**
 * readFiles
 * For homework 3, problem 1. Read a matrix and vector from the given filenames.
 *
 * @param A reference to Matrix object to populate
 * @param x reference to ColVector object to populate
 * @param matrixFile path to matrix text file
 * @param vectorFile path to vector text file
 * @return number of elements read for the matrix.
 */
int readFiles(Matrix &A, ColVector &x, string matrixFile, string vectorFile)
{
    //cout << "Rank 0 reading files " << matrixFile << " and " << vectorFile << endl;
    A.readFromFile(matrixFile);
    x.readFromFile(vectorFile);

    if(A.getColumnCount()==x.getCount())
      return x.getCount();
}



/**
 * badVectorCheck
 * Check if vector size has been set. If not, clean up MPI.
 *
 * @param rank MPI process rank
 * @param size received vector size
 * @return boolean whether check passed or not
 */
bool badVectorCheck(int rank, int size)
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



/**
 * sendVector
 * Rank 0 broadcasts vector to all other processes, which receive and
 * store the given vector.
 *
 * @param rank MPI process rank
 * @param size received vector size
 * @param x ColVector object to populate with received vector
 */
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



/**
 * getNrowsForRank
 * Compute number of rows to be processed for an MPI rank given the total
 * number of processes and the total number of rows to process.
 *
 * @param rank MPI process rank
 * @param nProcs total number of MPI processes
 * @param totalRows total number of rows to process
 * @return number of rows that the given rank will process
 */
int getNrowsForRank(int rank, int nProcs, int totalRows)
{
  int nRows = totalRows/nProcs;
  nRows += (rank < (totalRows - nRows*nProcs));
  return nRows;
}



/**
 * sendMatrixRows
 * Split up the given Matrix and send the rows out to all of the MPI
 * processes with rank >=1. It is assumed that MPI rank 0 process will
 * call this method.
 *
 * @param nProcs number of MPI procs to send to
 * @param A matrix to split and send
 */
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
