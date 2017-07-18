#include "hw3.h"


/**
 * main
 * Main function for HW3, problem 1. Performs matrix-vector multiplication
 * of a matrix and a vector read in from text files specified as command line
 * arguments. Problem is implemented as one procedural main function with
 * helper functions added for abstraction and readibility.
 */
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
  string matrixFile(argv[1]);
  string vectorFile(argv[2]);

  // rank0 read files and broadcast vector size
  Matrix A(0,0);
  ColVector x(0);
  int vecSize = -1;
  if(world_rank==0)
  {
    vecSize = readFiles(A, x, matrixFile, vectorFile);
    cout << "True answer b = \n";
    (A*x).print();
    cout << endl;
  }
  int finalSize = A.m;
  MPI_Bcast(&vecSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  A.n = vecSize;

  // dimension mismatch check
  if(!badVectorCheck(world_rank, vecSize))
    return 0;

  // send the vector
  sendVector(world_rank, vecSize, x);

  // rank0 send out matrix rows
  if(world_rank==0)
    sendMatrixRows(world_size, A);
  else
    receiveMatrixRows(world_rank, A);

  // process matrix rows
  ColVector result = A*x;

  // send results back to rank0 and write to file
  if(world_rank==0)
  {
    ColVector b = gatherResults(world_size, finalSize, result);
    b.writeToFile("p1Result");
  }
  else
    MPI_Send(result.getValueBuffer(), result.getCount(), MPI_INT, 0, 0, MPI_COMM_WORLD);

  // cleanup
  MPI_Finalize();

}
