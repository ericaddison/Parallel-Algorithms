#include "hw3.h"

int main(int argc, char** argv)
{

  // MPI init
  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // check command line args
  if(argc!=2)
  {
    if(world_rank==0)
      cerr << "\nUsage: " << argv[0] << " <input file>\n\n";
    MPI_Finalize();
    return 0;
  }

  // get command line args
  string vectorFile(argv[1]);

  // rank0 read files and broadcast vector size
  ColVector x(0);
  int vecSize = -1;
  if(world_rank==0)
  {
    x.readFromFile(vectorFile);
    vecSize = x.m;
  }
  int finalSize = A.m;
  MPI_Bcast(&vecSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  A.n = vecSize;

  // dimension mismatch check
  if(!dimensionCheck(world_rank, vecSize))
    return 0;

  // rank0 send out vector segments
  if(world_rank==0)
    sendVectorSegments(world_size, x);
  else
    receiveVectorSegments(world_rank, x);

  // Parallel HyperQuickSort

    // local recursive sort

    // parallel merge with partner


  // pass a token for writing to file
    // if !rank0
      // MPI_recv wait for token
    // write to file
    // if !rank(N-1)
      // send token to next rank



  // cleanup
  MPI_Finalize();

}
