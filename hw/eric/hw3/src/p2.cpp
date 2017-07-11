#include "hw3.h"

int main(int argc, char** argv)
{

  // MPI init
  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // HyperQuickSort uses a hypercube. Dismiss processes above a power of 2
  int nprocs = lastPow2(world_size);
  int ndims = log2(nprocs);

  // Dismiss procs greater than the last power of 2
  int color = world_rank>=nprocs;
  MPI_Comm fullCube_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &fullCube_comm);
  if(color)
  {
    MPI_Finalize();
    return 0;
  }

  int printInd = 0;
  if(argc>=2)
  {
    printInd = atoi(argv[1]);
  }

/*
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
    vecSize = x.getCount();
  }
  int finalSize = x.getCount;

  // dimension mismatch check
  if(!dimensionCheck(world_rank, vecSize))
    return 0;

  // rank0 send out vector segments
  if(world_rank==0)
    sendVectorSegments(nprocs, x);
  else
    receiveVectorSegments(world_rank, x);

  // each process sort locally
  quickSort(x.getValueBuffer(), x.getCount());
*/
  // Parallel HyperQuickSort

  // loop over dimensions
  int subcube = 0;
  MPI_Comm subCube_comm = fullCube_comm;
  int cube_rank;
  int cube_size;
  int pivot;
  if(world_rank<nprocs)
  {
    for(int i=nprocs/2; i>=1; i/=2)
    {
      // define new subcube MPI_comm
      MPI_Comm_split(subCube_comm, subcube, world_rank, &subCube_comm);
      MPI_Comm_rank(subCube_comm, &cube_rank);
      MPI_Comm_size(subCube_comm, &cube_size);
      if(world_rank==printInd)
        cout << "subCube: " << subcube << ": world_rank/world_size: " << world_rank << "/" << world_size << "\tcube_rank/cube_size: " << cube_rank << "/" << cube_size << "\n";

      // proc 0 pick pivot (median)
      if(cube_rank==0)
        pivot = 0;

      // broadcast median to other procs in subcube
      //MPI_Bcast();

      // each proc split sorted array in two: larger and smaller than pivot


      // each proc exchange low/hi arrays with partner in the ith direction


      // each proc merge two sorted lists


      // update subcube number
      subcube = 2*subcube + (cube_rank>=i);   // identify which subcube this process belongs to
      MPI_Barrier(subCube_comm);

      // repeat!
    }
  }
  // pass a token for writing to file
    // if !rank0
      // MPI_recv wait for token
    // write to file
    // if !rank(N-1)
      // send token to next rank



  // cleanup
  MPI_Finalize();

}
