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
  int dismiss = world_rank>=nprocs;
  MPI_Comm subCube_comm;
  MPI_Comm_split(MPI_COMM_WORLD, dismiss, world_rank, &subCube_comm);
  if(dismiss)
  {
    MPI_Finalize();
    return 0;
  }

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
  if(world_rank==0)
    x.readFromFile(vectorFile);


  // rank0 send out vector segments
  if(world_rank==0)
    sendVectorSegments(nprocs, x);
  else
    receiveVectorSegments(world_rank, x);


  // each process sort locally
  quickSort(x.getValueBuffer(), x.getCount());

  // Parallel HyperQuickSort

  // loop over dimensions
  int subcube = 0;
  int cube_rank = world_rank;
  int cube_size = nprocs;
  int pivot;
  for(int i=nprocs/2; i>=1; i/=2)
  {
    // define new subcube MPI_comm
    MPI_Comm oldComm = subCube_comm;
    MPI_Comm_split(oldComm, subcube, world_rank, &subCube_comm);
    MPI_Comm_rank(subCube_comm, &cube_rank);
    MPI_Comm_size(subCube_comm, &cube_size);
    MPI_Comm_free(&oldComm);

    // proc 0 pick pivot (median)
    if(cube_rank==0)
      pivot = x(x.m/2);

    // broadcast pivot to other procs in subcube
    MPI_Bcast(&pivot, 1, MPI_INT, 0, subCube_comm);

    // each proc split sorted array in two: larger and smaller than pivot
    int nLow = partition(x.getValueBuffer(), x.getCount(), pivot);
    int nHi = x.getCount()-nLow;

    // exchange vector segment with partner in ith direction
    exchangeVectorSegments(subCube_comm, i, nLow, nHi,  x);

    // update subcube number
    subcube = 2*subcube + (cube_rank>=i);   // identify which subcube this process belongs to
    MPI_Barrier(subCube_comm);
  }

  MPI_Comm_free(&subCube_comm);

  // write to file, one proc at a time
  writeSortedArrayToFile(x, nprocs, "sortedArray.txt");

  // cleanup
  MPI_Finalize();

}
