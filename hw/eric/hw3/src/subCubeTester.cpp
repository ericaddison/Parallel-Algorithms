#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cstdlib>

using std::cout;

int lastPow2(int n)
{
  int lp2=1;
  while((n>>=1)>0)
  {
    lp2<<=1;
  }
  return lp2;
}

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

  // loop over dimensions
  int subcube = 0;
  int cube_rank;
  int cube_size;
  MPI_Comm subCube_comm = fullCube_comm;

  if(world_rank<nprocs)
  {
    for(int i=nprocs/2; i>=1; i/=2)
    {
      // define new subcube MPI_comm
      MPI_Comm_split(subCube_comm, subcube, world_rank, &subCube_comm);
      MPI_Comm_rank(subCube_comm, &cube_rank);
      MPI_Comm_size(subCube_comm, &cube_size);

      if(world_rank==printInd)
        cout << "subCube " << subcube << ": world_rank/world_size: " << world_rank << "/" << world_size << "\tcube_rank/cube_size: " << cube_rank << "/" << cube_size << "\n";

      // update subcube number
      subcube = 2*subcube + (cube_rank>=i);   // identify which subcube this process belongs to
      MPI_Barrier(subCube_comm);
    }
  }

  // cleanup
  MPI_Finalize();

}
