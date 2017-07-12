#include "hw3.h"

int main(int argc, char** argv)
{

  // MPI init
  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  ofstream fout;
  char filename[30];
  sprintf(filename,"logs/log_rank_%d.log",world_rank);
  fout.open(filename);

  // HyperQuickSort uses a hypercube. Dismiss processes above a power of 2
  int nprocs = lastPow2(world_size);
  int ndims = log2(nprocs);

  // Dismiss procs greater than the last power of 2
  int color = world_rank>=nprocs;
  MPI_Comm subCube_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &subCube_comm);
  if(color)
  {
    fout << "Rank " << world_rank << " dismissed. Finalizing.\n";
    fout.close();
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
  {
    x.readFromFile(vectorFile);
  }
  int finalSize = x.getCount();

  fout << "rank " << world_rank << " has finalSize " << finalSize << endl;


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
  if(world_rank<nprocs)
  {
    //int i = nprocs/2;
    //for(int i=nprocs/2; i>=nprocs/4; i/=2)
    for(int i=nprocs/2; i>=1; i/=2)
    {
      fout << "-------------------------------------\n";
      fout << "Node bitmask i = " << i << "\n\n";
      fout << "subcube = " << subcube << "\n";
      fout << "My intital array:\n";
      x.print(fout);
      fout << "\n";
      // define new subcube MPI_comm
      MPI_Comm oldComm = subCube_comm;
      MPI_Comm_split(oldComm, subcube, world_rank, &subCube_comm);
      MPI_Comm_rank(subCube_comm, &cube_rank);
      MPI_Comm_size(subCube_comm, &cube_size);
      MPI_Comm_free(&oldComm);

      fout << "new subCube_comm has size " << cube_size << endl;

      // proc 0 pick pivot (median)
      if(cube_rank==0)
      {
        pivot = x(x.m/2);
        fout << "subcube " << subcube << " choosing pivot at x(" << (x.m/2) << ") = " << pivot << endl;
      }

      // broadcast median to other procs in subcube
      MPI_Bcast(&pivot, 1, MPI_INT, 0, subCube_comm);
      if(cube_rank>0)
      {
        fout << "(" << subcube << ", " << cube_rank << ") received pivot " << pivot << endl;
      }

      // each proc split sorted array in two: larger and smaller than pivot
      int nLow = partition(x.getValueBuffer(), x.getCount(), pivot);
      int nHi = x.getCount()-nLow;

      fout << "(" << subcube << ", " << cube_rank << ") found split index " << nLow << endl;

      // each proc exchange low/hi arrays with partner in the ith direction
      int partnerRank = cube_rank^i;
      int sendSize;
      int recvSize;
      int newSize;
      int * newVals;

      fout << "(" << subcube << ", " << cube_rank << ") has partnerRank " << partnerRank << endl;

      if(cube_rank<i) // lower half of cube, send first receive second
      {
        // exchange size info
        sendSize = nHi;   // number of elements in upper part of partitioned
        MPI_Send(&sendSize, 1, MPI_INT, partnerRank, 0, subCube_comm);
        MPI_Recv(&recvSize, 1, MPI_INT, partnerRank, 0, subCube_comm, MPI_STATUS_IGNORE);
        fout << "(" << subcube << ", " << cube_rank << ")" << " sent value " << sendSize << " and received value " << recvSize << "  **partner=" << partnerRank << endl;

        // exchange data
        newSize = recvSize+nLow;
        newVals = new int[newSize];
        fout << "Sending high side...\n";
        MPI_Send(x.getValueBuffer()+nLow, sendSize, MPI_INT, partnerRank, 0, subCube_comm);
        fout << "Receiving low side...\n";
        MPI_Recv(newVals, recvSize, MPI_INT, partnerRank, 0, subCube_comm, MPI_STATUS_IGNORE);
        fout << "Attempting to merge...\n";
        merge(newVals, newVals, recvSize, x.getValueBuffer(), nLow);
      }
      else
      {
        // exchange size info
        sendSize = nLow;
        MPI_Recv(&recvSize, 1, MPI_INT, partnerRank, 0, subCube_comm, MPI_STATUS_IGNORE);
        MPI_Send(&sendSize, 1, MPI_INT, partnerRank, 0, subCube_comm);
        fout << "(" << subcube << ", " << cube_rank << ")" << " sent value " << sendSize << " and received value " << recvSize << "  **partner=" << partnerRank << endl;

        // exchange data
        newSize = recvSize+nHi;
        newVals = new int[newSize];
        fout << "Receiving high side...\n";
        MPI_Recv(newVals, recvSize, MPI_INT, partnerRank, 0, subCube_comm, MPI_STATUS_IGNORE);
        fout << "Sending low side...\n";
        MPI_Send(x.getValueBuffer(), sendSize, MPI_INT, partnerRank, 0, subCube_comm);
        fout << "Attempting to merge...\n";
        merge(newVals, newVals, recvSize, x.getValueBuffer()+nLow, nHi);
      }

      fout << "Setting new buffer with size " << newSize << endl;
      x.setValueBuffer(newVals, newSize);

      fout << "(" << subcube << ", " << cube_rank << ") got array\n";
      x.print(fout);
      fout << endl;

      // update subcube number
      subcube = 2*subcube + (cube_rank>=i);   // identify which subcube this process belongs to
      MPI_Barrier(subCube_comm);
      fout << "done with iter\n\n\n";
    }
  }

  MPI_Comm_free(&subCube_comm);

  // write to file, one proc at a time
  int token = (world_rank==0);
  if(world_rank)
    MPI_Recv(&token, 1, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::ios_base::openmode appFlag = (world_rank) ? (ofstream::app) : ofstream::trunc;
  ofstream outFile("sortedArray.txt", ofstream::out | appFlag);
  x.printLinear(outFile);
  outFile.close();
  if(world_rank<(nprocs-1))
    MPI_Send(&token, 1, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD);


  // cleanup
  MPI_Finalize();
  fout.close();

}
