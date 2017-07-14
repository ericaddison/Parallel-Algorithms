#include "ParallelHyperQuickSorter.h"



int lastPow2(int n)
{
  int lp2=1;
  while((n>>=1)>0)
  {
    lp2<<=1;
  }
  return lp2;
}



ParallelHyperQuickSorter::ParallelHyperQuickSorter(MPI_Comm comm, ColVector &vec) : x(vec)
{
  initComm = comm;
  MPI_Comm_rank(initComm, &world_rank);
  MPI_Comm_size(initComm, &world_size);
  nprocs = lastPow2(world_size);
  dim = log2(nprocs);
}



int ParallelHyperQuickSorter::getNrowsForRank(int rank, int totalRows)
{
  int nRows = totalRows/nprocs;
  nRows += (rank < (totalRows - nRows*nprocs));
  return nRows;
}



void ParallelHyperQuickSorter::sendVectorSegments()
{
  int totalRows = x.m;
  x.m = getNrowsForRank(0, totalRows);
  int valCnt = x.m;

  for(int iRank=1; iRank<world_size; iRank++)
  {
    int nVals = getNrowsForRank(iRank, totalRows);
    int* vals = x.getValueBuffer() + valCnt;
    //cout << "sending " << nVals << " values to rank " << iRank << "\n";
    MPI_Send(&nVals, 1, MPI_INT, iRank, 0, initComm);
    if(nVals>0)
    {
      MPI_Send(vals, nVals, MPI_INT, iRank, 0, initComm);
      valCnt += nVals;
    }
  }
}



void ParallelHyperQuickSorter::receiveVectorSegments()
{
  int nVals;
  MPI_Recv(&nVals, 1, MPI_INT, 0, 0, initComm, MPI_STATUS_IGNORE);
  //cout << "Rnk " << rank << " expecting " << nVals << " rows \n";
  if(nVals>0)
  {
    int* vec = new int[nVals];
    MPI_Recv(vec, nVals, MPI_INT, 0, 0, initComm, MPI_STATUS_IGNORE);
    x.setValueBuffer(vec, nVals);
  }
}


void ParallelHyperQuickSorter::exchangeVectorSegments(int dimensionBit, int nLow, int nHi)
{
  // each proc exchange low/hi arrays with partner in the ith direction
  int partnerRank = cube_rank^dimensionBit;
  int sendSize;
  int recvSize;
  int newSize;
  int * newVals;

  if(cube_rank<dimensionBit) // lower half of cube, send first receive second
  {
    // exchange size info
    sendSize = nHi;   // number of elements in upper part of partitioned
    MPI_Send(&sendSize, 1, MPI_INT, partnerRank, 0, subCube_comm);
    MPI_Recv(&recvSize, 1, MPI_INT, partnerRank, 0, subCube_comm, MPI_STATUS_IGNORE);

    // exchange data
    newSize = recvSize+nLow;
    newVals = new int[newSize];
    MPI_Send(x.getValueBuffer()+nLow, sendSize, MPI_INT, partnerRank, 0, subCube_comm);
    MPI_Recv(newVals, recvSize, MPI_INT, partnerRank, 0, subCube_comm, MPI_STATUS_IGNORE);
    merge(newVals, newVals, recvSize, x.getValueBuffer(), nLow);
  }
  else
  {
    // exchange size info
    sendSize = nLow;
    MPI_Recv(&recvSize, 1, MPI_INT, partnerRank, 0, subCube_comm, MPI_STATUS_IGNORE);
    MPI_Send(&sendSize, 1, MPI_INT, partnerRank, 0, subCube_comm);

    // exchange data
    newSize = recvSize+nHi;
    newVals = new int[newSize];
    MPI_Recv(newVals, recvSize, MPI_INT, partnerRank, 0, subCube_comm, MPI_STATUS_IGNORE);
    MPI_Send(x.getValueBuffer(), sendSize, MPI_INT, partnerRank, 0, subCube_comm);
    merge(newVals, newVals, recvSize, x.getValueBuffer()+nLow, nHi);
  }

  x.setValueBuffer(newVals, newSize);

}



void ParallelHyperQuickSorter::sort()
{
  // HyperQuickSort uses a hypercube. Dismiss processes above a power of 2
  int nprocs = lastPow2(world_size);
  int ndims = log2(nprocs);

  // Dismiss procs greater than the last power of 2
  int inCube = world_rank<nprocs;
  MPI_Comm_split(initComm, inCube, world_rank, &subCube_comm);

  if(!inCube)
    return;

  // rank0 send out vector segments
  if(world_rank==0)
    sendVectorSegments();
  else
    receiveVectorSegments();

  // each process sort locally
  quickSort(x.getValueBuffer(), x.getCount());

  // loop over dimensions of hypercube
  int subcube = 0;
  cube_rank = world_rank;
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
    exchangeVectorSegments(i, nLow, nHi);

    // update subcube number
    subcube = 2*subcube + (cube_rank>=i);   // identify which subcube this process belongs to
    MPI_Barrier(subCube_comm);
  }

  MPI_Comm_free(&subCube_comm);
}



// merge the sorted arrays in1 and in2 into result
void ParallelHyperQuickSorter::merge(int *result, int *in1, int n1, int *in2, int n2)
{
  int c = n1+n2-1;
  int c1 = n1-1;
  int c2 = n2-1;
  while(c1>=0 && c2>=0)
  {
    if(in1[c1]>in2[c2])
      result[c--] = in1[c1--];
    else
      result[c--] = in2[c2--];
  }
  while(c1>=0)
    result[c--] = in1[c1--];
  while(c2>=0)
    result[c--] = in2[c2--];
}



void ParallelHyperQuickSorter::writeSortedArrayToFile(const string filename)
{

  if(world_rank>=nprocs)
    return;

  int token = (world_rank==0);
  if(world_rank)
    MPI_Recv(&token, 1, MPI_INT, world_rank-1, 0, initComm, MPI_STATUS_IGNORE);

  std::ios_base::openmode appFlag = (world_rank) ? (ofstream::app) : ofstream::trunc;
  ofstream outFile(filename.c_str(), ofstream::out | appFlag);
  x.printLinear(outFile);
  outFile.close();
  if(world_rank<(nprocs-1))
    MPI_Send(&token, 1, MPI_INT, world_rank+1, 0, initComm);
}
