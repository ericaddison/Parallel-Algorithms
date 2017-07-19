#include "ParallelHyperQuickSorter.h"



/**
 * lastPow2
 * Compute the largest power of 2 less than the given value.
 *
 * @param n value
 * @return largest power of 2 less than n
 */
int lastPow2(int n)
{
  int lp2=1;
  while((n>>=1)>0)
    lp2<<=1;
  return lp2;
}



/**
 * ParallelHyperQuickSorter Ctor
 * Construct a new ParallelHyperQuickSorter using the given MPI_Comm as
 * the base comm object.
 *
 * @param comm initial MPI_Comm
 */
ParallelHyperQuickSorter::ParallelHyperQuickSorter(MPI_Comm comm)
{
  initComm = comm;
  MPI_Comm_rank(initComm, &world_rank);
  MPI_Comm_size(initComm, &world_size);
  nprocs = lastPow2(world_size);
  dim = log2(nprocs);
}



/**
 * getNrowsForRank
 * Compute the correct number of rows (i.e. values) to be processed by a given
 * MPI_rank.
 *
 * @param rank the rank of an MPI process
 * @param totalRows the total number of rows that will be distributed
 * @return the number of rows (values) that the given rank will process
 */
int ParallelHyperQuickSorter::getNrowsForRank(int rank, int totalRows)
{
  int nRows = totalRows/nprocs;
  nRows += (rank < (totalRows - nRows*nprocs));
  return nRows;
}



/**
 * sort
 * Sort the provided ColVector (in-place) with the ParallelHyperQuickSort
 * algorithm. A hypercube topology is used based on the MPI processes associated
 * with the MPI_Comm provided to the constructor.
 *
 * @param in pointer to the ColVector to be sorted.
 */
void ParallelHyperQuickSorter::sort(ColVector *in)
{

  // create new MPI_Comm for remaining procs
  int inCube = world_rank<nprocs;
  MPI_Comm_split(initComm, inCube, world_rank, &subCube_comm);

  // Dismiss procs greater than the last power of 2
  if(!inCube)
    return;

  // store pointer to vector and make a reference
  vec = in;
  ColVector &x = *vec;
  final_size = x.m;

  // rank0 send out vector segments
  if(world_rank==0)
    sendVectorSegments();
  else
    receiveVectorSegments();

  // each process sort locally
  quickSort(x);

  // loop over dimensions of hypercube
  int subcube = 0;
  int cube_size = nprocs;
  cube_rank = world_rank;
  for(int i=nprocs/2; i>=1; i/=2)
  {
    // define new subcube MPI_comm
    MPI_Comm oldComm = subCube_comm;
    MPI_Comm_split(oldComm, subcube, world_rank, &subCube_comm);
    MPI_Comm_rank(subCube_comm, &cube_rank);
    MPI_Comm_size(subCube_comm, &cube_size);
    MPI_Comm_free(&oldComm);

    // broadcast pivot from proc 0 to other procs in subcube
    int pivot = x(x.m/2);
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
  gatherResults();
}



/**
 * sendVectorSegments
 * Split up the ColVector stored in the object field "x" and send the segments
 * out to the other MPI processes. Meant to be called by rank 0.
 */
void ParallelHyperQuickSorter::sendVectorSegments()
{
  ColVector &x = *vec;

  x.m = getNrowsForRank(0, final_size);
  int valCnt = x.m;

  for(int iRank=1; iRank<nprocs; iRank++)
  {
    int nVals = getNrowsForRank(iRank, final_size);
    int* vals = x.getValueBuffer() + valCnt;
    MPI_Send(&nVals, 1, MPI_INT, iRank, 0, initComm);
    if(nVals>0)
    {
      MPI_Send(vals, nVals, MPI_INT, iRank, 0, initComm);
      valCnt += nVals;
    }
  }
}



/**
 * receiveVectorSegments
 * Partner to sendVectorSegments(). To be called by procs with rank>0. Receive
 * segments of the input vector from rank 0.
 */
void ParallelHyperQuickSorter::receiveVectorSegments()
{
  ColVector &x = *vec;

  int nVals;
  MPI_Recv(&nVals, 1, MPI_INT, 0, 0, initComm, MPI_STATUS_IGNORE);

  if(nVals>0)
  {
    int* vec = new int[nVals];
    MPI_Recv(vec, nVals, MPI_INT, 0, 0, initComm, MPI_STATUS_IGNORE);
    x.setValueBuffer(vec, nVals);
  }
}



/**
 * exchangeVectorSegments
 * Sort the provided ColVector (in-place) with the ParallelHyperQuickSort
 * algorithm. A hypercube topology is used based on the MPI processes associated
 * with the MPI_Comm provided to the constructor.
 *
 * @param dimensionBit power-of-2 int representing the dimension of the
 *        hypercube to exchange across
 * @param nLow the number of elements in the "low" side of the proc's array
 * @param nHi the number of elements in the "high" side of the proc's array
 */
void ParallelHyperQuickSorter::exchangeVectorSegments(int dimensionBit, int nLow, int nHi)
{
  ColVector &x = *vec;

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



/**
 * merge
 * Merge two sorted arrays together into one result array.
 *
 * @param result array to store merged result
 * @param in1 first input array
 * @param n1 number of elements in first input array
 * @param in2 second input array
 * @param n2 number of elements in second input array
 */
void ParallelHyperQuickSorter::merge(int *result, int *in1, int n1, int *in2, int n2)
{
  int c = n1+n2-1;
  int c1 = n1-1;
  int c2 = n2-1;
  while(c1>=0 && c2>=0)
    result[c--] = (in1[c1]>in2[c2]) ? in1[c1--] : in2[c2--];
  while(c1>=0)
    result[c--] = in1[c1--];
  while(c2>=0)
    result[c--] = in2[c2--];
}



/**
 * gatherResults
 * Gather results from all processes back to rank 0.
 */
void ParallelHyperQuickSorter::gatherResults()
{
  // procs outside of hypercube return
  if(world_rank>=nprocs)
    return;

  ColVector &x = *vec;

  // pass token around to write in correct order
  if(world_rank>0)
  {
    MPI_Send(&x.m, 1, MPI_INT, 0, 0, initComm);
    if(x.m>0)
      MPI_Send(x.getValueBuffer(), x.m, MPI_INT, 0, 0, initComm);
  }
  else
  {
    int valCnt = x.m;
    int *temp = new int[x.m];
    std::copy(x.getValueBuffer(), x.getValueBuffer()+valCnt, temp);
    x.setValueBuffer(new int[final_size], final_size);
    std::copy(temp, temp+valCnt, x.getValueBuffer());
    delete[] temp;

    for(int iRank=1; iRank<nprocs; iRank++)
    {
      int nVals;
      MPI_Recv(&nVals, 1, MPI_INT, iRank, 0, initComm, MPI_STATUS_IGNORE);
      if(nVals>0)
      {
        MPI_Recv(x.getValueBuffer()+valCnt, nVals, MPI_INT, iRank, 0, initComm, MPI_STATUS_IGNORE);
        valCnt += nVals;
      }
    }
  }

}
