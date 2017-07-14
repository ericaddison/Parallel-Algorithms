#ifndef _PHQS
#define _PHQS

#include <mpi.h>
#include <fstream>
#include "hw3.h"

class ParallelHyperQuickSorter
{

  public:
    ParallelHyperQuickSorter(MPI_Comm comm, ColVector &vec);
    void writeSortedArrayToFile(const string filename);
    void sort();

  private:
    ColVector &x;
    MPI_Comm initComm;
    MPI_Comm subCube_comm;
    int dim;
    int nprocs;
    int world_size;
    int world_rank;
    int cube_rank;

    int getNrowsForRank(int rank, int totalRows);
    void sendVectorSegments();
    void receiveVectorSegments();
    void merge(int *result, int *in1, int n1, int *in2, int n2);
    void exchangeVectorSegments(int dimensionBit, int nLow, int nHi);

};

#endif
