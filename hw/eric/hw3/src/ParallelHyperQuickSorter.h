#ifndef _PHQS
#define _PHQS

#include <mpi.h>
#include <fstream>
#include "quickSort.h"
#include "hw3.h"


class ParallelHyperQuickSorter
{

  public:
    ParallelHyperQuickSorter(MPI_Comm comm);
    void sort(ColVector *vec);

  private:
    ColVector *vec;
    MPI_Comm initComm;
    MPI_Comm subCube_comm;
    int dim;
    int nprocs;
    int world_size;
    int world_rank;
    int cube_rank;
    int final_size;

    int getNrowsForRank(int rank, int totalRows);
    void sendVectorSegments();
    void receiveVectorSegments();
    void merge(int *result, int *in1, int n1, int *in2, int n2);
    void exchangeVectorSegments(int dimensionBit, int nLow, int nHi);
    void gatherResults();

};

#endif
