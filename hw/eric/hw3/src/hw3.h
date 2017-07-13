#ifndef _HW3
#define _HW3

#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include "matrix.h"

using std::cerr;
using std::cout;
using std::endl;

// p1_funcs.cpp
bool dimensionCheck(int rank, int size);
int getNrowsForRank(int rank, int nProcs, int totalRows);


// p2_funcs
int lastPow2(int n);
void sendVectorSegments(int world_size, ColVector &x);
void receiveVectorSegments(int rank, ColVector &x);
void quickSort(int *A, int n);
int partition(int *A, int n, int p);
void merge(int *result, int *in1, int n1, int *in2, int n2);
void parallelHyperQuickSort(MPI_Comm subCube_comm, int world_rank, int nprocs, ColVector x);
void writeSortedArrayToFile(ColVector &x, int nprocs, const string filename);
void exchangeVectorSegments(MPI_Comm subCube_comm, int bitmask, int nLow, int nHi, ColVector &x);
#endif
