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
int lastPow2(int n);
bool dimensionCheck(int rank, int size);
int getNrowsForRank(int rank, int nProcs, int totalRows);
void sendVectorSegments(int world_size, ColVector &x);
void receiveVectorSegments(int rank, ColVector &x);
void quickSort(int *A, int n);

#endif
