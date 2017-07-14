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


// quicksort.cpp
void quickSort(int *A, int n);
int partition(int *A, int n, int p);

#endif
