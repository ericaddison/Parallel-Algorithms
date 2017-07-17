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
bool badVectorCheck(int rank, int size);
int getNrowsForRank(int rank, int nProcs, int totalRows);
int readFiles(Matrix &A, ColVector &x, string matrixFile, string vectorFile);
void sendVector(int rank, int size, ColVector& x);
void sendMatrixRows(int world_size, Matrix &A);
void receiveMatrixRows(int rank, Matrix &A);
ColVector gatherResults(int world_size, int finalSize, ColVector &result, Matrix &A);


#endif
