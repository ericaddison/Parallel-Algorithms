#ifndef _HW3
#define _HW3

#include <iostream>
#include <mpi.h>
#include "matrix.h"

using std::cerr;
using std::cout;
using std::endl;

// p1_funcs.cpp
int readFiles(int, Matrix&, ColVector&, string, string);
bool dimensionCheck(int, int);
void sendVector(int, int, ColVector&);
void sendMatrixRows(int, int, Matrix&);
void receiveMatrixRows(int, Matrix&);
ColVector gatherResults(int, int, int, ColVector&, Matrix&);

#endif
