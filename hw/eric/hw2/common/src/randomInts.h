#ifndef _RANDINTS
#define _RANDINTS

#include "hw.h"

#define NDIGITS 3

void readIntsFromFile(const char* filename, int n, int* array);
void writeRandomFile(int n, const char* filename);

#endif
