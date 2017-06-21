#ifndef _RANDINTS
#define _RANDINTS

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NDIGITS 3

void readIntsFromFile(const char* filename, int n, int* array);
void writeRandomFile(int n, const char* filename);

#endif
