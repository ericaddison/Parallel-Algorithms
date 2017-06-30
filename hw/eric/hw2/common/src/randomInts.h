#ifndef _RANDINTS
#define _RANDINTS

#include "hw.h"

#define NDIGITS 3

typedef struct
{
    int n;
    int *A;
} randArray;


randArray readIntsFromFile(const char* filename);
void writeRandomFile(int n, const char* filename);

#endif
