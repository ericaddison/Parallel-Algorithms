#ifndef _P3_H
#define _P3_H

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "deviceFunctions.h"

#define MIN(x,y) ((x<y)?x:y)
#define MAX(x,y) ((x>y)?x:y)


__global__ void radix_sort_kernel(int *A, int n, int nDigits);
__global__ void parallel_merge_kernel(int *d_out, int *A, int treeLevel);


#endif
