#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "deviceFunctions.h"
#define MIN(x,y) ((x<y)?x:y)
#define MAX(x,y) ((x>y)?x:y)


// p2a.cu functions
__global__ void range_count_kernel_global(int * count, int * B, int * A, int n);
__global__ void reduce_add_kernel_global(int * B, int * A, int n);


// p2b.cu functions
__global__ void range_count_kernel_shared(int * count, int * A, int n);
__global__ void reduce_add_kernel_shared(int * B, int * A, int n);


// p2c.cu functions
__global__ void hs_scan_kernel(int *A, int n);

