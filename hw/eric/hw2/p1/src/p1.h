#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "deviceFunctions.h"

#define MIN(x,y) ((x<y)?x:y)
#define MAX(x,y) ((x>y)?x:y)
#define MAX_THREADS 1024

// p1a.cu declarations
typedef struct {
    int minVal;
} minResult;

__global__ void reduce_min_kernel(int * B, int * A, int n);
minResult find_min_seq(int* a, int n);
minResult find_min_cuda(int *a, int n);


// p1b.cu declarations
typedef struct {
    int* lastDigit;
} lastDigitResult;

__global__ void last_digit_kernel(int * d_out, int * d_in);
lastDigitResult last_digit_cuda(int *a, int n);
lastDigitResult last_digit_seq(int* a, int n);
