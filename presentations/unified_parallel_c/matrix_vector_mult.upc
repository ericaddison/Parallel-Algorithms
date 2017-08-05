// matrix_vector_mult.upc
#include <stdio.h>
#include <upc.h>

#define ROWS 3

shared int a[ROWS][THREADS];
shared int b[THREADS];
shared int c[THREADS];

void initArrays();
int printResult();

int main (void) {
	initArrays();
	upc_barrier;
	
	upc_forall(int i = 0 ; i < THREADS ; i++; i)
	{
		c[i] = 0;
		for (int j= 0 ; j < THREADS ; j++)
			c[i] += a[i][j]*b[j];
	}
	
	upc_barrier;
	
	if (MYTHREAD == 0)
		printResult();

	return 0;
}

void initArrays()
{
	if (MYTHREAD == 0){
		for (int i = 0; i < ROWS; i++){
			for (int j = 0; j < THREADS; j++){
				a[i][j] = 1;
			}
		}
		
		for (int i = 0; i < THREADS; i++){
			b[i] = 1;
		}
	}
}

int printResult()
{
	for (int i = 0; i < ROWS; i++){
			printf("%d ", c[i]);
		}
		
	printf("\n");
}
