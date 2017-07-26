// vect_add.upc
#include <stdio.h>
#include <upc.h>

#define N 100

shared int v1[N], v2[N], vResult[N];

void initVectors();
int printResult();

int main (void) {
	initArrays();
	upc_barrier;
	
	upc_forall(int i = 0 ; i < N; i++; i){
		vResult[i] = v1[i] + v2[i];
	}
	
	upc_barrier;	
	printResult();

	return 0;
}

void initVectors()
{
	if (MYTHREAD == 0){		
		for (int i = 0; i < N; i++){
			v1[i] = i;
			v2[i] = i;
		}
	}
}

int printResult()
{
	if (MYTHREAD == 0){
		for (int i = 0; i < N; i++){
				printf("%d ", vResult[i]);
			}
		
		printf("\n");
	}
}
