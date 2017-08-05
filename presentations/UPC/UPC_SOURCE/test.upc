#include <upc.h> 
#include <stdio.h>
int main()
{
	printf("Hello from %d of %d\n", MYTHREAD, THREADS);
}
