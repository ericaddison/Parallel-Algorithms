#include <stdio.h>
#include <upc.h>

int main()
{
	printf("Hello world: this is thread %d of %d\n", MYTHREAD, THREADS);
	return 0;
}
